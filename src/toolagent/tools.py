"""Declaring tools as typed Python functions and running them safely."""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from toolagent.schema import function_schema, validate

MAX_RESULT_CHARS = 20_000


class ToolError(Exception):
    """Raise from a tool to return a clean error message to the model (no traceback)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    requires_approval: bool = False
    timeout_s: float = 30.0

    def definition(self, *, strict: bool = True) -> dict[str, Any]:
        """The tool definition sent to the Messages API."""
        spec = {"name": self.name, "description": self.description, "input_schema": self.input_schema}
        if strict:
            spec["strict"] = True
        return spec


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    requires_approval: bool = False,
    timeout_s: float = 30.0,
):
    """Turn a typed, documented function into a :class:`Tool`.

    The docstring summary becomes the tool description and the ``Args:`` section documents each
    parameter, so the code and what the model reads can't drift apart::

        @tool
        def add(a: int, b: int) -> int:
            '''Add two integers.

            Args:
                a: First addend.
                b: Second addend.
            '''
            return a + b
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        description, schema = function_schema(f)
        if not description:
            raise ValueError(f"Tool {f.__name__!r} needs a docstring: it is the model's documentation")
        return Tool(name or f.__name__, description, schema, f, requires_approval, timeout_s)

    return wrap(fn) if fn is not None else wrap


@dataclass
class ToolOutcome:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool
    duration_ms: float
    approved: bool = True


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


class ToolRegistry:
    """The set of tools an agent may call, plus validated, time-limited execution."""

    def __init__(self, tools: Iterable[Tool] = (), *, max_result_chars: int = MAX_RESULT_CHARS) -> None:
        self.tools = {}
        self.max_result_chars = max_result_chars
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self.tools:
            raise ValueError(f"Duplicate tool name {t.name!r}")
        self.tools[t.name] = t

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def definitions(self, *, strict: bool = True) -> list[dict[str, Any]]:
        # Sorted for a stable request prefix (prompt caching is a byte-exact prefix match).
        return [self.tools[n].definition(strict=strict) for n in sorted(self.tools)]

    def execute(self, name: str, args: Any, *, approve: Callable[[str, dict], bool] | None = None) -> ToolOutcome:
        start = time.perf_counter()

        def done(output: str, is_error: bool, approved: bool = True) -> ToolOutcome:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolOutcome(name, args if isinstance(args, dict) else {}, output, is_error, elapsed, approved)

        t = self.tools.get(name)
        if t is None:
            return done(f"Unknown tool {name!r}. Available tools: {', '.join(sorted(self.tools))}", True)
        errors = validate(args, t.input_schema)
        if errors:
            return done("Invalid arguments: " + "; ".join(errors), True)
        if t.requires_approval and not (approve and approve(name, args)):
            return done("The user declined to run this tool. Do not retry it; continue without it.", True, False)

        # A timed-out tool is abandoned, not killed (Python threads can't be); tools that touch the
        # outside world should enforce their own timeouts as well.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            result = pool.submit(t.fn, **args).result(timeout=t.timeout_s)
        except concurrent.futures.TimeoutError:
            return done(f"Tool timed out after {t.timeout_s:g}s", True)
        except ToolError as exc:
            return done(str(exc), True)
        except Exception as exc:  # report, don't crash the agent loop
            return done(f"{type(exc).__name__}: {exc}", True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        text = _to_text(result)
        if len(text) > self.max_result_chars:
            omitted = len(text) - self.max_result_chars
            text = text[: self.max_result_chars] + f"\n…[truncated {omitted:,} characters]"
        return done(text, False)
