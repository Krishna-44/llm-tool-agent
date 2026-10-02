"""Model clients. The agent depends only on the small :class:`LLMClient` protocol.

* :class:`ClaudeClient` calls the Anthropic Messages API.
* :class:`ScriptedLLM` replays predefined turns. It drives the test suite and the offline demo, so
  the whole agent loop is exercised deterministically without an API key.

Assistant content is kept as plain dicts (exactly what the API returned, serialised), so
conversations can be saved, inspected and resumed, and are echoed back unchanged on the next
request — the history is append-only.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class ModelTurn:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class LLMClient(Protocol):
    def complete(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


class ClaudeClient:
    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            # Let the API retry on a fallback model if a safety classifier declines the request.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens or 0,
        }
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        return ModelTurn(content, response.stop_reason or "end_turn", usage, response.model)


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(name: str, args: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id or f"toolu_{name}", "name": name, "input": args}


class ScriptedLLM:
    """Replays a fixed list of turns, or calls ``policy(messages)`` to produce each turn.

    Each scripted turn is a list of content blocks (build them with :func:`text` and
    :func:`tool_use`); the stop reason is ``tool_use`` when the turn contains a tool call.
    """

    def __init__(
        self,
        turns: Sequence[list[dict[str, Any]]] | None = None,
        *,
        policy: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    ) -> None:
        if (turns is None) == (policy is None):
            raise ValueError("Pass exactly one of turns or policy")
        self._turns = iter(turns) if turns is not None else None
        self._policy = policy
        self._ids = itertools.count(1)
        self.requests: list[dict[str, Any]] = []

    def complete(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if self._policy is not None:
            blocks = self._policy(messages)
        else:
            try:
                blocks = next(self._turns)
            except StopIteration as exc:
                raise RuntimeError("ScriptedLLM ran out of scripted turns") from exc
        blocks = [
            {**b, "id": f"toolu_{next(self._ids):03d}"} if b.get("type") == "tool_use" else b for b in blocks
        ]
        stop = "tool_use" if any(b.get("type") == "tool_use" for b in blocks) else "end_turn"
        return ModelTurn(blocks, stop, {"input_tokens": 0, "output_tokens": 0}, "scripted")
