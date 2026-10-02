"""The agent loop: call the model, run the tools it asks for, feed results back, repeat.

Written by hand rather than with an SDK helper so every step is visible and testable:

* tool calls in one turn run concurrently, and all results go back in a single message;
* arguments are validated, tools time out, and failures come back as ``is_error`` results the model
  can recover from instead of exceptions that kill the run;
* side-effecting tools need explicit approval;
* a step limit bounds cost, and every model call and tool call is traced.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolagent.llm import LLMClient
from toolagent.tools import Tool, ToolOutcome, ToolRegistry

DEFAULT_SYSTEM = (
    "You complete tasks by using the tools provided. Use the calculator for arithmetic instead of "
    "calculating in your head, and read files before making claims about their contents. If a tool "
    "returns an error, use the message to adjust rather than repeating the same call. When you have "
    "the answer, reply with it directly and concisely."
)

Approver = Callable[[str, dict[str, Any]], bool]


@dataclass
class AgentResult:
    output: str
    stop_reason: str  # end_turn | max_steps | max_tokens | refusal | stop_sequence
    steps: int
    tool_calls: list[ToolOutcome]
    usage: dict[str, int]
    duration_ms: float = 0.0

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


@dataclass
class Tracer:
    """Collects structured events; optionally forwards each one to ``sink`` as it happens."""

    sink: Callable[[dict[str, Any]], None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(self, kind: str, **data: Any) -> None:
        # Parallel tool calls emit from worker threads; keep events and sink output ordered.
        with self._lock:
            event = {"t_ms": round((time.perf_counter() - self._start) * 1000, 1), "kind": kind, **data}
            self.events.append(event)
            if self.sink:
                self.sink(event)

    def write_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _preview(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | Iterable[Tool],
        *,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 12,
        approve: Approver | None = None,
        tracer: Tracer | None = None,
        parallel: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system = system
        self.max_steps = max_steps
        self.approve = approve
        self.tracer = tracer or Tracer()
        self.parallel = parallel
        self.messages: list[dict[str, Any]] = []

    # -- public API -------------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Start a fresh conversation and work on ``task`` until done."""
        self.messages = []
        return self.chat(task)

    def chat(self, message: str) -> AgentResult:
        """Continue the conversation with a new user message."""
        start = time.perf_counter()
        self.messages.append({"role": "user", "content": message})
        self.tracer.emit("user", text=_preview(message))
        calls: list[ToolOutcome] = []
        usage: Counter[str] = Counter()
        definitions = self.registry.definitions()

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            turn = self.llm.complete(system=self.system, messages=self.messages, tools=definitions)
            usage.update(turn.usage)
            # Echo the assistant turn back exactly as received (thinking blocks included).
            self.messages.append({"role": "assistant", "content": turn.content})
            uses = [b for b in turn.content if b.get("type") == "tool_use"]
            self.tracer.emit(
                "llm",
                step=step,
                stop_reason=turn.stop_reason,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                usage=turn.usage,
                text=_preview(self._text(turn.content)),
                tool_calls=[{"name": b["name"], "input": b.get("input", {})} for b in uses],
            )

            if turn.stop_reason == "pause_turn":
                continue  # the API paused a long turn; re-sending the history resumes it
            if turn.stop_reason == "tool_use" and uses:
                outcomes = self._run_tools(uses)
                calls.extend(outcomes)
                self.messages.append({"role": "user", "content": self._results(uses, outcomes)})
                continue

            if uses:
                # The turn ended (e.g. max_tokens) with tool calls we won't run: answer them so the
                # history stays valid for the next request.
                skipped = [ToolOutcome(b["name"], b.get("input", {}), "Not run: the response was cut off.",
                                       True, 0.0) for b in uses]
                self.messages.append({"role": "user", "content": self._results(uses, skipped)})
            output = self._text(turn.content)
            if turn.stop_reason == "refusal" and not output:
                output = "The model declined this request."
            return self._finish(output, turn.stop_reason, step, calls, usage, start)

        return self._finish(
            f"Stopped after {self.max_steps} steps without finishing. The partial work is in the "
            "conversation history.",
            "max_steps", self.max_steps, calls, usage, start,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"system": self.system, "messages": self.messages}, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.system, self.messages = data["system"], data["messages"]

    # -- internals --------------------------------------------------------------------------

    @staticmethod
    def _text(content: list[dict[str, Any]]) -> str:
        return "\n".join(b["text"] for b in content if b.get("type") == "text").strip()

    def _run_one(self, block: dict[str, Any]) -> ToolOutcome:
        outcome = self.registry.execute(block["name"], block.get("input", {}), approve=self.approve)
        self.tracer.emit(
            "tool", name=outcome.name, input=outcome.input, is_error=outcome.is_error,
            approved=outcome.approved, duration_ms=round(outcome.duration_ms, 1),
            output=_preview(outcome.output),
        )
        return outcome

    def _run_tools(self, uses: list[dict[str, Any]]) -> list[ToolOutcome]:
        needs_approval = any(
            self.registry.tools[b["name"]].requires_approval for b in uses if b["name"] in self.registry
        )
        if self.parallel and len(uses) > 1 and not needs_approval:
            with ThreadPoolExecutor(max_workers=min(8, len(uses))) as pool:
                return list(pool.map(self._run_one, uses))
        # Approval prompts are interactive, so those turns run one call at a time.
        return [self._run_one(b) for b in uses]

    @staticmethod
    def _results(uses: list[dict[str, Any]], outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
        results = []
        for block, outcome in zip(uses, outcomes, strict=True):
            result = {"type": "tool_result", "tool_use_id": block["id"], "content": outcome.output}
            if outcome.is_error:
                result["is_error"] = True
            results.append(result)
        return results

    def _finish(self, output, stop_reason, steps, calls, usage, start) -> AgentResult:
        result = AgentResult(output, stop_reason, steps, calls, dict(usage), (time.perf_counter() - start) * 1000)
        self.tracer.emit(
            "final", stop_reason=stop_reason, steps=steps, tool_calls=len(calls), usage=dict(usage),
            duration_ms=round(result.duration_ms, 1), output=_preview(output),
        )
        return result
