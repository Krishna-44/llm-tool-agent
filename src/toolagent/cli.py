"""Command-line interface: ``toolagent run | chat | demo | eval | tools``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from toolagent.agent import Agent, Tracer
from toolagent.builtin import default_tools
from toolagent.evals import CASES, format_report, run_suite
from toolagent.llm import DEFAULT_MODEL, ClaudeClient, ScriptedLLM, text, tool_use
from toolagent.tools import ToolRegistry

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
DIM, BOLD, GREEN, RED, RESET = (
    ("\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[0m") if _COLOR else ("",) * 5
)


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v)[:60]}" for k, v in args.items())


def print_event(event: dict[str, Any]) -> None:
    kind = event["kind"]
    if kind == "llm":
        calls = "  →  " + "; ".join(f"{c['name']}({_fmt_args(c['input'])})" for c in event["tool_calls"])
        said = event["text"] or ""
        print(f"{BOLD}◆ step {event['step']}{RESET} {said}{calls if event['tool_calls'] else ''}")
    elif kind == "tool":
        mark = f"{RED}✗{RESET}" if event["is_error"] else f"{GREEN}✓{RESET}"
        print(f"   {mark} {event['name']} {DIM}{event['duration_ms']:.1f} ms → {event['output'][:160]}{RESET}")
    elif kind == "final":
        usage = event["usage"]
        print(f"{DIM}■ {event['stop_reason']} after {event['steps']} steps, {event['tool_calls']} tool calls, "
              f"{usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out tokens, "
              f"{event['duration_ms'] / 1000:.1f}s{RESET}")


def ask_approval(name: str, args: dict[str, Any]) -> bool:
    preview = json.dumps(args, ensure_ascii=False)
    answer = input(f"\n{BOLD}Allow {name}{RESET}({preview[:400]})? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _agent(args: argparse.Namespace, tracer: Tracer) -> Agent:
    llm = ClaudeClient(model=args.model, effort=args.effort)
    tools = default_tools(args.workspace, allow_write=not args.read_only)
    approve = (lambda *_: True) if args.yes else ask_approval
    return Agent(llm, tools, max_steps=args.max_steps, approve=approve, tracer=tracer)


def cmd_run(args: argparse.Namespace) -> int:
    tracer = Tracer(sink=None if args.quiet else print_event)
    result = _agent(args, tracer).run(args.task)
    print(f"\n{result.output}")
    if args.trace:
        tracer.write_jsonl(args.trace)
    return 0 if result.completed else 1


def cmd_chat(args: argparse.Namespace) -> int:
    agent = _agent(args, Tracer(sink=print_event))
    print("Chat with the agent. Commands: /reset, /save FILE, /quit")
    while True:
        try:
            message = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message == "/quit":
            return 0
        if message == "/reset":
            agent.messages = []
            print("(conversation cleared)")
            continue
        if message.startswith("/save "):
            agent.save(message.split(maxsplit=1)[1])
            print("(saved)")
            continue
        print(f"\n{agent.chat(message).output}")


def demo_policy(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A scripted 'model' for the offline demo: plans, hits an error, recovers, answers."""
    turn = sum(1 for m in messages if m["role"] == "assistant")
    if turn == 0:
        return [
            text("I'll read the orders file and check the time in Tokyo at the same time."),
            tool_use("read_file", {"path": "data/orders.csv"}),
            tool_use("current_time", {"timezone": "Asia/Tokio"}),
        ]
    if turn == 1:
        return [
            text("The time-zone name was wrong; retrying with the correct one, and summing the order lines."),
            tool_use("calculator", {"expression": "12*4.50 + 3*19.99 + 7*12.25 + 20*0.75"}),
            tool_use("current_time", {"timezone": "Asia/Tokyo"}),
        ]
    results = {r["tool_use_id"]: r["content"] for r in messages[-1]["content"]}
    total, now = list(results.values())
    return [text(f"Total revenue in data/orders.csv is {total} USD. In Tokyo it is now {now}.")]


def cmd_demo(args: argparse.Namespace) -> int:
    print(f"{DIM}Offline demo: a scripted model drives the real agent loop and tools (no API key needed).{RESET}\n")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "data").mkdir()
        (Path(tmp) / "data" / "orders.csv").write_text(
            "item,quantity,unit_price\nwidget,12,4.50\ngadget,3,19.99\ngizmo,7,12.25\ndoohickey,20,0.75\n"
        )
        agent = Agent(ScriptedLLM(policy=demo_policy), default_tools(tmp), tracer=Tracer(sink=print_event))
        task = "What's the total revenue in data/orders.csv, and what time is it in Tokyo?"
        print(f"{BOLD}task:{RESET} {task}\n")
        result = agent.run(task)
    print(f"\n{result.output}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cases = [c for c in CASES if not args.cases or c.name in args.cases.split(",")]
    results = run_suite(lambda: ClaudeClient(model=args.model, effort=args.effort), cases,
                        max_steps=args.max_steps)
    print(format_report(results))
    return 0 if all(r.passed for r in results) else 1


def cmd_tools(args: argparse.Namespace) -> int:
    registry = ToolRegistry(default_tools(args.workspace or tempfile.gettempdir()))
    print(json.dumps(registry.definitions(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolagent", description="A tool-using LLM agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--max-steps", type=int, default=12)

    def agent_args(p: argparse.ArgumentParser) -> None:
        model_args(p)
        p.add_argument("--workspace", default=".", help="folder the file tools may access")
        p.add_argument("--read-only", action="store_true", help="don't offer the write_file tool")
        p.add_argument("--yes", action="store_true", help="approve every side-effecting tool call")

    run = sub.add_parser("run", help="complete one task")
    run.add_argument("task")
    agent_args(run)
    run.add_argument("--trace", help="write a JSONL trace of every model and tool call")
    run.add_argument("--quiet", action="store_true", help="print only the final answer")
    run.set_defaults(func=cmd_run)

    chat = sub.add_parser("chat", help="interactive multi-turn session")
    agent_args(chat)
    chat.set_defaults(func=cmd_chat)

    demo = sub.add_parser("demo", help="offline demo with a scripted model")
    demo.set_defaults(func=cmd_demo)

    ev = sub.add_parser("eval", help="run the task suite against Claude")
    model_args(ev)
    ev.add_argument("--cases", help="comma-separated case names (default: all)")
    ev.set_defaults(func=cmd_eval)

    tools = sub.add_parser("tools", help="print the generated tool definitions")
    tools.add_argument("--workspace")
    tools.set_defaults(func=cmd_tools)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
