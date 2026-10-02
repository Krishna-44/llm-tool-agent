import json

import pytest

from toolagent.agent import Agent, Tracer
from toolagent.builtin import calculator, default_tools
from toolagent.llm import ModelTurn, ScriptedLLM, text, tool_use
from toolagent.tools import tool


def test_single_tool_call_then_answer():
    llm = ScriptedLLM([[text("Let me compute."), tool_use("calculator", {"expression": "6*7"})], [text("42")]])
    agent = Agent(llm, [calculator])
    result = agent.run("What is 6 times 7?")
    assert result.output == "42" and result.completed and result.steps == 2
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    tool_result = agent.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["content"] == "42"
    assert tool_result["tool_use_id"] == agent.messages[1]["content"][1]["id"]
    # The model saw the tool definitions on every request.
    assert llm.requests[0]["tools"][0]["name"] == "calculator"


def test_parallel_calls_return_all_results_in_one_message_in_order():
    llm = ScriptedLLM([
        [tool_use("calculator", {"expression": "1+1"}), tool_use("calculator", {"expression": "2+2"})],
        [text("2 and 4")],
    ])
    agent = Agent(llm, [calculator])
    agent.run("Two sums")
    results = agent.messages[2]["content"]
    assert [r["content"] for r in results] == ["2", "4"]
    ids = [b["id"] for b in agent.messages[1]["content"]]
    assert [r["tool_use_id"] for r in results] == ids and len(set(ids)) == 2


def test_tool_errors_are_returned_to_the_model():
    llm = ScriptedLLM([[tool_use("calculator", {"expression": "1/0"})], [text("Division by zero.")]])
    agent = Agent(llm, [calculator])
    result = agent.run("1/0?")
    error = agent.messages[2]["content"][0]
    assert error["is_error"] is True and error["content"] == "Division by zero"
    assert result.tool_calls[0].is_error and result.completed


def test_step_limit_stops_runaway_loops():
    looping = ScriptedLLM(policy=lambda messages: [tool_use("calculator", {"expression": "1"})])
    result = Agent(looping, [calculator], max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps" and result.steps == 3 and len(result.tool_calls) == 3


def test_declined_approval_is_reported_and_not_executed(tmp_path):
    llm = ScriptedLLM([
        [tool_use("write_file", {"path": "x.txt", "content": "data"})],
        [text("Okay, I won't write it.")],
    ])
    asked = []
    agent = Agent(llm, default_tools(tmp_path), approve=lambda name, args: asked.append(name) or False)
    result = agent.run("write x.txt")
    assert asked == ["write_file"]
    assert not (tmp_path / "x.txt").exists()
    assert result.tool_calls[0].approved is False and result.completed


class CutOff:
    def complete(self, *, system, messages, tools):
        return ModelTurn([text("Calling"), tool_use("calculator", {"expression": "1"}, id="toolu_x")], "max_tokens")


def test_dangling_tool_calls_are_closed_so_history_stays_valid():
    agent = Agent(CutOff(), [calculator])
    result = agent.run("hi")
    assert result.stop_reason == "max_tokens" and result.output == "Calling"
    last = agent.messages[-1]
    assert last["role"] == "user" and last["content"][0]["tool_use_id"] == "toolu_x"
    assert last["content"][0]["is_error"] is True


def test_refusal_without_text_gets_a_message():
    class Refuses:
        def complete(self, **_):
            return ModelTurn([], "refusal")

    result = Agent(Refuses(), [calculator]).run("something")
    assert result.stop_reason == "refusal" and "declined" in result.output


def test_tracer_records_every_step(tmp_path):
    llm = ScriptedLLM([[tool_use("calculator", {"expression": "2*3"})], [text("6")]])
    seen = []
    tracer = Tracer(sink=seen.append)
    Agent(llm, [calculator], tracer=tracer).run("2*3")
    assert [e["kind"] for e in tracer.events] == ["user", "llm", "tool", "llm", "final"]
    assert seen == tracer.events
    path = tmp_path / "trace.jsonl"
    tracer.write_jsonl(path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[2]["name"] == "calculator" and lines[2]["output"] == "6"


def test_multi_turn_chat_keeps_history_and_round_trips(tmp_path):
    llm = ScriptedLLM([[text("Hi Krishna")], [text("You said your name is Krishna")]])
    agent = Agent(llm, [calculator])
    agent.chat("I'm Krishna")
    agent.chat("What's my name?")
    assert len(llm.requests[1]["messages"]) == 3
    path = tmp_path / "chat.json"
    agent.save(path)
    restored = Agent(ScriptedLLM([[text("ok")]]), [calculator])
    restored.load(path)
    assert restored.messages == agent.messages


def test_custom_tools_plug_in():
    @tool
    def shout(message: str) -> str:
        """Upper-case a message.

        Args:
            message: Text to shout.
        """
        return message.upper()

    llm = ScriptedLLM([[tool_use("shout", {"message": "hey"})], [text("HEY")]])
    result = Agent(llm, [shout]).run("shout hey")
    assert result.tool_calls[0].output == "HEY"


def test_scripted_llm_runs_out_loudly():
    with pytest.raises(RuntimeError):
        Agent(ScriptedLLM([[tool_use("calculator", {"expression": "1"})]]), [calculator]).run("x")
