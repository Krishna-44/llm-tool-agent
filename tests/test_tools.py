import time

import pytest

from toolagent.tools import ToolError, ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return a + b


@tool
def fails(kind: str) -> str:
    """Fail on purpose.

    Args:
        kind: "clean" raises ToolError, anything else raises ValueError.
    """
    if kind == "clean":
        raise ToolError("nothing found")
    raise ValueError("boom")


@tool(timeout_s=0.2)
def slow() -> str:
    """Sleep for a while."""
    time.sleep(2)
    return "done"


@tool(requires_approval=True)
def delete_everything() -> str:
    """A dangerous action."""
    return "deleted"


@tool
def big() -> str:
    """Return a lot of text."""
    return "x" * 50


@pytest.fixture()
def registry():
    return ToolRegistry([add, fails, slow, delete_everything, big], max_result_chars=20)


def test_definitions_are_sorted_and_strict(registry):
    defs = registry.definitions()
    assert [d["name"] for d in defs] == sorted(d["name"] for d in defs)
    assert all(d["strict"] is True for d in defs)
    assert defs[0]["input_schema"]["required"] == ["a", "b"]


def test_successful_call_returns_text(registry):
    out = registry.execute("add", {"a": 2, "b": 3})
    assert (out.output, out.is_error) == ("5", False)


def test_bad_arguments_and_unknown_tools_become_errors(registry):
    assert registry.execute("add", {"a": "2", "b": 3}).output.startswith("Invalid arguments")
    unknown = registry.execute("nope", {})
    assert unknown.is_error and "Available tools" in unknown.output


def test_exceptions_are_reported_not_raised(registry):
    assert registry.execute("fails", {"kind": "clean"}).output == "nothing found"
    assert registry.execute("fails", {"kind": "other"}).output == "ValueError: boom"


def test_timeouts(registry):
    out = registry.execute("slow", {})
    assert out.is_error and "timed out" in out.output
    assert out.duration_ms < 1500


def test_approval_gate(registry):
    declined = registry.execute("delete_everything", {})
    assert declined.is_error and not declined.approved and "declined" in declined.output
    approved = registry.execute("delete_everything", {}, approve=lambda name, args: True)
    assert approved.output == "deleted"


def test_long_results_are_truncated(registry):
    out = registry.execute("big", {}).output
    assert out.startswith("x" * 20) and "truncated 30 characters" in out


def test_tools_need_docstrings_and_unique_names():
    with pytest.raises(ValueError):
        tool(lambda: None)
    with pytest.raises(ValueError):
        ToolRegistry([add, add])
