import os

import pytest

from toolagent.builtin import Workspace, calculator, current_time
from toolagent.tools import ToolError, ToolRegistry


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3 * 4", "14"),
        ("round(2500 * (1 + 0.072/12) ** 36, 2)", "3100.75"),
        ("sqrt(16) + max(1, 7) - abs(-2)", "9"),
        ("-(2 ** 10) // 3", "-342"),
        ("pi", "3.14159265359"),
    ],
)
def test_calculator(expression, expected):
    assert calculator.fn(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "(1).__class__", "open('x')", "x + 1", "2 ** 100000", "1 / 0", "1 +"],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression):
    with pytest.raises(ToolError):
        calculator.fn(expression)


def test_current_time():
    assert "UTC" in current_time.fn()
    with pytest.raises(ToolError):
        current_time.fn("Mars/Olympus_Mons")


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hello')\n# TODO: tests\n")
    (root / "notes.md").write_text("line one\nline two\nline three\n")
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "secret.txt").write_text("top secret")
    return root


@pytest.fixture()
def tools(workspace):
    return ToolRegistry(Workspace(workspace).tools(), max_result_chars=10_000)


def run(tools, name, **args):
    return tools.execute(name, args, approve=lambda *_: True)


def test_list_read_and_search(tools):
    listing = run(tools, "list_files", recursive=True).output
    assert "src/app.py (" in listing and "src/" in listing
    excerpt = run(tools, "read_file", path="notes.md", start_line=2, max_lines=1).output
    assert excerpt == "    2  line two\n… 1 more lines"
    assert run(tools, "search_files", text="todo").output == "src/app.py:2: # TODO: tests"
    assert run(tools, "search_files", text="absent").output == "No matches."


def test_write_requires_approval(tools, workspace):
    declined = tools.execute("write_file", {"path": "out/new.txt", "content": "hi"})
    assert declined.is_error and not (workspace / "out" / "new.txt").exists()
    assert "Wrote 2 bytes" in run(tools, "write_file", path="out/new.txt", content="hi").output
    assert (workspace / "out" / "new.txt").read_text() == "hi"


@pytest.mark.parametrize("path", ["../secret.txt", "/etc/passwd", "src/../../secret.txt"])
def test_paths_outside_the_workspace_are_refused(tools, path):
    out = run(tools, "read_file", path=path)
    assert out.is_error and "outside the workspace" in out.output


def test_symlinks_cannot_escape(tools, workspace):
    os.symlink(workspace.parent / "secret.txt", workspace / "link.txt")
    out = run(tools, "read_file", path="link.txt")
    assert out.is_error and "top secret" not in out.output


def test_binary_files_are_reported(tools):
    assert "not a UTF-8 text file" in run(tools, "read_file", path="blob.bin").output


def test_read_only_mode_has_no_write_tool(workspace):
    assert "write_file" not in ToolRegistry(Workspace(workspace).tools(allow_write=False))
