"""Built-in tools: a safe calculator, the clock, and file tools sandboxed to a workspace."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from toolagent.tools import Tool, ToolError, tool

# --- calculator -------------------------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil, "abs": abs,
    "round": round, "min": min, "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or (abs(left) > 1e6 and right > 50)):
            raise ToolError("Exponent too large")
        return _BINARY[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(a) for a in node.args))
    raise ToolError(f"Unsupported expression element: {ast.dump(node)[:60]}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, instead of doing mental math.

    Supports + - * / // % ** and parentheses, the constants pi and e, and the functions sqrt, log,
    log10, exp, sin, cos, tan, floor, ceil, abs, round, min and max.

    Args:
        expression: The expression to evaluate, e.g. "round(2345.5 * 0.17, 2)".
    """
    if len(expression) > 500:
        raise ToolError("Expression too long (max 500 characters)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Could not parse expression: {exc.msg}") from exc
    try:
        result = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except (ValueError, OverflowError) as exc:
        raise ToolError(f"Math error: {exc}") from exc
    return f"{result:.12g}" if isinstance(result, float) else str(result)


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: IANA time zone name, e.g. "Asia/Kolkata" or "America/New_York".
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(f"Unknown time zone {timezone!r}") from exc
    return datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


# --- workspace file tools ---------------------------------------------------------------------

MAX_LIST_ENTRIES = 200
MAX_SEARCH_MATCHES = 50


class Workspace:
    """File tools confined to one directory.

    Every path is resolved (following symlinks) and must stay inside the workspace root, so the
    model can't read ``../../etc/passwd`` or escape through a symlink. Writing is a separate tool
    that requires the user's approval.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Workspace directory not found: {self.root}")

    def resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"Path {path!r} is outside the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def tools(self, *, allow_write: bool = True) -> list[Tool]:
        ws = self

        @tool
        def list_files(path: str = ".", recursive: bool = False) -> str:
            """List files and folders in the workspace.

            Args:
                path: Folder to list, relative to the workspace root.
                recursive: Also list everything inside sub-folders.
            """
            folder = ws.resolve(path)
            if not folder.is_dir():
                raise ToolError(f"{path!r} is not a folder")
            entries = sorted(folder.rglob("*") if recursive else folder.iterdir())
            lines = [
                f"{ws.relative(p)}/" if p.is_dir() else f"{ws.relative(p)} ({p.stat().st_size} bytes)"
                for p in entries[:MAX_LIST_ENTRIES]
            ]
            if len(entries) > MAX_LIST_ENTRIES:
                lines.append(f"… {len(entries) - MAX_LIST_ENTRIES} more entries")
            return "\n".join(lines) or "(empty folder)"

        @tool
        def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
            """Read a text file from the workspace, with line numbers.

            Args:
                path: File path relative to the workspace root.
                start_line: First line to return (1-based).
                max_lines: Maximum number of lines to return.
            """
            file = ws.resolve(path)
            if not file.is_file():
                raise ToolError(f"{path!r} is not a file")
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise ToolError(f"{path!r} is not a UTF-8 text file") from exc
            start = max(start_line, 1)
            chunk = lines[start - 1 : start - 1 + max(max_lines, 1)]
            body = "\n".join(f"{n:>5}  {text}" for n, text in enumerate(chunk, start=start))
            remaining = len(lines) - (start - 1 + len(chunk))
            footer = f"\n… {remaining} more lines" if remaining > 0 else ""
            return (body + footer) or "(empty file)"

        @tool
        def search_files(text: str, path: str = ".") -> str:
            """Find lines containing some text (case-insensitive) in workspace files.

            Args:
                text: The text to look for.
                path: Folder to search, relative to the workspace root.
            """
            folder = ws.resolve(path)
            needle = text.lower()
            matches = []
            for file in sorted(p for p in folder.rglob("*") if p.is_file()):
                try:
                    lines = file.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for n, line in enumerate(lines, start=1):
                    if needle in line.lower():
                        matches.append(f"{ws.relative(file)}:{n}: {line.strip()}")
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            return "\n".join(matches) + "\n… (more matches not shown)"
            return "\n".join(matches) or "No matches."

        @tool(requires_approval=True)
        def write_file(path: str, content: str) -> str:
            """Create or overwrite a text file in the workspace. The user must approve each write.

            Args:
                path: File path relative to the workspace root.
                content: Full text content to write.
            """
            file = ws.resolve(path)
            if file.is_dir():
                raise ToolError(f"{path!r} is a folder")
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(content, encoding="utf-8")
            return f"Wrote {len(content.encode('utf-8'))} bytes to {ws.relative(file)}"

        tools = [list_files, read_file, search_files]
        return tools + [write_file] if allow_write else tools


def default_tools(workspace: str | Path | None = None, *, allow_write: bool = True) -> list[Tool]:
    tools = [calculator, current_time]
    if workspace is not None:
        tools += Workspace(workspace).tools(allow_write=allow_write)
    return tools
