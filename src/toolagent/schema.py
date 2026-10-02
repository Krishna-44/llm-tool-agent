"""JSON Schema generation from Python signatures, and validation of model-supplied arguments.

Tool schemas are generated from type hints so a tool is just a typed, documented function. The
generated subset is the one strict tool use accepts: objects always set
``additionalProperties: false``, and there are no numeric or length constraints.

Arguments are validated before a tool runs. With ``strict: true`` the API already guarantees valid
arguments, but validating locally keeps the loop safe with any model or proxy and turns mistakes
into an error the model can read and correct.
"""

from __future__ import annotations

import inspect
import re
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

_PRIMITIVES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


class SchemaError(TypeError):
    """A tool function uses a parameter type that cannot be expressed as a tool schema."""


def type_to_schema(tp: Any) -> dict[str, Any]:
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    origin, args = get_origin(tp), get_args(tp)
    if origin is Literal:
        values = list(args)
        kinds = {type(v) for v in values}
        if len(kinds) != 1 or next(iter(kinds)) not in _PRIMITIVES:
            raise SchemaError(f"Literal values must share one primitive type: {tp!r}")
        return {"type": _PRIMITIVES[next(iter(kinds))], "enum": values}
    if origin in (list, tuple) and args:
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            raise SchemaError(f"Only homogeneous tuples are supported: {tp!r}")
        return {"type": "array", "items": type_to_schema(args[0])}
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return type_to_schema(non_none[0])  # Optional[X]: optional-ness lives in "required"
    raise SchemaError(f"Unsupported parameter type: {tp!r}")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"^\s*(?:Args|Arguments|Parameters):\s*$", doc, maxsplit=1, flags=re.MULTILINE)
    summary = " ".join(parts[0].split())
    params: dict[str, str] = {}
    if len(parts) == 2:
        current = None
        for line in parts[1].splitlines():
            if re.match(r"^\s*(Returns|Raises|Examples?):", line):
                break
            match = re.match(r"^\s*(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
            if match:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    return summary, params


def function_schema(fn) -> tuple[str, dict[str, Any]]:
    """Return (description, input_schema) for a typed function."""
    summary, param_docs = parse_docstring(fn.__doc__)
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise SchemaError(f"{fn.__name__}: *args/**kwargs are not supported")
        if name not in hints:
            raise SchemaError(f"{fn.__name__}: parameter {name!r} needs a type hint")
        prop = type_to_schema(hints[name])
        description = param_docs.get(name, "")
        if param.default is not inspect.Parameter.empty and param.default is not None:
            description = f"{description} (default: {param.default!r})".strip()
        if description:
            prop["description"] = description
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    return summary, schema


def validate(value: Any, schema: dict[str, Any], path: str = "input") -> list[str]:
    """Validate ``value`` against the schema subset produced above. Returns error messages."""
    kind = schema.get("type")
    errors: list[str] = []
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{k} is not an allowed field" for k in value if k not in props]
        for name, sub in props.items():
            if name in value:
                errors += validate(value[name], sub, f"{path}.{name}")
        return errors
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        for i, item in enumerate(value):
            errors += validate(item, schema.get("items", {}), f"{path}[{i}]")
        return errors
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
    }
    if kind in checks and not checks[kind](value):
        return [f"{path} must be of type {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']}"]
    return errors
