from typing import Literal

import pytest

from toolagent.schema import SchemaError, function_schema, parse_docstring, validate


def search(query: str, limit: int = 5, exact: bool = False, tags: list[str] | None = None,
           sort: Literal["new", "top"] = "top", score: float | None = None) -> str:
    """Search the catalogue.

    Args:
        query: Words to search for.
        limit: Maximum number of results,
            counted after filtering.
        sort: Ordering of results.
    Returns:
        Matching items.
    """
    return query


def test_schema_from_signature_and_docstring():
    description, schema = function_schema(search)
    assert description == "Search the catalogue."
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["query"] == {"type": "string", "description": "Words to search for."}
    assert props["limit"]["type"] == "integer"
    assert props["limit"]["description"] == "Maximum number of results, counted after filtering. (default: 5)"
    assert props["exact"] == {"type": "boolean", "description": "(default: False)"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["sort"]["enum"] == ["new", "top"]
    assert props["score"] == {"type": "number"}


def test_unsupported_signatures_are_rejected():
    def no_hint(x):
        """Doc."""

    def var_args(*items: str):
        """Doc."""

    def bad_type(x: dict):
        """Doc."""

    for fn in (no_hint, var_args, bad_type):
        with pytest.raises(SchemaError):
            function_schema(fn)


def test_parse_docstring_without_args_section():
    assert parse_docstring("Just a summary\nspanning lines.") == ("Just a summary spanning lines.", {})


@pytest.mark.parametrize(
    "value, errors",
    [
        ({"query": "x"}, []),
        ({"query": "x", "limit": 3, "tags": ["a"], "sort": "new"}, []),
        ({}, ["input.query is required"]),
        ({"query": 1}, ["input.query must be of type string"]),
        ({"query": "x", "limit": True}, ["input.limit must be of type integer"]),
        ({"query": "x", "tags": ["a", 2]}, ["input.tags[1] must be of type string"]),
        ({"query": "x", "sort": "old"}, ["input.sort must be one of ['new', 'top']"]),
        ({"query": "x", "extra": 1}, ["input.extra is not an allowed field"]),
        ("not an object", ["input must be an object"]),
    ],
)
def test_validate(value, errors):
    _, schema = function_schema(search)
    assert validate(value, schema) == errors
