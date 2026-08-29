"""Request manual confirmation after validation succeeds."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema(
    {
        "summary": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
    },
    required=["summary", "changes"],
)


def run(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": not state.issues,
        "ready": not state.issues,
        "summary": arguments.get("summary"),
        "changes": arguments.get("changes"),
        "error": None if not state.issues else "manifest still has validator issues",
        "issues": state.packets,
    }
