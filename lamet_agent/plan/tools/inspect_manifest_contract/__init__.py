"""Inspect the contract for an existing or proposed manifest path."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema({"path": {"type": "string"}}, required=["path"])


def run(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return {"ok": False, "error": "path must be a nonempty dotted manifest path"}
    return state.contract_view(path)
