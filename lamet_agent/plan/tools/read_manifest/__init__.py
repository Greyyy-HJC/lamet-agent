"""Read the current in-memory manifest."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema({"path": {"type": "string"}})


def run(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, **state.manifest_view(str(arguments.get("path", "")))}
