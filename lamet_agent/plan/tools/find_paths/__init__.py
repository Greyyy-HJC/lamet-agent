"""Discover candidate project paths for manifest values."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema(
    {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    required=["query"],
)


def run(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str):
        return {"ok": False, "error": "query must be a string"}
    maximum = arguments.get("max_results", 30)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 100:
        return {"ok": False, "error": "max_results must be between 1 and 100"}
    return {"ok": True, "paths": state.find_paths(query, maximum)}
