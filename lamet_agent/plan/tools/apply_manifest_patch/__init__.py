"""Apply guarded JSON Patch operations to the authored candidate."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import PATCH_SCHEMA, object_schema

PARAMETERS = object_schema({"patches": PATCH_SCHEMA}, required=["patches"])


def run(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    patches = arguments.get("patches")
    if not isinstance(patches, list):
        return {"ok": False, "error": "patches must be a list"}
    return state.apply(patches)
