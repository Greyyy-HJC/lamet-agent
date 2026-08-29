"""Shared JSON-schema helpers for Plan tools."""

from __future__ import annotations

from typing import Any, Mapping


def object_schema(
    properties: Mapping[str, Any] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
        "additionalProperties": False,
    }


PATCH_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "replace", "remove"]},
            "path": {"type": "string"},
            "value": {},
        },
        "required": ["op", "path"],
        "additionalProperties": False,
    },
}


__all__ = ["PATCH_SCHEMA", "object_schema"]
