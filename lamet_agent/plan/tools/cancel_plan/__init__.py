"""Cancel Plan without entering Run mode."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema()


def run(_state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, "cancelled": True}
