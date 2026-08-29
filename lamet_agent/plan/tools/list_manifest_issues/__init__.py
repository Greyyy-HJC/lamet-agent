"""Return the current lossless validator evidence."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema()


def run(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, "issue_count": len(state.issues), "issues": state.packets}
