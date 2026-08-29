"""Save the current candidate without accepting the Plan."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema()


def run(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    saved = state.save()
    return {
        "ok": True,
        "saved_path": str(saved),
        "remaining_issue_count": len(state.issues),
    }
