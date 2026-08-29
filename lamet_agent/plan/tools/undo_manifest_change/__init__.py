"""Undo the latest successful manifest update."""

from __future__ import annotations

from typing import Any, Mapping

from .._schema import object_schema

PARAMETERS = object_schema()


def run(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    undone = state.undo()
    return {
        "ok": undone,
        "undone": undone,
        "remaining_issue_count": len(state.issues),
        "issues": state.packets,
        "error": None if undone else "nothing to undo",
    }
