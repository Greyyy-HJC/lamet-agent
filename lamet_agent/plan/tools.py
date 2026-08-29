"""Planning tool registry with prompts, schemas, and guarded handlers.

Purpose: keep every Plan-mode LLM tool definition and implementation in one
independently testable registry.
Inputs: a PlanState-compatible object, tool name, and decoded JSON arguments.
Outputs: provider schemas, controller guidance, and JSON observations.
Example: ``run_planning_tool(state, "list_manifest_issues", {})``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


def _object_schema(
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


@dataclass(frozen=True)
class PlanningTool:
    """One LLM-visible planning tool and its complete usage contract."""

    name: str
    prompt: str
    parameters: Mapping[str, Any]
    handler: Callable[[Any, Mapping[str, Any]], dict[str, Any]]

    @property
    def provider_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.prompt,
                "parameters": dict(self.parameters),
            },
        }


def _read_manifest(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, **state.manifest_view(str(arguments.get("path", "")))}


def _list_manifest_issues(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, "issue_count": len(state.issues), "issues": state.packets}


def _inspect_manifest_contract(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return {"ok": False, "error": "path must be a nonempty dotted manifest path"}
    return state.contract_view(path)


def _apply_manifest_patch(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    patches = arguments.get("patches")
    if not isinstance(patches, list):
        return {"ok": False, "error": "patches must be a list"}
    return state.apply(patches)


def _find_paths(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str):
        return {"ok": False, "error": "query must be a string"}
    maximum = arguments.get("max_results", 30)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 100:
        return {"ok": False, "error": "max_results must be between 1 and 100"}
    return {"ok": True, "paths": state.find_paths(query, maximum)}


def _undo_manifest_change(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    undone = state.undo()
    return {
        "ok": undone,
        "undone": undone,
        "remaining_issue_count": len(state.issues),
        "issues": state.packets,
        "error": None if undone else "nothing to undo",
    }


def _save_draft(state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    saved = state.save()
    return {
        "ok": True,
        "saved_path": str(saved),
        "remaining_issue_count": len(state.issues),
    }


def _cancel_plan(_state: Any, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, "cancelled": True}


def _finish_plan(state: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": not state.issues,
        "ready": not state.issues,
        "summary": arguments.get("summary"),
        "changes": arguments.get("changes"),
        "error": None if not state.issues else "manifest still has validator issues",
        "issues": state.packets,
    }


_PATCH_SCHEMA = {
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

PLANNING_TOOLS = (
    PlanningTool(
        "read_manifest",
        "Read the current in-memory authored manifest or one JSON Pointer subtree. Use this for natural-language "
        "requests to show current settings; never infer a value from stale conversation text.",
        _object_schema({"path": {"type": "string"}}),
        _read_manifest,
    ),
    PlanningTool(
        "list_manifest_issues",
        "Return current validator Issues, values, allowed siblings, and all related contract child rules. Call this "
        "before asking when the active Issue subtree is not already present in context.",
        _object_schema(),
        _list_manifest_issues,
    ),
    PlanningTool(
        "inspect_manifest_contract",
        "Inspect all contract rules below an existing or proposed dotted manifest path, even when validation reports "
        "no Issue there. Use this before implementing an explicit request to add or change a valid stage, job, "
        "systematics declaration, strategy, or parameter set.",
        _object_schema({"path": {"type": "string"}}, required=["path"]),
        _inspect_manifest_contract,
    ),
    PlanningTool(
        "apply_manifest_patch",
        "Apply guarded RFC 6902 add, replace, or remove operations to the in-memory authored candidate. Use only "
        "after the user answer establishes the values. The observation contains a fresh validator result; inspect and "
        "reorder the remaining Issues, then ask one concise question or a manageable group of short questions.",
        _object_schema({"patches": _PATCH_SCHEMA}, required=["patches"]),
        _apply_manifest_patch,
    ),
    PlanningTool(
        "find_paths",
        "Find project paths below the manifest directory or metadata.root_directory. Use this whenever a file or "
        "directory value is missing or ambiguous; do not guess paths.",
        _object_schema(
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            required=["query"],
        ),
        _find_paths,
    ),
    PlanningTool(
        "undo_manifest_change",
        "Undo the most recent successful manifest update. Natural-language requests such as 'undo that change' "
        "must call this tool immediately.",
        _object_schema(),
        _undo_manifest_change,
    ),
    PlanningTool(
        "save_draft",
        "Save the current candidate without accepting the plan or entering run mode. Use for natural-language "
        "requests to save progress; validation Issues may remain.",
        _object_schema(),
        _save_draft,
    ),
    PlanningTool(
        "cancel_plan",
        "Cancel planning without entering run mode. Use for an explicit natural-language request to quit or cancel.",
        _object_schema(),
        _cancel_plan,
    ),
    PlanningTool(
        "finish_plan",
        "Request manual plan confirmation only after all validator Issues are resolved. Summarize the accepted "
        "physics and configuration changes in natural language for a physicist; never provide a code diff.",
        _object_schema(
            {
                "summary": {"type": "string"},
                "changes": {"type": "array", "items": {"type": "string"}},
            },
            required=["summary", "changes"],
        ),
        _finish_plan,
    ),
)

_BY_NAME = {tool.name: tool for tool in PLANNING_TOOLS}


def planning_tool_schemas() -> list[dict[str, Any]]:
    """Return provider-ready schemas in stable registry order."""
    return [tool.provider_schema for tool in PLANNING_TOOLS]


def planning_controller_prompt() -> str:
    """Compose global policy with the registry-owned tool usage prompts."""
    base = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    catalog = "\n".join(f"- `{tool.name}`: {tool.prompt}" for tool in PLANNING_TOOLS)
    return f"{base}\n\n# Planning tools\n\n{catalog}"


def run_planning_tool(state: Any, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one registered tool and label its JSON observation."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return {"tool": name, "ok": False, "error": f"unknown planning tool {name!r}"}
    return {"tool": name, **tool.handler(state, arguments)}


__all__ = [
    "PLANNING_TOOLS",
    "PlanningTool",
    "planning_controller_prompt",
    "planning_tool_schemas",
    "run_planning_tool",
]
