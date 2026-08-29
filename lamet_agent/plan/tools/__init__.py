"""Discoverable registry for the independently packaged Plan tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from . import (
    apply_manifest_patch,
    cancel_plan,
    find_paths,
    finish_plan,
    inspect_manifest_contract,
    list_manifest_issues,
    read_manifest,
    save_draft,
    undo_manifest_change,
)


@dataclass(frozen=True)
class PlanningTool:
    """One packaged Plan tool with its prompt, provider schema, and handler."""

    name: str
    prompt: str
    parameters: Mapping[str, Any]
    handler: Callable[[Any, Mapping[str, Any]], dict[str, Any]]

    @property
    def provider_schema(self) -> dict[str, Any]:
        summary = self.prompt.split("\n\n", 1)[0].replace("\n", " ")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": summary,
                "parameters": dict(self.parameters),
            },
        }


def _load_tool(module: ModuleType) -> PlanningTool:
    name = module.__name__.rsplit(".", 1)[-1]
    prompt_path = Path(module.__file__).with_name("prompts.md")
    if not prompt_path.is_file():
        raise ValueError(f"Plan tool '{name}' requires prompts.md")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Plan tool '{name}' requires nonempty prompts.md")
    parameters = getattr(module, "PARAMETERS", None)
    if not isinstance(parameters, Mapping):
        raise TypeError(f"Plan tool '{name}' must export mapping PARAMETERS")
    handler = getattr(module, "run", None)
    if not callable(handler):
        raise TypeError(f"Plan tool '{name}' must export callable run")
    return PlanningTool(name, prompt, parameters, handler)


PLANNING_TOOLS = tuple(
    _load_tool(module)
    for module in (
        read_manifest,
        list_manifest_issues,
        inspect_manifest_contract,
        apply_manifest_patch,
        find_paths,
        undo_manifest_change,
        save_draft,
        cancel_plan,
        finish_plan,
    )
)
_BY_NAME = {tool.name: tool for tool in PLANNING_TOOLS}


def planning_tool_schemas() -> list[dict[str, Any]]:
    """Return provider-ready schemas in stable registry order."""
    return [tool.provider_schema for tool in PLANNING_TOOLS]


def planning_controller_prompt() -> str:
    """Compose the controller policy with every tool-owned prompts.md file."""
    base = Path(__file__).parents[1].joinpath("prompt.md").read_text(encoding="utf-8").strip()
    catalog = "\n\n".join(f"## `{tool.name}`\n\n{tool.prompt}" for tool in PLANNING_TOOLS)
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
