"""Agent runtime loop for staged LaMET workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .core.prompting import build_stage_prompt
from .core.stages import select_stage_sequence
from .manifest import AnalysisManifest


@dataclass
class AgentState:
    """In-memory agent state for one run."""

    run_id: str
    completed_stages: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)


def call_llm_api(prompt: str, *, model: str = "mock") -> dict[str, Any]:
    """Call the LLM API and return one structured action.

    Current behavior:
    - `model="mock"`: return a deterministic placeholder action
    - non-mock models: raise NotImplementedError until provider integration is added
    """
    if model == "mock":
        return {
            "action": "call_tool",
            "tool_name": "mock_tool",
            "args": {"note": "Replace with real tool execution."},
            "reason": "Scaffold mode: deterministic mock action.",
        }

    raise NotImplementedError(
        "Real LLM API integration is not implemented. "
        "Add provider logic in `call_llm_api`."
    )


def run_agent(
    manifest: AnalysisManifest,
    *,
    resume_from: str | None = None,
    model: str = "mock",
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run stage loop and collect structured actions."""
    stages = select_stage_sequence(manifest.goal)
    if resume_from is not None:
        if resume_from in stages:
            start_index = stages.index(resume_from)
            stages = stages[start_index:]
        else:
            stages = [resume_from]

    state = AgentState(run_id=manifest.run_id)

    for step_count, stage in enumerate(stages, start=1):
        if step_count > max_steps:
            return {
                "run_id": manifest.run_id,
                "status": "stopped-max-steps",
                "completed_stages": state.completed_stages,
                "actions": state.actions,
            }

        prompt = build_stage_prompt(
            stage,
            manifest,
            completed_stages=state.completed_stages.copy(),
        )
        action = call_llm_api(prompt, model=model)
        state.actions.append({"stage": stage, "prompt": prompt, "action": action})
        state.completed_stages.append(stage)

    return {
        "run_id": manifest.run_id,
        "status": "completed",
        "model": model,
        "completed_stages": state.completed_stages,
        "actions": state.actions,
        "summary": json.dumps(
            {
                "run_id": manifest.run_id,
                "stage_count": len(state.completed_stages),
                "action_count": len(state.actions),
            }
        ),
    }
