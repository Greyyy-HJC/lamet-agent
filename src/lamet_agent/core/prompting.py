"""Prompt composition utilities used by the agent loop."""

from __future__ import annotations

import json
from importlib import import_module

from lamet_agent.manifest import AnalysisManifest

from .stages import resolve_stage_package

SYSTEM_PROMPT = """
You are a LaMET analysis agent.
Drive each stage by emitting one JSON action at a time.
All numerical work goes through the listed stage tools; do not invent tools.
Reference data produced by earlier tools through their 'out' keys.
If required inputs are missing, ask for user input.
""".strip()

ACTION_OUTPUT_HINT = (
    'Return JSON with keys: "action" (one of "call_tool", "request_user_input", '
    '"finish"), "reason", optional "tool_name", optional "args".'
)


def _stage_module(stage: str, kind: str):
    package_name = resolve_stage_package(stage)
    if not package_name:
        return None
    return import_module(f"lamet_agent.stages.{package_name}.{kind}")


def get_stage_instruction(stage: str) -> str:
    """Resolve one stage instruction text from stage prompt modules."""
    module = _stage_module(stage, "prompts")
    if module is None:
        return "Run this stage carefully."
    return getattr(module, "STAGE_PROMPT", "Run this stage carefully.")


def get_stage_skill(stage: str) -> str:
    """Resolve the stage skill guidance and tool catalog, if available."""
    module = _stage_module(stage, "skills")
    if module is None:
        return ""
    skill = getattr(module, "STAGE_SKILL", "")
    catalog_fn = getattr(module, "tool_catalog", None)
    catalog = catalog_fn() if callable(catalog_fn) else ""
    parts = []
    if skill:
        parts.append(f"Stage skill:\n{skill}")
    if catalog:
        parts.append(f"Available tools:\n{catalog}")
    return "\n\n".join(parts)


def build_stage_static_prompt(
    stage: str,
    manifest: AnalysisManifest,
    *,
    completed_stages: list[str],
    input_issues: list[str] | None = None,
) -> str:
    """Build the static stage context (no tool observations)."""
    stage_prompt = get_stage_instruction(stage)
    stage_skill = get_stage_skill(stage)
    correlator_ids = [
        {"dataset_id": item.dataset_id, "kind": item.kind, "path": item.path}
        for item in manifest.correlators
    ]
    kernel_ids = [item.kernel_id for item in manifest.kernels]

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Run ID: {manifest.run_id}\n"
        f"Goal: {manifest.goal}\n"
        f"Current stage: {stage}\n"
        f"Completed stages: {completed_stages}\n"
        f"Correlators: {json.dumps(correlator_ids)}\n"
        f"Kernels: {kernel_ids}\n\n"
        f"Metadata: {json.dumps(manifest.metadata)}\n\n"
        f"Input issues: {json.dumps(input_issues or [])}\n\n"
        f"Stage instruction: {stage_prompt}\n\n"
        f"{stage_skill}\n\n"
        f"{ACTION_OUTPUT_HINT}\n"
    )


def format_tool_observation(observation: dict) -> str:
    """Format one tool result as a compact follow-up user turn."""
    return "Tool result:\n" + json.dumps(observation, indent=2)


def build_stage_prompt(
    stage: str,
    manifest: AnalysisManifest,
    *,
    completed_stages: list[str],
    input_issues: list[str] | None = None,
    observations: list[dict] | None = None,
) -> str:
    """Build one monolithic prompt (static context plus optional observation dump)."""
    static = build_stage_static_prompt(
        stage,
        manifest,
        completed_stages=completed_stages,
        input_issues=input_issues,
    )
    if not observations:
        return static
    observation_text = "Tool results so far:\n" + json.dumps(observations, indent=2) + "\n\n"
    return static + "\n" + observation_text
