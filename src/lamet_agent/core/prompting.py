"""Prompt composition utilities used by the agent loop."""

from __future__ import annotations

import json
from importlib import import_module

from lamet_agent.manifest import AnalysisManifest, StageJob

from .stages import resolve_stage_package

SYSTEM_PROMPT = """
You are a LaMET analysis agent.
Drive each stage by emitting one JSON action at a time.
All numerical work goes through the listed stage tools; do not invent tools.
Each job has an isolated store. Upstream inputs are already available under the
role names shown in Job inputs. External artifact inputs declared in the manifest
are pre-loaded into the job store before tools run. The stage's primary result
must be stored as store['output'] by its terminal tool.
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
    job: StageJob,
    effective_params: dict,
    completed_stages: list[str],
    input_issues: list[str] | None = None,
) -> str:
    """Build the static context for one stage job."""
    stage_prompt = get_stage_instruction(stage)
    stage_skill = get_stage_skill(stage)
    correlators = [
        item.model_dump()
        for item in manifest.correlators
        if item.correlator_id in job.correlator_ids
    ]

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Run ID: {manifest.run_id}\n"
        f"Current stage: {stage}\n"
        f"Current job: {job.id}\n"
        f"Completed stages: {completed_stages}\n"
        f"Run metadata: {json.dumps(manifest.metadata.model_dump())}\n"
        f"Correlators: {json.dumps(correlators)}\n"
        f"Job inputs: {json.dumps(job.inputs)}\n"
        f"Effective job parameters: {json.dumps(effective_params)}\n\n"
        f"Input issues: {json.dumps(input_issues or [])}\n\n"
        f"Stage instruction: {stage_prompt}\n\n"
        f"{stage_skill}\n\n"
        f"{ACTION_OUTPUT_HINT}\n"
    )


def format_tool_observation(observation: dict) -> str:
    """Format one tool result as a compact follow-up user turn."""
    llm_observation = {key: value for key, value in observation.items() if key != "ignored_args"}
    return "Tool result:\n" + json.dumps(llm_observation, indent=2)
