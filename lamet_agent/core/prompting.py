"""Prompt composition utilities used by the agent loop."""

from __future__ import annotations

import json
from importlib.resources import files

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import (
    MANIFEST_PARAMETER_MAINTENANCE_POLICY,
    stage_contract_guidance,
)

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
Treat the injected stage parameter contract as authoritative for manifest
parameter meanings and compatibility; never invent an unsupported parameter.
""".strip()

ACTION_OUTPUT_HINT = (
    'Return JSON with keys: "action" (one of "call_tool", "request_user_input", '
    '"finish"), "reason", optional "tool_name", optional "args".'
)


def get_stage_instruction(stage: str) -> str:
    """Resolve one stage instruction from its packaged Markdown resource."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return "Run this stage carefully."
    prompt_file = files(f"lamet_agent.stages.{package_name}").joinpath("prompts.md")
    return prompt_file.read_text(encoding="utf-8").strip()


def build_stage_static_prompt(
    stage: str,
    manifest: AnalysisManifest,
    *,
    job: StageJob,
    effective_params: dict,
    completed_stages: list[str],
    input_issues: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
) -> str:
    """Build the static context for one stage job."""
    stage_prompt = get_stage_instruction(stage)
    parameter_contract = stage_contract_guidance(stage)
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
        f"Tools allowed for this job: {json.dumps(allowed_tool_names or [])}\n"
        "Do not call tools outside this job-specific list.\n\n"
        f"Manifest parameter contract: {json.dumps(parameter_contract, indent=2)}\n\n"
        f"Repository maintenance policy: {MANIFEST_PARAMETER_MAINTENANCE_POLICY}\n\n"
        f"Stage instruction: {stage_prompt}\n\n"
        f"{ACTION_OUTPUT_HINT}\n"
    )


def format_tool_observation(observation: dict) -> str:
    """Format one tool result as a compact follow-up user turn."""
    llm_observation = {key: value for key, value in observation.items() if key != "ignored_args"}
    return "Tool result:\n" + json.dumps(llm_observation, indent=2)
