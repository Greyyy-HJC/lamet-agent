"""Prompt composition utilities used by the agent loop."""

from __future__ import annotations

from importlib import import_module

from lamet_agent.manifest import AnalysisManifest

from .stages import resolve_stage_package

SYSTEM_PROMPT = """
You are a LaMET analysis agent.
Output one JSON action per stage. Be explicit and deterministic.
If required inputs are missing, ask for user input.
""".strip()

ACTION_OUTPUT_HINT = (
    'Return JSON with keys: "action", "reason", optional "tool_name", optional "args".'
)


def get_stage_instruction(stage: str) -> str:
    """Resolve one stage instruction text from stage prompt modules."""
    package_name = resolve_stage_package(stage)
    if not package_name:
        return "Run this stage carefully."

    module = import_module(f"lamet_agent.stages.{package_name}.prompts")
    return getattr(module, "STAGE_PROMPT", "Run this stage carefully.")


def build_stage_prompt(
    stage: str,
    manifest: AnalysisManifest,
    *,
    completed_stages: list[str],
) -> str:
    """Build one prompt payload for a stage."""
    stage_prompt = get_stage_instruction(stage)
    correlator_ids = [item.dataset_id for item in manifest.correlators]
    kernel_ids = [item.kernel_id for item in manifest.kernels]

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Run ID: {manifest.run_id}\n"
        f"Goal: {manifest.goal}\n"
        f"Current stage: {stage}\n"
        f"Completed stages: {completed_stages}\n"
        f"Correlators: {correlator_ids}\n"
        f"Kernels: {kernel_ids}\n\n"
        f"Stage instruction: {stage_prompt}\n"
        f"{ACTION_OUTPUT_HINT}\n"
    )
