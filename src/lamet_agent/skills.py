"""Minimal stage/skill helpers used by the agent loop."""

from __future__ import annotations

from .manifest import AnalysisManifest
from .prompts import ACTION_OUTPUT_HINT, STAGE_PROMPTS, SYSTEM_PROMPT

DEFAULT_STAGES = [
    "correlator_analysis",
    "renormalization",
    "fourier_transform",
    "perturbative_matching",
    "extrapolation",
]


def select_stage_sequence(goal: str) -> list[str]:
    """Resolve stage sequence for a goal."""
    if goal == "custom":
        return []
    return DEFAULT_STAGES.copy()


def build_stage_prompt(
    stage: str,
    manifest: AnalysisManifest,
    *,
    completed_stages: list[str],
) -> str:
    """Build one prompt payload for a stage."""
    stage_prompt = STAGE_PROMPTS.get(stage, "Run this stage carefully.")
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
