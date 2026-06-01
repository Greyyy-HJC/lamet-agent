"""Stage sequencing and stage-package routing helpers."""

from __future__ import annotations

DEFAULT_STAGES = [
    "correlator_analysis",
    "renormalization",
    "fourier_transform",
    "perturbative_matching",
    "extrapolation",
]

STAGE_TO_PACKAGE = {
    "correlator_analysis": "correlator",
    "renormalization": "renorm",
    "fourier_transform": "fourier",
    "perturbative_matching": "matching",
    "extrapolation": "extrapolation",
}


def select_stage_sequence(goal: str) -> list[str]:
    """Resolve stage sequence for a goal."""
    if goal == "custom":
        return []
    return DEFAULT_STAGES.copy()


def resolve_stage_package(stage: str) -> str:
    """Map a stage id to its stage package name."""
    return STAGE_TO_PACKAGE.get(stage, "")
