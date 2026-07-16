"""Lightweight stage-id to package routing."""

from __future__ import annotations


STAGE_TO_PACKAGE = {
    "correlator_analysis": "correlator",
    "renormalization": "renorm",
    "fourier_transform": "fourier",
    "perturbative_matching": "matching",
    "extrapolation": "extrapolation",
    "review": "review",
}


def resolve_stage_package(stage: str) -> str:
    """Map a stage id to its stage package name."""
    return STAGE_TO_PACKAGE.get(stage, "")
