"""Stage sequencing and stage-package routing helpers."""

from __future__ import annotations

from pathlib import Path

STAGE_TO_PACKAGE = {
    "correlator_analysis": "correlator",
    "renormalization": "renorm",
    "fourier_transform": "fourier",
    "perturbative_matching": "matching",
    "extrapolation": "extrapolation",
    "review": "review",
}

STAGE_ARTIFACT_DIRECTORIES = {
    "correlator_analysis": "1_correlator_analysis",
    "renormalization": "2_renormalization",
    "fourier_transform": "3_fourier_transform",
    "perturbative_matching": "4_perturbative_matching",
    "extrapolation": "5_extrapolation",
    "review": "6_review",
}


def resolve_stage_package(stage: str) -> str:
    """Map a stage id to its stage package name."""
    return STAGE_TO_PACKAGE.get(stage, "")


def stage_artifact_directory_name(stage: str) -> str:
    """Return the fixed numbered artifact-directory name for a stage id."""
    try:
        return STAGE_ARTIFACT_DIRECTORIES[stage]
    except KeyError as exc:
        raise ValueError(f"unknown stage id {stage!r}") from exc


def resolve_stage_artifacts_directory(artifacts_directory: str | Path, stage: str) -> Path:
    """Resolve one stage's numbered directory below the artifact root."""
    return Path(artifacts_directory) / stage_artifact_directory_name(stage)
