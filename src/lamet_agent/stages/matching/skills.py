"""Stage-local helpers for perturbative matching."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only."""
    if manifest.kernels:
        return []
    return ["No matching kernel references were provided."]
