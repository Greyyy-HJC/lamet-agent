"""Stage-local helpers for extrapolation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return stage-local issues only."""
    del manifest
    value = job.inputs.get("lightcone")
    if not isinstance(value, list):
        return ["An extrapolation job requires a list input role named lightcone."]
    if not value:
        return ["An extrapolation job requires at least one perturbative_matching input."]
    return []

STAGE_SKILL = "Fit matched light-cone distributions to IMF and/or continuum limits."
