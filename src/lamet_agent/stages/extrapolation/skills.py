"""Stage-local helpers for extrapolation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return stage-local issues only."""
    del manifest
    if "momenta" in job.inputs:
        return ["The extrapolation stage is a placeholder and is not implemented yet."]
    return ["An extrapolation job requires a momenta input role."]
