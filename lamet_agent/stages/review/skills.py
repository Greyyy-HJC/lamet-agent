"""Stage-local validation for review generation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    del manifest, job
    return []
