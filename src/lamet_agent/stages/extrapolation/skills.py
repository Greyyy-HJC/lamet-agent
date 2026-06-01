"""Stage-local helpers for extrapolation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only."""
    if manifest.correlators:
        return []
    return ["No analysis outputs were provided for extrapolation."]
