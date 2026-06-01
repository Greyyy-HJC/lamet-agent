"""Stage-local helpers for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return human-readable issues for this stage only."""
    if manifest.correlators:
        return []
    return ["No correlator datasets were provided."]
