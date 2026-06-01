"""Execution placeholders for extrapolation stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def build_stage_context(manifest: AnalysisManifest) -> dict[str, object]:
    """Return minimal context used by prompt/tool layers."""
    return {
        "metadata_keys": sorted(manifest.metadata.keys()),
        "correlator_count": len(manifest.correlators),
    }
