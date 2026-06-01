"""Execution placeholders for correlator analysis stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def build_stage_context(manifest: AnalysisManifest) -> dict[str, object]:
    """Return minimal context used by prompt/tool layers."""
    return {
        "correlator_count": len(manifest.correlators),
        "dataset_ids": [item.dataset_id for item in manifest.correlators],
    }
