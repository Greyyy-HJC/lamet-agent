"""Execution placeholders for renormalization stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def build_stage_context(manifest: AnalysisManifest) -> dict[str, object]:
    """Return minimal context used by prompt/tool layers."""
    return {
        "kernel_count": len(manifest.kernels),
        "kernel_ids": [item.kernel_id for item in manifest.kernels],
    }
