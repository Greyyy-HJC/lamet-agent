"""Execution placeholders for Fourier-transform stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def build_stage_context(manifest: AnalysisManifest) -> dict[str, object]:
    """Return minimal context used by prompt/tool layers."""
    return {
        "run_id": manifest.run_id,
        "goal": manifest.goal,
    }
