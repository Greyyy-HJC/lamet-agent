"""Execution placeholders for perturbative matching stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


def build_stage_context(manifest: AnalysisManifest) -> dict[str, object]:
    """Return minimal context used by prompt/tool layers."""
    return {"kernel_functions": [item.function for item in manifest.kernels]}
