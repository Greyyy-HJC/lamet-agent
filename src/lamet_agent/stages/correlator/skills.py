"""Stage-local skill guidance and tool catalog for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


STAGE_SKILL = """
Correlator-analysis physics:
- The 2pt correlator is symmetric about t = Lt/2; fit only the first half
  (tmax <= Lt//2, tmin >= 1).
- Ratio R(tsep, tau) = C_3pt / C_2pt(tsep); a two-state ratio fit uses tau in
  [tau_cut, tsep + 1 - tau_cut) with tau_cut >= 1.
- The bare matrix element is O00/(2*E0); it is invariant under the 2pt rescale.

Fit-quality rules:
- Always resample at read time ('bs' or 'jk'); never fit a single configuration mean.
- joint fits 2pt+ratio together; chained fits 2pt first, then anchors the ratio
  with E0/z0 from the 2pt posterior. Prefer Q > 0.05 and stable plateaus.
""".strip()

TOOL_CATALOG = {
    "inspect_correlator_scale": "inspect_correlator_scale(...) -> 2pt magnitude diagnostics for correlator_rescale.",
    "tune_ground_state": "tune_ground_state(...) -> per-window 2pt diagnostics and stored E0_avg/z0_avg.",
    "tune_bare_matrix": "tune_bare_matrix(...) -> ranked 3pt/2pt window candidates on sample-average data.",
    "fit_bare_matrix_grid": "fit_bare_matrix_grid(...) -> apply one shared setting to all z/samples and write bare-matrix artifacts.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return human-readable issues for this stage only."""
    has_2pt = any(item.kind == "2pt" for item in manifest.correlators)
    has_3pt = any(item.kind == "3pt" for item in manifest.correlators)
    if has_3pt and not has_2pt:
        return ["3pt ratio analysis requires at least one 2pt correlator dataset."]
    if not has_2pt:
        return ["No 2pt correlator datasets were provided for ground-state analysis."]
    return []
