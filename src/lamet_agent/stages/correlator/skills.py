"""Stage-local skill guidance and tool catalog for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


STAGE_SKILL = """
Correlator-analysis physics:
- The 2pt correlator is symmetric about t = Lt/2; fit only the first half
  (tmax <= Lt//2, tmin >= 1).
- Ratio R(tsep, tau) = C_3pt / C_2pt(tsep); a two-state ratio fit uses
  tau in [tau_cut, tsep + 1 - tau_cut) with tau_cut >= 1, and needs enough combined
  re+im points across the chosen tseps.
- The bare matrix element is O00/(2*E0); it is invariant under the 2pt rescale.

Strategy:
- Always resample at read time ('bs' or 'jk'); never fit a single configuration mean.
- Inspect 2pt magnitudes and pick a power-of-ten correlator_rescale so fitted 2pt data
  are ~0.0001..0.01.
- Tune one shared fit setting on sample-average data, then apply it to every sample:
  this keeps all z (and b) on the same window instead of per-z choices.
- joint strategy fits 2pt+ratio together; chained fits 2pt first, then the ratio with
  E0/z0 anchored from the 2pt posterior. Prefer Q > 0.05 and stable plateaus.
""".strip()

TOOL_CATALOG = {
    "inspect_correlator_scale": "inspect_correlator_scale(pt2_path, momentum, source_sink, gamma, pt2_windows?) -> 2pt magnitude diagnostics for choosing correlator_rescale.",
    "tune_ground_state": "tune_ground_state(pt2_path, pt2_windows, correlator_rescale, ...) -> per-window 2pt diagnostics + plot; window_indices+model_average store E0_avg/z0_avg.",
    "tune_bare_matrix": "tune_bare_matrix(pt2_path, pt3_paths, tsep_ls, momentum, fit_strategy, correlator_rescale, pt2_windows, pt3_tau_cuts, tune_z?) -> ranked candidate windows with O00/(2E0) on sample-average data + tuning plot.",
    "fit_bare_matrix_grid": "fit_bare_matrix_grid(pt2_path, pt3_paths, tsep_ls, z_values, ensemble, tag, momentum, fit_strategy, correlator_rescale, pt2_window|pt2_windows, pt3_window|pt3_tau_cuts, model_average?) -> applies one shared setting to all z/samples; writes bare_qpdf txt, sample-0 PDFs, split logs, summary PDF + report.",
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
