"""Stage-local helpers and skill guidance for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


STAGE_SKILL = """
Correlator-analysis skill: 2pt ground-state extraction and optional 3pt/2pt
ratio fits for the bare matrix element.

2pt physics:
- The periodic/anti-periodic 2pt correlator is symmetric about t = Lt/2.
- Fit only the first half: tmax <= Lt//2 and tmin < Lt/2.

3pt physics:
- Ratio R(tsep, tau) = C_3pt / C_2pt(tsep); tau_cut >= 1; fit tau in [tau_cut, tsep+1-tau_cut).
- read_pt3 once per manifest 3pt file so keys 4,6,8,10 are available; you pick
  tsep_ls (subset) and tau_cut per fit_pt3_window call.
- Two-state ratio fit needs >= 10 re+im data points total across chosen tseps.

Strategy:
- Always resample ('bs' or 'jk'); never use a single configuration mean.
- 2pt: fit_window (<=6 windows on scan); model_average E0, log(dE1), z0, z1;
  plot_fit_on_data.
- 3pt: fit_pt3_window (<=2 trials) anchors E0,z0 to 2pt BMA (5x widened errors); log(dE), z1, O_ij use broad priors;
  then model_average O00_re and O00_im on trusted window_indices; plot shows
  model-averaged O00/(2*E0) bands on both ratio re and im panels.
- Prefer Q > 0.05 and stable plateaus on both stages.
""".strip()

TOOL_CATALOG = {
    "read_pt2": "read_pt2(path, ...) -> pt2_samples (real) and pt2_imag_samples, shape (n_cfg, Lt).",
    "read_pt3": "read_pt3(path, append=True) -> pt3_samples_re/im; do not pass out=.",
    "compute_pt3_ratio": "compute_pt3_ratio() -> ratio_samples_re/im.",
    "resample_to_gvar": "resample_to_gvar(samples='pt2_samples', mode='bs'|'jk') -> pt2_gv.",
    "resample_ratio_to_gvar": "resample_ratio_to_gvar(mode='bs'|'jk') -> ratio_real_gv, ratio_imag_gv.",
    "fit_window": "fit_window(pt2_gv, tmin, tmax, Lt, out='scan', append=True) -> 2pt window scan (max 6).",
    "fit_pt3_window": "fit_pt3_window(tsep_ls, tau_cut, append=True); Lt optional; autofills E0_avg,z0_avg; log(dE1),z1 use broad 3pt priors.",
    "model_average": "model_average(scan, param, window_indices) -> logGBF-weighted gvar; pass a subset of indices.",
    "plot_fit_on_data": "plot_fit_on_data(pt2_gv, scan, window_indices, E0_avg, Lt) -> C2pt/meff PDFs.",
    "plot_pt3_fit_on_data": "plot_pt3_fit_on_data(ratio_real_gv, ratio_imag_gv, scan='pt3_scan', window_indices, O00_re_avg, Lt) -> ratio PDFs.",
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
