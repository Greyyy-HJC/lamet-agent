"""Stage-local helpers and skill guidance for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest


STAGE_SKILL = """
Correlator-analysis skill: extract a ground-state energy and matrix element
from a two-point correlator.

Physics:
- The periodic/anti-periodic 2pt correlator is symmetric about t = Lt/2
  (mirror relation from exp(-E t) + exp(-E (Lt-t))).
- Fit only the first half: keep tmax <= Lt//2 and tmin < Lt/2. Do not extend
  the fit window into t >= Lt/2 (redundant information, misleading plateaus).

Strategy:
- Always resample first ('bs' bootstrap or 'jk' jackknife) so every result
  carries a statistical error; never collapse to a single mean.
- Explore fit ranges with fit_window(append=True): at most six windows;
  tmin >= 1, tmax <= Lt//2, and tmax - tmin >= 2*nstate (enough points for
  the fit). Prefer fixed tmax=Lt//2 and a few tmin values; watch chi2/dof,
  Q, and E0 stability.
- Prefer windows with Q > 0.05 and an E0 stable across nearby windows.
- Pick window_indices for the windows you trust, then logGBF-weighted
  model_average on that subset; spread across windows is systematic error.
- Report model-averaged E0 and z0 with statistical and systematic errors.
- plot_fit_on_data with the same window_indices after model_average; inspect
  per-window bands on C2pt and meff plus the E0 horizontal band on meff.
""".strip()

TOOL_CATALOG = {
    "read_pt2": "read_pt2(path, source_sink='SS', gamma='5', momentum='PX0PY0PZ0', out='pt2_samples') -> reads a 2pt dataset into (n_cfg, Lt) real samples.",
    "resample_to_gvar": "resample_to_gvar(samples='pt2_samples', mode='bs'|'jk', n_samples=200, out='pt2_gv') -> reduces samples to a gvar correlator array.",
    "fit_window": "fit_window(pt2_gv='pt2_gv', tmin=int, tmax=int, Lt=int, nstate=2, out='scan', append=True) -> one [tmin,tmax) window; at most 6 appends; tmin>=1, tmax<=Lt//2, tmax-tmin>=2*nstate.",
    "model_average": "model_average(scan='scan', param='E0'|'z0', window_indices=[...]|None, out=None) -> logGBF-weighted average with stat/sys errors.",
    "plot_fit_on_data": "plot_fit_on_data(pt2_gv='pt2_gv', scan='scan', window_indices=[...]|None, E0_avg='E0_avg', Lt=int, boundary='periodic') -> C2pt/meff per-window bands + E0 band on meff; PDFs saved under artifacts/.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return human-readable issues for this stage only."""
    if any(item.kind == "2pt" for item in manifest.correlators):
        return []
    return ["No 2pt correlator datasets were provided for ground-state analysis."]
