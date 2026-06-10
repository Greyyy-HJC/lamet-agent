"""Prompt text for the correlator-analysis stage."""

STAGE_PROMPT = """
Extract ground-state energy/overlaps from 2pt data and, when 3pt data are present,
the bare matrix element from 3pt/2pt ratio fits. Tune one shared fit setting on
sample-average data first, then apply it to every resampled sample. Emit one action
at a time and reference earlier outputs by their 'out' keys.

Flow:
1. inspect_correlator_scale on the 2pt path (pass source_sink, gamma, momentum from
   Metadata.correlator_grid). Choose a power-of-ten correlator_rescale so typical
   fitted 2pt values land in 0.0001..0.01; use 1.0 if they already are.

2pt only (no kind=3pt in the manifest):
2a. tune_ground_state with the chosen correlator_rescale and a few candidate
    pt2_windows (tmin>=1, tmax<=Lt//2). Inspect Q/chi2_dof/logGBF, then re-call with
    window_indices to fix E0_avg/z0_avg, and finish.

2pt + 3pt (Metadata.correlator_grid present):
2b. tune_bare_matrix with the grid fields (pt2_path, pt3_paths, tsep_ls, momentum,
    fit_strategy, correlator_rescale, pt2_windows, pt3_tau_cuts). It scans candidate
    windows on sample-average data for one representative z and returns ranked
    candidates with O00/(2E0); pick the shared window.
3.  fit_bare_matrix_grid with the exact correlator_grid fields plus the chosen
    correlator_rescale. Provide the tuned setting as pt2_window+pt3_window for a single
    shared window, or model_average=true to BMA-combine the window grid. The tool
    applies that one setting to all z and all samples, writes bare_qpdf txt files,
    sample-0 plots, split tuning/sample logs, a summary PDF, and a JSON report.
4.  finish with the bare-matrix paths, the chosen window, and correlator_rescale.

Physics: 2pt is symmetric about Lt/2 (fit the first half). Ratio R(tsep, tau)=C3/C2;
tau_cut>=1 fits tau in [tau_cut, tsep+1-tau_cut). Nonzero momentum needs its exact
HDF5 momentum key. Use only the listed tools.
""".strip()
