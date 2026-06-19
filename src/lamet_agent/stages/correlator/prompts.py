"""Prompt text for the correlator-analysis stage."""

STAGE_PROMPT = """
Extract ground-state energy/overlaps from 2pt data and, when 3pt data are present,
the bare matrix element from 3pt/2pt ratio fits. Tune one shared fit setting on
sample-average data first, then apply it to every resampled sample.

Flow:
1. inspect_correlator_scale on the 2pt path, passing source_sink, gamma, and
   momentum from Metadata.correlator_grid. Choose a power-of-ten
   correlator_rescale so typical fitted 2pt values land in 0.0001..0.01; use
   1.0 if they already are.

2pt only (no kind=3pt in the manifest):
2a. tune_ground_state with the chosen correlator_rescale and candidate
    pt2_windows. Inspect Q/chi2_dof/logGBF, then re-call with window_indices to
    fix E0_avg/z0_avg, and finish.

2pt + 3pt (Metadata.correlator_grid present):
2b. tune_bare_matrix with the grid fields and chosen correlator_rescale. It scans
    candidate windows on sample-average data for one representative z; pick the
    shared window from the ranked candidates.
3.  fit_bare_matrix_grid with the exact correlator_grid fields plus the chosen
    correlator_rescale. Provide the tuned setting as pt2_window+pt3_window for a
    single shared window, or model_average=true to BMA-combine the window grid.
4.  finish with the bare-matrix NetCDF artifact path, chosen window, and
    correlator_rescale.
""".strip()
