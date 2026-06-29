"""Prompt text for one correlator-analysis job."""

STAGE_PROMPT = """
Analyze only the correlators listed for the current job. Manifest-derived paths,
selectors, resampling mode, candidate windows, nstate_values, prior_width, and
fit_strategies are injected into tool calls when omitted.

1. Call inspect_correlator_scale and choose a power-of-ten correlator_rescale that
   puts typical fitted 2pt values in 0.0001..0.01.
2. Call tune_bare_matrix with that scale. Compare the returned candidates across
   nstate, prior_width, fit_scope, fit_strategy, windows, Q, chi2/dof,
   n_data, and n_params. For data-window selection, only consider candidates
   with Q above q_min and n_data > n_params; do not rank different data
   windows by raw logGBF. Prefer a good chi2/dof, and when chi2/dof is
   comparable prefer the window with more data points.
3. Call fit_bare_matrix_grid with the selected fit_scope and fit_strategy, the
   same scale, and either the selected pt2_window/pt3_window or the tool's
   sample-average window selection. If passing a selected data window, use
   pt2_window={"tmin": ..., "tmax": ...} and
   pt3_window={"tsep_ls": [...], "tau_cut": ...}; do not pass bare tmin/tmax
   or tau_cut keys. The manifest-controlled model_average setting controls
   fit-function averaging only; do not override model_average. When
   model_average is true, do not pass a scalar nstate or prior_width selected
   from tune_bare_matrix; leave them omitted so the manifest nstate_values and
   prior_width scan remain active.
4. Finish with the NetCDF and diagnostic PDF paths.
""".strip()
