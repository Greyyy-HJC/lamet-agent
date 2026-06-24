"""Prompt text for one correlator-analysis job."""

STAGE_PROMPT = """
Analyze only the correlators listed for the current job. Manifest-derived paths,
selectors, resampling mode, candidate windows, nstate_values, and fit_strategies
are injected into tool calls when omitted.

1. Call inspect_correlator_scale and choose a power-of-ten correlator_rescale that
   puts typical fitted 2pt values in 0.0001..0.01.
2. Call tune_bare_matrix with that scale. Compare the returned candidates across
   nstate, fit_scope, fit_strategy, windows, Q, chi2/dof, and logGBF.
3. Call fit_bare_matrix_grid with the selected scalar nstate, fit_scope, and
   fit_strategy, the same scale, and either the selected pt2_window/pt3_window or
   the manifest-controlled model_average setting. This writes the job NetCDF
   and store['output']; do not override model_average.
4. Finish with the NetCDF and diagnostic PDF paths.
""".strip()
