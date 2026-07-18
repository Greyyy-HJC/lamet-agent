"""Prompt text for one correlator-analysis job."""

STAGE_PROMPT = """
Analyze only the correlators listed for the current job. Manifest-derived paths,
selectors, resampling mode, candidate windows, nstate_values, prior_width, and
fit_strategies are injected into tool calls when omitted.

fit_scope selects exactly one analysis family for a job. 3pt_ratio, FH, and
3pt_ratio+FH consume 3pt data. qda_ratio constructs C_qDA(bz,P,t)/C2(P,t)
from a nonlocal qDA 2pt and an optional ordinary local-source/local-sink 2pt.
When the ordinary input is absent, the qDA operator's bz=0 correlator supplies
C2 and uses the mixed overlap z_n*zprime_n; the extracted matrix element is
O00/zprime0 instead of O00/z0. qda_ratio has no 3pt data, tsep, tau_cut, or
current operator.

1. Call inspect_correlator_scale and choose a power-of-ten correlator_rescale that
   puts typical fitted 2pt values in 0.0001..0.01.
2. Call tune_bare_matrix with that scale and required tune_z_values. Choose
   tune_z_values from the job bz list in the stage context: include the minimum z,
   at least one mid-range z, and the maximum z; use 3-5 values when the grid is
   wide. Put the smallest or most trusted z first in tune_z_values.
   For qda_ratio, choose representative values directly from the qDA input's
   bz grid. With the bz=0 fallback denominator, the bz=0 ratio is naturally
   normalized because numerator and denominator use the same resampled data.
   Compare returned candidates across nstate, prior_width, fit_scope,
   fit_strategy, windows, Q, chi2/dof, n_data, n_params, and cross-z feasibility.
   For data-window selection:
   - only consider candidates with feasible_at_all_tune_z=true;
   - prefer recommended_robust_index; do not pick recommended_index if that
     candidate fails any tune z;
   - among feasible candidates, prefer higher min_Q, lower worst_chi2_dof, then
     more n_data; do not rank different data windows by raw logGBF;
   - if no candidate is feasible at all tune z values, call request_user_input
     instead of guessing a primary-z-best window.
3. Call fit_bare_matrix_grid with the selected fit_scope and fit_strategy, the
   same scale, and the selected pt2_window/pt3_window from the robust candidate.
   Use pt2_window={"tmin": ..., "tmax": ...} and
   pt3_window={"tsep_ls": [...], "tau_cut": ...} for 3pt/FH scopes; qda_ratio
   uses only pt2_window. Do not pass bare tmin/tmax or tau_cut keys. The
   manifest-controlled model_average setting controls
   fit-function averaging only; do not override model_average. When
   model_average is true, do not pass a scalar nstate or prior_width selected
   from tune_bare_matrix; leave them omitted so the manifest nstate_values and
   prior_width scan remain active.
4. Finish with the NetCDF and diagnostic PDF paths.
""".strip()
