"""Prompt text for one correlator-analysis job."""

STAGE_PROMPT = """
Analyze only the correlators listed for the current job. Manifest-derived paths,
selectors, resampling mode, nstate_values, prior_width, and fit_strategies are
injected into tool calls when omitted. When pt2_windows, pt3_windows, and
pt3_tau_cuts are absent, the tools generate bounded automatic window candidates
from the resampled 2pt signal and available tsep grid; explicit windows are exact
overrides.

fit_strategy selects joint (fit 2pt with ratio), chained (fit 2pt first, then
anchor the ratio prior), or independent (fit ratio/FH/qda_ratio alone with no
2pt channel). fit_scope selects exactly one analysis family for a job.
3pt_ratio, FH, and 3pt_ratio+FH consume 3pt data. qda_ratio constructs
C_qDA(bz,P,t)/C2(P,t) from a nonlocal qDA 2pt and an optional ordinary
local-source/local-sink 2pt.
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
   bz grid. With an ordinary local denominator, include z=0 in tune/fit as usual.
   With the nonlocal bz=0 fallback denominator only: do not include z=0 in
   tune_z_values (choose min/mid/max among z>0); the tools skip fitting z=0
   because the ratio is identically one and write bare ME=1 at z=0 in the
   output NetCDF. Compare returned candidates across nstate, prior_width,
   fit_scope, fit_strategy, windows, Q, chi2/dof, n_data, n_params, and
   cross-z feasibility.
   For data-window selection:
   - only consider candidates with feasible_at_all_tune_z=true;
   - prefer recommended_robust_index; do not pick recommended_index if that
     candidate fails any tune z;
   - among feasible candidates, prefer higher min_Q, lower worst_chi2_dof, then
     more n_data; do not rank different data windows by raw logGBF;
   - if status is "no_common_feasible_candidate" (or no candidate is feasible at
     all tune z values), retry tune_bare_matrix at least once with a narrower
     tune_z_values list: keep the minimum (nonzero, for nonlocal_bz0) z and one
     mid-range z; drop the largest tune z first. Use succeeded_counts_by_z and
     retry_hint from the observation. Only after that retry still fails, call
     request_user_input instead of guessing a primary-z-best window.
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
