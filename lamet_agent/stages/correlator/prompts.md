# Correlator Analysis

## Basic Procedure

Analyze only the correlators listed for the current job. Manifest-derived paths,
selectors, resampling mode, nstate_values, prior_width, and fit_strategies are
injected into tool calls when omitted. When pt2_windows and pt3_windows are
absent, the tools generate bounded automatic window candidates from the
resampled 2pt signal and available tsep grid; explicit windows are exact
overrides.
Use the injected manifest parameter contract for the authoritative meanings of
fit_scope, fit_strategy, fitting_form, model_average, and their compatibility
rules.

1. Call `inspect_correlator_scale` and choose a power-of-ten correlator_rescale that
   puts typical fitted 2pt values in 0.0001..0.01.
2. Call `tune_bare_matrix` with that scale and required tune_z_values. Choose
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
     all tune z values), retry `tune_bare_matrix` at least once with a narrower
     tune_z_values list: keep the minimum (nonzero, for nonlocal_bz0) z and one
     mid-range z; drop the largest tune z first. Use succeeded_counts_by_z and
     retry_hint from the observation. Only after that retry still fails, call
     request_user_input instead of guessing a primary-z-best window.
3. Call `fit_bare_matrix_grid` with the selected fit_scope and fit_strategy, the
   same scale, and the selected pt2_window/pt3_window from the robust candidate.
   Use pt2_window={"tmin": ..., "tmax": ...} and
   pt3_window={"tsep_ls": [...], "tau_cut": ...} for 3pt/FH scopes; qda_ratio
   uses only pt2_window. Do not pass bare tmin/tmax or tau_cut keys. The
   manifest-controlled model_average setting controls
   fit-function averaging only; do not override model_average. When
   model_average is true, do not pass a scalar nstate or prior_width selected
   from `tune_bare_matrix`; leave them omitted so the manifest nstate_values and
   prior_width scan remain active.
4. Finish with the NetCDF and diagnostic PDF paths.

## Stage Skill

Correlator-analysis physics:
- Fit the symmetric 2pt correlator only in the first half of the lattice.
- Form 3pt/2pt ratios after resampling both correlators with shared indices.
- Construct FH data by summing ratio data over tau after the configured cuts and
  finite-differencing neighboring source-sink separations.
- Tune data windows on sample-average data at multiple representative z values
  chosen by the agent. `fit_bare_matrix_grid` then keeps one shared window and
  applies the contract-selected fit-function policy sample by sample.
- When manifest windows are omitted, generate bounded 2pt candidates from the
  first-half resampled signal and 3pt candidates from the available tsep grid.
  Explicit pt2_windows and pt3_windows remain exact overrides.
- A shared data window must pass sample-average fits at every tune z the
  agent selects; a good chi2/dof at only the smallest tune z is not sufficient.
- Data-window candidates with different pt2/pt3 points should not be ranked by
  raw logGBF. Choose windows after the Q and n_data > n_params gates, favoring
  cross-z feasibility, good chi2/dof, and more data points when chi2/dof values
  are comparable.
- Preserve the estimator normalization declared by the contract; all exported
  bare matrix elements remain invariant under the common 2pt rescaling.

## Available Tools

- `inspect_correlator_scale`: Inspect the selected job's 2pt magnitude.
- `tune_ground_state`: Optionally scan 2pt-only windows and model-average the
  selected ground-state fits.
- `tune_bare_matrix`: Scan every configured nstate, prior_width, fit strategy, and explicit or automatic fit window
  at LLM-supplied tune_z_values; return cross-z feasibility and
  recommended_robust_index. For qda_ratio, when no shared window works at
  every tune z, returns status='no_common_feasible_candidate' with
  succeeded_counts_by_z and retry_hint instead of raising. For
  nonlocal_bz0, z=0 is dropped from tune_z_values automatically.
- `fit_bare_matrix_grid`: Apply one shared data window, optionally model-average fit functions per sample,
  and write store['output']; the runner writes one stage report with fit_logs links.
  For nonlocal_bz0 qda_ratio, skips fitting z=0 and assigns bare ME=1 there.
