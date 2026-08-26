## LLM Calls

- Execution model: each agent turn issues one `backend.complete(...)` request
  and normally selects one deterministic stage tool. Tools do not call the LLM
  themselves. The only separate stage-owned path is the conditional correlator
  `pt2_windows` null-hook. Malformed-protocol retries are repeated attempts, not
  new logical stage decisions.
- Labels: **UNNECESSARY** = validated params/state determine the call;
  **PARTIAL** = only some arguments need scientific judgment; **KEEP** = a real
  scientific or narrative choice remains.
- Four reference examples, excluding Review and protocol retries:
  `pion_pdf_cg=12`, `pion_pdf_gi=12`, `pion_da_gi=224`,
  `kaon_da_gi=224`; total `472` logical requests.

### Correlator Analysis

- Conditional `recommend_pt2_windows -> return_parameter_estimate` when
  `lsqfit.pt2_windows` is absent: **KEEP/PARTIAL** because plateau selection is
  data-dependent, but consolidate it with tuning instead of opening a separate
  conversation. The four current manifests author it and do not trigger this.
- Ordinary least-squares/qDA path, currently three calls per job:
  1. `inspect_correlators`: **UNNECESSARY**; inspection and rescaling are
     mandatory/deterministic.
  2. `fit_matrix_element` (`qda_ratio`) or `fit_matrix_element_model` (ordinary):
     **PARTIAL**. The LLM chooses `tune_z_values`; the tool deterministically
     scans every authored scope, strategy, state, prior, and window. Keep one
     constrained nonzero tuning-coordinate decision until a deterministic policy
     is approved.
  3. `publish_correlator_result(candidate_id=...)`: **UNNECESSARY**; the grid
     result already supplies the original-rule recommendation and publishing
     performs deterministic full-grid preflight/reselection.
- Direct spectrum path, `inspect_correlators -> fit_spectrum ->
  publish_correlator_result`: inspection/publication are **UNNECESSARY**;
  `fit_spectrum` is **KEEP** because its range and ordered prior means/widths are
  currently model decisions.
- Lanczos path, `inspect_lanczos_inputs -> run_lanczos_analysis`: both are
  **UNNECESSARY**; method/scope and all moment-grid/resampling choices are
  explicit or derived deterministically.

### Renormalization

- `inspect_renormalization -> apply_renormalization`: both **UNNECESSARY**;
  scheme, strategy, inputs, scale, normalization, and hybrid switch are explicit.
- `inspect_renormalization -> fit_self_renormalization`: both **UNNECESSARY**;
  operation, fit/remap parameters, and numerical fitting are deterministic.

### Fourier Transform

- Reference `scheme_scan` path, `inspect_long_distance -> run_fourier_scan`:
  both **UNNECESSARY**. Provenance determines conventions and the scan applies
  the original range-selection/model-averaging rule.
- Generic non-`scheme_scan` path:
  1. `inspect_long_distance`: **UNNECESSARY**.
  2. One or more `fit_tail_candidate(...)`: **KEEP/PARTIAL** while the model may
     choose model/range/priors outside an authored scan; use a deterministic scan
     when all candidate axes are authored.
  3. `transform_distribution(candidate_id=...)`: **KEEP/PARTIAL** only while no
     approved deterministic generic-candidate selection rule exists; otherwise
     it is **UNNECESSARY** as an LLM call.

### Perturbative Matching

- `inspect_kernel -> apply_matching`, two calls per job: both **UNNECESSARY**.
  Kernel identity, scheme, scales, grids, component, and parameters are validated;
  document loading and matrix application are deterministic.

### Continuum Extrapolation

- Fit path, currently four calls per job:
  1. `inspect_scaling`: **UNNECESSARY** mandatory alignment/provenance checking.
  2. `fit_extrapolation_candidate(terms, excluded_ensembles)`: **PARTIAL** in the
     generic engine, but **UNNECESSARY for all four reference examples** because
     `allowed_terms=[]`, required terms fix one model, and no ensemble is excluded.
  3. `compare_extrapolations(candidate_ids)`: **UNNECESSARY in the current
     implementation**; it selects one candidate with weight 1 and performs no
     real multi-model comparison.
  4. `publish_extrapolation`: **UNNECESSARY** after selection.
- `operation="systematics_budget" -> publish_systematics_budget`: one
  **UNNECESSARY** call; groups, envelope, and quadrature are authored.

### Review

- Existing sequence: `inspect_results -> check_consistency -> list_literature ->
  read_papers -> write_review`; literature calls depend on catalog/use. Review is
  outside the current parity milestone.
- `inspect_results`, `check_consistency`: **UNNECESSARY** deterministic evidence.
- `list_literature`: **UNNECESSARY** when filters derive from result provenance;
  keep judgment only for intentional scope broadening.
- `read_papers(paper_ids)`: **KEEP** as bounded source selection.
- `write_review(title, analysis, conclusion)`: **KEEP** for synthesis/prose;
  factual tables and findings remain deterministic inputs.

### Proposed reduction order

- [ ] P0: remove all LLM orchestration from Renormalization, reference
  `scheme_scan` Fourier, Perturbative Matching, and systematics budget.
- [ ] P1: directly execute mandatory inspect/publish tools around remaining
  correlator, generic Fourier, and generic extrapolation decisions.
- [ ] P1: reduce reference correlator jobs from three requests to one constrained
  `tune_z_values` decision; merge an unresolved `pt2_windows` hook into it.
- [ ] P1: run reference extrapolation deterministically when required terms fix
  the sole candidate; design a real comparison rule before retaining
  `compare_extrapolations`.
- [ ] P2: make Review inspection, checks, and provenance-scoped literature list
  deterministic; retain paper selection and narrative calls.

## Contract

- [ ] Better physics comments
- [ ] After all four reference examples pass, restore conservative median uncertainty `max(P(50+34.1344746)-P50, P50-P(50-34.1344746))` in `lamet_agent/data.py`; the temporary parity definition is the symmetric half-width between those one-sigma percentiles

## Correlator Analysis

- [ ] After parity evaluation, consider moving full-grid candidate retry policy out of publishing while preserving fail-early diagnostics

## Renormalization

- [ ] Replace the internal fixed self-renormalization ZMS model with the original
  explicit renormalization `kernel_id` selection (`ZMSbar_pdf` or `ZMSbar_da`);
  this is not a `Recommends` default because the observable changes the kernel.
- [ ] Restore the original `z_coverage_policy` choices (`strict`, `intersection`,
  `extrapolate`) before exposing its original `extrapolate` default through
  `Recommends`; Neo currently implements only extrapolation.

## Perturbative Matching

- [ ] Move generation and source/paper consistency validation of kernel `.md` formula documents into the literature module; Matching only consumes the paired document

## Continuum Extrapolation

- [ ] After four-example parity, replace repeated global `lsqfit.nonlinear_fit` calls for the linear joint-x model with a correlated linear solve that reuses the fixed design/covariance/prior factorization across resamples
- [ ] Avoid serializing the full 2700 x 1201 extrapolation design matrix separately to every worker batch

## Review

- [ ] Report
- [ ] Implement review
- [ ] Summary of all stages
- [ ] Represent reference `literature=false` explicitly; `catalog="builtin"` currently corresponds only to `literature=true`
