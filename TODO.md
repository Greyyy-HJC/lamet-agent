## LLM Calls

Deferred Fourier scan and generic Extrapolation design is recorded in
[`WORKFLOW_DESIGN.md`](WORKFLOW_DESIGN.md).

- Execution model: Correlator, Renormalization, Matching, and reference
  Extrapolation jobs run through stage-owned workflows. The LLM receives typed
  parameter-estimation requests only when a fit parameter is missing; it does
  not select inspect, fit, comparison, publish, or reporting functions. Fourier
  and Review retain their previous tool loops pending separate design work.
- Labels: **UNNECESSARY** = validated params/state determine the call;
  **PARTIAL** = only some arguments need scientific judgment; **KEEP** = a real
  scientific or narrative choice remains.
- Four reference examples after the first workflow migration, excluding Review,
  protocol retries, and authored-away null hooks: `pion_pdf_cg=4`,
  `pion_pdf_gi=4`, `pion_da_gi=63`, `kaon_da_gi=63`; total `134`
  logical requests. Of these, 22 are typed Correlator fit suggestions and 112
  remain in the deferred Fourier `scheme_scan` tool loop.

### Correlator Analysis

- Conditional `recommend_pt2_windows -> return_parameter_estimate` when
  `pt2_windows` is absent: **KEEP/PARTIAL** because plateau selection is
  data-dependent, but consolidate it with tuning instead of opening a separate
  conversation. The four current manifests author it and do not trigger this.
- Ordinary least-squares/qDA path now performs inspection, the candidate scan,
  deterministic selection, and publication in the stage workflow. The only LLM
  communication is one typed `tune_z_values` suggestion per job.
- Direct spectrum jobs request one typed range/state/prior suggestion; inspection,
  fitting, selection, and publication remain workflow-owned.
- Lanczos is fully deterministic and performs no LLM communication.

### Renormalization

- Renormalization is fully workflow-owned and performs no LLM communication.

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

- Matching is fully workflow-owned and performs no LLM communication.

### Continuum Extrapolation

- Reference fit jobs (`allowed_terms=[]`) and systematics budgets are fully
  workflow-owned and perform no LLM communication. Single-candidate selection is
  ordinary Python logic rather than a tool. Generic term selection is deferred.

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

- [x] Remove LLM orchestration from Renormalization, Perturbative Matching, and
  systematics budget. Fourier `scheme_scan` remains deferred.
- [x] Execute Correlator inspect/selection/publish in the stage workflow.
- [x] Reduce reference Correlator jobs to one typed fit-parameter suggestion.
  Merging an unresolved `pt2_windows` hook into that request remains open.
- [x] Run reference Extrapolation deterministically and remove the single-model
  comparison tool. Generic model selection remains deferred.
- [ ] P2: make Review inspection, checks, and provenance-scoped literature list
  deterministic; retain paper selection and narrative calls.

## Contract

- [ ] Better physics comments
- [ ] After all four reference examples pass, restore conservative median uncertainty `max(P(50+34.1344746)-P50, P50-P(50-34.1344746))` in `lamet_agent/data.py`; the temporary parity definition is the symmetric half-width between those one-sigma percentiles

## Correlator Analysis

- [ ] Add one bounded typed parameter re-suggestion when every authored fit
  candidate fails numerically or remains below the accepted quality policy.
- [ ] After parity evaluation, consider moving full-grid candidate retry policy out of publishing while preserving fail-early diagnostics

## Renormalization

- [x] Replace the internal fixed self-renormalization ZMS model with the original
  explicit renormalization `kernel_id` selection (`z_msbar_pdf_nlo` or `z_msbar_da_nlo`);
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
