# Deferred Workflow Design

This document records unresolved workflow design decisions. It is descriptive,
not an implementation specification; the affected paths remain deferred until
their contracts and statistical policies are approved.

## Fourier Tail Fitting

### Current `scheme_scan` behavior

`scheme_scan` is a two-phase tail-fit and Fourier pipeline rather than a scan of
physical renormalization schemes.

1. Build range candidates from the allowed tail models and every ordered
   `(zmin_fm, zmax_fm)` pair, truncated in authored order by `max_schemes`.
2. Fit every range at the ensemble center using only `order[0]` and
   `posterior_prior_error_scale[0]`.
3. Select one range using `q_min`, Q, and logGBF according to the reference
   range-selection rule.
4. On that fixed range, fit every authored `(order, prior width)` model for all
   resamples.
5. Select a model per resample or perform evidence averaging, according to
   `model_average`, and then execute the Fourier transform.

The first order and prior width therefore have a privileged role in range
selection. `max_schemes` is an ordered truncation, not an unbiased sampling of
the candidate space. These behaviors may be retained for parity, but they must
be documented as algorithmic choices.

### Current contract boundary

The implementation reads `scheme_scan` unconditionally. It is therefore a
required dependency in the current parity workflow. The generic path remains
disabled until an explicit scan-versus-suggested provider is approved; it must
not be represented by making `scheme_scan` optional again.

### Resolved recommendation boundary

`scheme_scan` remains required, fixed, and deterministic. The LLM never selects
order, sector, prior width, tail model, model averaging, or candidate ranking.
Only missing `zmin_fm`/`zmax_fm` lists are jointly recommended through null
hooks. Every valid range is crossed with the complete authored scan.

When no candidate passes `q_min`, one job-bounded recommendation may revise the
runtime ranges using the complete parameter-to-Q/chi2 mapping. User-authored
ranges are fixed on the first attempt but may be temporarily overridden after
that attempt fails; the resolved manifest is never changed. The per-job request
budget is `1 + metadata.parameter_recommendation_retries`.

## Generic Extrapolation

### Current behavior

The contract describes a model space through `required_terms`, `allowed_terms`,
and `max_terms`, but it does not define how a model is chosen from that space.
The migrated workflow supports only the fixed reference case:

```text
allowed_terms=[] -> one model -> weight 1 -> publish
```

Nonempty `allowed_terms` currently raises an explicit not-implemented error.
The old comparison tool was not a multi-model implementation: it required
exactly one candidate and assigned weight 1, so it has been replaced by ordinary
single-candidate selection logic.

`excluded_ensembles` is also not a current decision surface. The fitting code
rejects every nonempty exclusion list.

### Missing statistical policy

A generic model space requires an explicit policy for:

- deterministic enumeration versus one typed LLM suggestion;
- candidate quality rejection;
- AIC, logGBF, Q, or another selection criterion;
- evidence/model averaging versus single-model selection;
- between-model uncertainty;
- global versus per-resample model selection;
- whether ensemble exclusions are authored variations or prohibited.

### Proposed model

Introduce an explicit policy such as:

```text
fixed      one authored model; deterministic
scan       enumerate legal term subsets; requires an approved comparison rule
suggested  request one typed term set; no claim of multi-model comparison
```

Only `fixed` is currently implemented. `suggested` can be added without a
comparison layer because it produces one model. `scan` must remain disabled
until its model-selection and uncertainty semantics are approved. Arbitrary LLM
ensemble exclusions should remain prohibited; exclusions, if restored, should
be authored systematic variations.

## LLM Boundary

For non-Review workflows, the LLM should never select or invoke inspection,
numerical fitting, comparison, publication, artifact, or reporting functions.
It may provide a typed fit-parameter suggestion when parameters are absent, and
later may provide one bounded revision when every allowed candidate fails or
misses an approved quality policy.

The suggestion boundary is:

```text
EnsembleData
  -> JSON evidence containing coordinates, mean, uncertainty, and constraints
  -> prompt-only recommendation definition
  -> typed JSON response
  -> contract validation
  -> deterministic stage workflow
```

Free-form text is not a parameter source. Numerical data and artifacts remain
`EnsembleData` or stage-owned files throughout the workflow.
