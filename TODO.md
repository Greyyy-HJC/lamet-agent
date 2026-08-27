## Contract

- [ ] Better physics comments
- [x] `max(P84-P50, P50-P16)`

## Artifacts

- [x] Compare artifacts with original
- [ ] Flat folder structure
- [ ] Plots and reports

## Correlator Analysis

- [x] Add one bounded typed parameter re-suggestion when every authored fit candidate fails numerically or remains below the accepted quality policy.

## Renormalization

- [x] Expose kernel parameters
- [x] Restore the original `z_coverage_policy` choices (`strict`, `intersection`, `extrapolate`) before exposing its original `extrapolate` default through `Recommends`.
  - [ ] Or ask LLM

## Fourier Transform

- [x] LLM suggested `z_min` and `z_max` lists
- [x] Repeated LLM requests if failed on fitting

## Perturbative Matching

- [x] New `_alpha_s` from Fei Yao

## Continuum Extrapolation

- [x] Performance with "variance" and "one_sigma" error mode
- [ ] Contract of systematic parameters (Unexpected parameters)

### Codex Hints
- [ ] Avoid serializing the full 2700 x 1201 extrapolation design matrix separately to every worker batch
- [ ] After four-example parity, replace repeated global `lsqfit.nonlinear_fit` calls for the linear joint-x model with a correlated linear solve that reuses the fixed design/covariance/prior factorization across resamples

## Review

- [ ] Implement review
- [ ] Include all previous reports
- [ ] Include related papers (https://ar5iv.labs.arxiv.org/html/)
- [ ] Rely on `literature` module
- [ ] Prompts engineering
