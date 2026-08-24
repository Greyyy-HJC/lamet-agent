Fit one matrix-element spectral model using an authored window. Choose
`joint`, `chained`, or `independent` only from `lsqfit.fit_strategy`, and select
the physical scope from `lsqfit.fit_scope`. This tool fits
only the sample average at `lsqfit.tune_z` so the authored windows can be
compared. It must not fit the full z grid or individual samples. Numerical
correlator rescaling is selected automatically from the inspected two-point
magnitude and is reused for every candidate in the job. A numerically unusable
fit is stored as a rejected candidate, so continue evaluating the remaining
authored candidates instead of retrying the same settings.
