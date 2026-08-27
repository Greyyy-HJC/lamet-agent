For an authored `qda_ratio` job, return exactly the requested fit parameters.
Choose representative available **nonzero** z coordinates as `tune_z_values`;
z=0 is the exact qDA denominator and must never be used for tuning. When
`pt2_windows` is requested, select a compact ordered set of stable half-open
two-point windows with enough points for every authored state count. Parameters
listed under `fixed_parameters` are user-authored for the initial attempt and
must not be changed. When `previous_attempts` is present, make conservative
runtime adjustments using every combination's Q and chi2 diagnostics.
