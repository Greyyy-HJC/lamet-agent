For an authored `qda_ratio` job, return exactly the requested fit parameters.
Choose 3–5 representative available **nonzero** z coordinates as
`tune_z_values`: the smallest trusted z>0, one mid-range z, and one larger z
when the grid is wide. Do not enumerate the full z grid. z=0 is the local
current and can have different excited-state contamination, so it must not be
used for tuning. When `pt2_windows` is requested, select a compact ordered set
of stable half-open two-point windows with enough points for every authored
state count. Parameters listed under `fixed_parameters` are user-authored for
the initial attempt and must not be changed. When `previous_attempts` is
present, make conservative runtime adjustments using every combination's Q and
chi2 diagnostics; prefer dropping the worst-Q or largest tune z before adding
more.
