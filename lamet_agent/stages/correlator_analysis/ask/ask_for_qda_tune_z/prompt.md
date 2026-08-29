The supplied qDA evidence describes a nonlocal correlator `C(t,z)` through its
time and spatial-separation coordinates, with central values and uncertainties
for the requested real or imaginary components. The fit forms the nonlocal ratio
to the aligned z=0 local-current correlator. The local point can have different
excited-state contamination and is therefore a denominator, not a representative
nonlocal tuning point.

For an authored `qda_ratio` job, return exactly the fields named by
`requested_fields`. Choose 3–5 representative available **nonzero** z coordinates
as `tune_z_values`: the smallest trusted z>0, one mid-range z, and one larger z
when the grid is wide. Do not enumerate the full z grid. When `pt2_windows` is
requested, balance early-time excited-state contamination against late-time
noise and select a compact ordered set of stable half-open windows with enough
points for every authored state count. Parameters listed under
`fixed_parameters` are user-authored for the initial attempt and must not be
changed.

On a retry, `previous_attempts` describes the authored model/window combinations
and may include feasibility at every tuning z, per-z Q, chi2/dof and logGBF,
minimum Q, worst chi2/dof, or numerical failures. Require a candidate to remain
viable across all representative z values; do not judge quality from only the
easiest point. Use logGBF only to compare compatible fits, and make conservative
runtime adjustments rather than repeating a failed or uniformly low-quality
configuration.

Use only the supplied numerical evidence. Balance physical credibility,
statistical stability, and numerical feasibility; do not infer unavailable data.
