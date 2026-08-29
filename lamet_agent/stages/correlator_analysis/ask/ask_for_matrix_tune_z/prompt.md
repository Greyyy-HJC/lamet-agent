The supplied ordinary matrix-element evidence contains two-point correlators
`C2(t)` that constrain energies and overlaps, together with three-point data
`C3(tsep,tau,z)` used by the authored ratio or Feynman-Hellmann fit scope. Values
are given as component-wise central values with uncertainties on their declared
time, source-sink-separation, insertion-time, and spatial-separation coordinates.
Small source-sink separations and insertion times near either endpoint are more
susceptible to excited-state contamination, while larger separations are usually
noisier.

Return exactly the fields named by `requested_fields`. Choose 3–5 representative
available z coordinates as `tune_z_values`: include the smallest or most trusted
z, one mid-range z, and the largest z when the grid is wide. Do not enumerate the
full z grid. When requested, choose stable half-open `pt2_windows` and compatible
`pt3_windows`. Every three-point window contains `tsep_ls` and a nonnegative
`tau_cut`; retain enough insertion-time support after the endpoint cut and obey
`2*tau_cut <= tsep`. Parameters listed under `fixed_parameters` are user-authored
for the initial attempt and must not be changed.

On a retry, `previous_attempts` describes every authored strategy, scope, state,
prior-width, and window combination. It may include feasibility at every tuning
z, per-z Q, chi2/dof and logGBF, minimum Q, worst chi2/dof, or numerical failures.
Require viability across all representative z values and judge the weakest point,
not only the easiest one. Use logGBF only to compare compatible fits, and make
conservative runtime adjustments rather than repeating a failed or uniformly
low-quality configuration.

Use only the supplied numerical evidence. Balance physical credibility,
statistical stability, and numerical feasibility; do not infer unavailable data.
