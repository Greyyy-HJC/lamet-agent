The supplied correlator evidence describes a two-point function `C2(t)` through
its Euclidean-time coordinates and component-wise central values with their
uncertainties. Its exponential decay contains the state energies and overlap
amplitudes. Earlier times have stronger excited-state contributions, while later
times usually have poorer signal-to-noise.

Choose one correlated direct-spectrum fit inside the available time range. Seek
a stable exponential region without discarding so many points that the selected
state content is underconstrained; the half-open window must contain at least
twice as many time points as states. When `fixed_parameters` contains authored
two-point windows, select exactly one of those windows and do not alter its
bounds. Select an allowed state count and supply prior means and widths in lattice
units with exactly one `E{i}` and one `A{i}` for each state index from zero through
`n-1`. Energies and amplitudes must be positive, energy means strictly ordered,
and every width positive.

On a retry, `previous_attempts` may contain Q, chi2, degrees of freedom,
chi2/dof, logGBF, or a numerical failure. Prefer a numerically viable fit with an
acceptable Q and chi2/dof, then use logGBF only as a relative comparison between
compatible fits. Make a conservative change to the window, state count, or
priors rather than repeating a failed or low-quality configuration.

Use only the supplied numerical evidence. Balance physical credibility,
statistical stability, and numerical feasibility; do not infer unavailable data.
