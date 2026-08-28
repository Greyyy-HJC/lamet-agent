Return exactly the requested ordinary matrix-element fit parameters. Choose
3–5 representative available z coordinates as `tune_z_values`: include the
smallest or most trusted z, one mid-range z, and the largest z when the grid is
wide. Do not enumerate the full z grid. When requested, choose stable half-open
`pt2_windows` and compatible `pt3_windows`, where every three-point window
contains `tsep_ls` and a nonnegative `tau_cut` satisfying `2*tau_cut <= tsep`.
Parameters listed under `fixed_parameters` are user-authored for the initial
attempt and must not be changed. When `previous_attempts` is present, make
conservative runtime adjustments using every combination's Q and chi2
diagnostics; prefer dropping the worst-Q or largest tune z before adding more.
