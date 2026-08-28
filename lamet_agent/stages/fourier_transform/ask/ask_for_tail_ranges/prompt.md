Recommend compact candidate lists for the missing Fourier tail-fit boundaries.
Use only coordinates present in the supplied positive-z grid. Each valid pair
must satisfy zmin_fm < zmax_fm, retain enough points for the configured tail
models, and avoid ranges dominated by uncertainty. Fixed parameters are
authored values and must not be changed on the initial attempt. When previous
attempts are supplied, make a conservative runtime adjustment using their Q and
chi2 diagnostics; do not alter scheme_scan.
