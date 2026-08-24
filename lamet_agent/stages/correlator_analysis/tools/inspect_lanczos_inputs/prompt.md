Use this first and only for `analysis_method="lanczos"`. It loads the raw,
configuration-aligned correlators and reports the automatically selected `t0`,
transfer-matrix time step, Lanczos order, and effective two-/three-point moment
grid. For `3pt_matrix`, explicitly report `point_usage_warning` before running
the terminal analysis; ordinary `(tsep,tau)` points outside the complete square
do not enter the estimator.
