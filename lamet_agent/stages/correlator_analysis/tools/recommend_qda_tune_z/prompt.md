For an authored `qda_ratio` job, choose a nonempty set of representative
available **nonzero** z coordinates as `tune_z_values`; z=0 is the exact qDA
denominator and must never be used for tuning. The workflow will scan the
complete strategy, state-count, prior-width, and two-point-window grid and apply
the original robust qDA recommendation. Return only the representative
coordinates; do not enumerate windows yourself.
