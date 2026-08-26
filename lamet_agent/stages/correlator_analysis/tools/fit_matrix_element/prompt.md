For an authored `qda_ratio` job, choose a nonempty set of representative
available **nonzero** z coordinates as `tune_z_values`; z=0 is the exact qDA
denominator and must never be used for tuning. Then scan the complete strategy,
state-count, prior-width, and two-point-window grid on their sample averages.
Call this tool exactly once after inspection.
It reports the original robust qDA recommendation; do not enumerate windows
yourself.
