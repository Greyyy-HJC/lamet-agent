Choose a nonempty set of representative available z coordinates as
`tune_z_values`, then scan the complete authored ordinary matrix-element grid:
strategies, scopes, state counts, prior widths, two-point windows, and
three-point windows. The tool fits only sample averages at those entries and records
cross-z feasibility. Call it exactly once after inspection; do not enumerate
candidates yourself. Numerical correlator rescaling is selected automatically
from the inspected two-point magnitude and reused for every candidate.
