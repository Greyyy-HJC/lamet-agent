# Extrapolation

The workflow inspects all explicitly listed matched distributions and aligns by physical `x`
coordinates and provenance before fitting. `x_independent_terms` and
`x_dependent_terms` define one exact authored model; no runtime model selection
is performed before publishing the continuum and infinite-momentum result.
Only models containing pion-mass terms also evaluate an explicit physical mass.
For `operation="systematics_budget"`, the workflow directly reproduces the
reference envelope and quadrature prescription from the ordered input groups.
