# Extrapolation

The workflow inspects all explicitly listed matched distributions and aligns by physical `x`
coordinates and provenance before fitting. Candidate term sets must include every
required term and may use only authored allowed terms under `max_terms`. Compare
candidate fits before publishing the continuum and infinite-momentum result.
Only models containing pion-mass terms also evaluate an explicit physical mass.
For `operation="systematics_budget"`, the workflow directly reproduces the
reference envelope and quadrature prescription from the ordered input groups.
