# Extrapolation

Inspect all explicitly listed matched distributions and align by physical `x`
coordinates and provenance before fitting. Candidate term sets must include every
required term and may use only authored allowed terms under `max_terms`. Compare
candidate fits before publishing the continuum and infinite-momentum result.
Only models containing pion-mass terms also evaluate an explicit physical mass.
For `operation="systematics_budget"`, do not run scaling fits. Call
`publish_systematics_budget` directly; its ordered input groups reproduce the
reference envelope and quadrature prescription.
