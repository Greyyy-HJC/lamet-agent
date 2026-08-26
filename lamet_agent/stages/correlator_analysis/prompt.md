# Correlator analysis

Load the declared correlator descriptor, inspect its coordinates and resampling
coverage, then follow the implementation selected by `analysis_method`.

For `analysis_method="lsqfit"`, choose only authored `fit_scope` and
`fit_strategy` candidates. Fit windows and priors are model decisions inside the
available correlator time coordinates; they are not guessed
from missing manifest values. Numerical correlator rescaling is computed during
inspection from each job's two-point magnitudes and is not an LLM decision or a
manifest parameter. Spectral-model candidate tools deterministically scan every
authored strategy, scope, state count, prior width, and fit window using only
sample averages at the tool-call `tune_z_values`; qDA tuning must use nonzero z
because z=0 is its exact denominator. Candidate tools do not fit every sample
or the full z grid. Call the one grid tool appropriate to the authored scope,
then pass its exact recommended candidate id to
`publish_correlator_result`. Publishing applies the original
information-preserving data-window rule and fits only the selected candidate to
the full z grid and all samples, then stores the result.

For `analysis_method="lanczos"`, do not use the least-squares
tools. Call `inspect_lanczos_inputs`, report its effective moment grid and any
discarded ordinary `(tsep,tau)` points, then call `run_lanczos_analysis`.
Lanczos consumes configuration-level correlators and owns its nested resampling:
the manifest-selected jackknife/bootstrap is the outer distribution, while each
outer sample receives the independent inner bootstrap declared by
`lanczos.inner_samples`. The Lanczos order is inferred from the available
two-point times and complete three-point square; it is never selected by the
model.
