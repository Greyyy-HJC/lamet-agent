# Correlator analysis

Load the declared correlator descriptor, inspect its coordinates and resampling
coverage, then follow the implementation selected by `analysis_method`.

For `analysis_method="lsqfit"`, choose only authored `fit_scope` and
`fit_strategy` candidates. Fit windows and priors are model decisions inside the
authored `lsqfit.time_range`; they are not guessed
from missing manifest values. Numerical correlator rescaling is computed during
inspection from each job's two-point magnitudes and is not an LLM decision or a
manifest parameter. Spectral-model candidate tools compare the authored
windows using only the sample average at `lsqfit.tune_z`; they do not fit every
sample or the full z grid. Call `publish_correlator_result` only after every
authored candidate has been evaluated. Publishing applies the original
information-preserving data-window rule and fits the selected candidate once to
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
