Use this terminal tool only after `inspect_lanczos_inputs` and only for
`analysis_method="lanczos"`. It performs the manifest-selected outer
jackknife/bootstrap over aligned, binned raw configurations. Every outer sample
then receives an independent inner bootstrap for transfer-matrix construction,
Cullum-Willoughby filtering, and median aggregation. It writes the spectrum or
ground-state matrix element, detailed state matrices for 3pt analysis, and the
standard plot/report artifacts.
