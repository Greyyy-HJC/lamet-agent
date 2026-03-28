# Workflow Smoke Data

- `two_point_qpdf_ft_demo.txt`: tiny raw two-point correlator with 32 time slices and 16 configurations.
- `three_point_qpdf_ft_demo_zXX.txt`: tracked raw three-point correlators for `z=0..5`.
- These files back `examples/workflow_smoke_manifest.json`.
- Their purpose is only to exercise the full `correlator_analysis -> renormalization -> fourier_transform -> perturbative_matching -> physical_limit` stage chain with small repository-tracked inputs.
