# qPDF FT Toy Demo Data

- `two_point_qpdf_ft_demo.txt`: toy raw two-point correlator with 32 time slices and 16 configurations.
- `three_point_qpdf_ft_demo_zXX.txt`: toy raw three-point correlators for `z=0..5`.
- These files are small tracked inputs for the default `examples/demo_manifest.json`.
- The demo uses them to exercise the sample-wise `correlator_analysis -> renormalization -> fourier_transform -> perturbative_matching -> physical_limit` workflow without depending on the larger local-only bare-qPDF data package.
