# Perturbative Matching

## Basic Procedure

Convert the current job's quasi-PDF into a light-cone PDF sample by sample.

1. Call `load_quasi_pdf` without a path. It consumes the job's in-memory Fourier
   output (or an external artifact if declared) and selects the manifest component.
2. Call `build_matching_kernel` without overriding kernel_id, momentum_gev, mu, zs_fm, or lc_x_ls;
   the framework resolves the logical kernel declaration and scheme.
3. Call `apply_matching` once to produce the matched EnsembleData and primary job
   NetCDF under store['output'].
4. Call `plot_matched_pdf`, then finish with the NetCDF, PDF, and SVG paths. A single
   language-selected stage report is written after all matching jobs finish; do not call
   `report_matching_result` unless explicitly asked to regenerate a per-job report.

## Stage Skill

Perturbative matching applies the selected NLO kernel matrix independently to
every quasi-PDF sample. The stage-owned scheme is ratio, hybrid, or msbar and
must match the token in the exact declared kernel_id. Hybrid kernels use zs_fm
and momentum_gev to form z_s P_z.

The report integrates quasi and matched over the range this job actually matched and
states no expected value: whether that integral is 1 depends on whether the matrix
element was normalized at z=0 upstream. Report the numbers as numbers; a value near 1
is not a passed check and a value away from 1 is not a failure.

Two grids, both optional and both taken from the manifest. quasi_y_ls is the grid
the kernel integrates over (its columns); it defaults to the grid the Fourier stage
produced, must stay inside that grid's range, must not contain zero (the kernels
carry a 1/y measure), and must be uniformly spaced. Setting it to anything else
linearly interpolates every quasi sample onto it. lc_x_ls is the grid the matched
PDF comes out on (the kernel's rows); it defaults to the quasi grid and is
otherwise unconstrained -- it may contain zero and need not be uniform.

## Available Tools

- `load_quasi_pdf`: Select the requested real/imaginary component from the job's in-memory or external Fourier input.
- `build_matching_kernel`: Build the manifest-selected NLO matching matrix.
- `apply_matching`: Apply the kernel sample by sample and write the job NetCDF to store['output'].
- `plot_matched_pdf`: Plot quasi and matched PDFs.
- `report_matching_result`: Regenerate an optional per-job English/Chinese report; the runner writes one stage report after all matching jobs.
