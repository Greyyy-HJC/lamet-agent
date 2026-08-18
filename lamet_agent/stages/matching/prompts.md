# Perturbative Matching

## Basic Procedure

Convert the current job's quasi-PDF into a light-cone PDF sample by sample.

1. Call `load_quasi_pdf` without a path. It consumes the job's in-memory Fourier
   output (or an external artifact if declared) and selects the manifest component.
2. Call `build_matching_kernel` without overriding kernel_id, momentum_gev, mu, zs_fm, or lc_x_ls;
   the framework resolves the logical kernel declaration and scheme. `lc_x_ls` is a
   `{start, stop}` window on the Fourier artifact x grid.
3. Call `apply_matching` once to produce the matched EnsembleData and primary job
   NetCDF under store['output'].
4. Call `plot_matched_pdf`, then finish with the NetCDF, PDF, and SVG paths. A single
   language-selected stage report is written after all matching jobs finish; do not call
   `report_matching_result` unless explicitly asked to regenerate a per-job report.

## Stage Skill

Perturbative matching applies the selected NLO kernel matrix independently to
every quasi-PDF sample. Use the injected manifest contract for the authoritative
scheme, kernel, hybrid-scale, component, sector, and grid definitions.

The report integrates quasi and matched over the range this job actually matched and
states no expected value: whether that integral is 1 depends on whether the matrix
element was normalized at z=0 upstream. Report the numbers as numbers; a value near 1
is not a passed check and a value away from 1 is not a failure.

## Available Tools

- `load_quasi_pdf`: Select the requested real/imaginary component from the job's in-memory or external Fourier input.
- `build_matching_kernel`: Build the manifest-selected NLO matching matrix.
- `apply_matching`: Apply the kernel sample by sample and write the job NetCDF to store['output'].
- `plot_matched_pdf`: Plot quasi and matched PDFs.
- `report_matching_result`: Regenerate an optional per-job English/Chinese report; the runner writes one stage report after all matching jobs.
