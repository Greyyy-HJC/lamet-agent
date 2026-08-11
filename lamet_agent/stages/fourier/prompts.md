# Fourier Transform

## Basic Procedure

Transform the current job's renormalized coordinate-space matrix elements into a
quasi-distribution while preserving every resampled sample.

1. External artifact inputs are pre-loaded before tools run. Do not call
   `load_renormalized_matrix_element_samples` when the input is already in memory.
   Call `run_fourier_transform` directly.
2. Call `run_fourier_transform` once. Job defaults/params and source metadata supply
   y_grid, scheme_scan, method, observable, order, sector, coordinate units, lattice
   spacing, momentum, output paths, and fit controls. Do not override them.
3. The run tool writes the primary NetCDF, fit-info NetCDF, plots, and registers
   store['output']. A single language-selected stage report is written after all Fourier
   jobs finish. Finish by reporting the NetCDF/plot paths plus selected-range
   and fit-model diagnostics; do not call the individual plot/report tools again.

## Stage Skill

Fourier transformation extends finite coordinate-space matrix elements with the
configured asymptotic model, transforms every resampled sample, and preserves
the sample axis in an EnsembleData(x) output. Fit ranges are selected once from
sample-average tail-fit diagnostics over the configured zmin/zmax grid. After
that range is fixed, scheme_scan.model_average controls per-sample averaging
over fit-model candidates defined by order and posterior_prior_error_scale;
the method argument is a fixed theory choice and is not scanned.
For DA only, symmetry_guarantee defaults to true: rotate by exp(+i*z*Pz/2),
discard the rotated imaginary part, rotate the retained real part back by
exp(-i*z*Pz/2), then run the ordinary extension and Fourier transform.
Set it false to use the DA input unchanged. It has no effect for PDF or GPD.

## Available Tools

- `load_renormalized_matrix_element_samples`: Load the external NetCDF source for a partial run; skip this for an in-memory upstream input.
- `run_fourier_transform`: Run tail fits, Fourier transform, plots, and write the job NetCDF to store['output']; the runner writes one stage report after all Fourier jobs.
