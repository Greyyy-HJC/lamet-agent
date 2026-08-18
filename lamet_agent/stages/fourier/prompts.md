# Fourier Transform

## Basic Procedure

Transform the current job's renormalized coordinate-space matrix elements into a
quasi-distribution while preserving every resampled sample.

1. External artifact inputs are pre-loaded before tools run. Do not call
   `load_renormalized_matrix_element_samples` when the input is already in memory.
   Call `run_fourier_transform` directly.
2. Call `run_fourier_transform` once. Job defaults/params and source metadata supply
   y_grid, scheme_scan, method, order, sector, hadron, momentum, output paths, and fit
   controls. Coordinates and fit ranges are fixed in fm; target and parton come from
   run metadata, and the tool derives the observable. Do not override them.
3. The run tool writes the primary NetCDF, fit-info NetCDF, plots, and registers
   store['output']. A single language-selected stage report is written after all Fourier
   jobs finish. Finish by reporting the NetCDF/plot paths plus selected-range
   and fit-model diagnostics; do not call the individual plot/report tools again.

## Stage Skill

Fourier transformation extends finite coordinate-space matrix elements with the
configured asymptotic model, transforms every resampled sample, and preserves
the sample axis in an EnsembleData(x) output. Use the injected manifest contract
for the tail-range, model-averaging, sector, and DA-symmetry definitions; all
input and scan coordinates are physical distances in fm. This
instruction only determines tool order.

## Available Tools

- `load_renormalized_matrix_element_samples`: Load the external NetCDF source for a partial run; skip this for an in-memory upstream input.
- `run_fourier_transform`: Run tail fits, Fourier transform, plots, and write the job NetCDF to store['output']; the runner writes one stage report after all Fourier jobs.
