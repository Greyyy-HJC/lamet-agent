# Fourier Transform

## Basic Procedure

Transform the current job's renormalized coordinate-space matrix elements into a
quasi-distribution while preserving every resampled sample.

1. External artifact inputs are pre-loaded before tools run. Do not call
   `load_renormalized_matrix_element_samples` when the input is already in memory.
   Call `run_fourier_transform` directly.
2. Call `run_fourier_transform` once. Job defaults/params and source metadata supply
   quasi_y_ls, scheme_scan, gfix, order, hadron, momentum, output paths, and fit
   controls. A named `sector` resolves the projection automatically; without it,
   `part`, `output_scale`, and `im_flip_for_ft` define the manual projection.
   Correlator-backed jobs inherit `gfix`; jobs reading an external artifact
   declare it in Fourier defaults or job params. Coordinates and fit ranges are fixed
   in fm; target and parton come from run metadata, and the tool derives the observable.
   Do not override them.
   For GPD, `bilocal_anchor` records the bilocal layout (default `mid_at_0`),
   and a nonforward job consumes the exchanged-momentum `hermitian_partner` flow.
   PDF and DA jobs do not use either GPD-only setting. GPD sector outputs are
   projected from the full complex paired Fourier result, not from a pre-fit
   real/imaginary channel selection.
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
