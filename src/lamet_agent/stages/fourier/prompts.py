"""Prompt text for one Fourier-transform job."""

STAGE_PROMPT = """
Transform the current job's renormalized coordinate-space matrix elements into a
quasi-distribution while preserving every resampled sample.

1. External artifact inputs are pre-loaded before tools run. Do not call
   load_renormalized_matrix_element_samples when the input is already in memory.
   Call run_fourier_transform directly.
2. Call run_fourier_transform once. Job defaults/params and source metadata supply
   y_grid, scheme_scan, method, observable, order, sector, coordinate units, lattice
   spacing, momentum, output paths, and fit controls. Do not override them.
3. The run tool writes the primary NetCDF, fit-info NetCDF, plots, and registers
   store['output']. A single bilingual stage report is written after all Fourier
   jobs finish. Finish by reporting the NetCDF/plot paths plus selected-range
   and fit-model diagnostics; do not call the individual plot/report tools again.
""".strip()
