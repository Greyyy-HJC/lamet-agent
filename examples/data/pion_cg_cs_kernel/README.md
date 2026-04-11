# Pion CG Collins-Soper Kernel Example Data

This directory holds the input data for `examples/pion_cg_cs_kernel_manifest.json`.

The data are post-ground-state-fit bare quasi TMDWF matrix elements for the pion
computed with a Coulomb-gauge-fixed HISQ ensemble (a ≈ 0.06 fm, 48³×64, Nf=3,
mπ ≈ 300 MeV).  They are the outputs of a two-state joint ratio-FH fit to
2pt + 3pt correlators and serve as the starting point for the renormalization,
Fourier-transform, and CS-kernel-extraction stages.

## Layout

### Finite-momentum jackknife samples

`bare_quasi_p{px}_b{b}_{re|im}.txt`

- One file per (px, b) combination, for `px ∈ {8, 9, 10}` (lattice units) and
  `b ∈ {0, 2, 4, 6, 8, 10}` (transverse separation in lattice units).
- Shape on disk: `(n_z, n_jk)` — rows are z-displacement indices `z = 0..20`,
  columns are jackknife samples (N ≈ 553).
- `re` files hold the real part; `im` files hold the imaginary part.
- Load with `numpy.loadtxt`.

### p=0 reference (mean ± sdev)

`bare_quasi_p0_b{b}_{re|im}_meansdev.txt`

- One file per b, for `b ∈ {0, 2, 4, 6, 8, 10}`.
- Shape on disk: `(n_z, 2)` — column 0 is the mean, column 1 is the standard
  deviation, both derived from the gvar posterior of the ground-state fit.
- The prepare script (`scripts/prepare_cs_kernel_data.py`) draws Gaussian samples
  from these mean/sdev values to generate a synthetic jackknife ensemble for the
  p=0 normalization reference.

### Literature comparison reference

`cs_kernel_literature_updated.gv`

- Compiled CS kernel results from phenomenology groups and other lattice calculations
  (N³LO, N³LL, MAP22, ART23, DWF24, LPC23, etc.) stored in gvar binary format.
- Tracked by git (small, < 1 MB).
- Used by the `cs_kernel_momentum_fit` evaluation method to produce the
  `cs_kernel_comparison.pdf` plot.

## Access restrictions

The `*.txt` files above are **local-only** (gitignored) because they contain
unpublished measurement data.  Restore them from a trusted backup if they are
absent on a new machine.

## Physical parameters

| Parameter | Value |
|---|---|
| Lattice action | CG52bxyp20 (HISQ Nf=3) |
| Lattice spacing a | 0.06 fm |
| Spatial extent | 48³ |
| Temporal extent | 64 |
| Pion mass (valence) | 300 MeV |
| Momenta px (lattice) | 8, 9, 10 |
| Transverse separations b (lattice) | 0, 2, 4, 6, 8, 10 |
| z displacements | 0..20 |
| Operator | γ₁₄ (g14), u−d |
