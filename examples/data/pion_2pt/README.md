# Pion Two-Point Demo Data

This directory holds the input data for `examples/pion_2pt_manifest.json`.

The data are raw two-point correlator measurements for a pion with zero momentum,
intended for testing and demonstrating the `correlator_analysis` stage.

## Layout

### `two_point_raw_demo.txt`

- Shape on disk: `(64, 315)` — rows are Euclidean time slices `t = 0..63`,
  columns are individual configuration measurements (314 configs) preceded by
  the integer time index in column 0.
- Column 0: integer time separation `t = 0, ..., 63`
- Columns 1–314: raw measurements from 314 lattice configurations
- Space-separated; load with `numpy.loadtxt`.

The stage loader interprets this layout as:

- `axis = raw[:, 0]`
- `samples = raw[:, 1:]`
- `values = samples.mean(axis=1)`

## Physical Parameters

| Parameter | Value |
|---|---|
| Hadron | Pion |
| Momentum | (0, 0, 0) |
| Smearing | SS (source-sink) |
| Boundary condition | Periodic |
| Temporal extent | 64 |
| Configurations | 314 |
| Lattice spacing a | 0.09 fm (demo) |
| Pion mass | 300 MeV (demo) |
