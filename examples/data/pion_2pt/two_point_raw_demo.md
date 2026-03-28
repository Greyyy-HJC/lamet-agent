# Two-Point Raw Demo Data

`two_point_raw_demo.csv` is a raw two-point correlator sample intended for
testing the `correlator_analysis` stage.

## Layout

- Shape on disk: `(64, 315)`
- Column 0: integer Euclidean time separation `t = 0, ..., 63`
- Columns 1-314: raw measurements from 314 lattice configurations

The file is intentionally left as a plain numeric matrix so it can be consumed
directly by `numpy.loadtxt(..., delimiter=",", comments="#")`.

The stage loader interprets this layout as:

- `axis = raw[:, 0]`
- `samples = raw[:, 1:]`
- `values = samples.mean(axis=1)`
