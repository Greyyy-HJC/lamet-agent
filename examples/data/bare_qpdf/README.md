# Bare qPDF Example Data

- `two_point_ss_p440.txt`: SS two-point raw data for `p=(4,4,0)`, built from the averaged `Tg0/Tg8` real correlator in the legacy non-zero-momentum dataset.
- `three_point_ss_ud_gt_p440_b0_zXX.txt`: SS `u-d` `gamma_t` three-point raw data for `p=(4,4,0)`, `b=0`, and `z=0..20`.
- The packaged three-point files follow the `proton_cg_pdf` `PX4_PY4_PZ0` workflow: `tsep in [8, 9, 10, 11, 12]`.
- `tsep=8,10,12` come from the legacy non-zero-momentum HDF5 files, while `tsep=9,11` come from the newer combined-data HDF5 files.
- Each three-point file stores the first 12 tau slots for every `tsep`, so the TXT loader can reconstruct one rectangular `(tsep, tau, cfg)` array.
- All files are unresampled raw measurements.
- The repository manifest uses `correlators[].expand` to generate the full
  `b/z` family from one three-point entry instead of listing every file
  explicitly.
- `examples/bare_qpdf_manifest.json` uses these files for the coordinate-space
  bare-qPDF workflow.
- `examples/qpdf_ft_manifest.json` reuses the same raw data for the sample-wise
  asymptotic extrapolation and Fourier-transform workflow.
