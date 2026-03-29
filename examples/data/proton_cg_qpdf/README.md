# Proton CG qPDF Example Data

- `two_point_ss_p440.txt`: SS two-point raw data for `p=(4,4,0)`. The TXT layout is `t`, then all real-part configurations, then all imaginary-part configurations.
- `three_point_ss_ud_gt_p440_b0_z{z}.txt`: SS `u-d` `gamma_t` three-point raw data for `p=(4,4,0)`, `b=0`, and `z=0..20`.
- The three-point files cover `tsep in [8, 9, 10, 11, 12]`.
- Each three-point file stores the first 12 tau slots for every `tsep`, so the TXT loader reconstructs one rectangular `(tsep, tau, cfg)` array.
- All files are unresampled raw measurements.
- These TXT files are local-only real-data inputs and must not be committed.
- `examples/proton_cg_qpdf_manifest.json` is the canonical local-data proton CG qPDF workflow and replaces the older split `bare_qpdf` / `qpdf_ft` example naming.
