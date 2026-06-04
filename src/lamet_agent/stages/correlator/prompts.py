"""Prompt text for correlator analysis stage."""

STAGE_PROMPT = """
Goal: when 2pt data are present, extract ground-state energy and overlaps with
model-averaged 2pt fits and fit-on-data plots. When 3pt data are also present,
extract the bare matrix element from 3pt/2pt ratio fits after the 2pt step.

2pt: the correlator is symmetric about t = Lt/2. Fit only the first half:
tmax <= Lt//2 (tmax exclusive), typically tmin from 1 or 2 upward.

3pt: read every kind=3pt correlator in the manifest (typically tsep 4,6,8,10).
You choose tsep_ls as a non-empty subset of loaded keys and tau_cut for each
fit_pt3_window call (tau_cut >= 1). Tau fit points: tau in [tau_cut, tsep+1-tau_cut). Need at
least 10 combined re+im points for a two-state fit.

Emit one action at a time.

Phase A (2pt, if manifest includes kind=2pt):
1. read_pt2 on the 2pt path (stores pt2_samples and pt2_imag_samples; note Lt).
2. resample_to_gvar -> pt2_gv.
3. fit_window up to six times (append=True); tmin>=1, tmax<=Lt//2,
   tmax-tmin>=2*nstate; judge Q, chi2/dof, E0.
4. Choose window_indices; model_average on scan for E0, log(dE1), z0, and z1
   (creates E0_avg, log(dE1)_avg, z0_avg, z1_avg in the store).
5. plot_fit_on_data with the same window_indices and E0_avg='E0_avg'.

Phase B (3pt, if manifest includes kind=3pt):
6. read_pt3 once per 3pt dataset path in the manifest (append=True); load all
   available tsep (e.g. 4, 6, 8, 10) before fitting.
7. compute_pt3_ratio using pt2_samples, pt2_imag_samples, pt3_samples_re/im.
8. resample_ratio_to_gvar -> ratio_real_gv, ratio_imag_gv.
9. fit_pt3_window at most TWO times (append=True, out='pt3_scan'); each call
   picks one (tsep_ls, tau_cut). Shared 2pt parameters are pinned automatically
   to E0_avg and z0_avg from step 4 (default use_pt2_avg_prior); log(dE1), z1 stay broad.
10. Pick only trustworthy window_indices (prefer Q > 0.05, stable O00_re);
    model_average O00_re and O00_im on that subset of pt3_scan — do not average all windows.
11. plot_pt3_fit_on_data(scan='pt3_scan', window_indices=..., O00_re_avg, Lt=...).

12. finish with 2pt and 3pt results, chosen windows, and artifact PDF paths.

Use only listed tools. Reference earlier outputs by their 'out' keys.
""".strip()
