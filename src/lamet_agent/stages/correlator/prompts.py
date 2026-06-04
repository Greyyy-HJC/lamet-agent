"""Prompt text for correlator analysis stage."""

STAGE_PROMPT = """
Goal: extract the ground-state energy and matrix element from the 2pt
correlator, with a model-averaged result and a fit-on-data plot.

The 2pt correlator is symmetric about t = Lt/2. Fit only the first half:
tmax <= Lt//2 (tmax is exclusive), typically tmin from 1 or 2 upward.

Do this by emitting one action at a time:
1. read_pt2 on the 2pt dataset path (note Lt).
2. resample_to_gvar to obtain a gvar correlator.
3. fit_window at most six times (append=True) with [tmin, tmax) in the first
   half (tmin >= 1, tmax <= Lt//2, tmax - tmin >= 2*nstate); use index, Q,
   chi2/dof, and E0 to judge quality. Prefer a fixed tmax=Lt//2 and a few tmin.
4. Choose window_indices for the trustworthy windows.
5. model_average E0 and z0 with window_indices=...
6. plot_fit_on_data with the same window_indices and E0_avg='E0_avg' (PDFs are
   written under artifacts/ automatically; do not choose a directory).
7. finish, reporting model-averaged E0 and z0 (stat/sys), chosen windows,
   and plot paths under artifacts/.

Use only the listed tools. Reference earlier outputs by their 'out' keys.
""".strip()
