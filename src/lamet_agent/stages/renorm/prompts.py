"""Prompt text for renormalization stage."""

STAGE_PROMPT = """
Apply ratio/hybrid-scheme renormalization deterministically and preserve every
resampled matrix-element sample. Emit one action at a time.

Flow for CG qPDF ratio renormalization:
1. Call load_bare_matrix_element_grid for the target bare matrix elements with
   out='target_bare_matrix_element'. If the correlator stage ran earlier in this
   same agent process, omit report_json so the tool uses store['bare_matrix_grid_report'].
2. Call load_bare_matrix_element_grid for the P=0 denominator with
   out='denominator_bare_matrix_element'. The manifest metadata supplies
   renormalization.denominator_report_json when available.
3. Call apply_ratio_scheme_renormalization with target, denominator, zs, delta_m,
   and m0 from metadata. Defaults are zs=4, delta_m=0, m0=0, z0=0, matching the
   Coulomb-gauge setup of Eq. 15 in arXiv:2306.14960. The tool writes a
   sample-preserving NPZ and stores matrix_element_data for Fourier.
4. Call plot_renormalized_matrix_element.
5. Finish with the renormalized NPZ path and PDF path.

Use only the listed tools. Do not average samples except inside the plotting tool.
""".strip()
