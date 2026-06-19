"""Prompt text for renormalization stage."""

STAGE_PROMPT = """
Apply ratio/hybrid-scheme renormalization while preserving every resampled
matrix-element sample.

Flow for CG qPDF ratio renormalization:
1. Call load_bare_matrix_element_grid for the target bare matrix elements with
   out='target_bare_matrix_element'. If the correlator stage ran earlier in this
   same agent process, omit netcdf_path so the tool uses the stored EnsembleData.
2. Call load_bare_matrix_element_grid for the P=0 denominator with
   out='denominator_bare_matrix_element'. Use
   metadata.renormalization.denominator_netcdf_path when available.
3. Call apply_ratio_scheme_renormalization with target, denominator, zs, delta_m,
   and m0 from metadata. Defaults are zs=4, delta_m=0, m0=0, z0=0.
4. Call plot_renormalized_matrix_element.
5. Finish with the renormalized NetCDF path and PDF path.
""".strip()
