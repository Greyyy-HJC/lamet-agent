"""Prompt text for one renormalization job."""

STAGE_PROMPT = """
The current job store already contains the bare EnsembleData roles declared by
this job. Do not reload them. When manifest defaults set normalization=true, the
runner has already divided each bare matrix element by its lattice z=0 value.

Follow the scheme declared in the job parameters:

hybrid_ratio:
1. Call apply_ratio_scheme_renormalization without overriding target,
   denominator, scheme, scheme_parameters, or save_path.
2. Call plot_renormalized_matrix_element; it plots store['output'] to the job PDF.
3. Finish with the NetCDF and PDF paths.

self_renormalization fit job (inputs exactly {reference}):
1. Call fit_self_renormalization_factor without overriding reference, kernel_id,
   d, m0_gev, mu, svdcut, or save_path. Job params.d is required (fixed for the
   gz fit and initial zR). params.m0_gev is optional: omit to fit m0 from
   short-distance g(z); set it to freeze. It writes store['zR'], store['output'],
   and store['self_renorm_fit'], plus the zR NetCDF.
2. Call plot_self_renormalization_diagnostics without overriding mode or
   save_path. Fit mode writes fit-only diagnostic PDFs once (no fit_vs_data,
   no m0 panel).
3. Finish with the zR NetCDF and fit diagnostic PDF paths.

self_renormalization apply job (inputs exactly {target, zR}):
1. Call apply_self_renormalization without overriding target, zR, kernel_id,
   d, m0_gev, or save_path. Upstream zR is already in the store; do not re-fit.
   Optional params.d / params.m0_gev remap that zR onto this operator (e.g.
   PDF-fit zR → DA d/m0) before H/(zR*ZMSbar). It writes store['output'] plus
   the job NetCDF.
2. Call plot_self_renormalization_diagnostics without overriding mode,
   save_path, sibling_artifacts, or include_discrete_effect. Apply mode writes
   zmsbar_compare; discrete_effect is emitted only on the last apply job when
   sibling NetCDFs are available, as stage-level discrete_effect_re/im (no
   job-id prefix).
3. Call plot_renormalized_matrix_element; it plots store['output'] to the job PDF.
4. Finish with the NetCDF and PDF paths.
""".strip()
