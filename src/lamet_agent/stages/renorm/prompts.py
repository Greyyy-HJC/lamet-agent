"""Prompt text for one renormalization job."""

STAGE_PROMPT = """
The current job store already contains the bare EnsembleData roles declared by
this job. Do not reload them. When manifest defaults set normalization=true, the
runner has already divided each bare matrix element by its lattice z=0 value.

Follow the scheme and strategy declared in the job parameters:

strategy=ratio (scheme=ratio or hybrid):
1. Call apply_ratio_scheme_renormalization without overriding target,
   denominator, scheme, scheme_parameters, or save_path.
   ratio divides target(z) by denominator(z) on the complete grid and ignores
   hybrid-only parameters. scheme=hybrid uses the declared zs_fm switch.
2. Call plot_renormalized_matrix_element; it plots store['output'] to the job PDF.
3. Finish with the NetCDF and PDF paths.

strategy=self_renormalization fit job (inputs exactly {reference}):
1. Call fit_self_renormalization_factor with no arguments. The runner binds
   reference, kernel_id, d, mu, LambdaQCD_gev, svdcut, and save_path.
   scheme_parameters.d is required and fixed
   for the gz fit and zR. The reference-operator m0 is fitted from short-distance g(z), and the
   fit never extrapolates outside the reference grid. It writes store['zR'],
   store['output'], and store['self_renorm_fit'], plus the zR NetCDF.
2. Call plot_self_renormalization_diagnostics without overriding mode or
   save_path. Fit mode writes fit-only diagnostic PDFs once (no fit_vs_data,
   no m0 panel).
3. Finish with the zR NetCDF and fit diagnostic PDF paths.

strategy=self_renormalization apply job:
ratio/msbar inputs are exactly {target, zR}; hybrid inputs are exactly
{target, denominator, zR}.
1. Call apply_self_renormalization with no arguments. The runner binds target,
   zR, kernel_id, d, m0_gev, mu, LambdaQCD_gev, z_coverage_policy, and save_path. Upstream zR is
   already in the store; do not re-fit.
   Optional scheme_parameters.d / scheme_parameters.m0_gev remap that zR onto this operator (e.g.
   PDF-fit zR → DA d/m0). ratio computes H/(zR*ZMSbar), msbar computes H/zR,
   and hybrid uses target/denominator below zs_fm and target/(zR*ZT) above it,
   with ZT fixed by continuity. It writes store['output'] plus
   the job NetCDF. With z_coverage_policy=extrapolate, the tool automatically
   extends the inferred long-distance f1 and rebuilds zR only where the target
   grid exceeds the fitted zR range.
2. Call plot_self_renormalization_diagnostics without overriding mode,
   save_path, sibling_artifacts, or include_discrete_effect. Apply mode writes
   zmsbar_compare; discrete_effect is emitted only on the last apply job when
   sibling NetCDFs are available, as stage-level momentum-specific
   discrete_effect_<momentum>_re/im plots (no job-id prefix).
3. Call plot_renormalized_matrix_element; it plots store['output'] to the job PDF.
4. Finish with the NetCDF and PDF paths.
""".strip()
