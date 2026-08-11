The current job store already contains the bare EnsembleData roles declared by
this job. Do not reload them. When manifest defaults set normalization=true, the
runner has already divided each bare matrix element by its lattice z=0 value.

Follow the scheme and strategy declared in the job parameters:

strategy=external_denominator (scheme=ratio or hybrid):
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

Stage skill:
Renormalization consumes bare EnsembleData from the current job store.
When normalization=true, the runner already divides each bare matrix element by
its lattice z=0 value before any tool calls.

Physical scheme and execution strategy are independent. scheme is ratio,
hybrid, or msbar; strategy is external_denominator or self_renormalization.

The external_denominator strategy uses roles target and denominator. ratio
divides them pointwise on the complete z grid. hybrid additionally requires
flat job/defaults parameter zs_fm; m0_gev and delta_m_gev remain in
scheme_parameters.
The self_renormalization strategy combines a full-z self-renormalization fit with
short-distance MSbar matching to fix the finite renormalization. It splits into:
- scheme_parameters.LambdaQCD_gev is required and is used by the
  self-renormalization ansatz; use one value throughout the fit/apply chain.
- fit job inputs {reference}: require scheme_parameters.d; fit the reference-operator m0 from
  short-distance g(z), use one discretization family, and never extrapolate
  beyond the reference grid. Writes store['output']/store['zR'].
- ratio/msbar apply inputs are {target, zR}; ratio applies H/(zR*ZMSbar)
  while msbar applies H/zR.
- hybrid apply inputs are {target, denominator, zR}; short distances use
  target/denominator and long distances use target/(zR*ZT), with constant ZT
  fixed at zs_fm for continuity.
- Optional scheme_parameters.d / m0_gev remap upstream zR (e.g. PDF-fit factor
  to DA d/m0).
  scheme_parameters.z_coverage_policy extrapolates the single-family
  long-distance f1 by default when the target
  extends past zR; strict and intersection remain explicit alternatives.

Available tools:
- apply_ratio_scheme_renormalization: external_denominator strategy: consume target/denominator and apply the ratio or hybrid scheme.
- fit_self_renormalization_factor: self_renormalization fit job: fit zR using scheme_parameters (including required LambdaQCD_gev and d); short-distance MSbar matching fixes m0.
- apply_self_renormalization: self_renormalization apply job: apply the declared ratio, hybrid, or msbar scheme; optional scheme_parameters d/m0_gev remap zR.
- plot_self_renormalization_diagnostics: self_renormalization: fit-job panels, or apply-job zmsbar_compare (+ stage-level discrete_effect once).
- plot_renormalized_matrix_element: Plot store['output'] to PDF (apply jobs).
