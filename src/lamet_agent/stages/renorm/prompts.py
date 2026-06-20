"""Prompt text for one renormalization job."""

STAGE_PROMPT = """
The current job store already contains EnsembleData under the roles 'target' and
'denominator'. Do not load either input again.

1. Call apply_ratio_scheme_renormalization without overriding target,
   denominator, scheme, scheme_parameters, or save_path. It applies the declared
   hybrid_ratio scheme and writes store['output'] plus the job NetCDF.
2. Call plot_renormalized_matrix_element; it plots store['output'] to the job PDF.
3. Finish with the NetCDF and PDF paths.
""".strip()
