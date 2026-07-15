"""Prompt text for one perturbative-matching job."""

STAGE_PROMPT = """
Convert the current job's quasi-PDF into a light-cone PDF sample by sample.

1. Call load_quasi_pdf without a path. It consumes the job's in-memory Fourier
   output (or an external artifact if declared) and selects the manifest component.
2. Call build_matching_kernel without overriding kernel_id, momentum_gev, mu, or zs_fm;
   the framework resolves the logical kernel declaration and scheme.
3. Call apply_matching once to produce the matched EnsembleData and primary job
   NetCDF under store['output'].
4. Call plot_matched_pdf, then finish with the NetCDF, PDF, and SVG paths. A single
   language-selected stage report is written after all matching jobs finish; do not call
   report_matching_result unless explicitly asked to regenerate a per-job report.
""".strip()
