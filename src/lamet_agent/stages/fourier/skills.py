"""Stage-local helpers for Fourier-transform stage."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest

STAGE_SKILL = """
Fourier-transform skill: extend coordinate-space matrix elements and transform
them to momentum space while preserving resampling samples.

Strategy:
- Load real/imaginary matrix-element samples from either an NPZ file with coord,
  re_samples, and im_samples, or an HDF5 file with a group such as Pz=4 that
  contains z_ary, Re, and Im. For HDF5 bootstrap files where Re/Im are shaped
  (n_z, n_sample), pass sample_axis=1 to run_fourier_transform.
- Use explicit units. coord_unit='lambda' means the transform coordinate is
  already Ioffe time. coord_unit='fm', 'gev_inv', or 'lattice' requires pz_gev
  for the Fourier phase; lattice also requires a_fm. For physical coordinate
  units, k_grid is the momentum fraction y in Pz dz/(2*pi) exp(i y Pz z) h(z).
- Choose method='GI' or 'CG' and order='LA', 'NLA', or 'Empirical'. CG adds a
  power-law exponent in the asymptotic tail for LA/NLA. Empirical uses Eq. (6)
  of arXiv:2208.08008 in the large-lambda region.
- Choose observable to select the LA/NLA formula block from arXiv:2601.12189:
  pion_quark_quasi_pdf uses Eqs. (2.1)/(2.2);
  nucleon_quark_unpolarized_quasi_pdf uses Eqs. (2.3)/(2.4);
  nucleon_quark_transversity_quasi_pdf uses Eqs. (2.5)/(2.6);
  meson_quasi_da uses Eqs. (2.7)/(2.8);
  pion_quark_quasi_gpd uses Eqs. (2.9)/(2.10);
  nucleon_quark_quasi_gpd uses Eqs. (2.11)/(2.12).
  For GPD observables, pass pz_prime_gev if P'^z differs from P^z.
- zmin must be positive because NLA and CG forms are singular at zero.
- Scan a small list of schemes with zmin, zmax, z_ext_max, smooth='linear' or
  'none'. Linear smoothing uses data before blend_start and fit after zmax.
- If scheme_scan is provided, pass it to run_fourier_transform instead of
  manually selecting one scheme. The tool will generate zmin/zmax combinations,
  compute data-vs-fit chi2/dof, compute y-range roughness from the Fourier
  curve, and model-average schemes with score-based weights.
- After run_fourier_transform, call summarize_fourier_result, plot_fourier_result,
  and plot_fourier_extension_quality_result. Finish with the NPZ artifact path,
  plot paths, best scheme, scheme weights, chi2/dof, roughness scores, fit
  failure counts, and stat/sys errors.
""".strip()

TOOL_CATALOG = {
    "load_renormalized_matrix_element_samples": "load_renormalized_matrix_element_samples(path, input_format='npz'|'h5', h5_group=None, h5_pz=None, coord_key='coord' or 'z_ary', re_key='re_samples' or 'Re', im_key='im_samples' or 'Im', out='matrix_element') -> load renormalized coordinate-space samples from NPZ or HDF5.",
    "run_fourier_transform": "run_fourier_transform(samples='matrix_element', k_grid=[...] or {start,stop,num/step}, schemes=[{zmin,zmax,z_ext_max,label}] or scheme_scan={zmin_values/zmin_start,zmax_values/zmax_start,z_ext_max,y_range,roughness_weight}, method='GI'|'CG', order='LA'|'NLA'|'Empirical', observable='pion_quark_quasi_pdf'|'nucleon_quark_unpolarized_quasi_pdf'|'nucleon_quark_transversity_quasi_pdf'|'meson_quasi_da'|'pion_quark_quasi_gpd'|'nucleon_quark_quasi_gpd', coord_unit='lambda'|'fm'|'gev_inv'|'lattice', pz_gev=None, pz_prime_gev=None, a_fm=None, sample_axis=0, out='fourier_result') -> run LaMETLat workflow, score schemes, model-average results, and write artifacts/fourier_result.npz.",
    "summarize_fourier_result": "summarize_fourier_result(result='fourier_result', out='fourier_summary') -> compact mean/stat/sys arrays plus best scheme, scheme weights, chi2/dof, and roughness diagnostics for reporting.",
    "plot_fourier_result": "plot_fourier_result(result='fourier_result', save_path=None, title='Fourier result') -> plot artifacts/fourier_result.npz and write artifacts/fourier_result.pdf.",
    "plot_fourier_extension_quality_result": "plot_fourier_extension_quality_result(samples='matrix_element', result='fourier_result', scheme_index=None, save_path=None) -> write artifacts/fourier_extension_re.pdf and artifacts/fourier_extension_im.pdf for coordinate-space data and smoothed extension with fit-range markers for the best weighted scheme by default.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available Fourier-stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only."""
    if manifest.metadata.get("fourier_input"):
        return []
    if any(item.metadata.get("matrix_element") for item in manifest.correlators):
        return []
    return [
        "No coordinate-space matrix-element samples were declared. Provide "
        "manifest.metadata.fourier_input or a correlator with metadata.matrix_element=true."
    ]
