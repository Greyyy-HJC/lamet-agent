"""Stage-local helpers for Fourier-transform stage."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.manifest import AnalysisManifest

OBSERVABLE_ORDER_HELP = (
    "Missing metadata.fourier.observable/order. Fill observable with one of: "
    "pion_quark_quasi_pdf (2601.12189 2.1/2.2), "
    "nucleon_quark_unpolarized_quasi_pdf (2.3/2.4), "
    "nucleon_quark_transversity_quasi_pdf (2.5/2.6), "
    "meson_quasi_da (2.7/2.8), "
    "pion_quark_quasi_gpd (2.9/2.10), "
    "nucleon_quark_quasi_gpd (2.11/2.12). "
    "Fill order with LA, NLA, or Empirical; Empirical uses arXiv:2208.08008 Eq. (6)."
)

COORD_UNIT_HELP = (
    "Missing metadata.fourier.coord_unit. Use lambda for input lambda=zPz, fm or gev_inv for physical z "
    "(also set pz_gev), or lattice for z/a (also set a_fm and pz_gev)."
)

METHOD_HELP = (
    "Missing metadata.fourier.method. Use GI for gauge-invariant tails, or CG for the tail with an extra power n."
)

K_GRID_HELP = (
    "Missing metadata.fourier.k_grid. This is the output x grid, e.g. {'start': -2.0, 'stop': 2.0, 'num': 401}."
)

SCHEME_HELP = (
    "metadata.fourier.scheme_scan is optional. If omitted or incomplete, the tool chooses four large zmax "
    "candidates ending at the last stable long-distance point before the central values visibly jitter or the "
    "error bars grow sharply, then chooses zmin candidates by scanning upward at fixed zmax until the selected "
    "method/order/observable tail fit has stable chi2/dof and Q. It fills min_width, "
    "z_ext_max, and smooth='linear'. It also defaults y_range=[-2,2], roughness_weight=1.0, and "
    "model_average=true when omitted. Use explicit scheme_scan to override fit-range values."
)

INPUT_FORMAT_HELP = (
    "Missing metadata.fourier.input_format and the suffix is unclear. Use npz for coord/re_samples/im_samples, "
    "or h5 for Pz=*/z_ary, Re, Im."
)

VALID_METHODS = {"gi", "cg"}
VALID_ORDERS = {"la", "nla", "empirical"}
VALID_OBSERVABLES = {
    "pion_quark_quasi_pdf",
    "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_quark_transversity_quasi_pdf",
    "meson_quasi_da",
    "pion_quark_quasi_gpd",
    "nucleon_quark_quasi_gpd",
}
VALID_COORD_UNITS = {"lambda", "fm", "gev_inv", "lattice"}

STAGE_SKILL = """
Fourier-transform skill: extend coordinate-space matrix elements and transform
them to momentum space while preserving resampling samples.

Strategy:
- Load real/imaginary matrix-element samples from either an NPZ file with coord,
  re_samples, and im_samples, or an HDF5 file with a group such as Pz=4 that
  contains z_ary, Re, and Im. The loader normalizes samples into EnsembleData
  with dimensions (resample,z); users do not need to pass sample_axis.
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
  'none'. Linear smoothing starts at each scheme's zmin and reaches pure fit at zmax.
- If scheme_scan is provided, pass it to run_fourier_transform. If scheme_scan
  is omitted or incomplete, let the tool auto-generate a four-by-four scan:
  choose zmax as large as possible before central-value jitter or sharply
  growing error bars, then choose zmin by fixing each zmax and increasing zmin
  until the selected method/order/observable tail fit has stable chi2/dof and
  Q. Model-average schemes using chi2/dof, Fourier roughness, and fit-failure
  penalties. Missing y_range, roughness_weight, and model_average default to
  [-2,2], 1.0, and true.
- After run_fourier_transform, call summarize_fourier_result, plot_fourier_result,
  and plot_fourier_extension_quality_result. Finish with the NPZ artifact path,
  plot paths, best scheme, scheme weights, chi2/dof, roughness scores, fit
  failure counts, and stat/sys errors.
""".strip()

TOOL_CATALOG = {
    "load_renormalized_matrix_element_samples": "load_renormalized_matrix_element_samples(path, input_format='npz'|'h5', h5_group=None, coord_key='coord' or 'z_ary', re_key='re_samples' or 'Re', im_key='im_samples' or 'Im') -> load renormalized coordinate-space samples from NPZ or HDF5.",
    "run_fourier_transform": "run_fourier_transform(k_grid=[...] or {start,stop,num/step}, optional scheme_scan={zmin_values/zmin_start,zmax_values/zmax_start,z_ext_max,y_range,roughness_weight}; if omitted or incomplete, choose large stable zmax values and zmin values from stable tail-fit chi2/dof and Q diagnostics; method='GI'|'CG', order='LA'|'NLA'|'Empirical', observable='pion_quark_quasi_pdf'|'nucleon_quark_unpolarized_quasi_pdf'|'nucleon_quark_transversity_quasi_pdf'|'meson_quasi_da'|'pion_quark_quasi_gpd'|'nucleon_quark_quasi_gpd', coord_unit='lambda'|'fm'|'gev_inv'|'lattice', pz_gev=None, pz_prime_gev=None, a_fm=None) -> run the local Fourier workflow, score schemes, model-average results, and write artifacts/fourier_result.npz.",
    "summarize_fourier_result": "summarize_fourier_result() -> compact mean/stat/sys arrays plus best scheme, scheme weights, chi2/dof, and roughness diagnostics for reporting.",
    "plot_fourier_result": "plot_fourier_result(save_path=None, title='Fourier result') -> plot artifacts/fourier_result.npz and write artifacts/fourier_result.pdf.",
    "plot_fourier_extension_quality_result": "plot_fourier_extension_quality_result(scheme_index=None, save_path=None) -> write artifacts/fourier_extension_re.pdf and artifacts/fourier_extension_im.pdf for coordinate-space data and smoothed extension with fit-range markers for the best weighted scheme by default.",
}


def tool_catalog() -> str:
    """Return a human-readable catalog of available Fourier-stage tools."""
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest) -> list[str]:
    """Return stage-local issues only."""
    issues = []
    if not manifest.metadata.get("fourier_input") and not any(
        item.metadata.get("matrix_element") for item in manifest.correlators
    ):
        issues.append(
            "No coordinate-space matrix-element samples were declared. Provide "
            "manifest.metadata.fourier_input or a correlator with metadata.matrix_element=true."
        )
        return issues

    fourier = manifest.metadata.get("fourier", {})
    if not isinstance(fourier, dict):
        return ["metadata.fourier must be an object containing Fourier-stage options."]

    input_path = str(manifest.metadata.get("fourier_input", ""))
    suffix = Path(input_path).suffix.lower()
    if suffix not in {"", ".npz", ".h5", ".hdf5"} and not fourier.get("input_format"):
        issues.append(INPUT_FORMAT_HELP)

    if "method" not in fourier:
        issues.append(METHOD_HELP)
    elif str(fourier["method"]).lower() not in VALID_METHODS:
        issues.append(f"metadata.fourier.method={fourier['method']!r} is invalid. {METHOD_HELP}")

    if "order" not in fourier or "observable" not in fourier:
        issues.append(OBSERVABLE_ORDER_HELP)
    else:
        order = str(fourier["order"]).lower()
        observable_value = str(fourier["observable"]).lower()
        if order not in VALID_ORDERS:
            issues.append(f"metadata.fourier.order={fourier['order']!r} is invalid. {OBSERVABLE_ORDER_HELP}")
        if observable_value not in VALID_OBSERVABLES:
            issues.append(
                f"metadata.fourier.observable={fourier['observable']!r} is invalid. {OBSERVABLE_ORDER_HELP}"
            )

    if "coord_unit" not in fourier:
        issues.append(COORD_UNIT_HELP)
    else:
        coord_unit = str(fourier["coord_unit"]).lower()
        if coord_unit not in VALID_COORD_UNITS:
            issues.append(f"metadata.fourier.coord_unit={fourier['coord_unit']!r} is invalid. {COORD_UNIT_HELP}")
        if coord_unit in {"fm", "gev_inv", "lattice"} and "pz_gev" not in fourier:
            issues.append(
                "metadata.fourier.pz_gev is required when coord_unit is fm, gev_inv, or lattice. "
                "It is the hadron momentum P_z in GeV and sets lambda=z P_z and the Fourier phase."
            )
        if coord_unit == "lattice" and "a_fm" not in fourier:
            issues.append(
                "metadata.fourier.a_fm is required when coord_unit='lattice'. It is the lattice spacing in fm, "
                "used to convert lattice z to physical distance before forming lambda=z P_z."
            )

    observable = str(fourier.get("observable", "")).lower()
    if "gpd" in observable and "pz_prime_gev" not in fourier:
        issues.append(
            "metadata.fourier.pz_prime_gev is required for quasi-GPD observables unless P'^z is known to equal P^z. "
            "It is the final-state hadron momentum in GeV and enters the GPD phases in arXiv:2601.12189 "
            "Eqs. (2.9)-(2.12). If P'^z=P^z, set pz_prime_gev equal to pz_gev explicitly."
        )

    if "k_grid" not in fourier:
        issues.append(K_GRID_HELP)
    if issues:
        issues.insert(
            0,
            "Fourier stage needs more metadata before running tools.",
        )
    return issues
