"""Stage-local helpers for Fourier-transform stage."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.manifest import AnalysisManifest

OBSERVABLE_ORDER_HELP = (
    "Missing metadata.fourier.observable/order. Fill observable with one of: "
    "pion_quark_quasi_pdf (2601.12189 2.1/2.2), "
    "nucleon_quark_unpolarized_quasi_pdf (2.3/2.4), "
    "nucleon_quark_transversity_quasi_pdf (2.5/2.6), "
    "pion_gluon_quasi_pdf (F.8/F.9), "
    "nucleon_gluon_quasi_pdf (F.6/F.7), "
    "meson_quasi_da (2.7/2.8), "
    "pion_quark_quasi_gpd (2.9/2.10), "
    "nucleon_quark_quasi_gpd (2.11/2.12). "
    "Fill order with LA or NLA."
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
    "error bars grow sharply, then chooses zmin candidates by scanning upward from z ~= 0.5 fm at fixed zmax "
    "until the selected method/order/observable/part tail fit has stable chi2/dof and Q. It fills "
    "z_ext_max and smooth='linear'. Missing z_ext_max defaults to lambda_max + 8, converted "
    "back to the input coordinate unit. It also defaults y_range=[-2,2], roughness_weight=1.0, "
    "and model_average=true when omitted. Use explicit scheme_scan to override fit-range values."
)

INPUT_FORMAT_HELP = (
    "Missing metadata.fourier.input_format and the suffix is unclear. Use nc for EnsembleData NetCDF, "
    "npz for legacy coord/re_samples/im_samples, or h5 for Pz=*/z_ary, Re, Im."
)

VALID_METHODS = {"gi", "cg"}
VALID_ORDERS = {"la", "nla"}
VALID_OBSERVABLES = {
    "pion_quark_quasi_pdf",
    "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_quark_transversity_quasi_pdf",
    "pion_gluon_quasi_pdf",
    "nucleon_gluon_quasi_pdf",
    "meson_quasi_da",
    "pion_quark_quasi_gpd",
    "nucleon_quark_quasi_gpd",
}
VALID_COORD_UNITS = {"lambda", "fm", "gev_inv", "lattice"}

STAGE_SKILL = """
Fourier-transform skill: extend coordinate-space matrix elements and transform
them to momentum space while preserving resampling samples.

Strategy:
- Load real/imaginary matrix-element samples from either an EnsembleData NetCDF
  file, a legacy NPZ file with coord, re_samples, and im_samples, or an HDF5
  file with a group such as Pz=4 that contains z_ary, Re, and Im. The loader normalizes samples into EnsembleData
  with dimensions (resample,z); users do not need to pass sample_axis. Pass
  resample_mode='bs' or 'jk' to the loader when the manifest declares the input
  resampling. Do not pass resample_mode to run_fourier_transform; it uses the
  resampling mode recorded on the loaded EnsembleData.
- Use explicit units. coord_unit='lambda' means the transform coordinate is
  already Ioffe time. coord_unit='fm', 'gev_inv', or 'lattice' requires pz_gev
  for the Fourier phase; lattice also requires a_fm. For physical coordinate
  units, k_grid is the momentum fraction y in Pz dz/(2*pi) exp(i y Pz z) h(z).
- Choose method='GI' or 'CG' and order='LA' or 'NLA'. CG adds a power-law
  exponent in the asymptotic tail.
- Choose observable to select the LA/NLA formula block from arXiv:2601.12189:
  pion_quark_quasi_pdf uses Eqs. (2.1)/(2.2);
  nucleon_quark_unpolarized_quasi_pdf uses Eqs. (2.3)/(2.4);
  nucleon_quark_transversity_quasi_pdf uses Eqs. (2.5)/(2.6);
  pion_gluon_quasi_pdf uses Appendix F Eqs. (F.8)/(F.9);
  nucleon_gluon_quasi_pdf uses Appendix F Eqs. (F.6)/(F.7);
  meson_quasi_da uses Eqs. (2.7)/(2.8);
  pion_quark_quasi_gpd uses Eqs. (2.9)/(2.10);
  nucleon_quark_quasi_gpd uses Eqs. (2.11)/(2.12).
  For GPD observables, pass pz_prime_gev if P'^z differs from P^z.
- zmin must be positive because NLA and CG forms are singular at zero.
- Lambda0 optionally sets the fitted large-distance Lambda lower bound; default
  is 0.1. Fit windows must contain at least as many coordinate points as the
  selected large-distance model has parameters.
- posterior_prior_error_scale optionally inflates the sample-average fit
  parameter errors when using that posterior as a weak prior for
  bootstrap/jackknife sample fits; default is 3.0.
- fit_error_mode='diagonal' fits with pointwise standard deviations; use
  'covariance' to fit with the full real/imag covariance estimated from
  bootstrap/jackknife samples. The default is 'diagonal'.
- part='both' preserves the standard real+imaginary tail fit. Use part='re' for
  real-only observables and part='im' for imaginary-only observables; the
  inactive coordinate-space channel is fixed to zero before the Fourier transform.
- output_scale multiplies only the final Fourier-space samples, central values,
  and statistical/systematic uncertainties. Use output_scale=2.0 with part='re'
  for the valence q-qbar convention obtained from the real-part transform.
- Scan a small list of schemes with zmin, zmax, z_ext_max, smooth='linear' or
  'none'. Linear smoothing starts at each scheme's zmin and reaches pure fit at zmax.
- If scheme_scan is provided, pass it to run_fourier_transform. If scheme_scan
  is omitted or incomplete, let the tool auto-generate a four-by-four scan:
  choose zmax as large as possible before central-value jitter or sharply
  growing error bars, then choose zmin by fixing each zmax and increasing zmin
  from z ~= 0.5 fm until the selected method/order/observable/part tail fit has
  stable chi2/dof and Q. Model-average schemes using chi2/dof, Fourier roughness, and fit-failure
  penalties. Missing z_ext_max defaults to lambda_max + 8; missing y_range,
  roughness_weight, and model_average default to [-2,2], 1.0, and true.
- Pass run_fourier_transform save_path when the manifest requests a non-default
  Fourier NetCDF artifact name.
- run_fourier_transform already writes the summary, plots, and English/Chinese
  Markdown reports. After it returns, finish with the report paths, NetCDF
  artifact path, PDF/PNG plot paths, best scheme, scheme weights, chi2/dof,
  roughness scores, fit failure counts, and stat/sys errors. Call the standalone
  summarize/plot/report tools only when the user explicitly asks to regenerate
  one artifact.
""".strip()

TOOL_CATALOG = {
    "load_renormalized_matrix_element_samples": "load_renormalized_matrix_element_samples(path, input_format='nc'|'npz'|'h5', h5_group=None, coord_key='coord' or 'z_ary', re_key='re_samples' or 'Re', im_key='im_samples' or 'Im', resample_mode='bs'|'jk') -> load renormalized coordinate-space samples from NetCDF, legacy NPZ, or HDF5 into EnsembleData.",
    "run_fourier_transform": "run_fourier_transform(k_grid=[...] or {start,stop,num/step}, optional scheme_scan={zmin_values/zmin_start,zmax_values/zmax_start,z_ext_max,smooth,y_range,roughness_weight,model_average,max_schemes}; if omitted or incomplete, choose large stable zmax values and zmin values from stable tail-fit chi2/dof and Q diagnostics starting near z=0.5 fm; missing z_ext_max defaults to lambda_max + 8; method='GI'|'CG', order='LA'|'NLA', observable='pion_quark_quasi_pdf'|'nucleon_quark_unpolarized_quasi_pdf'|'nucleon_quark_transversity_quasi_pdf'|'pion_gluon_quasi_pdf'|'nucleon_gluon_quasi_pdf'|'meson_quasi_da'|'pion_quark_quasi_gpd'|'nucleon_quark_quasi_gpd', coord_unit='lambda'|'fm'|'gev_inv'|'lattice', pz_gev=None, pz_prime_gev=None, a_fm=None, im_flip_for_ft=False, Lambda0=0.1, posterior_prior_error_scale=3.0, fit_error_mode='diagonal'|'covariance', part='both'|'re'|'im', output_scale=1.0, save_path=None) -> run the local Fourier workflow using the resampling mode stored on EnsembleData, score schemes, model-average results, optionally scale final Fourier-space outputs, and write artifacts/fourier_result.nc.",
    "summarize_fourier_result": "summarize_fourier_result() -> compact mean/stat/sys arrays plus best scheme, scheme weights, chi2/dof, and roughness diagnostics for reporting.",
    "plot_fourier_result": "plot_fourier_result(save_path=None, title='Fourier result') -> plot artifacts/fourier_result.nc and write artifacts/fourier_result.pdf plus a PNG companion for Markdown embedding.",
    "plot_fourier_extension_quality_result": "plot_fourier_extension_quality_result(scheme_index=None, save_path=None) -> write artifacts/fourier_extension_re.pdf and artifacts/fourier_extension_im.pdf plus PNG companions for coordinate-space data and extrapolation-band quality with fit-range markers for the best weighted scheme by default.",
    "report_fourier_result": "report_fourier_result(save_path=None) -> write an English report at artifacts/report_fourier.md and a Chinese companion report at artifacts/report_fourier_CN.md with the physical quantity, implemented LA/NLA tail form, fit diagnostics, Fourier transform formula, embedded PNG plots, PDF artifact links, NetCDF contents, and artifact paths.",
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

    fourier = dict(manifest.stages.get("fourier_transform", {}).get("defaults", {}))
    if "component" in fourier and "part" not in fourier:
        fourier["part"] = fourier.pop("component")
    if isinstance(manifest.metadata.get("fourier"), dict):
        explicit_fourier = dict(manifest.metadata["fourier"])
        if "component" in explicit_fourier and "part" not in explicit_fourier:
            explicit_fourier["part"] = explicit_fourier.pop("component")
        fourier.update(explicit_fourier)
    elif "fourier" in manifest.metadata:
        return ["metadata.fourier must be an object containing Fourier-stage options."]
    input_correlators = manifest.inputs.get("correlators", [])
    if not isinstance(input_correlators, list):
        input_correlators = []
    if "hadron" not in fourier and "hadron" in manifest.metadata:
        fourier["hadron"] = manifest.metadata["hadron"]
    if "gfix" not in fourier and "gfix" in manifest.metadata:
        fourier["gfix"] = manifest.metadata["gfix"]
    if "pz_gev" not in fourier and "pz_gev" in manifest.metadata:
        fourier["pz_gev"] = manifest.metadata["pz_gev"]
    if "a_fm" not in fourier and "a_fm" in manifest.metadata:
        fourier["a_fm"] = manifest.metadata["a_fm"]
    for key in ("hadron", "gfix", "a_fm", "pz_gev"):
        if key in fourier:
            continue
        values = [
            item[key]
            for item in input_correlators
            if isinstance(item, dict) and key in item and item[key] not in (None, "")
        ]
        if key == "pz_gev":
            values = [
                value
                for value in values
                if not isinstance(value, int | float) or float(value) != 0.0
            ]
        if values and len({str(value).lower() for value in values}) == 1:
            fourier[key] = values[0]
    if "method" not in fourier:
        gfix = str(fourier.get("gfix", "")).upper()
        if gfix in {"CG", "GI"}:
            fourier["method"] = gfix
    if "observable" not in fourier:
        target = str(
            fourier.get("target_observable") or manifest.metadata.get("target_observable", "")
        ).lower()
        target = target.replace("-", "_")
        hadron = str(fourier.get("hadron", "")).lower()
        parton = str(fourier.get("parton") or fourier.get("parton_type") or "").lower()
        is_pion = hadron in {"pion", "pi", "meson"}
        is_nucleon = hadron in {"nucleon", "proton", "neutron"}
        if target in {"pdf", "qpdf", "quasi_pdf"}:
            if is_pion and parton == "quark":
                fourier["observable"] = "pion_quark_quasi_pdf"
            elif is_pion and parton == "gluon":
                fourier["observable"] = "pion_gluon_quasi_pdf"
            elif is_nucleon and parton == "quark":
                fourier["observable"] = "nucleon_quark_unpolarized_quasi_pdf"
            elif is_nucleon and parton == "gluon":
                fourier["observable"] = "nucleon_gluon_quasi_pdf"
        elif target in {"da", "quasi_da"} and is_pion:
            fourier["observable"] = "meson_quasi_da"
        elif target in {"gpd", "quasi_gpd"}:
            if is_pion and parton == "quark":
                fourier["observable"] = "pion_quark_quasi_gpd"
            elif is_nucleon and parton == "quark":
                fourier["observable"] = "nucleon_quark_quasi_gpd"

    input_path = str(manifest.metadata.get("fourier_input", ""))
    suffix = Path(input_path).suffix.lower()
    if suffix not in {"", ".nc", ".npz", ".h5", ".hdf5"} and not fourier.get("input_format"):
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
