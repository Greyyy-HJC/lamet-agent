"""Fourier manifest parameter contract, planning guidance, and validation."""

from __future__ import annotations

import math
from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import (
    ConstraintSpec,
    ParameterSpec,
    RuleViolation,
    StageParamContract,
    StageValidationContext,
    resolve_stage_params,
    merge_stage_params,
)


INFERRED_OBSERVABLES = {
    (target, "quark", hadron): f"{hadron}_quark_quasi_{target}"
    for target in ("pdf", "gpd")
    for hadron in ("pion", "nucleon")
}
INFERRED_OBSERVABLES.update(
    {
        ("pdf", "gluon", "pion"): "pion_gluon_quasi_pdf",
        ("pdf", "gluon", "nucleon"): "nucleon_gluon_quasi_pdf",
    }
)
PUBLIC_OBSERVABLES = frozenset({*INFERRED_OBSERVABLES.values(), "meson_quasi_da"})


def _parameter(summary: str, physics: str, **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _validate_y_grid(value: Any) -> str | None:
    if isinstance(value, list):
        if not value:
            return "y_grid must not be empty."
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
            return "y_grid list values must be finite numbers."
        return None
    if not isinstance(value, dict):
        return "y_grid must be a numeric list or an object."
    if not {"start", "stop"}.issubset(value):
        return "y_grid object requires start and stop."
    selectors = {key for key in ("num", "step") if key in value}
    if len(selectors) != 1:
        return "y_grid object requires exactly one of num or step."
    if "num" in value and (type(value["num"]) is not int or value["num"] < 2):
        return "y_grid num must be an integer of at least 2."
    if "step" in value and (
        isinstance(value["step"], bool)
        or not isinstance(value["step"], (int, float))
        or not math.isfinite(float(value["step"]))
        or float(value["step"]) <= 0.0
    ):
        return "y_grid step must be a finite positive number."
    return None


def _validate_scheme_scan(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "scheme_scan must be an object."
    missing = [key for key in ("z_ext_max", "smooth", "model_average") if key not in value]
    if missing:
        return "scheme_scan requires " + ", ".join(missing) + "."
    for bound in ("zmin", "zmax"):
        has_values = f"{bound}_values" in value
        has_range = any(f"{bound}_{suffix}" in value for suffix in ("start", "stop", "step"))
        if not has_values and not has_range:
            return f"scheme_scan requires {bound}_values or a complete {bound}_start/{bound}_stop range."
        values = value.get(f"{bound}_values")
        if has_values and (not isinstance(values, list) or not values):
            return f"scheme_scan {bound}_values must be a non-empty numeric list."
        if has_values and any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in values):
            return f"scheme_scan {bound}_values must be a non-empty numeric list."
        if has_values and has_range:
            return f"scheme_scan {bound}_values cannot be combined with {bound}_start/stop/step."
        if has_range and not {f"{bound}_start", f"{bound}_stop"}.issubset(value):
            return f"scheme_scan {bound} range requires both {bound}_start and {bound}_stop."
    for key in ("step", "zmin_step", "zmax_step"):
        item = value.get(key)
        if key in value and not isinstance(item, bool) and isinstance(item, (int, float)) and float(item) <= 0.0:
            return f"scheme_scan {key} must be positive."
    if "max_schemes" in value and type(value["max_schemes"]) is int and value["max_schemes"] < 1:
        return "scheme_scan max_schemes must be a positive integer."
    return None


def _violation(
    context: StageValidationContext,
    *,
    message: str,
    path: str,
    cause: str,
    parameters: tuple[str, ...] = (),
) -> RuleViolation:
    return RuleViolation(message=message, path=path, cause=cause, parameters=parameters)


def _check_input_role(context: StageValidationContext) -> RuleViolation | None:
    roles = set(context.inputs)
    if roles == {"input"}:
        return None
    return _violation(
        context,
        message="A fourier_transform job requires exactly one input role named input.",
        path=f"{context.job_path}.inputs",
        cause=f"The effective input roles are {sorted(roles)}.",
        parameters=("inputs.input",),
    )


def _check_momentum(context: StageValidationContext) -> RuleViolation | None:
    if context.params.get("momentum_gev") is not None:
        return None
    return _violation(
        context,
        message=f"Fourier job {context.job_id!r} has no derivable physical momentum.",
        path=f"{context.job_path}.inputs",
        cause="The upstream source does not provide a complete momentum, volume, and lattice-spacing triple.",
        parameters=("momentum_gev",),
    )


def _check_polarization(context: StageValidationContext) -> RuleViolation | None:
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    if target not in {"pdf", "gpd"} or str(context.params.get("polarization", "")).lower():
        return None
    return _violation(
        context,
        message=f"Fourier job {context.job_id!r} has no polarization for its {target.upper()} observable.",
        path=context.parameter_path("polarization"),
        cause="Neither stage params nor upstream metadata declares the spin channel.",
        parameters=("polarization",),
    )


def _check_observable(context: StageValidationContext) -> RuleViolation | None:
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    parton = str(context.metadata.get("parton", "quark")).lower()
    hadron = str(context.params.get("hadron", "")).lower()
    hadron = "nucleon" if hadron == "proton" else hadron
    if (
        target not in {"pdf", "gpd"}
        or (parton == "gluon" and target != "pdf")
        or "observable" in context.params
        or (target, parton, hadron) in INFERRED_OBSERVABLES
    ):
        return None
    return _violation(
        context,
        message=f"Fourier job {context.job_id!r} has no explicit or derivable observable.",
        path=context.parameter_path("observable"),
        cause=f"No backend is registered for target={target!r}, parton={parton!r}, hadron={hadron!r}.",
        parameters=("observable",),
    )


def _check_gluon_backend(context: StageValidationContext) -> RuleViolation | None:
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    parton = str(context.metadata.get("parton", "quark")).lower()
    polarization = str(context.params.get("polarization", "")).lower()
    if parton != "gluon" or (target == "pdf" and polarization in {"", "unpolarized"}):
        return None
    return _violation(
        context,
        message="The Fourier backend currently supports only unpolarized gluon PDF observables.",
        path=f"{context.job_path}.params",
        cause=f"The effective selection is target={target!r}, parton='gluon', polarization={polarization!r}.",
    )


def _check_target_metadata(context: StageValidationContext) -> RuleViolation | None:
    if "target_observable" not in context.params:
        return None
    target = str(context.metadata.get("target_observable", "")).lower()
    if str(context.params["target_observable"]).lower() == target:
        return None
    return _violation(
        context,
        message="Fourier target_observable must agree with metadata.target_observable.",
        path=context.parameter_path("target_observable"),
        cause=f"Stage value {context.params['target_observable']!r} conflicts with metadata value {target!r}.",
    )


def _check_sector(context: StageValidationContext) -> RuleViolation | None:
    if "sector" not in context.params:
        return None
    sector = str(context.params["sector"]).lower()
    if sector not in {"sea", "valence", "singlet", "full"}:
        return None
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    parton = str(context.metadata.get("parton", "quark")).lower()
    allowed = (
        {"full"}
        if parton == "gluon" or target == "da"
        else {"sea", "valence", "singlet", "full"}
    )
    if sector in allowed:
        return None
    return _violation(
        context,
        message=f"Fourier sector must be one of {sorted(allowed)}.",
        path=context.parameter_path("sector"),
        cause=f"The effective sector is {sector!r} for target={target!r}, parton={parton!r}.",
    )


def _check_sector_manual_projection(context: StageValidationContext) -> RuleViolation | None:
    manual = sorted({"part", "output_scale", "im_flip_for_ft"}.intersection(context.authored_params))
    if "sector" not in context.params or not manual:
        return None
    return _violation(
        context,
        message="Fourier sector cannot be combined with manual projection controls.",
        path=f"{context.job_path}.params",
        cause=f"sector is set together with {manual}.",
    )


def _check_component_part(context: StageValidationContext) -> RuleViolation | None:
    if "component" not in context.authored_params or "part" not in context.authored_params:
        return None
    return _violation(
        context,
        message="Fourier component and part cannot both be set.",
        path=f"{context.job_path}.params",
        cause=(
            f"component={context.params['component']!r} and part={context.params['part']!r} "
            "select the same channel."
        ),
    )


def _check_scheme_scan(context: StageValidationContext) -> RuleViolation | None:
    message = _validate_scheme_scan(context.params.get("scheme_scan"))
    if message is None:
        return None
    return _violation(
        context,
        message=message,
        path=context.parameter_path("scheme_scan"),
        cause=f"The effective scheme_scan is {context.params.get('scheme_scan')!r}.",
    )


def _check_da_requirements(context: StageValidationContext) -> list[RuleViolation] | None:
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    if target != "da":
        return None
    required = ("symmetry_guarantee", "psi1_flavor_class", "psi2_flavor_class")
    issues: list[RuleViolation] = []
    for parameter in required:
        if parameter not in context.params:
            issues.append(_violation(
                context,
                message=f"DA Fourier jobs require {parameter}.",
                path=context.parameter_path(parameter),
                cause=f"The DA-specific parameter {parameter!r} is absent from the effective configuration.",
                parameters=(parameter,),
            ))
    return issues or None


_GRID_FIELDS = {
    "num": _parameter(
        "Number of uniformly spaced transform points.",
        "A denser grid changes output sampling, not the information content of the finite coordinate-space data.",
        expected=int,
    ),
    "start": _parameter(
        "First transform-grid coordinate.",
        "Together with stop, this selects the quasi-distribution domain represented in the output artifact.",
        expected=float,
    ),
    "step": _parameter(
        "Positive transform-grid spacing.",
        "This is an alternative to num; specifying both would make the requested discretization ambiguous.",
        expected=float,
    ),
    "stop": _parameter(
        "Last transform-grid coordinate.",
        "Together with start, this selects the quasi-distribution domain represented in the output artifact.",
        expected=float,
    ),
}


_SCHEME_SCAN_FIELDS = {
    "max_schemes": _parameter(
        "Maximum number of tail-range candidates.",
        "This bounds runtime without changing the definition of any individual tail model.",
        expected=int,
        default=200,
    ),
    "model_average": _parameter(
        "Whether to average successful tail models per resampled sample.",
        "The fit range is selected once from sample-average diagnostics. False then uses one selected tail model for all resamples; true averages successful order/prior-width models per resample and propagates their spread.",
        expected=bool,
        choices=(False, True),
        choice_descriptions={
            False: "Use one sample-average-selected tail model after the fit range is fixed.",
            True: "Average successful order and posterior_prior_error_scale candidates for each resampled sample.",
        },
    ),
    "smooth": _parameter(
        "Interpolation used to join data and the asymptotic tail.",
        "The join prescription controls how sharply the finite-data region transitions into the fitted long-distance model.",
        expected=str,
        choices=("linear", "none"),
        choice_descriptions={
            "linear": "Linearly blend the measured region into the fitted extension.",
            "none": "Join the measured data and fitted extension without a smoothing interval.",
        },
    ),
    "step": _parameter(
        "Fallback spacing shared by zmin and zmax scans.",
        "The spacing uses the same coordinate unit as the renormalized matrix element.",
        expected=float,
    ),
    "z_ext_max": _parameter(
        "Coordinate through which the fitted tail is extended.",
        "The extension must control transform truncation and uses the same unit as the input coordinate.",
        expected=float,
    ),
    "zmax_start": _parameter("First maximum fit coordinate.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmax_step": _parameter("Maximum-coordinate scan spacing.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmax_stop": _parameter("Last maximum fit coordinate.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmax_values": _parameter(
        "Explicit maximum fit-coordinate candidates.",
        "Each value truncates the data region used to constrain the long-distance tail.",
        expected=list,
        items=float,
    ),
    "zmin_start": _parameter("First minimum fit coordinate.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmin_step": _parameter("Minimum-coordinate scan spacing.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmin_stop": _parameter("Last minimum fit coordinate.", "Fit-range coordinates use the input coord_unit.", expected=float),
    "zmin_values": _parameter(
        "Explicit minimum fit-coordinate candidates.",
        "These values control where the asymptotic ansatz begins to describe the matrix element.",
        expected=list,
        items=float,
    ),
}

_PLOT_FIELDS = {
    "save_path": _parameter("Plot file name.", "Plot placement remains inside the job artifact directory.", expected=str),
    "title": _parameter("Optional plot title.", "This changes presentation only.", expected=str),
}


FOURIER_CONSTRAINTS = (
    ConstraintSpec(
        code="fourier.inputs.exactly_one",
        parameters=("inputs.input",),
        rule="Each job has exactly one input role named input.",
        physics="A Fourier job transforms one renormalized coordinate-space matrix element into one quasi-distribution.",
        suggested_fix='Set job inputs to {"input": "<renormalization job or artifact>"}.',
        check=_check_input_role,
    ),
    ConstraintSpec(
        code="fourier.kinematics.momentum_required",
        parameters=("inputs.input", "derived.momentum_gev"),
        rule="Physical momentum must be derivable from the upstream job or artifact.",
        physics="Converting coordinate separation to Ioffe time requires the hadron momentum in physical units.",
        suggested_fix="Declare discrete momentum, volume, and lattice_spacing_fm on the upstream source or partial-run artifact.",
        check=_check_momentum,
    ),
    ConstraintSpec(
        code="fourier.pdf_gpd.polarization_required",
        parameters=("polarization", "metadata.target_observable"),
        rule="PDF and GPD jobs require explicit or upstream polarization.",
        physics="Unpolarized, helicity, and transversity observables use different symmetry projections and partonic extensions.",
        suggested_fix="Declare polarization as unpolarized, helicity, or transversity when upstream provenance cannot supply it.",
        check=_check_polarization,
    ),
    ConstraintSpec(
        code="fourier.observable.required",
        parameters=("observable", "hadron", "metadata.target_observable", "metadata.parton"),
        rule="The short observable name must be explicit unless it can be inferred from target, parton, and hadron.",
        physics="The observable selects the asymptotic backend; polarization remains a separate physical label.",
        suggested_fix="Declare a supported short observable or provide supported upstream hadron metadata.",
        check=_check_observable,
    ),
    ConstraintSpec(
        code="fourier.gluon.backend_boundary",
        parameters=("metadata.target_observable", "metadata.parton", "polarization"),
        rule="The current gluon backend supports only an unpolarized PDF.",
        physics="Gluon helicity and GPD tails require operator-specific asymptotic formulae that are not implemented by the current backend.",
        suggested_fix="Use target_observable=pdf with polarization=unpolarized, or select a supported quark observable.",
        check=_check_gluon_backend,
    ),
    ConstraintSpec(
        code="fourier.target.metadata_conflict",
        parameters=("target_observable", "metadata.target_observable"),
        rule="A stage target_observable override must agree with metadata.target_observable.",
        physics="A single run target must select the same symmetry and tail backend during planning, validation, and execution.",
        suggested_fix="Remove the stage override or set it equal to metadata.target_observable.",
        check=_check_target_metadata,
    ),
    ConstraintSpec(
        code="fourier.sector.compatibility",
        parameters=("sector", "metadata.target_observable", "metadata.parton"),
        rule="DA and gluon jobs use sector=full; quark PDF/GPD jobs also support sea, valence, and singlet.",
        physics="Quark/antiquark sector projections do not apply to the current DA or gluon transform backends.",
        suggested_fix="Use sector=full for DA/gluon, or choose a supported quark PDF/GPD sector.",
        check=_check_sector,
    ),
    ConstraintSpec(
        code="fourier.sector.manual_projection_conflict",
        parameters=("sector", "part", "output_scale", "im_flip_for_ft"),
        rule="sector cannot be combined with manual part, output_scale, or im_flip_for_ft controls.",
        physics="A named sector already determines the active real/imaginary channel, normalization, and negative-x convention.",
        suggested_fix="Prefer sector and remove the manual projection controls, or omit sector and configure the manual controls explicitly.",
        check=_check_sector_manual_projection,
    ),
    ConstraintSpec(
        code="fourier.component_part.conflict",
        parameters=("component", "part"),
        rule="component and part are aliases and cannot both be set.",
        physics="Both fields select the same real/imaginary transform channel; two values have no independent physical meaning.",
        suggested_fix="Keep part and remove component, or keep component as the legacy alias.",
        check=_check_component_part,
    ),
    ConstraintSpec(
        code="fourier.scheme_scan.coordinates",
        parameters=("scheme_scan", "coord_unit"),
        rule="Tail-scan coordinates use coord_unit, and values form is exclusive with start/stop form.",
        physics="Mixing lattice sites and physical distances changes the fitted long-distance window and therefore the Fourier systematic.",
        suggested_fix="Use one range representation per bound and express every range value in coord_unit.",
        check=_check_scheme_scan,
    ),
    ConstraintSpec(
        code="fourier.da.required",
        parameters=("symmetry_guarantee", "psi1_flavor_class", "psi2_flavor_class"),
        rule="DA jobs require explicit symmetry and constituent flavor-class choices.",
        physics="DA symmetry projection and unequal-mass asymptotics depend on these analysis choices.",
        suggested_fix="Declare symmetry_guarantee, psi1_flavor_class, and psi2_flavor_class in stage defaults or job params.",
        check=_check_da_requirements,
    ),
)


def _normalize_draft(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize deterministic Fourier constraints in a mutable planning draft."""
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    force_full = str(metadata.get("target_observable", "")).lower() == "da" or str(metadata.get("parton", "")).lower() == "gluon"
    if not force_full:
        return []
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    stage = stages.get("fourier_transform", {}) if isinstance(stages, dict) else {}
    if not isinstance(stage, dict):
        return []
    edits: list[dict[str, Any]] = []
    defaults = stage.get("defaults", {})
    if isinstance(defaults, dict) and "sector" in defaults and str(defaults.get("sector", "")).lower() != "full":
        old = defaults.get("sector")
        defaults["sector"] = "full"
        edits.append({"path": "stages.fourier_transform.defaults.sector", "old": old, "new": "full", "note": "DA and gluon Fourier sectors are fixed to full."})
    jobs = stage.get("jobs", [])
    if isinstance(jobs, list):
        for job in jobs:
            params = job.get("params", {}) if isinstance(job, dict) else {}
            if isinstance(params, dict) and "sector" in params and str(params.get("sector", "")).lower() != "full":
                old = params.get("sector")
                params["sector"] = "full"
                edits.append({"path": f"stages.fourier_transform.jobs.{job.get('id', '')}.params.sector", "old": old, "new": "full", "note": "DA and gluon Fourier sectors are fixed to full."})
    return edits


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="fourier",
    summary="Finite-distance extension and Fourier transformation of a renormalized matrix element.",
    physics=(
        "The stage fits the long-distance coordinate-space tail, extends the finite lattice signal, "
        "and transforms every resampled sample onto a declared momentum-fraction grid."
    ),
    input_roles=("input",),
    input_role_descriptions={
        "input": "One renormalized coordinate-space matrix element to extend and Fourier transform.",
    },
    normalize_draft=_normalize_draft,
    schema={
        "Lambda0_gev": _parameter(
            "Infrared scale entering the tail exponent prior.",
            "It shifts the lower prior location of the asymptotic power parameter without changing the perturbative matching scale.",
            expected=float,
            unit="GeV",
            required=True,
        ),
        "component": _parameter(
            "Legacy alias for part.",
            "It selects the real, imaginary, or combined transform channel.",
            expected=str,
            choices=("re", "im", "both"),
            choice_descriptions={
                "re": "Use the real coordinate-space channel.",
                "im": "Use the imaginary coordinate-space channel.",
                "both": "Retain both channels.",
            },
        ),
        "coord_key": _parameter("NPZ/HDF5 coordinate dataset key.", "This maps an external file layout onto the stage coordinate axis.", expected=str, default="coord"),
        "coord_unit": _parameter(
            "Unit of the input coordinate axis.",
            "The runner converts fm, lattice, or inverse-GeV separations to Ioffe time; lambda is already dimensionless Ioffe time.",
            expected=str,
            choices=("fm", "lattice", "gev_inv", "lambda"),
            choice_descriptions={
                "fm": "Coordinates are physical distances in femtometers and are multiplied by momentum/(hbar*c).",
                "lattice": "Coordinates are lattice-site separations and require lattice_spacing_fm plus momentum.",
                "gev_inv": "Coordinates are inverse-GeV distances and are multiplied by physical momentum.",
                "lambda": "Coordinates are already dimensionless Ioffe time.",
            },
            default="fm",
        ),
        "gfix": _parameter("Gauge-link treatment inherited from the input.", "CG and GI select the corresponding tail method when method is omitted.", expected=str),
        "h5_group": _parameter("HDF5 group containing one momentum channel.", "This is file-layout metadata and does not alter the Fourier prescription.", expected=str),
        "hadron": _parameter("Hadron identity used for observable inference.", "Hadron and parton labels select the public quasi-observable backend.", expected=str),
        "im_flip_for_ft": _parameter(
            "Manual sign flip for the negative-coordinate imaginary part.",
            "This changes the imposed Hermiticity extension and should be used only when no named sector supplies the convention.",
            expected=bool,
            default=False,
        ),
        "im_key": _parameter("NPZ/HDF5 imaginary-sample dataset key.", "This maps an external file layout onto complex matrix-element samples.", expected=str, default="im_samples"),
        "input_format": _parameter(
            "External matrix-element file format.",
            "The format changes loading only; all supported formats are normalized to EnsembleData before analysis.",
            expected=str,
            choices=("nc", "netcdf", "npz", "h5", "hdf5"),
        ),
        "method": _parameter(
            "Long-distance tail ansatz family.",
            "GI and CG use different asymptotic parameterizations; the choice is fixed theory input and is not model-averaged.",
            expected=str,
            required=True,
            choices=("GI", "CG"),
            choice_descriptions={
                "GI": "Use the gauge-invariant asymptotic parameterization.",
                "CG": "Use the Coulomb-gauge asymptotic parameterization.",
            },
        ),
        "observable": _parameter(
            "Short public quasi-observable name without polarization.",
            "It selects the tail backend; polarization independently selects the physical spin channel.",
            expected=str,
            choices=tuple(sorted(PUBLIC_OBSERVABLES)),
        ),
        "order": _parameter(
            "Tail ansatz orders included as fit-model candidates.",
            "LA keeps the leading asymptotic structure; NLA adds the next term and can enlarge the model systematic.",
            expected=(str, list),
            items=str,
            choices=("LA", "NLA"),
            choice_descriptions={
                "LA": "Keep the leading asymptotic term.",
                "NLA": "Include the next asymptotic term as a more flexible model candidate.",
            },
            required=True,
        ),
        "output_scale": _parameter("Final manual multiplicative scale.", "This rescales the transformed distribution and its uncertainties.", expected=float, default=1.0),
        "part": _parameter(
            "Manual real/imaginary transform channel.",
            "The selected channel controls which coordinate-space component constrains the output when sector is absent.",
            expected=str,
            choices=("re", "im", "both"),
            choice_descriptions={
                "re": "Transform the real channel only.",
                "im": "Transform the imaginary channel only.",
                "both": "Transform both channels.",
            },
            default="both",
        ),
        "symmetry_guarantee": _parameter(
            "Apply the DA phase rotation and symmetry projection.",
            "For DA, true applies a phase rotation by exp(+i z Pz/2), discards the rotated imaginary part, rotates the retained real part back, and only then extends and transforms the signal. False preserves the DA input unchanged. The setting has no effect for PDF/GPD.",
            expected=bool,
            choices=(False, True),
            choice_descriptions={
                False: "Use the DA matrix element unchanged.",
                True: "Project the phase-rotated DA matrix element onto its expected real symmetry channel.",
            },
        ),
        "plot_extension": _parameter(
            "Tail-extension diagnostic plot settings.",
            "These settings affect presentation only.",
            expected=dict,
            schema={
                **_PLOT_FIELDS,
                "scheme_index": _parameter("Candidate index shown in the diagnostic.", "This selects presentation of one fitted range.", expected=int),
            },
        ),
        "plot_fourier": _parameter("Fourier-result plot settings.", "These settings affect presentation only.", expected=dict, schema=_PLOT_FIELDS),
        "posterior_prior_error_scale": _parameter(
            "Prior-width candidate or candidates for tail fits.",
            "Each value scales the sample-average posterior width used as the prior for resampled tail fits; multiple values create model candidates whose spread can enter scheme_scan.model_average.",
            expected=(float, list),
            items=float,
            required=True,
        ),
        "polarization": _parameter(
            "Physical spin channel.",
            "Unpolarized, helicity, and transversity channels use different symmetry relations and sector projections.",
            expected=str,
            choices=("unpolarized", "helicity", "transversity"),
            choice_descriptions={
                "unpolarized": "Use the vector/unpolarized symmetry and negative-x convention.",
                "helicity": "Use the helicity convention, including its distinct quark/antiquark extension.",
                "transversity": "Use the tensor/transversity symmetry convention.",
            },
        ),
        "psi1_flavor_class": _parameter("First meson constituent mass class.", "For DA, light/heavy assignments constrain which asymptotic amplitudes are related or vanish.", expected=str, choices=("light", "heavy")),
        "psi2_flavor_class": _parameter("Second meson constituent mass class.", "For DA, light/heavy assignments constrain which asymptotic amplitudes are related or vanish.", expected=str, choices=("light", "heavy")),
        "re_key": _parameter("NPZ/HDF5 real-sample dataset key.", "This maps an external file layout onto complex matrix-element samples.", expected=str, default="re_samples"),
        "report": _parameter(
            "Optional per-job report settings.",
            "Reporting summarizes diagnostics but does not alter numerical results.",
            expected=dict,
            schema={
                "enabled": _parameter("Enable the optional per-job report.", "This affects reporting only.", expected=bool),
                "report_language": _parameter("Report language.", "This affects reporting only.", expected=str, choices=("en", "ch")),
                "save_path": _parameter("Report file name.", "Report placement remains inside the job artifact directory.", expected=str),
            },
        ),
        "scheme_scan": _parameter(
            "Tail fit-range scan and model-averaging configuration.",
            "zmin/zmax candidates explicitly select the measured coordinate range used to constrain the tail, in coord_unit; z_ext_max controls the subsequent extension. Range variation estimates the finite-distance systematic.",
            expected=dict,
            required=True,
            schema=_SCHEME_SCAN_FIELDS,
        ),
        "sector": _parameter(
            "Partonic projection of a quark PDF/GPD, or full distribution.",
            "For quarks it selects the negative-x extension and active complex channel; DA and gluon backends support full only.",
            expected=str,
            required=True,
            choices=("sea", "valence", "singlet", "full"),
            choice_descriptions={
                "sea": "Construct the antiquark/sea projection from the negative-x extension.",
                "valence": "Construct the quark-minus-antiquark projection; the active complex channel depends on polarization.",
                "singlet": "Construct the quark-plus-antiquark projection; the active complex channel depends on polarization.",
                "full": "Keep the full signed-x distribution; this is the only supported sector for DA and gluon backends.",
            },
        ),
        "target_observable": _parameter("Run target override.", "It must agree with metadata.target_observable so validation and execution select the same physics.", expected=str, choices=("pdf", "da", "gpd")),
        "zmin_shift": _parameter(
            "Symmetric index shift used to generate low/high tail-window systematics branches.",
            "A nonzero magnitude asks manifest expansion to clone the Fourier job with negative and positive shifts of the automatically selected minimum tail-fit coordinate; the central job uses zero. Prefer explicit scheme_scan ranges when a fixed physical window is intended.",
            expected=int,
            default=0,
        ),
        "y_grid": _parameter(
            "Dimensionless momentum-fraction grid for the transformed quasi-distribution.",
            "This required grid declares the momentum-fraction coordinates where the Fourier result is sampled. Use an explicit numeric list, or an object with start, stop, and exactly one of num or positive step. Increasing its density refines output discretization but cannot create information absent from the finite coordinate-space signal.",
            expected=(list, dict),
            items=float,
            required=True,
            schema=_GRID_FIELDS,
            examples=([-1.0, 0.0, 1.0], {"start": -1.0, "stop": 1.0, "num": 101}),
            validator=_validate_y_grid,
            suggested_fix='For example, use {"start": -1.0, "stop": 1.0, "num": 101}.',
        ),
    },
    removed={
        "Lambda0": "is no longer supported; use Lambda0_gev.",
        "distribution_type": "is no longer supported; use polarization.",
    },
    constraints=FOURIER_CONSTRAINTS,
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved Fourier context consumed by the shared evaluator."""
    stage = manifest.stages["fourier_transform"]
    authored = merge_stage_params(stage.defaults, job.params)
    resolved = resolve_stage_params("fourier_transform", stage.defaults, job.params)
    params = {**derive_job_kinematics(manifest, job), **resolved}
    context = StageValidationContext(
        stage="fourier_transform",
        job_id=job.id,
        job_path=f"stages.fourier_transform.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        authored_params=authored,
    )
    return context


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise messages for the agent loop."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
