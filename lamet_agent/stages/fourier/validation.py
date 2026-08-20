"""Fourier manifest parameter contract, planning guidance, and validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics, job_input_refs
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


def resolve_grid_spec(spec: list[float] | dict[str, Any], *, name: str = "grid") -> list[float]:
    """Turn a manifest grid spec into an explicit list of points.

    A spec is either an explicit numeric list or a compact dict: ``{start, stop, num}``
    (linspace, ``stop`` inclusive) or ``{start, stop, step}`` (arange, ``stop``
    inclusive). ``name`` only labels error messages.
    """
    if isinstance(spec, dict):
        start = float(spec["start"])
        stop = float(spec["stop"])
        if "num" in spec:
            num = int(spec["num"])
            if num < 2:
                raise ValueError(f"{name} num must be at least 2")
            return np.linspace(start, stop, num).tolist()
        step = float(spec["step"])
        if step <= 0:
            raise ValueError(f"{name} step must be positive")
        return np.arange(start, stop + 0.5 * step, step).tolist()
    return [float(item) for item in spec]


def quasi_y_ls_error(grid: np.ndarray, *, eps: float = 1e-12) -> str | None:
    """Return a message if ``grid`` cannot be used as a matching integration grid."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2:
        return "quasi_y_ls must resolve to at least 2 points."
    if not np.all(np.isfinite(values)):
        return "quasi_y_ls must contain only finite values."
    if np.any(np.abs(values) <= eps):
        return (
            "quasi_y_ls must not contain 0: matching kernels carry a 1/y measure, "
            "so a y = 0 point is singular. With a symmetric {start, stop, num} spec an "
            "even num avoids the midpoint (num=100 does, num=101 does not)."
        )
    spacing = np.diff(values)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=eps):
        return "quasi_y_ls must be uniformly spaced."
    return None


def _validate_quasi_y_ls(value: Any) -> str | None:
    if isinstance(value, list):
        if not value:
            return "quasi_y_ls must not be empty."
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
            return "quasi_y_ls list values must be finite numbers."
    elif not isinstance(value, dict):
        return "quasi_y_ls must be a numeric list or an object."
    else:
        if not {"start", "stop"}.issubset(value):
            return "quasi_y_ls object requires start and stop."
        selectors = {key for key in ("num", "step") if key in value}
        if len(selectors) != 1:
            return "quasi_y_ls object requires exactly one of num or step."
        if "num" in value and (type(value["num"]) is not int or value["num"] < 2):
            return "quasi_y_ls num must be an integer of at least 2."
        if "step" in value and (
            isinstance(value["step"], bool)
            or not isinstance(value["step"], (int, float))
            or not math.isfinite(float(value["step"]))
            or float(value["step"]) <= 0.0
        ):
            return "quasi_y_ls step must be a finite positive number."
    try:
        grid = np.asarray(resolve_grid_spec(value, name="quasi_y_ls"), dtype=float)
    except (TypeError, ValueError, KeyError) as exc:
        return str(exc)
    return quasi_y_ls_error(grid)


def _validate_scheme_scan(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
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


def _check_input_role(context: StageValidationContext) -> RuleViolation | list[RuleViolation] | None:
    roles = set(context.inputs)
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    initial = context.params.get("initial_momentum")
    final = context.params.get("final_momentum")
    nonforward_gpd = target == "gpd" and initial is not None and final is not None and initial != final
    expected = {"input", "hermitian_partner"} if nonforward_gpd else {"input"}
    issues = []
    if roles != expected:
        issues.append(
            _violation(
                context,
                message=f"A {target.upper()} Fourier job requires input roles {sorted(expected)}.",
                path=f"{context.job_path}.inputs",
                cause=f"The effective input roles are {sorted(roles)}.",
                parameters=("inputs.input", "inputs.hermitian_partner"),
            )
        )
    if target != "gpd" and "bilocal_anchor" in (context.authored_params or {}):
        issues.append(
            _violation(
                context,
                message="bilocal_anchor is only valid for GPD Fourier transforms.",
                path=context.parameter_path("bilocal_anchor"),
                cause=f"The run target_observable is {target!r}.",
                parameters=("bilocal_anchor", "metadata.target_observable"),
            )
        )
    partner = context.resources.get("partner_kinematics", {})
    if nonforward_gpd and "hermitian_partner" in roles and (
        partner.get("initial_momentum") != final or partner.get("final_momentum") != initial
    ):
        issues.append(
            _violation(
                context,
                message="The GPD hermitian_partner must exchange the initial and final momenta.",
                path=f"{context.job_path}.inputs.hermitian_partner",
                cause=(
                    f"The target flow is {initial}->{final}, while the partner flow is "
                    f"{partner.get('initial_momentum')}->{partner.get('final_momentum')}."
                ),
                parameters=("inputs.hermitian_partner",),
            )
        )
    return issues


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


def _check_gfix_provenance(context: StageValidationContext) -> RuleViolation | None:
    source = context.resources.get("gfix_source")
    inherited = context.resources.get("inherited_gfix")
    authored = (context.authored_params or {}).get("gfix")
    if source == "correlator" and authored is not None:
        return _violation(
            context,
            message="Fourier gfix is inherited from correlator provenance and must not be redeclared.",
            path=context.parameter_path("gfix"),
            cause=f"The correlator declares gfix={inherited!r}, while Fourier declares {authored!r}.",
            parameters=("gfix",),
        )
    if source == "artifact" and inherited is not None and authored is not None and inherited != authored:
        return _violation(
            context,
            message="Fourier gfix conflicts with the external artifact provenance.",
            path=context.parameter_path("gfix"),
            cause=f"The artifact records gfix={inherited!r}, while Fourier declares {authored!r}.",
            parameters=("gfix",),
        )
    return None


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


def _check_hadron(context: StageValidationContext) -> RuleViolation | None:
    target = str(context.metadata.get("target_observable", "pdf")).lower()
    parton = str(context.metadata.get("parton", "quark")).lower()
    hadron = str(context.params.get("hadron", "")).lower()
    hadron = "nucleon" if hadron == "proton" else hadron
    if (parton == "gluon" and target != "pdf") or (target == "da" and hadron) or (target, parton, hadron) in INFERRED_OBSERVABLES:
        return None
    return _violation(
        context,
        message=f"Fourier job {context.job_id!r} has no supported hadron for observable inference.",
        path=context.parameter_path("hadron"),
        cause=f"No backend is registered for target={target!r}, parton={parton!r}, hadron={hadron!r}.",
        parameters=("hadron",),
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


def _float_list(value: Any) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return [float(item) for item in value]
    return None


def _bz_max_for_correlator_job(manifest: AnalysisManifest, job: StageJob) -> int | None:
    values: list[int] = []
    for correlator in manifest.correlators:
        if correlator.correlator_id not in job.correlator_ids:
            continue
        if correlator.bz:
            values.extend(int(item) for item in correlator.bz)
    return max(values) if values else None


def upstream_z_last_fm(manifest: AnalysisManifest, job: StageJob) -> float | None:
    """Return max(bz)*a from an upstream correlator, or None when the z grid is unknown."""
    jobs = {
        candidate.id: (stage_id, candidate)
        for stage_id, config in manifest.stages.items()
        for candidate in config.jobs
    }
    artifacts = {artifact.id for artifact in manifest.inputs.artifacts}

    def from_job(stage_id: str, candidate: StageJob, seen: set[str]) -> float | None:
        if stage_id == "correlator_analysis":
            spacing = derive_job_kinematics(manifest, candidate).get("lattice_spacing_fm")
            bz_max = _bz_max_for_correlator_job(manifest, candidate)
            if spacing is None or bz_max is None:
                return None
            return float(bz_max) * float(spacing)
        for role in ("input", "quasi", "target", "reference", "denominator"):
            for reference in job_input_refs(candidate.inputs.get(role)):
                if reference in seen or reference in artifacts:
                    continue
                found = jobs.get(reference)
                if found is None:
                    continue
                resolved = from_job(*found, seen | {reference})
                if resolved is not None:
                    return resolved
        return None

    stage = next((stage_id for stage_id, config in manifest.stages.items() if job in config.jobs), None)
    if stage is None:
        return None
    return from_job(stage, job, {job.id})


def _check_scheme_scan_grid_range(context: StageValidationContext) -> list[RuleViolation] | None:
    spec = context.params.get("scheme_scan")
    if not isinstance(spec, dict):
        return None
    issues: list[RuleViolation] = []
    zmin_values = _float_list(spec["zmin_fm"]) if "zmin_fm" in spec else None
    zmax_values = _float_list(spec["zmax_fm"]) if "zmax_fm" in spec else None
    zmax_ext = spec.get("zmax_ext_fm")
    if (
        zmax_values
        and isinstance(zmax_ext, (int, float))
        and not isinstance(zmax_ext, bool)
        and float(zmax_ext) < max(zmax_values)
    ):
        issues.append(
            _violation(
                context,
                message="scheme_scan zmax_ext_fm must be greater than or equal to every zmax_fm candidate.",
                path=context.parameter_path("scheme_scan.zmax_ext_fm"),
                cause=f"zmax_ext_fm={float(zmax_ext)} is below max(zmax_fm)={max(zmax_values)}.",
                parameters=("scheme_scan",),
            )
        )
    z_last = context.resources.get("z_last_fm")
    spacing = context.params.get("lattice_spacing_fm")
    if z_last is not None and spacing is not None:
        upper = float(z_last) + 0.5 * float(spacing)
        for key, values in (("zmin_fm", zmin_values), ("zmax_fm", zmax_values)):
            if not values:
                continue
            for item in values:
                if item <= 0.0 or item > upper:
                    issues.append(
                        _violation(
                            context,
                            message=(
                                f"scheme_scan {key}={item} lies outside the available z grid "
                                f"(0, {float(z_last)} fm]. If this was a lattice-site index, convert with "
                                "n * lattice_spacing_fm."
                            ),
                            path=context.parameter_path(f"scheme_scan.{key}"),
                            cause=f"Available z ends at {float(z_last)} fm (max bz * lattice_spacing_fm).",
                            parameters=("scheme_scan",),
                        )
                    )
    return issues or None


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
    "zmax_ext_fm": _parameter(
        "Physical distance through which the fitted tail is extended.",
        "The extension endpoint controls transform truncation and is always expressed in fm.",
        expected=float,
    ),
    "zmax_fm": _parameter(
        "Maximum fit-distance candidates in fm.",
        "Each physical distance truncates the data region used to constrain the long-distance tail.",
        expected=list,
        items=float,
    ),
    "zmin_fm": _parameter(
        "Minimum fit-distance candidates in fm.",
        "These physical distances control where the asymptotic ansatz begins to describe the matrix element.",
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
        code="fourier.inputs.observable_contract",
        parameters=("inputs.input", "inputs.hermitian_partner", "bilocal_anchor"),
        rule="PDF, DA, and forward GPD jobs use input; nonforward GPD jobs also use the exchanged-flow hermitian_partner.",
        physics="Nonforward GPD Hermiticity relates the negative-z branch to the flow with exchanged initial and final momenta.",
        suggested_fix='Use {"input": "<flow>", "hermitian_partner": "<exchanged flow>"} for a nonforward GPD job.',
        check=_check_input_role,
    ),
    ConstraintSpec(
        code="fourier.gfix.provenance",
        parameters=("gfix", "inputs.input"),
        rule="Correlator-backed jobs inherit gfix; external jobs declare it explicitly and agree with artifact provenance.",
        physics="CG and GI matrix elements use different long-distance tail parameterizations, so the Fourier choice must match the gauge-link construction of its input.",
        suggested_fix="Remove a redundant correlator-backed gfix, or declare the artifact-compatible CG/GI value for an external input.",
        check=_check_gfix_provenance,
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
        code="fourier.hadron.required",
        parameters=("hadron", "metadata.target_observable", "metadata.parton"),
        rule="A supported hadron must be authored or inherited so the short observable can be inferred.",
        physics="Target, parton, and hadron select the asymptotic backend; polarization remains a separate physical label.",
        suggested_fix="Declare hadron in Fourier params when upstream provenance cannot supply it.",
        check=_check_hadron,
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
        code="fourier.scheme_scan.coordinates",
        parameters=("scheme_scan",),
        rule="Tail-scan coordinates zmin_fm, zmax_fm, and zmax_ext_fm are physical distances in fm.",
        physics="A fixed physical-distance convention prevents lattice sites and Ioffe time from being mixed into the fitted long-distance window.",
        suggested_fix="Express every Fourier fit-range value in fm.",
        check=_check_scheme_scan,
    ),
    ConstraintSpec(
        code="fourier.scheme_scan.grid_range",
        parameters=("scheme_scan",),
        rule="Authored zmin_fm and zmax_fm must lie on the available coordinate grid, and zmax_ext_fm must be at least the largest zmax_fm.",
        physics="Tail windows are physical distances on the measured z grid; a lattice-site index written in fm falls far outside that grid.",
        suggested_fix="Convert lattice indices with n * lattice_spacing_fm, or omit the range keys and let the stage auto-fill from the data.",
        check=_check_scheme_scan_grid_range,
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
    target = str(metadata.get("target_observable", "")).lower()
    force_full = target == "da" or str(metadata.get("parton", "")).lower() == "gluon"
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    stage = stages.get("fourier_transform", {}) if isinstance(stages, dict) else {}
    if not isinstance(stage, dict):
        return []
    edits: list[dict[str, Any]] = []
    defaults = stage.get("defaults", {})
    if target != "gpd" and isinstance(defaults, dict) and "bilocal_anchor" in defaults:
        old = defaults.pop("bilocal_anchor")
        edits.append({"path": "stages.fourier_transform.defaults.bilocal_anchor", "old": old, "new": None, "note": "Removed the GPD-only bilocal anchor."})
    jobs = stage.get("jobs", [])
    if target != "gpd" and isinstance(jobs, list):
        for job in jobs:
            params = job.get("params", {}) if isinstance(job, dict) else {}
            if isinstance(params, dict) and "bilocal_anchor" in params:
                old = params.pop("bilocal_anchor")
                edits.append({"path": f"stages.fourier_transform.jobs.{job.get('id', '')}.params.bilocal_anchor", "old": old, "new": None, "note": "Removed the GPD-only bilocal anchor."})
            inputs = job.get("inputs", {}) if isinstance(job, dict) else {}
            if isinstance(inputs, dict) and "hermitian_partner" in inputs:
                old = inputs.pop("hermitian_partner")
                edits.append({"path": f"stages.fourier_transform.jobs.{job.get('id', '')}.inputs.hermitian_partner", "old": old, "new": None, "note": "Removed the GPD-only Hermitian partner."})
    if not force_full:
        return edits
    if isinstance(defaults, dict) and "sector" in defaults and str(defaults.get("sector", "")).lower() != "full":
        old = defaults.get("sector")
        defaults["sector"] = "full"
        edits.append({"path": "stages.fourier_transform.defaults.sector", "old": old, "new": "full", "note": "DA and gluon Fourier sectors are fixed to full."})
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
    planning_notes=(
        "Only GPD jobs author bilocal_anchor; an omitted GPD value resolves to mid_at_0 without materializing a PDF/DA parameter.",
        "Each nonforward GPD job uses a job-specific hermitian_partner whose initial and final momenta exchange those of input.",
    ),
    input_roles=("input", "hermitian_partner"),
    input_role_descriptions={
        "input": "One renormalized coordinate-space matrix element to extend and Fourier transform.",
        "hermitian_partner": "The GPD matrix element with exchanged initial and final momenta used to reconstruct negative z.",
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
        "bilocal_anchor": _parameter(
            "Location fixed at the origin in a GPD bilocal operator.",
            "mid_at_0 uses the centered bilocal; barpsi_at_0 fixes the barred field; psi_at_0 fixes the unbarred field and reverses the canonical separation.",
            expected=str,
            choices=("mid_at_0", "barpsi_at_0", "psi_at_0"),
            choice_descriptions={
                "mid_at_0": "Use barpsi(-z/2) Gamma W psi(z/2).",
                "barpsi_at_0": "Use barpsi(0) Gamma W(0,z) psi(z).",
                "psi_at_0": "Use barpsi(z) Gamma W(z,0) psi(0).",
            },
        ),
        "coord_key": _parameter("NPZ/HDF5 coordinate dataset key.", "This maps an external file layout onto the stage coordinate axis.", expected=str, default="coord"),
        "gfix": _parameter(
            "Gauge-link treatment and long-distance tail family.",
            "CG and GI matrix elements use their corresponding asymptotic parameterizations; correlator-backed jobs inherit this value, while external jobs declare it explicitly.",
            expected=str,
            required=True,
            choices=("CG", "GI"),
            choice_descriptions={
                "CG": "Use the Coulomb-gauge asymptotic parameterization.",
                "GI": "Use the gauge-invariant asymptotic parameterization.",
            },
        ),
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
            "The selected part controls which coordinate-space channel constrains the output when sector is absent.",
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
            "zmin_fm/zmax_fm select the measured physical-distance range used to constrain the tail; zmax_ext_fm controls the subsequent extension. Omitting range keys lets the runtime infer bounded candidates from the fm coordinate grid. Range variation estimates the finite-distance systematic.",
            expected=dict,
            required=True,
            schema=_SCHEME_SCAN_FIELDS,
        ),
        "sector": _parameter(
            "Partonic projection of a quark PDF/GPD, or full distribution.",
            "PDF sectors select the negative-x extension and active complex channel. GPD sectors are projected sample by sample from the full complex result after the paired Fourier transform. DA and gluon backends support full only.",
            expected=str,
            choices=("sea", "valence", "singlet", "full"),
            choice_descriptions={
                "sea": "Construct the antiquark/sea projection from the negative momentum-fraction branch.",
                "valence": "Construct the quark-minus-antiquark projection; GPD keeps both complex channels through the transform.",
                "singlet": "Construct the quark-plus-antiquark projection; GPD keeps both complex channels through the transform.",
                "full": "Keep the full signed-x distribution; this is the only supported sector for DA and gluon backends.",
            },
        ),
        "zmin_shift": _parameter(
            "Symmetric index shift used to generate low/high tail-window systematics branches.",
            "A nonzero magnitude asks manifest expansion to clone the Fourier job with negative and positive shifts of the automatically selected minimum tail-fit coordinate; the central job uses zero. Prefer explicit scheme_scan ranges when a fixed physical window is intended.",
            expected=int,
            default=0,
        ),
        "quasi_y_ls": _parameter(
            "Dimensionless momentum-fraction grid for the transformed quasi-distribution.",
            "This required grid declares the momentum-fraction coordinates of the Fourier output, which matching then uses as its integration measure. It must be uniform and exclude zero because matching kernels carry a 1/y singularity. Use an explicit numeric list, or an object with start, stop, and exactly one of num or positive step. On a symmetric range an even num avoids the midpoint (num=100 does, num=101 does not). Increasing its density refines output discretization but cannot create information absent from the finite coordinate-space signal.",
            expected=(list, dict),
            items=float,
            required=True,
            schema=_GRID_FIELDS,
            examples=([-1.0, -0.5, 0.5, 1.0], {"start": -2.0, "stop": 2.0, "num": 100}),
            validator=_validate_quasi_y_ls,
            suggested_fix='For example, use {"start": -2.0, "stop": 2.0, "num": 100}.',
        ),
    },
    removed={
        "Lambda0": "is no longer supported; use Lambda0_gev.",
        "distribution_type": "is no longer supported; use polarization.",
        "y_grid": "is no longer supported; use quasi_y_ls.",
    },
    constraints=FOURIER_CONSTRAINTS,
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved Fourier context consumed by the shared evaluator."""
    stage = manifest.stages["fourier_transform"]
    authored = merge_stage_params(stage.defaults, job.params)
    resolved = resolve_stage_params("fourier_transform", stage.defaults, job.params)
    derived = derive_job_kinematics(manifest, job)
    reference = job.inputs.get("input")
    jobs = {
        candidate.id: (stage_id, candidate)
        for stage_id, config in manifest.stages.items()
        for candidate in config.jobs
    }
    artifacts = {artifact.id for artifact in manifest.inputs.artifacts}
    seen: set[str] = set()
    gfix_source = None
    while isinstance(reference, str) and reference not in seen:
        if reference in artifacts:
            gfix_source = "artifact"
            break
        found = jobs.get(reference)
        if found is None:
            break
        stage_id, candidate = found
        if stage_id == "correlator_analysis":
            gfix_source = "correlator"
            break
        seen.add(reference)
        reference = next(
            (
                candidate.inputs[key]
                for key in ("target", "input", "reference", "quasi")
                if isinstance(candidate.inputs.get(key), str)
            ),
            None,
        )
    inherited_gfix = derived.get("gfix")
    params = {**derived, **resolved}
    if gfix_source == "artifact" and "gfix" not in authored:
        params.pop("gfix", None)
    partner_kinematics = {}
    partner_reference = job.inputs.get("hermitian_partner")
    if isinstance(partner_reference, str):
        partner_artifact = next(
            (artifact for artifact in manifest.inputs.artifacts if artifact.id == partner_reference),
            None,
        )
        if partner_artifact is not None:
            partner_kinematics = {
                key: partner_artifact.resolved_metadata[key]
                for key in ("initial_momentum", "final_momentum")
                if partner_artifact.resolved_metadata.get(key) is not None
            }
        elif partner_reference in jobs:
            partner_kinematics = derive_job_kinematics(manifest, jobs[partner_reference][1])
    context = StageValidationContext(
        stage="fourier_transform",
        job_id=job.id,
        job_path=f"stages.fourier_transform.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        resources={
            "z_last_fm": upstream_z_last_fm(manifest, job),
            "gfix_source": gfix_source,
            "inherited_gfix": inherited_gfix,
            "partner_kinematics": partner_kinematics,
        },
        authored_params=authored,
    )
    return context


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise messages for the agent loop."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
