"""Executable manifest contract for correlator analysis."""

from __future__ import annotations

import math
from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import (
    ConstraintSpec,
    ListItems,
    ParameterSpec,
    RuleViolation,
    StageParamContract,
    StageValidationContext,
    merge_stage_params,
    resolve_stage_params,
)


def _parameter(summary: str, physics: str, **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _scopes(context: StageValidationContext) -> list[str]:
    raw = context.params.get("fit_scope", [])
    return [str(item) for item in raw] if isinstance(raw, list) else [str(raw)]


def _selected(context: StageValidationContext) -> list[Any]:
    return list(context.resources.get("selected_correlators", []))


def _violation(
    context: StageValidationContext,
    message: str,
    *,
    parameter: str,
    cause: str,
) -> RuleViolation:
    path = (
        f"{context.job_path}.correlator_ids"
        if parameter == "correlator_ids"
        else context.parameter_path(parameter)
    )
    return RuleViolation(message, path, cause, (parameter,))


def _validate_fit_scope(value: Any) -> str | None:
    raw = value if isinstance(value, list) else [value]
    allowed = {"3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"}
    if not raw or any(not isinstance(scope, str) or scope not in allowed for scope in raw):
        return "fit_scope must contain only '3pt_ratio', 'FH', '3pt_ratio+FH', or 'qda_ratio'."
    return None


def _positive_number_message(name: str):
    def validate(value: Any) -> str | None:
        values = value if isinstance(value, list) else [value]
        valid = bool(values) and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0.0
            for item in values
        )
        return None if valid else f"{name} must contain only finite positive values."

    return validate


def _nstate_message(value: Any) -> str | None:
    values = value if isinstance(value, list) else [value]
    valid = bool(values) and all(type(item) is int and item >= 1 for item in values)
    return None if valid else "nstate must contain only positive integer state counts."


def _q_min_message(value: Any) -> str | None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
    return None if valid else "q_min must be a finite probability between 0 and 1."


def _check_scope_compatibility(context: StageValidationContext) -> RuleViolation | None:
    scopes = _scopes(context)
    fitting_form = str(context.params.get("fitting_form", ""))
    if "qda_ratio" in scopes and len(scopes) != 1:
        return _violation(
            context,
            "fit_scope='qda_ratio' cannot be mixed with 3pt/FH scopes in one job.",
            parameter="fit_scope",
            cause=f"The effective fit scopes are {scopes!r}.",
        )
    if scopes == ["qda_ratio"] and fitting_form != "Breit":
        return _violation(
            context,
            "fit_scope='qda_ratio' requires fitting_form 'Breit'.",
            parameter="fitting_form",
            cause=f"The effective fitting_form is {fitting_form!r}.",
        )
    if fitting_form == "NonBreit" and any("FH" in scope for scope in scopes):
        return _violation(
            context,
            "fit_scope values containing 'FH' currently require fitting_form 'Breit'.",
            parameter="fitting_form",
            cause=f"The effective fit scopes are {scopes!r} with NonBreit kinematics.",
        )
    return None


def _check_fh_state_count(context: StageValidationContext) -> RuleViolation | None:
    if not any("FH" in scope for scope in _scopes(context)):
        return None
    raw = context.params.get("nstate", [])
    values = raw if isinstance(raw, list) else [raw]
    if not any(type(value) is int and value > 2 for value in values):
        return None
    return _violation(
        context,
        "FH correlator fits currently support nstate values no larger than 2.",
        parameter="nstate",
        cause=f"The effective nstate candidates are {values!r}.",
    )


def _check_qda_inputs(context: StageValidationContext) -> RuleViolation | None:
    if _scopes(context) != ["qda_ratio"]:
        return None
    momentum = context.params.get("momentum")
    if not isinstance(momentum, str):
        return _violation(
            context,
            "A qda_ratio correlator_analysis job requires scalar params.momentum.",
            parameter="momentum",
            cause=f"The effective momentum is {momentum!r}.",
        )
    pt2 = [item for item in _selected(context) if _field(item, "correlator_type") == "2pt"]
    matching = [item for item in pt2 if momentum in (_field(item, "momentum", []) or [])]
    qda = [
        item
        for item in matching
        if _field(item, "bz") is not None
        and (
            "_nonlocal" in str(_field(item, "source_operator", ""))
            or "_nonlocal" in str(_field(item, "sink_operator", ""))
        )
    ]
    local = [
        item
        for item in matching
        if "_nonlocal" not in str(_field(item, "source_operator", ""))
        and "_nonlocal" not in str(_field(item, "sink_operator", ""))
    ]
    if len(qda) != 1 or len(local) > 1:
        return _violation(
            context,
            "A qda_ratio job requires exactly one nonlocal qDA 2pt correlator with a bz grid and at most one ordinary local-source/local-sink 2pt correlator.",
            parameter="correlator_ids",
            cause=f"Found {len(qda)} qDA candidates and {len(local)} local candidates at {momentum}.",
        )
    qda_input = qda[0]
    bt = _field(qda_input, "bT")
    if bt is None or len(bt) != 1:
        return _violation(
            context,
            "A qda_ratio qDA 2pt correlator must declare exactly one bT value.",
            parameter="correlator_ids",
            cause=f"The selected qDA bT metadata is {bt!r}.",
        )
    if not local and 0 not in (_field(qda_input, "bz", []) or []):
        return _violation(
            context,
            "A qda_ratio job without an ordinary local 2pt correlator requires bz=0 in the nonlocal qDA 2pt grid.",
            parameter="correlator_ids",
            cause=f"The selected qDA bz grid is {_field(qda_input, 'bz')!r}.",
        )
    source = str(_field(qda_input, "source_operator", ""))
    sink = str(_field(qda_input, "sink_operator", ""))
    if any(token in source or token in sink for token in ("<bz>", "{bz}")):
        return _violation(
            context,
            "qDA source_operator and sink_operator must not encode bz placeholders.",
            parameter="correlator_ids",
            cause=f"The selected operators are source={source!r}, sink={sink!r}.",
        )
    if local:
        provenance_fields = (
            "ensemble",
            "hadron",
            "gfix",
            "volume",
            "lattice_spacing_fm",
            "temporal_extent",
        )
        qda_provenance = tuple(_field(qda_input, key) for key in provenance_fields)
        local_provenance = tuple(_field(local[0], key) for key in provenance_fields)
        if qda_provenance != local_provenance:
            return _violation(
                context,
                "The qDA and ordinary 2pt correlators must have matching ensemble provenance.",
                parameter="correlator_ids",
                cause=f"qDA provenance {qda_provenance!r} differs from local provenance {local_provenance!r}.",
            )
    return None


def _ordinary_selection(context: StageValidationContext) -> tuple[list[Any], list[Any], RuleViolation | None]:
    if _scopes(context) == ["qda_ratio"]:
        return [], [], None
    selected = _selected(context)
    pt2 = [item for item in selected if _field(item, "correlator_type") == "2pt"]
    pt3 = [item for item in selected if _field(item, "correlator_type") == "3pt"]
    fitting_form = str(context.params.get("fitting_form", ""))
    if fitting_form == "Breit":
        momentum = context.params.get("momentum")
        if not isinstance(momentum, str):
            return [], [], _violation(
                context,
                "A Breit correlator_analysis job requires scalar params.momentum.",
                parameter="momentum",
                cause=f"The effective momentum is {momentum!r}.",
            )
        selected_pt2 = [item for item in pt2 if momentum in (_field(item, "momentum", []) or [])]
        if not selected_pt2:
            return [], [], _violation(
                context,
                f"No selected 2pt correlator declares momentum {momentum!r}.",
                parameter="correlator_ids",
                cause="None of the selected 2pt momentum grids contains the requested momentum.",
            )
        return selected_pt2, [item for item in pt3 if momentum in (_field(item, "momentum", []) or [])], None
    initial = context.params.get("initial_momentum")
    final = context.params.get("final_momentum")
    if not isinstance(initial, str) or not isinstance(final, str):
        return [], [], _violation(
            context,
            "A NonBreit correlator_analysis job requires params.initial_momentum and params.final_momentum.",
            parameter="initial_momentum",
            cause=f"The effective values are initial={initial!r}, final={final!r}.",
        )
    if not any(initial in (_field(item, "momentum", []) or []) for item in pt2):
        return [], [], _violation(
            context,
            f"No selected 2pt correlator declares initial_momentum {initial!r}.",
            parameter="correlator_ids",
            cause="The initial state is absent from the selected 2pt momentum grids.",
        )
    if not any(final in (_field(item, "momentum", []) or []) for item in pt2):
        return [], [], _violation(
            context,
            f"No selected 2pt correlator declares final_momentum {final!r}.",
            parameter="correlator_ids",
            cause="The final state is absent from the selected 2pt momentum grids.",
        )
    selected_pt2 = [item for item in pt2 if initial in (_field(item, "momentum", []) or []) or final in (_field(item, "momentum", []) or [])]
    selected_pt3 = [item for item in pt3 if final in (_field(item, "momentum", []) or [])]
    return selected_pt2, selected_pt3, None


def _check_ordinary_inputs(context: StageValidationContext) -> RuleViolation | None:
    selected_pt2, selected_pt3, issue = _ordinary_selection(context)
    if issue is not None or _scopes(context) == ["qda_ratio"]:
        return issue
    if not selected_pt3:
        return _violation(
            context,
            "A correlator_analysis job requires at least one 3pt correlator.",
            parameter="correlator_ids",
            cause="No selected 3pt correlator contains the requested final momentum.",
        )
    scopes = _scopes(context)
    tseps = {tsep for item in selected_pt3 for tsep in (_field(item, "tsep", []) or [])}
    if any("FH" in scope for scope in scopes) and len(tseps) < 2:
        return _violation(
            context,
            "FH correlator_analysis jobs require at least two 3pt tsep values.",
            parameter="correlator_ids",
            cause=f"The selected 3pt data provide tsep values {sorted(tseps)!r}.",
        )
    if any(_field(item, "bT") is None or len(_field(item, "bT")) != 1 for item in selected_pt3):
        return _violation(
            context,
            "The current correlator stage requires exactly one bT value per 3pt correlator.",
            parameter="correlator_ids",
            cause="At least one selected 3pt correlator has a missing or non-scalar bT grid.",
        )
    reference = selected_pt3[0]
    operator_fields = ("source_operator", "sink_operator", "current_operator", "bz_direction", "bT", "bz")
    reference_operators = tuple(_field(reference, key) for key in operator_fields)
    if any(tuple(_field(item, key) for key in operator_fields) != reference_operators for item in selected_pt3[1:]):
        return _violation(
            context,
            "Selected 3pt correlators must use the same operators, bz_direction, bT, and bz grid.",
            parameter="correlator_ids",
            cause="The selected 3pt operator metadata are not identical.",
        )
    source_sink = (_field(reference, "source_operator"), _field(reference, "sink_operator"))
    if any((_field(item, "source_operator"), _field(item, "sink_operator")) != source_sink for item in selected_pt2):
        return _violation(
            context,
            "Selected 2pt and 3pt correlators must use the same source and sink operators.",
            parameter="correlator_ids",
            cause="At least one selected 2pt source/sink pair differs from the 3pt reference.",
        )
    provenance_fields = ("ensemble", "hadron", "gfix", "volume", "lattice_spacing_fm")
    provenance = tuple(_field(reference, key) for key in provenance_fields)
    if any(tuple(_field(item, key) for key in provenance_fields) != provenance for item in [*selected_pt2, *selected_pt3]):
        return _violation(
            context,
            "Selected correlators must use the same ensemble, hadron, gfix, volume, and lattice spacing.",
            parameter="correlator_ids",
            cause="The selected data do not share one ensemble provenance tuple.",
        )
    return None


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="correlator",
    summary="Selection and joint fitting of two- and three-point lattice correlators.",
    physics="The stage isolates ground-state matrix elements while controlling excited-state contamination and preserving common ensemble/operator provenance.",
    planning_notes=(
        "Omit fit windows and fit-control candidates unless the user explicitly requests them; automatic tuning supplies bounded candidates.",
        "Momentum fields are job-specific because each job selects one kinematic channel.",
    ),
    job_parameters=("momentum", "initial_momentum", "final_momentum"),
    schema={
        "component": _parameter(
            "Complex component used in the fit.",
            "Real and imaginary matrix-element components carry different operator information; fitting both preserves both channels in one job.",
            expected=str,
            choices=("re", "im", "both"),
            required=True,
            choice_descriptions={
                "re": "Fit only the real component.",
                "im": "Fit only the imaginary component.",
                "both": "Fit and export both components.",
            },
        ),
        "correlator_rescale": _parameter("Finite positive numerical rescaling applied before fitting.", "A common rescaling improves conditioning without changing the recovered physical matrix element.", expected=float, default=1.0, validator=_positive_number_message("correlator_rescale")),
        "final_momentum": _parameter("Final-state lattice momentum for NonBreit kinematics.", "The final momentum fixes the outgoing hadron state in a non-forward matrix element.", expected=str),
        "fit_scope": _parameter(
            "Correlator observable families fitted by the job.",
            "The scope chooses the estimator and required data: ordinary scopes use 2pt/3pt correlators, while qda_ratio uses a nonlocal qDA 2pt numerator and a local or bz=0 denominator.",
            expected=(str, list),
            items=str,
            choices=("3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"),
            choice_descriptions={
                "3pt_ratio": "Fit the resampled three-point/two-point ratio.",
                "FH": "Fit the Feynman-Hellmann estimator formed from summed ratios and neighboring source-sink separations.",
                "3pt_ratio+FH": "Fit ratio and Feynman-Hellmann channels in one correlated likelihood.",
                "qda_ratio": "Fit a nonlocal qDA two-point ratio; use O00/z0 with a local denominator or O00/zprime0 with the qDA bz=0 fallback. In fallback mode z=0 is identically one and is not fitted. This exclusive scope has no 3pt, tsep, tau-cut, or current insertion.",
            },
            required=True,
            validator=_validate_fit_scope,
            coerce_scalar_to_list=True,
        ),
        "fit_strategy": _parameter(
            "Joint, chained, or independent fit strategy candidates.",
            "The strategy changes how spectral information and covariance propagate from the two-point channel into the matrix-element estimator, not the target observable itself.",
            expected=(str, list),
            items=str,
            choices=("joint", "chained", "independent"),
            required=True,
            choice_descriptions={
                "joint": "Fit the selected 2pt and ratio/FH/qDA channels together with their shared covariance.",
                "chained": "Fit the 2pt channel first, then use its widened posterior as the spectral prior for the matrix-element fit.",
                "independent": "Fit ratio/FH/qDA data without a 2pt likelihood or transferred 2pt prior.",
            },
            coerce_scalar_to_list=True,
        ),
        "fitting_form": _parameter(
            "Breit or NonBreit kinematic form.",
            "Breit uses one momentum channel; NonBreit represents a non-forward matrix element with distinct incoming and outgoing hadron states.",
            expected=str,
            choices=("Breit", "NonBreit"),
            choice_descriptions={
                "Breit": "Use one scalar momentum for the initial and final state.",
                "NonBreit": "Use separate initial_momentum and final_momentum channels; FH scopes are not implemented for this form.",
            },
            required=True,
        ),
        "initial_momentum": _parameter("Initial-state lattice momentum for NonBreit kinematics.", "The initial momentum fixes the incoming hadron state in a non-forward matrix element.", expected=str),
        "model_average": _parameter(
            "Average successful fit-function candidates inside the selected data window.",
            "False applies one sample-average-selected nstate/prior-width model to every resample; true combines successful nstate/prior-width fits per sample with log-evidence weights and records their spread as model uncertainty. Data-window selection remains a separate tuning decision.",
            expected=bool,
            choices=(False, True),
            choice_descriptions={
                False: "Use one tuned fit function for every coordinate and resampled sample.",
                True: "Average successful fit functions sample by sample after one shared window is fixed.",
            },
            required=True,
        ),
        "momentum": _parameter("Scalar lattice momentum selected by a Breit or qDA job.", "The requested momentum must be present in the selected correlator data.", expected=str),
        "nstate": _parameter("Positive spectral state-count candidate or candidates.", "Additional states parameterize excited-state contamination; FH ansatz implementations currently support at most two states.", expected=(int, list), items=int, required=True, validator=_nstate_message, coerce_scalar_to_list=True),
        "posterior_prior_error_scale": _parameter(
            "Scale used to build per-sample priors from the sample-average posterior.",
            "The per-sample prior width is the sample-average posterior width multiplied by this scale and the selected prior_width, controlling how strongly noisy resamples are anchored.",
            expected=float,
            required=True,
            validator=_positive_number_message("posterior_prior_error_scale"),
        ),
        "prior_width": _parameter("Finite positive prior-width candidate or candidates.", "Prior-width variation tests the stability of excited-state removal.", expected=(float, list), items=float, default=[1], validator=_positive_number_message("prior_width"), coerce_scalar_to_list=True),
        "pt2_windows": ListItems(
            {"tmin": _parameter("First two-point fit time.", "Removing early times suppresses excited states.", expected=int), "tmax": _parameter("Last two-point fit time.", "Late-time reach balances ground-state isolation against noise.", expected=int)},
            summary="Candidate two-point fit windows.",
            physics="Window variation measures the stability of ground-state isolation against early-time contamination and late-time noise.",
        ),
        "pt3_windows": ListItems(
            {"tau_cut": _parameter("Source/sink exclusion for one candidate.", "The cut removes time slices closest to source and sink.", expected=int), "tsep_ls": _parameter("Source-sink separations in one fit candidate.", "Several separations resolve excited-state time dependence.", expected=list, items=int)},
            summary="Candidate three-point fit windows.",
            physics="Each window trades excited-state suppression against the statistical information retained across source-sink separations.",
        ),
        "q_min": _parameter("Minimum accepted fit p-value.", "The threshold rejects statistically incompatible fit models.", expected=float, required=True, validator=_q_min_message),
        "svdcut": _parameter("Finite positive relative covariance singular-value cut.", "Regularization stabilizes inversion of noisy correlated data.", expected=float, default=1e-12, validator=_positive_number_message("svdcut")),
    },
    removed={
        "variant": "is not a supported correlator_analysis parameter.",
        "pt3_tau_cuts": "was replaced by pt3_windows; declare {tsep_ls, tau_cut} candidates instead of a tau-cut list.",
    },
    constraints=(
        ConstraintSpec("correlator.scope.compatibility", ("fit_scope", "fitting_form"), "qDA is exclusive and requires Breit; FH currently requires Breit.", "These estimators use different kinematic and time-dependence models.", "Choose one compatible fit_scope/fitting_form combination.", _check_scope_compatibility),
        ConstraintSpec("correlator.fh.nstate", ("fit_scope", "nstate"), "FH scopes currently support nstate <= 2.", "The implemented summed-ratio finite-difference ansatz contains ground and first-excited-state terms only.", "Use nstate 1 or 2 for every FH candidate.", _check_fh_state_count),
        ConstraintSpec("correlator.qda.inputs", ("momentum", "correlator_ids"), "A qDA job selects one nonlocal qDA 2pt and at most one matching local 2pt.", "The nonlocal numerator and local or bz=0 denominator must describe the same hadron ensemble.", "Select the required qDA/local correlators with matching momentum and provenance.", _check_qda_inputs),
        ConstraintSpec("correlator.ordinary.inputs", ("momentum", "initial_momentum", "final_momentum", "correlator_ids"), "Ordinary jobs require compatible 2pt/3pt momentum, operator, separation, and provenance metadata.", "A correlated spectral fit is meaningful only when every input represents the same operator and ensemble channel.", "Select compatible correlators or correct the job kinematics.", _check_ordinary_inputs),
    ),
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved correlator context consumed by the shared evaluator."""
    stage = manifest.stages["correlator_analysis"]
    authored_params = merge_stage_params(stage.defaults, job.params)
    params = resolve_stage_params("correlator_analysis", stage.defaults, job.params)
    selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
    return StageValidationContext(
        stage="correlator_analysis",
        job_id=job.id,
        job_path=f"stages.correlator_analysis.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        resources={"selected_correlators": selected},
        authored_params=authored_params,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
