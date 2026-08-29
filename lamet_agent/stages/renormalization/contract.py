"""Manifest contract for renormalization."""

from __future__ import annotations

import inspect
import math
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from lamet_agent.contract import CheckContext, Depends, Issue, Provides, Recommends, Source, Value, stage_job_rules
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.ui import warning


def _annotation_accepts(annotation: Any, value: Any) -> bool:
    """Return whether a JSON value matches one supported kernel annotation."""
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts(candidate, value) for candidate in arguments)
    if origin is Literal:
        return any(type(value) is type(choice) and value == choice for choice in arguments)
    if annotation is np.ndarray:
        return isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is type(None):
        return value is None
    if origin is list:
        return isinstance(value, list) and all(_annotation_accepts(arguments[0], item) for item in value)
    return False


def _kernel_parameter_issues(kernel: Any, values: dict[str, Any]) -> list[Issue]:
    """Validate authored renormalization kernel parameters against its signature."""
    physics = "Kernel parameters must match the selected renormalization kernel() signature."
    parameters = inspect.signature(kernel).parameters
    z_fm = parameters.get("z_fm")
    mu = parameters.get("mu")
    if (
        z_fm is None
        or z_fm.kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or mu is None
        or mu.kind is inspect.Parameter.POSITIONAL_ONLY
        or any(
            parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in parameters.values()
        )
    ):
        return [Issue("kernel_id", "kernel must explicitly accept z_fm and keyword mu", physics)]
    issues = [
        Issue("kernel_parameters.z_fm", "is supplied by input data and cannot be overridden", physics)
        for _name in values
        if _name == "z_fm"
    ]
    issues.extend(
        Issue(f"kernel_parameters.{name}", "is not accepted by the selected kernel signature", physics)
        for name in values
        if name not in parameters
    )
    issues.extend(
        Issue(f"kernel_parameters.{name}", "is required by the selected kernel signature", physics)
        for name, parameter in parameters.items()
        if name not in {"z_fm", "mu"} and parameter.default is inspect.Parameter.empty and name not in values
    )
    try:
        annotations = get_type_hints(kernel)
    except (NameError, TypeError) as exc:
        return [*issues, Issue("kernel_id", f"kernel annotations cannot be resolved: {exc}", physics)]
    for name, value in values.items():
        if name not in parameters or name == "z_fm":
            continue
        annotation = annotations.get(name, parameters[name].annotation)
        if annotation is inspect.Parameter.empty:
            issues.append(Issue(f"kernel_parameters.{name}", "has no type annotation", physics))
        elif not _annotation_accepts(annotation, value):
            expected = getattr(annotation, "__name__", str(annotation).replace("typing.", ""))
            issues.append(Issue(f"kernel_parameters.{name}", f"must match kernel annotation {expected}", physics))
    return issues


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _positive_int(value: int) -> bool:
    return value > 0


def _nonnegative_finite(value: int | float) -> bool:
    return math.isfinite(value) and value >= 0


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Depends("", "strategy", physics="Renormalization factors are either supplied through an external denominator or extracted by fitting reference matrix elements in the self-renormalization workflow."),
    Value("strategy", Literal["external_denominator", "self_renormalization"], physics="Selects whether the denominator factor is externally determined or fitted from reference matrix elements."),
    Provides("", "external_denominator", "strategy", physics="External-denominator jobs apply a renormalization factor determined outside this workflow to the target matrix element."),
    Provides("", "self_renormalization", "strategy", physics="Self-renormalization jobs extract a reusable factor from reference matrix elements and apply it to target matrix elements."),
    Depends("", "normalization", physics="Selects whether matrix-element inputs are normalized at the origin z=0 before the renormalization prescription is applied."),
    Value("normalization", bool, physics="Origin normalization divides each sample by its own value at the unique z=0 point, setting that local normalization to one."),
    Depends("external_denominator", "scheme", physics="The external scheme determines how the supplied renormalization factor is interpreted across the short- and long-distance regions."),
    Recommends("external_denominator", "type", physics="External factors are not fitted in this stage, so external-denominator jobs use type='apply'.", default="apply"),
    Value("external_denominator.type", Literal["apply"], physics="External-denominator jobs apply an authored renormalization prescription rather than fitting one."),
    Value("external_denominator.scheme", Literal["ratio", "hybrid", "msbar"], physics="The supplied denominator represents the selected ratio, hybrid, or MSbar renormalization prescription."),
    Provides("external_denominator", "hybrid", "external_denominator.scheme", physics="External hybrid renormalization separates perturbative short-distance treatment from predominantly nonperturbative long-distance mass subtraction."),
    Depends("external_denominator.hybrid", "zs_fm", physics="The switching distance marks the boundary between the short-range and long-range hybrid prescriptions."),
    Depends("external_denominator.hybrid", "m0_gev", physics="The long-distance hybrid correction includes an explicit renormalon-related linear mass contribution."),
    Depends("external_denominator.hybrid", "delta_m_gev", physics="The long-distance hybrid correction includes the linear-power-divergence mass contribution associated with the Wilson line."),
    Value("external_denominator.hybrid.zs_fm", (int, float), physics="The hybrid short/long-distance boundary is a finite positive physical distance in fm and must be represented on the z grid.", validator=_positive),
    Value("external_denominator.hybrid.m0_gev", (int, float), physics="Renormalon-related linear mass parameter for the long-distance hybrid correction, in GeV.", validator=_finite),
    Value("external_denominator.hybrid.delta_m_gev", (int, float), physics="Linear-power-divergence mass contribution for the long-distance hybrid correction, in GeV.", validator=_finite),
    Provides("external_denominator", "msbar", "external_denominator.scheme", physics="External MSbar jobs use a denominator determined in the MSbar prescription and record its perturbative scale."),
    Depends("external_denominator.msbar", "mu", physics="An externally determined MSbar factor must be associated with an explicit perturbative scale."),
    Value("external_denominator.msbar.mu", (int, float), physics="The external MSbar perturbative scale is finite and positive.", validator=_positive),
    Depends("self_renormalization", "scheme", physics="Self-renormalization selects ratio, hybrid, or MSbar for the fitted factor and its target application."),
    Recommends("self_renormalization", "type", physics="Self-renormalization uses type='fit' to extract a reusable factor from reference matrix elements and type='apply' to use it on target matrix elements.", default="apply"),
    Value("self_renormalization.type", Literal["fit", "apply"], physics="A self-renormalization job either extracts the factor from references or applies it to a target."),
    Provides("self_renormalization", "fit", "self_renormalization.type", physics="Fit jobs determine the reference operator's finite correction and reusable self-renormalization factor."),
    Provides("self_renormalization", "apply", "self_renormalization.type", physics="Apply jobs remap the reusable factor to the target operator using target-specific finite corrections."),
    Value("self_renormalization.scheme", Literal["ratio", "hybrid", "msbar"], physics="The self-renormalization scheme records whether the fitted factor is used for ratio, hybrid, or MSbar renormalization."),
    Depends("self_renormalization", "kernel_id", physics="Self-renormalization requires an explicit coordinate-space MSbar conversion kernel for short-distance matching and finite-term determination."),
    Value("self_renormalization.kernel_id", str, physics="The kernel id selects the operator- and channel-specific coordinate-space MSbar conversion formula, including its perturbative order."),
    Recommends("self_renormalization", "kernel_parameters", physics="Kernel parameters expose non-coordinate inputs of the selected conversion formula; coordinate z is supplied by the data and the stage scale may be explicitly overridden.", default={}),
    Value("self_renormalization.kernel_parameters", dict, physics="Kernel parameter overrides form the explicit mapping passed to the selected conversion formula."),
    Depends("self_renormalization", "mu", physics="Self-renormalization uses an explicit perturbative scale for short-distance matching and factor provenance."),
    Depends("self_renormalization", "LambdaQCD_gev", physics="Fit and apply use one QCD scale so the UV running and self-renormalization remapping share the same prescription."),
    Recommends("self_renormalization", "svdcut", physics="The covariance singular-value cutoff regularizes ill-conditioned directions in the correlated self-renormalization fit; the default is 1e-12.", default=1e-12),
    Recommends("self_renormalization", "z_coverage_policy", physics="Target coverage relative to the fitted factor is either required in full, restricted to the intersection, or extended only at the long-distance upper end.", default="extrapolate"),
    Depends("self_renormalization.fit", "d", physics="The reference fit uses the finite perturbative correction associated with the reference operator."),
    Depends("self_renormalization.apply", "d", physics="Application remaps the fitted factor with the target operator's finite perturbative correction, which may differ from the fit value."),
    Depends("self_renormalization.apply", "m0_gev", physics="Application remaps the fitted factor with the target operator's renormalon-related mass contribution."),
    Value("self_renormalization.mu", (int, float), physics="The perturbative matching scale is finite and positive.", validator=_positive),
    Value("self_renormalization.LambdaQCD_gev", (int, float), physics="Lambda_QCD is the finite positive QCD scale used by the common self-renormalization prescription.", validator=_positive),
    Value("self_renormalization.svdcut", (int, float), physics="The covariance singular-value cutoff used to stabilize the self-renormalization fit is finite and positive.", validator=_positive),
    Value("self_renormalization.z_coverage_policy", Literal["strict", "intersection", "extrapolate"], physics="Coverage is strict, limited to the target/factor intersection, or extrapolated only toward larger long-distance coordinates."),
    Value("self_renormalization.fit.d", (int, float), physics="The reference operator's finite perturbative correction coefficient is finite.", validator=_finite),
    Value("self_renormalization.apply.d", (int, float), physics="The target operator's finite perturbative correction coefficient is finite and may differ from the reference-fit value.", validator=_finite),
    Value("self_renormalization.apply.m0_gev", (int, float), physics="The target operator's renormalon-related mass parameter is finite and is used for factor remapping.", validator=_finite),
    Provides("self_renormalization", "hybrid", "self_renormalization.scheme", physics="Self-hybrid renormalization uses an external ratio at short distance and the reusable self-renormalization factor at long distance, with a transfer normalization enforcing continuity at the switch."),
    Depends("self_renormalization.hybrid", "zs_fm", physics="The switching distance marks the boundary between short-distance ratio treatment and long-distance self-renormalization."),
    Value("self_renormalization.hybrid.zs_fm", (int, float), physics="The self-hybrid switching distance is a finite positive physical distance in fm and must lie on the z grid.", validator=_positive),
)

INPUT_RULES = ()

JOB_CONDITIONAL_RULES = (
    Depends("external_denominator", "inputs", physics="External renormalization applies the supplied denominator factor to one target matrix element."),
    Depends("external_denominator.inputs", "target", physics="The target is the bare matrix element selected for renormalization."),
    Depends("external_denominator.inputs", "denominator", physics="The denominator is the externally determined renormalization factor, supplied as a coordinate-dependent source or a nonzero constant."),
    Source("external_denominator.inputs.target", physics="The target comes from one prior job or external file source."),
    Source("external_denominator.inputs.denominator", physics="The external renormalization factor comes from one prior job, file, or nonzero constant.", allow_constant=True),
    Depends("self_renormalization.fit", "inputs", physics="Self-renormalization fitting extracts a reusable factor from reference matrix elements at multiple lattice spacings."),
    Depends("self_renormalization.fit.inputs", "reference", physics="The reference matrix elements used to extract the factor must be explicit."),
    Source("self_renormalization.fit.inputs.reference", physics="The fit reference is one (a,z) source or an ordered source list at different lattice spacings sharing the physical z grid.", allow_list=True),
    Depends("self_renormalization.apply", "inputs", physics="Self-renormalization application combines one target matrix element with one fitted renormalization factor."),
    Depends("self_renormalization.apply.inputs", "target", physics="The target is the bare matrix element to be renormalized."),
    Depends("self_renormalization.apply.inputs", "zR", physics="zR is the fitted reusable self-renormalization factor applied to the target."),
    Source("self_renormalization.apply.inputs.target", physics="The target comes from one prior job or external file source."),
    Source("self_renormalization.apply.inputs.zR", physics="The reusable factor comes from one prior self-renormalization fit or external file source."),
    Provides("self_renormalization.apply", "hybrid", "self_renormalization.scheme", physics="Self-hybrid application combines short-distance external ratio renormalization with long-distance self-renormalization while preserving continuity at the switch."),
    Depends("self_renormalization.apply.hybrid.inputs", "denominator", physics="The short-distance denominator anchors the ratio branch and its continuity with the long-distance fitted factor."),
    Source("self_renormalization.apply.hybrid.inputs.denominator", physics="The short-distance denominator comes from one prior job or external file source."),
)
# fmt: on
# ruff: enable[E501]


def check_path(context: CheckContext) -> Issue | None:
    physics = "Only declared renormalization strategy, type, and scheme combinations define a valid numerical path; the renormalon-related mass term is extracted by fitting and supplied when applying the factor."
    params = context.params
    strategy = params.get("strategy")
    job_type = params.get("type")
    if job_type == "fit" and strategy != "self_renormalization":
        return Issue("type", "fit is available only for self_renormalization", physics)
    if job_type == "fit" and "d" not in params:
        return Issue("d", "is required by a self-renormalization fit", physics)
    if job_type == "fit" and "m0_gev" in params:
        return Issue("m0_gev", "is fitted from the reference and must be omitted", physics)
    return None


def check_hybrid(context: CheckContext) -> Issue | None:
    params = context.params
    inputs = context.inputs
    physics = "Hybrid renormalization separates short- and long-distance treatments at an explicit switching distance; the external branch additionally requires its mass-subtraction parameters and coordinate-dependent denominator."
    if params.get("scheme") == "hybrid":
        missing = ["zs_fm"] if "zs_fm" not in params else []
        if missing:
            return Issue("zs_fm", "is required by scheme='hybrid'", physics)
        if params.get("strategy") == "external_denominator" and "denominator" not in inputs:
            return Issue("inputs.denominator", "is required for hybrid external renormalization", physics)
        if params.get("strategy") == "external_denominator":
            missing = [key for key in ("m0_gev", "delta_m_gev") if key not in params]
            if missing:
                return Issue(missing[0], "is required for hybrid external renormalization", physics)
        elif "delta_m_gev" in params:
            return Issue("delta_m_gev", "delta_m_gev is an external-denominator hybrid parameter", physics)
    return None


def check_inputs(context: CheckContext) -> Issue | None:
    physics = "The strategy, job type, and scheme determine whether target, denominator, reference, and fitted factor roles are physically required; no additional input roles are meaningful."
    params = context.params
    strategy = params.get("strategy")
    scheme = params.get("scheme")
    job_type = params.get("type")
    expected = (
        {"target", "denominator"}
        if strategy == "external_denominator"
        else {"reference"}
        if job_type == "fit"
        else {"target", "denominator", "zR"}
        if scheme == "hybrid"
        else {"target", "zR"}
    )
    active = set(context.inputs)
    if active != expected:
        return Issue(
            "inputs",
            f"must provide exactly {sorted(expected)}; missing={sorted(expected - active)}, unexpected={sorted(active - expected)}",
            physics,
        )
    for role in active:
        source = context.inputs[role]
        values = source if isinstance(source, list) else [source]
        if not values or any(
            not isinstance(value, str)
            and not (isinstance(value, dict) and set(value) == {"file"} and isinstance(value["file"], str))
            and not (
                role == "denominator"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) != 0.0
            )
            for value in values
        ):
            return Issue(f"inputs.{role}", "contains an invalid job, file, list, or constant source", physics)
    if params.get("strategy") == "external_denominator":
        if params.get("scheme") == "hybrid" and not isinstance(context.inputs.get("denominator"), (str, dict)):
            return Issue("inputs.denominator", "hybrid requires a coordinate-dependent source", physics)
        if "zs_fm" in params and params.get("scheme") != "hybrid":
            return Issue("zs_fm", "is valid only for hybrid application", physics)
        if any(key in params for key in ("m0_gev", "delta_m_gev")) and params.get("scheme") != "hybrid":
            return Issue("m0_gev", "m0_gev and delta_m_gev are valid only for hybrid application", physics)
    return None


def check_kernel(context: CheckContext) -> list[Issue] | Issue | None:
    params = context.params
    if params.get("strategy") != "self_renormalization":
        return None
    kernel_id = params.get("kernel_id")
    if not isinstance(kernel_id, str):
        return None
    physics = "Self-renormalization matches the selected operator's short-distance behavior to an explicit coordinate-space MSbar kernel; its signature and authored overrides must be consistent."
    try:
        kernel = load_renormalization_kernel(kernel_id)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return Issue("kernel_id", str(exc), physics)
    values = params.get("kernel_parameters")
    if not isinstance(values, dict):
        return None
    issues = _kernel_parameter_issues(kernel, values)
    overridden = sorted({"mu"}.intersection(values))
    if not issues and overridden:
        warning(f"renormalization kernel_parameters overrides stage context: {overridden}")
    return issues


def _check_scale(context: CheckContext) -> Issue | None:
    params = context.params
    scheme = params.get("scheme")
    strategy = params.get("strategy")
    is_fit = params.get("type") == "fit"
    requires = is_fit or strategy == "self_renormalization" or scheme == "msbar"
    if requires and "mu" not in params:
        return Issue(
            "mu",
            "is required for this renormalization path",
            "Scale-dependent renormalization paths require a positive perturbative scale; pure external ratio division does not.",
        )
    if (is_fit or strategy == "self_renormalization") and "LambdaQCD_gev" not in params:
        return Issue(
            "LambdaQCD_gev",
            "is required for self-renormalization fitting and application",
            "The reusable zR fit and its target remap must use one shared QCD scale so their UV running and finite remapping remain in the same prescription.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES, JOB_CONDITIONAL_RULES)

CHECKS = (check_path, check_hybrid, check_inputs, check_kernel, _check_scale)
