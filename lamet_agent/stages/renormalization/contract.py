"""Manifest contract for renormalization."""

from __future__ import annotations

import inspect
import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Provides, Recommends, Source, Value, stage_job_rules
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.stages.renormalization.parameters import effective_params


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
    Depends("", "strategy", physics="Strategy selects external or self-renormalization data flow."),
    Value("strategy", Literal["external_denominator", "self_renormalization"], physics="Strategy is external_denominator or self_renormalization."),
    Provides("", "external_denominator", "strategy", physics="External-denominator jobs own direct ratio, hybrid, and MSbar parameters."),
    Provides("", "self_renormalization", "strategy", physics="Self-renormalization jobs own factor fitting and remapping parameters."),
    Depends("", "normalization", physics="Origin normalization is explicit."),
    Value("normalization", bool, physics="Origin normalization is boolean."),
    Depends("external_denominator", "scheme", physics="External renormalization selects ratio, hybrid, or MSbar."),
    Value("external_denominator.scheme", Literal["ratio", "hybrid", "msbar"], physics="External scheme is ratio, hybrid, or msbar."),
    Provides("external_denominator", "hybrid", "external_denominator.scheme", physics="External hybrid renormalization owns switch and mass-gap parameters."),
    Depends("external_denominator.hybrid", "zs_fm", physics="Hybrid switching distance is explicit."),
    Depends("external_denominator.hybrid", "m0_gev", physics="Hybrid long-distance mass correction is explicit."),
    Depends("external_denominator.hybrid", "delta_m_gev", physics="Hybrid mass difference is explicit."),
    Value("external_denominator.hybrid.zs_fm", (int, float), physics="Hybrid switching distance is finite and positive.", validator=_positive),
    Value("external_denominator.hybrid.m0_gev", (int, float), physics="The hybrid mass parameter is finite.", validator=_finite),
    Value("external_denominator.hybrid.delta_m_gev", (int, float), physics="The hybrid mass difference is finite.", validator=_finite),
    Provides("external_denominator", "msbar", "external_denominator.scheme", physics="External MSbar renormalization owns its perturbative scale."),
    Depends("external_denominator.msbar", "mu", physics="MSbar external renormalization requires an explicit scale."),
    Value("external_denominator.msbar.mu", (int, float), physics="The MSbar scale is finite and positive.", validator=_positive),
    Depends("self_renormalization", "scheme", physics="Self-renormalization selects ratio, hybrid, or MSbar."),
    Recommends("self_renormalization", "type", physics="Self-renormalization applies a reusable factor unless fit is selected explicitly.", default="apply"),
    Value("self_renormalization.type", Literal["fit", "apply"], physics="Self-renormalization job type is fit or apply."),
    Provides("self_renormalization", "fit", "self_renormalization.type", physics="Fit jobs own the reference-operator finite coefficient."),
    Provides("self_renormalization", "apply", "self_renormalization.type", physics="Apply jobs own target-operator remapping coefficients."),
    Value("self_renormalization.scheme", Literal["ratio", "hybrid", "msbar"], physics="Self-renormalization scheme is ratio, hybrid, or msbar."),
    Depends("self_renormalization", "kernel_id", physics="Self-renormalization explicitly selects its coordinate-space conversion formula."),
    Value("self_renormalization.kernel_id", str, physics="The renormalization kernel id is one exact public filename stem."),
    Depends("self_renormalization", "mu", physics="Self-renormalization uses an explicit perturbative scale."),
    Depends("self_renormalization", "LambdaQCD_gev", physics="Self-renormalization fitting and remapping share one explicitly authored QCD scale."),
    Recommends("self_renormalization", "svdcut", physics="The original self-renormalization reference fit defaults to a 1e-12 covariance singular-value cut.", default=1e-12),
    Depends("self_renormalization.fit", "d", physics="The reference fit uses one explicit finite logarithmic coefficient."),
    Depends("self_renormalization.apply", "d", physics="Application remaps one explicit finite logarithmic coefficient."),
    Depends("self_renormalization.apply", "m0_gev", physics="Application remaps the fitted residual mass."),
    Value("self_renormalization.mu", (int, float), physics="The scale is finite and positive.", validator=_positive),
    Value("self_renormalization.LambdaQCD_gev", (int, float), physics="Lambda_QCD is finite and positive.", validator=_positive),
    Value("self_renormalization.svdcut", (int, float), physics="The self-renormalization covariance cutoff is finite and positive.", validator=_positive),
    Value("self_renormalization.fit.d", (int, float), physics="The fit logarithmic correction is finite.", validator=_finite),
    Value("self_renormalization.apply.d", (int, float), physics="The apply logarithmic correction is finite.", validator=_finite),
    Value("self_renormalization.apply.m0_gev", (int, float), physics="The remapped residual mass is finite.", validator=_finite),
    Provides("self_renormalization", "hybrid", "self_renormalization.scheme", physics="Hybrid self-renormalization owns its switching distance."),
    Depends("self_renormalization.hybrid", "zs_fm", physics="Hybrid switching distance is explicit."),
    Value("self_renormalization.hybrid.zs_fm", (int, float), physics="Hybrid switching distance is finite and positive.", validator=_positive),
)

INPUT_RULES = ()

JOB_CONDITIONAL_RULES = (
    Depends("external_denominator", "inputs", physics="External renormalization consumes a target and denominator."),
    Depends("external_denominator.inputs", "target", physics="External renormalization consumes one target source."),
    Depends("external_denominator.inputs", "denominator", physics="External renormalization consumes one denominator source or constant."),
    Source("external_denominator.inputs.target", physics="The target is one prior job or external file source."),
    Source("external_denominator.inputs.denominator", physics="The denominator is one prior job, file, or nonzero constant.", allow_constant=True),
    Depends("self_renormalization.fit", "inputs", physics="Self-renormalization fitting consumes reference matrix elements."),
    Depends("self_renormalization.fit.inputs", "reference", physics="The fit reference is explicit."),
    Source("self_renormalization.fit.inputs.reference", physics="The fit reference is one source or an ordered source list.", allow_list=True),
    Depends("self_renormalization.apply", "inputs", physics="Self-renormalization application consumes a target and fitted factor."),
    Depends("self_renormalization.apply.inputs", "target", physics="Self-renormalization application consumes one target source."),
    Depends("self_renormalization.apply.inputs", "zR", physics="Self-renormalization application consumes one fitted factor source."),
    Source("self_renormalization.apply.inputs.target", physics="The target is one prior job or external file source."),
    Source("self_renormalization.apply.inputs.zR", physics="The fitted factor is one prior job or external file source."),
    Provides("self_renormalization.apply", "hybrid", "self_renormalization.scheme", physics="Hybrid self-renormalization application also consumes a short-distance denominator."),
    Depends("self_renormalization.apply.hybrid.inputs", "denominator", physics="Hybrid self-renormalization consumes one short-distance denominator source."),
    Source("self_renormalization.apply.hybrid.inputs.denominator", physics="The hybrid denominator is one prior job or external file source."),
)
# fmt: on
# ruff: enable[E501]


def check_path(context: CheckContext) -> Issue | None:
    physics = "Only the declared renormalization operation/scheme/strategy combinations have one numerical path."
    params = effective_params(context.params)
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
    params = effective_params(context.params)
    inputs = context.inputs
    physics = "Hybrid application requires a switching distance and its mass correction."
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
    physics = "The explicit job type, strategy, and scheme determine one exact set of input roles."
    params = effective_params(context.params)
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


def check_kernel(context: CheckContext) -> Issue | None:
    params = effective_params(context.params)
    if params.get("strategy") != "self_renormalization":
        return None
    kernel_id = params.get("kernel_id")
    if not isinstance(kernel_id, str):
        return None
    physics = "Self-renormalization formulas are explicit callables selected by filename stem."
    try:
        kernel = load_renormalization_kernel(kernel_id)
        signature = inspect.signature(kernel)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return Issue("kernel_id", str(exc), physics)
    parameters = list(signature.parameters.values())
    mu = signature.parameters.get("mu")
    if (
        not parameters
        or parameters[0].name != "z_fm"
        or parameters[0].kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or mu is None
        or mu.kind is inspect.Parameter.POSITIONAL_ONLY
        or any(
            parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in parameters
        )
    ):
        return Issue("kernel_id", "kernel must explicitly accept positional z_fm and keyword mu", physics)
    required = [
        parameter.name
        for parameter in parameters[1:]
        if parameter.default is inspect.Parameter.empty and parameter.name != "mu"
    ]
    if required:
        return Issue("kernel_id", f"kernel has unsupported required parameters: {required}", physics)
    return None


def _check_scale(context: CheckContext) -> Issue | None:
    params = effective_params(context.params)
    scheme = params.get("scheme")
    strategy = params.get("strategy")
    is_fit = params.get("type") == "fit"
    requires = is_fit or strategy == "self_renormalization" or scheme == "msbar"
    if requires and "mu" not in params:
        return Issue(
            "mu",
            "is required for this renormalization path",
            "Scale-dependent renormalization paths require a positive scale; ratio external division does not.",
        )
    if (is_fit or strategy == "self_renormalization") and "LambdaQCD_gev" not in params:
        return Issue(
            "LambdaQCD_gev",
            "is required for self-renormalization fitting and application",
            "The reusable zR fit and its target remap must use one shared QCD scale.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES, JOB_CONDITIONAL_RULES)

CHECKS = (check_path, check_hybrid, check_inputs, check_kernel, _check_scale)
