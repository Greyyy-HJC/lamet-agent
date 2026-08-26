"""Manifest contract for renormalization."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Provides, Recommends, Source, Value, stage_job_rules
from lamet_agent.stages.renormalization.parameters import effective_params


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _positive_int(value: int) -> bool:
    return value > 0


def _nonnegative_finite(value: int | float) -> bool:
    return math.isfinite(value) and value >= 0


def _valid_denominator(value: object) -> bool:
    return (
        isinstance(value, (str, dict))
        or isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value != 0
    )


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
    Value("self_renormalization.scheme", Literal["ratio", "hybrid", "msbar"], physics="Self-renormalization scheme is ratio, hybrid, or msbar."),
    Depends("self_renormalization", "mu", physics="Self-renormalization uses an explicit perturbative scale."),
    Depends("self_renormalization", "LambdaQCD_gev", physics="Self-renormalization fitting and remapping share one explicitly authored QCD scale."),
    Recommends("self_renormalization", "svdcut", physics="The original self-renormalization reference fit defaults to a 1e-12 covariance singular-value cut.", default=1e-12),
    Depends("self_renormalization", "d", physics="Self-renormalization may remap one flat finite logarithmic coefficient.", required=False),
    Depends("self_renormalization", "m0_gev", physics="Self-renormalization application may remap the fitted residual mass.", required=False),
    Value("self_renormalization.mu", (int, float), physics="The scale is finite and positive.", validator=_positive),
    Value("self_renormalization.LambdaQCD_gev", (int, float), physics="Lambda_QCD is finite and positive.", validator=_positive),
    Value("self_renormalization.svdcut", (int, float), physics="The self-renormalization covariance cutoff is finite and positive.", validator=_positive),
    Value("self_renormalization.d", (int, float), physics="The finite logarithmic correction is finite.", validator=_finite),
    Value("self_renormalization.m0_gev", (int, float), physics="The remapped residual mass is finite.", validator=_finite),
    Provides("self_renormalization", "hybrid", "self_renormalization.scheme", physics="Hybrid self-renormalization owns its switching distance."),
    Depends("self_renormalization.hybrid", "zs_fm", physics="Hybrid switching distance is explicit."),
    Value("self_renormalization.hybrid.zs_fm", (int, float), physics="Hybrid switching distance is finite and positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "target", physics="Application needs a target matrix element.", required=False),
    Depends("", "denominator", physics="External application may use a coordinate-dependent denominator.", required=False),
    Depends("", "zR", physics="Self-renormalization application consumes the fitted zR factor.", required=False),
    Depends("", "reference", physics="Self-renormalization fitting may use an explicit reference.", required=False),
    Source("target", physics="Target is one prior job or external file source."),
    Source("denominator", physics="Denominator is one source or finite nonzero constant.", allow_constant=True),
    Value("denominator", (str, dict, int, float), physics="A constant denominator is finite and nonzero.", validator=_valid_denominator),
    Source("zR", physics="zR is one prior job or external file source."),
    Source("reference", physics="Reference is one source or a nonempty source list.", allow_list=True),
)
# fmt: on
# ruff: enable[E501]


def check_path(context: CheckContext) -> Issue | None:
    physics = "Only the declared renormalization operation/scheme/strategy combinations have one numerical path."
    params = effective_params(context.params)
    strategy = params.get("strategy")
    is_fit = set(context.inputs) == {"reference"}
    if is_fit and strategy != "self_renormalization":
        return Issue("strategy", "a reference-only fit requires self_renormalization", physics)
    if is_fit and "d" not in params:
        return Issue("d", "is required by a reference-only self-renormalization fit", physics)
    if is_fit and "m0_gev" in params:
        return Issue("m0_gev", "is fitted from the reference and must be omitted", physics)
    if not is_fit and "target" not in context.inputs:
        return Issue("inputs.target", "is required by a renormalization apply job", physics)
    return None


def check_hybrid(context: CheckContext) -> Issue | None:
    params = effective_params(context.params)
    physics = "Hybrid application requires a switching distance and its mass correction."
    if params.get("scheme") == "hybrid":
        missing = ["zs_fm"] if "zs_fm" not in params else []
        if missing:
            return Issue("zs_fm", "is required by scheme='hybrid'", physics)
        if params.get("strategy") == "external_denominator" and "denominator" not in context.inputs:
            return Issue("inputs.denominator", "is required for hybrid external renormalization", physics)
        if params.get("strategy") == "external_denominator":
            missing = [key for key in ("m0_gev", "delta_m_gev") if key not in params]
            if missing:
                return Issue(f"params.{missing[0]}", "is required for hybrid external renormalization", physics)
        elif any(key in params for key in ("m0_gev", "delta_m_gev")):
            return Issue("m0_gev", "m0_gev and delta_m_gev are external-denominator hybrid parameters", physics)
    return None


def check_inputs(context: CheckContext) -> Issue | None:
    physics = "External ratio/MSbar consumes one denominator; self-renormalization consumes a fitted factor."
    if set(context.inputs) == {"reference"}:
        return None
    params = effective_params(context.params)
    scheme = params.get("scheme")
    strategy = params.get("strategy")
    if strategy == "external_denominator":
        if scheme in {"ratio", "msbar"}:
            if "denominator" not in context.inputs:
                return Issue("inputs.denominator", "is required for ratio/MSbar external renormalization", physics)
        if scheme == "hybrid" and not isinstance(context.inputs.get("denominator"), (str, dict)):
            return Issue("inputs.denominator", "hybrid requires a coordinate-dependent source", physics)
        if "zR" in context.inputs or "reference" in context.inputs:
            role = "zR" if "zR" in context.inputs else "reference"
            return Issue(f"inputs.{role}", "must be omitted for external renormalization", physics)
        if "zs_fm" in params and scheme != "hybrid":
            return Issue("zs_fm", "is valid only for hybrid application", physics)
        if any(key in params for key in ("m0_gev", "delta_m_gev")) and scheme != "hybrid":
            return Issue("m0_gev", "m0_gev and delta_m_gev are valid only for hybrid application", physics)
    else:
        if "zR" not in context.inputs:
            return Issue("inputs.zR", "is required for self-renormalization application", physics)
        if scheme == "hybrid" and "denominator" not in context.inputs:
            return Issue("inputs.denominator", "is required for hybrid self-renormalization", physics)
        if scheme in {"ratio", "msbar"} and "denominator" in context.inputs:
            return Issue("inputs.denominator", "must be omitted for ratio/MSbar self-renormalization", physics)
    return None


def _check_scale(context: CheckContext) -> Issue | None:
    params = effective_params(context.params)
    scheme = params.get("scheme")
    strategy = params.get("strategy")
    is_fit = set(context.inputs) == {"reference"}
    requires = is_fit or strategy == "self_renormalization" or scheme == "msbar"
    if requires and "mu" not in params:
        return Issue(
            "mu",
            "is required for this renormalization path",
            "Scale-dependent renormalization paths require a positive scale; ratio external division does not.",
        )
    if (is_fit or strategy == "self_renormalization") and "LambdaQCD_gev" not in params:
        return Issue(
            "params.LambdaQCD_gev",
            "is required for self-renormalization fitting and application",
            "The reusable zR fit and its target remap must use one shared QCD scale.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_path, check_hybrid, check_inputs, _check_scale)
