"""Manifest contract for renormalization."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Value


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _positive_int(value: int) -> bool:
    return value > 0


def _nonnegative_finite(value: int | float) -> bool:
    return math.isfinite(value) and value >= 0


def _valid_denominator(value: object) -> bool:
    return isinstance(value, dict) or math.isfinite(value) and value != 0


PARAM_RULES = (
    Depends("", "operation", physics="The job either fits a reusable factor or applies one."),
    Depends("", "scheme", physics="Renormalization scheme is a physical convention."),
    Depends("", "strategy", physics="Strategy selects external or self-renormalization data flow."),
    Depends("", "normalization", physics="Origin normalization is explicit."),
    Depends("", "mu", physics="The perturbative scale is required by scale-dependent operations.", required=False),
    Depends("", "zs_fm", physics="Hybrid switching distance is a physical job parameter.", required=False),
    Depends("", "m0_gev", physics="Hybrid long-distance mass correction is explicit.", required=False),
    Depends("", "delta_m_gev", physics="Hybrid mass difference is explicit.", required=False),
    Depends("", "self_renormalization", physics="Fit-factor formula and spacing choices are explicit.", required=False),
    Depends("", "factor_remap", physics="Applying a reusable self-renormalization factor may change operator finite terms.", required=False),
    Depends("self_renormalization", "k", physics="The logarithmic divergence coefficient is explicit."),
    Depends("self_renormalization", "LambdaQCD_gev", physics="The QCD scale in the fit formula is explicit."),
    Depends("self_renormalization", "d", physics="The finite logarithmic correction is explicit."),
    Depends("self_renormalization", "n_f", physics="The active-flavor count is explicit."),
    Depends("self_renormalization", "zms_model", physics="The short-distance matching model is explicit."),
    Depends("self_renormalization", "lattice_spacing_range_fm", physics="The lattice-spacing fit range is explicit."),
    Depends("self_renormalization", "short_distance_range_fm", physics="The short-distance fit range is explicit."),
    Depends("self_renormalization", "reference_coord_unit", physics="Legacy-free reference coordinates declare their physical unit explicitly."),
    Depends("self_renormalization.lattice_spacing_range_fm", "min", physics="The lower lattice-spacing fit bound is explicit."),
    Depends("self_renormalization.lattice_spacing_range_fm", "max", physics="The upper lattice-spacing fit bound is explicit."),
    Depends("self_renormalization.short_distance_range_fm", "min", physics="The lower short-distance fit bound is explicit."),
    Depends("self_renormalization.short_distance_range_fm", "max", physics="The upper short-distance fit bound is explicit."),
    Depends("factor_remap", "d", physics="The target operator finite logarithmic correction is explicit."),
    Depends("factor_remap", "m0_gev", physics="The target operator finite mass slope is explicit."),
    Depends("factor_remap", "LambdaQCD_gev", physics="The target operator remap uses an explicit QCD scale."),
    Depends("factor_remap", "zms_model", physics="The final coordinate-space conversion matches the target observable."),
    Value("operation", Literal["fit_factor", "apply"], physics="Operation is fit_factor or apply."),
    Value("scheme", Literal["ratio", "hybrid", "msbar"], physics="Scheme is ratio, hybrid, or msbar."),
    Value("strategy", Literal["external_denominator", "self_renormalization"], physics="Strategy is external_denominator or self_renormalization."),
    Value("normalization", bool, physics="Origin normalization is boolean."),
    Value("mu", (int, float), physics="The scale is finite and positive.", validator=_positive),
    Value("zs_fm", (int, float), physics="Hybrid switching distance is finite and positive.", validator=_positive),
    Value("m0_gev", (int, float), physics="The hybrid mass parameter is finite.", validator=_finite),
    Value("delta_m_gev", (int, float), physics="The hybrid mass difference is finite.", validator=_finite),
    Value("self_renormalization.k", (int, float), physics="The logarithmic divergence coefficient is finite.", validator=_finite),
    Value("self_renormalization.LambdaQCD_gev", (int, float), physics="Lambda_QCD is finite and positive.", validator=_positive),
    Value("self_renormalization.d", (int, float), physics="The finite correction is finite.", validator=_finite),
    Value("self_renormalization.n_f", int, physics="The active-flavor count is positive.", validator=_positive_int),
    Value("self_renormalization.zms_model", Literal["pdf_nlo"], physics="The initial short-distance model is pdf_nlo."),
    Value("self_renormalization.lattice_spacing_range_fm.min", (int, float), physics="The lower lattice-spacing bound is finite and positive.", validator=_positive),
    Value("self_renormalization.lattice_spacing_range_fm.max", (int, float), physics="The upper lattice-spacing bound is finite and positive.", validator=_positive),
    Value("self_renormalization.short_distance_range_fm.min", (int, float), physics="The lower short-distance bound is finite and nonnegative.", validator=_nonnegative_finite),
    Value("self_renormalization.short_distance_range_fm.max", (int, float), physics="The upper short-distance bound is finite and positive.", validator=_positive),
    Value("self_renormalization.reference_coord_unit", Literal["fm"], physics="Self-renormalization reference z coordinates are in fm."),
    Value("factor_remap.d", (int, float), physics="The target finite correction is finite.", validator=_finite),
    Value("factor_remap.m0_gev", (int, float), physics="The target mass slope is finite.", validator=_finite),
    Value("factor_remap.LambdaQCD_gev", (int, float), physics="Lambda_QCD is finite and positive.", validator=_positive),
    Value("factor_remap.zms_model", Literal["pdf_nlo", "da_nlo"], physics="The final conversion is PDF or DA NLO."),
)

INPUT_RULES = (
    Depends("", "target", physics="Application needs a target matrix element.", required=False),
    Depends("", "denominator", physics="External application may use a coordinate-dependent denominator.", required=False),
    Depends("", "factor", physics="Self-renormalization application consumes a fitted factor.", required=False),
    Depends("", "reference", physics="Self-renormalization fitting may use an explicit reference.", required=False),
    Value("target", dict, physics="Target is exactly one input source."),
    Value("denominator", (dict, int, float), physics="Denominator is one source or finite nonzero constant.", validator=_valid_denominator),
    Value("factor", dict, physics="Factor is exactly one input source."),
    Value("reference", (dict, list), physics="Reference is an input source."),
)


def check_operation(context: CheckContext) -> Issue | None:
    physics = "Only the declared renormalization operation/scheme/strategy combinations have one numerical path."
    operation = context.params.get("operation")
    scheme = context.params.get("scheme")
    strategy = context.params.get("strategy")
    if operation == "fit_factor" and (scheme != "msbar" or strategy != "self_renormalization"):
        return Issue("params.operation", "fit_factor requires scheme='msbar' and strategy='self_renormalization'", physics)
    if operation == "apply" and "target" not in context.inputs:
        return Issue("inputs.target", "is required by operation='apply'", physics)
    if operation == "fit_factor" and "reference" not in context.inputs:
        return Issue("inputs.reference", "is required by operation='fit_factor'", physics)
    if operation == "fit_factor" and "target" in context.inputs:
        return Issue("inputs.target", "must be omitted for operation='fit_factor'", physics)
    if operation == "fit_factor" and "self_renormalization" not in context.params:
        return Issue("params.self_renormalization", "is required by operation='fit_factor'", physics)
    if operation == "apply" and "self_renormalization" in context.params:
        return Issue("params.self_renormalization", "is valid only for operation='fit_factor'", physics)
    if operation == "fit_factor" and "factor_remap" in context.params:
        return Issue("params.factor_remap", "is valid only for operation='apply'", physics)
    if operation == "apply" and strategy == "self_renormalization" and "factor_remap" not in context.params:
        return Issue("params.factor_remap", "is required for self-renormalization application", physics)
    if operation == "apply" and strategy != "self_renormalization" and "factor_remap" in context.params:
        return Issue("params.factor_remap", "is valid only for self-renormalization application", physics)
    return None


def check_hybrid(context: CheckContext) -> Issue | None:
    physics = "Hybrid application requires a switching distance and its mass correction."
    if context.params.get("scheme") == "hybrid":
        missing = ["zs_fm"] if "zs_fm" not in context.params else []
        if missing:
            return Issue("params.zs_fm", "is required by scheme='hybrid'", physics)
        if context.params.get("strategy") == "external_denominator" and "denominator" not in context.inputs:
            return Issue("inputs.denominator", "is required for hybrid external renormalization", physics)
        if context.params.get("strategy") == "external_denominator":
            missing = [key for key in ("m0_gev", "delta_m_gev") if key not in context.params]
            if missing:
                return Issue(f"params.{missing[0]}", "is required for hybrid external renormalization", physics)
        elif any(key in context.params for key in ("m0_gev", "delta_m_gev")):
            return Issue("params.m0_gev", "m0_gev and delta_m_gev are external-denominator hybrid parameters", physics)
    return None


def check_inputs(context: CheckContext) -> Issue | None:
    physics = "External ratio/MSbar consumes one denominator; self-renormalization consumes a fitted factor."
    if context.params.get("operation") != "apply":
        return None
    scheme = context.params.get("scheme")
    strategy = context.params.get("strategy")
    if strategy == "external_denominator":
        if scheme in {"ratio", "msbar"}:
            if "denominator" not in context.inputs:
                return Issue("inputs.denominator", "is required for ratio/MSbar external renormalization", physics)
        if scheme == "hybrid" and not isinstance(context.inputs.get("denominator"), dict):
            return Issue("inputs.denominator", "hybrid requires a coordinate-dependent source", physics)
        if "factor" in context.inputs or "reference" in context.inputs:
            role = "factor" if "factor" in context.inputs else "reference"
            return Issue(f"inputs.{role}", "must be omitted for external renormalization", physics)
        if "zs_fm" in context.params and scheme != "hybrid":
            return Issue("params.zs_fm", "is valid only for hybrid application", physics)
        if any(key in context.params for key in ("m0_gev", "delta_m_gev")) and scheme != "hybrid":
            return Issue("params.m0_gev", "m0_gev and delta_m_gev are valid only for hybrid application", physics)
    else:
        if "factor" not in context.inputs:
            return Issue("inputs.factor", "is required for self-renormalization application", physics)
        if scheme == "hybrid" and "denominator" not in context.inputs:
            return Issue("inputs.denominator", "is required for hybrid self-renormalization", physics)
        if scheme in {"ratio", "msbar"} and "denominator" in context.inputs:
            return Issue("inputs.denominator", "must be omitted for ratio/MSbar self-renormalization", physics)
    return None


def _check_scale(context: CheckContext) -> Issue | None:
    operation = context.params.get("operation")
    scheme = context.params.get("scheme")
    strategy = context.params.get("strategy")
    requires = operation == "fit_factor" or strategy == "self_renormalization" or scheme in {"msbar", "hybrid"}
    if requires and "mu" not in context.params:
        return Issue("params.mu", "is required for this renormalization path", "Scale-dependent renormalization paths require a positive scale; ratio external division does not.")
    return None


def check_fit_ranges(context: CheckContext) -> Issue | None:
    settings = context.params.get("self_renormalization")
    if settings is None:
        return None
    for name in ("lattice_spacing_range_fm", "short_distance_range_fm"):
        value = settings[name]
        if value["min"] >= value["max"]:
            return Issue(f"params.self_renormalization.{name}.min", f"must be smaller than {name}.max", "The two self-renormalization fit ranges must be ordered.")
    return None


CHECKS = (check_operation, check_hybrid, check_inputs, _check_scale, check_fit_ranges)
