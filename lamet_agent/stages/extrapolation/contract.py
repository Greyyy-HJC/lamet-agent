"""Manifest contract for extrapolation."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, List, Value


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _unit_interval(value: int | float) -> bool:
    return _finite(value) and 0 <= value <= 1


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


def _pair(value: list[object]) -> bool:
    return len(value) == 2


def _valid_priors(value: dict[object, object]) -> bool:
    return all(
        isinstance(prior, dict)
        and set(prior) == {"mean", "sdev"}
        and isinstance(prior["mean"], (int, float))
        and not isinstance(prior["mean"], bool)
        and math.isfinite(prior["mean"])
        and isinstance(prior["sdev"], (int, float))
        and not isinstance(prior["sdev"], bool)
        and math.isfinite(prior["sdev"])
        and prior["sdev"] > 0
        for prior in value.values()
    )


def _boolean_values(value: dict[object, object]) -> bool:
    return all(isinstance(item, bool) for item in value.values())


PARAM_RULES = (
    Depends("", "required_terms", physics="Required correction terms are always included."),
    Depends("", "allowed_terms", physics="The agent may select only authored additional terms."),
    Depends("", "max_terms", physics="Candidate complexity has an explicit upper bound."),
    Depends("", "fit_ranges", physics="Ensemble, momentum, and x ranges are explicit."),
    Depends("", "priors", physics="Correction-coefficient prior policy is explicit."),
    Depends("", "x_dependence", physics="Each correction coefficient declares whether it varies with x."),
    Depends("", "physical_pion_mass_gev", physics="The published point has an authored physical pion mass."),
    Depends("", "model_selection", physics="Candidate comparison criterion is explicit."),
    List("required_terms", "required", physics="Required terms are a list."),
    List("allowed_terms", "allowed", physics="Allowed terms are a list."),
    Value("required_terms.required", Literal["a", "a2", "a4", "ap2", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Required correction term ids are supported basis terms."),
    Value("allowed_terms.allowed", Literal["a", "a2", "a4", "ap2", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Allowed correction term ids are supported basis terms."),
    Value("max_terms", int, physics="Maximum term count is positive.", validator=_positive),
    Value("priors", dict, physics="Priors map term ids to finite mean and positive sdev values.", validator=_valid_priors),
    Value("x_dependence", dict, physics="Coefficient x-dependence maps term ids to booleans.", validator=_boolean_values),
    Value("physical_pion_mass_gev", (int, float), physics="Physical pion mass is finite and positive.", validator=_positive),
    Depends("fit_ranges", "x", physics="The x fit range is explicit."),
    Depends("fit_ranges", "lattice_spacing_fm", physics="The lattice-spacing fit range is explicit."),
    Depends("fit_ranges", "momentum_gev", physics="The momentum fit range is explicit."),
    Depends("fit_ranges", "allowed_exclusions", physics="Allowed ensemble exclusions are explicit."),
    List("fit_ranges.x", "item", physics="The x range is a two-value numeric list.", validator=_pair),
    Value("fit_ranges.x.item", (int, float), physics="Every x range endpoint is finite.", validator=_finite),
    List("fit_ranges.lattice_spacing_fm", "item", physics="The lattice-spacing range is a two-value numeric list.", validator=_pair),
    Value("fit_ranges.lattice_spacing_fm.item", (int, float), physics="Every lattice-spacing endpoint is finite.", validator=_finite),
    List("fit_ranges.momentum_gev", "item", physics="The momentum range is a two-value numeric list.", validator=_pair),
    Value("fit_ranges.momentum_gev.item", (int, float), physics="Every momentum endpoint is finite.", validator=_finite),
    List("fit_ranges.allowed_exclusions", "item", physics="Allowed exclusions are a list."),
    Value("fit_ranges.allowed_exclusions.item", str, physics="Allowed exclusions are ensemble ids."),
    Depends("model_selection", "min_Q", physics="The minimum fit quality is explicit."),
    Depends("model_selection", "stability_sigma", physics="The stability threshold is explicit."),
    Value("model_selection.min_Q", (int, float), physics="Minimum Q lies in [0,1].", validator=_unit_interval),
    Value("model_selection.stability_sigma", (int, float), physics="Stability sigma is finite and positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "distributions", physics="Extrapolation consumes a nonempty ordered list of matched distributions."),
    List("distributions", "distribution", physics="Distributions are nonempty and preserve authored order.", validator=_nonempty),
    Value("distributions.distribution", dict, physics="Each distribution is an input source."),
)


def check_extrapolation_relations(context: CheckContext) -> Issue | None:
    physics = "Extrapolation terms, ranges, priors, and model policy form one closed authored fit contract."
    required = context.params.get("required_terms")
    allowed = context.params.get("allowed_terms")
    maximum = context.params.get("max_terms")
    if len(set(required)) != len(required) or len(set(allowed)) != len(allowed):
        return Issue("params.required_terms", "required_terms and allowed_terms must not contain duplicates", physics)
    if set(required) & set(allowed):
        return Issue("params.allowed_terms", "must be disjoint from required_terms", physics)
    if maximum < len(required) or maximum > len(set(required) | set(allowed)):
        return Issue("params.max_terms", "must lie between required term count and total authored term count", physics)
    ranges = context.params["fit_ranges"]
    for name in ("x", "lattice_spacing_fm", "momentum_gev"):
        if ranges[name][0] >= ranges[name][1]:
            return Issue(f"params.fit_ranges.{name}", f"fit_ranges.{name}.min must be smaller than max", physics)
    priors = context.params.get("priors")
    expected = set(required) | set(allowed)
    if set(priors) != expected:
        return Issue("params.priors", "must contain exactly one entry for every required or allowed term", physics)
    x_dependence = context.params.get("x_dependence")
    if set(x_dependence) != expected:
        return Issue("params.x_dependence", "must contain exactly one boolean for every required or allowed term", physics)
    return None


CHECKS = (check_extrapolation_relations,)
