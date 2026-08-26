"""Manifest contract for extrapolation."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, List, Provides, Recommends, Source, Value, stage_job_rules


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
    return (
        set(value) == {"mean", "sdev"}
        and isinstance(value["mean"], (int, float))
        and not isinstance(value["mean"], bool)
        and math.isfinite(value["mean"])
        and isinstance(value["sdev"], (int, float))
        and not isinstance(value["sdev"], bool)
        and math.isfinite(value["sdev"])
        and value["sdev"] > 0
    )


def _boolean_values(value: dict[object, object]) -> bool:
    return all(isinstance(item, bool) for item in value.values())


def _valid_systematics_groups(value: dict[object, object]) -> bool:
    return (
        set(value)
        == {"main", "zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations"}
        and isinstance(value["main"], int)
        and not isinstance(value["main"], bool)
        and all(
            isinstance(value[key], list)
            and all(isinstance(index, int) and not isinstance(index, bool) for index in value[key])
            for key in ("zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations")
        )
    )


PARAM_RULES = (
    Recommends("", "operation", physics="Authored extrapolation jobs fit; generated budget jobs override this operation.", default="fit"),
    Value("operation", Literal["fit", "systematics_budget"], physics="The extrapolation operation is controlled."),
    Provides("", "fit", "operation", physics="Fit jobs own continuum-model and diagnostic parameters."),
    Provides("", "systematics_budget", "operation", physics="Budget jobs own grouping and combination parameters."),
    Depends("fit", "required_terms", physics="Required correction terms are always included."),
    Recommends("fit", "allowed_terms", physics="The reference examples add no optional terms beyond their exact authored model.", default=[]),
    Recommends("fit", "max_terms", physics="The reference central model contains four correction terms.", default=4),
    Recommends("fit", "priors", physics="All initial linear coefficients share the reference zero-centered width-three prior unless explicitly overridden.", default={"mean": 0.0, "sdev": 3.0}),
    Depends("fit", "x_dependence", physics="Each correction coefficient declares whether it varies with x."),
    Depends("fit", "pdep_gev", physics="Requested finite momenta are used only for the post-fit momentum-dependence diagnostic."),
    Depends("fit", "physical_pion_mass_gev", physics="Mass-dependent extensions require an explicit physical pion mass.", required=False),
    Depends("fit", "posterior_prior_error_scale", physics="Per-resample fits use an explicitly authored widening of the sample-average posterior."),
    List("fit.required_terms", "required", physics="Required terms are a list."),
    List("fit.allowed_terms", "allowed", physics="Allowed terms are a list."),
    Value("fit.required_terms.required", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Required correction term ids are supported basis terms."),
    Value("fit.allowed_terms.allowed", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Allowed correction term ids are supported basis terms."),
    Value("fit.max_terms", int, physics="Maximum term count is positive.", validator=_positive),
    Value("fit.priors", dict, physics="The shared initial prior has one finite mean and positive sdev.", validator=_valid_priors),
    Value("fit.x_dependence", dict, physics="Coefficient x-dependence maps term ids to booleans.", validator=_boolean_values),
    List("fit.pdep_gev", "momentum", physics="Momentum-dependence diagnostics use a nonempty authored list.", validator=_nonempty),
    Value("fit.pdep_gev.momentum", (int, float), physics="Every diagnostic momentum is finite and positive.", validator=_positive),
    Value("fit.physical_pion_mass_gev", (int, float), physics="Physical pion mass is finite and positive.", validator=_positive),
    Value("fit.posterior_prior_error_scale", (int, float), physics="Posterior-prior widening is finite and positive.", validator=_positive),
    Recommends("systematics_budget", "systematics_prescription", physics="The reference budget uses variant envelopes with independent components combined in quadrature.", default="variant_envelope_quadrature"),
    Depends("systematics_budget", "systematics_groups", physics="Budget jobs map ordered distribution inputs to uncertainty components."),
    Value("systematics_budget.systematics_prescription", Literal["variant_envelope_quadrature"], physics="The supported budget prescription is the reference envelope-plus-quadrature rule."),
    Value("systematics_budget.systematics_groups", dict, physics="Systematics groups use valid ordered input indices.", validator=_valid_systematics_groups),
)

INPUT_RULES = (
    Depends("", "distributions", physics="Extrapolation consumes a nonempty ordered list of matched distributions."),
    List("distributions", "distribution", physics="Distributions are nonempty and preserve authored order.", validator=_nonempty),
    Source("distributions.distribution", physics="Each distribution is a prior job or external file source."),
)


def check_extrapolation_relations(context: CheckContext) -> Issue | None:
    physics = "Extrapolation terms, ranges, priors, and model policy form one closed authored fit contract."
    if context.params["operation"] == "systematics_budget":
        groups = context.params["systematics_budget"]["systematics_groups"]
        count = len(context.inputs.get("distributions", []))
        indices = [groups["main"], *groups["zs"], *groups["lambda_extrapolation"], *groups["lamet_scale"], *groups["other_extrapolations"]]
        if not indices or min(indices) < 0 or max(indices) >= count:
            return Issue("systematics_groups", "contains an index outside the ordered distributions input", physics)
        if len(indices) != len(set(indices)) or set(indices) != set(range(count)):
            return Issue("systematics_groups", "must assign every ordered distribution input to exactly one budget role", physics)
        return None
    fit = context.params["fit"]
    required = fit.get("required_terms")
    allowed = fit.get("allowed_terms")
    maximum = fit.get("max_terms")
    if len(set(required)) != len(required) or len(set(allowed)) != len(allowed):
        return Issue("required_terms", "required_terms and allowed_terms must not contain duplicates", physics)
    if set(required) & set(allowed):
        return Issue("allowed_terms", "must be disjoint from required_terms", physics)
    if maximum < len(required) or maximum > len(set(required) | set(allowed)):
        return Issue("max_terms", "must lie between required term count and total authored term count", physics)
    expected = set(required) | set(allowed)
    x_dependence = fit.get("x_dependence")
    if set(x_dependence) != expected:
        return Issue("x_dependence", "must contain exactly one boolean for every required or allowed term", physics)
    pdep = fit.get("pdep_gev")
    if len(set(float(value) for value in pdep)) != len(pdep):
        return Issue("pdep_gev", "must contain unique momenta", "Each requested diagnostic curve needs one distinct physical momentum.")
    mass_terms = {"mpi2", "mpi4_log_mpi2"}
    if expected & mass_terms and "physical_pion_mass_gev" not in fit:
        return Issue("physical_pion_mass_gev", "is required by pion-mass correction terms", physics)
    if not expected & mass_terms and "physical_pion_mass_gev" in fit:
        return Issue("physical_pion_mass_gev", "must be omitted when no pion-mass term is selected", physics)
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_extrapolation_relations,)
