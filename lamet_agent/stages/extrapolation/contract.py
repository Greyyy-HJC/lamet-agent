"""Manifest contract for extrapolation."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import (
    CheckContext,
    Depends,
    Issue,
    List,
    Provides,
    Recommends,
    Source,
    Value,
    stage_job_rules,
)


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


def _valid_systematics_groups(value: dict[object, object]) -> bool:
    return (
        set(value) == {"main", "zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations"}
        and isinstance(value["main"], int)
        and not isinstance(value["main"], bool)
        and all(
            isinstance(value[key], list)
            and all(isinstance(index, int) and not isinstance(index, bool) for index in value[key])
            for key in ("zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations")
        )
    )


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Recommends("", "operation", physics="Authored extrapolation jobs fit; generated budget jobs override this operation.", default="fit"),
    Value("operation", Literal["fit", "systematics_budget"], physics="The extrapolation operation is controlled."),
    Provides("", "fit", "operation", physics="Fit jobs own continuum-model and diagnostic parameters."),
    Provides("", "systematics_budget", "operation", physics="Budget jobs own grouping and combination parameters."),
    Recommends("fit", "x_independent_terms", physics="These correction coefficients are shared across the complete x grid.", default=[]),
    Recommends("fit", "x_dependent_terms", physics="These correction coefficients are fitted independently at every x.", default=[]),
    Recommends("fit", "priors", physics="All initial linear coefficients share the reference zero-centered width-three prior unless explicitly overridden.", default={"mean": 0.0, "sdev": 3.0}),
    Recommends("fit", "x_covariance", physics="Cross-x covariance may be retained within each ensemble source.", default=False),
    Depends("fit", "pdep_gev", physics="Requested finite momenta are used only for the post-fit momentum-dependence diagnostic."),
    Recommends("fit", "physical_pion_mass_gev", physics="Pion-mass terms use the shared isospin-limit physical point.", default=0.135),
    Depends("fit", "posterior_prior_error_scale", physics="Per-resample fits use an explicitly authored widening of the sample-average posterior."),
    List("fit.x_independent_terms", "term", physics="Shared correction terms preserve authored order."),
    List("fit.x_dependent_terms", "term", physics="x-dependent correction terms preserve authored order."),
    Value("fit.x_independent_terms.term", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Shared correction term ids are supported basis terms."),
    Value("fit.x_dependent_terms.term", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="x-dependent correction term ids are supported basis terms."),
    Value("fit.priors", dict, physics="The shared initial prior has one finite mean and positive sdev.", validator=_valid_priors),
    Value("fit.x_covariance", bool, physics="Cross-x covariance is retained only when explicitly enabled."),
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
# fmt: on
# ruff: enable[E501]


def check_extrapolation_relations(context: CheckContext) -> Issue | None:
    physics = "Extrapolation terms, ranges, priors, and model policy form one closed authored fit contract."
    if context.params["operation"] == "systematics_budget":
        groups = context.params["systematics_groups"]
        count = len(context.inputs.get("distributions", []))
        indices = [
            groups["main"],
            *groups["zs"],
            *groups["lambda_extrapolation"],
            *groups["lamet_scale"],
            *groups["other_extrapolations"],
        ]
        if not indices or min(indices) < 0 or max(indices) >= count:
            return Issue("systematics_groups", "contains an index outside the ordered distributions input", physics)
        if len(indices) != len(set(indices)) or set(indices) != set(range(count)):
            return Issue(
                "systematics_groups", "must assign every ordered distribution input to exactly one budget role", physics
            )
        return None
    fit = context.params
    independent = fit.get("x_independent_terms")
    dependent = fit.get("x_dependent_terms")
    if len(set(independent)) != len(independent) or len(set(dependent)) != len(dependent):
        return Issue("x_independent_terms", "term lists must not contain duplicates", physics)
    if set(independent) & set(dependent):
        return Issue("x_dependent_terms", "must be disjoint from x_independent_terms", physics)
    if not independent and not dependent:
        return Issue("x_dependent_terms", "at least one extrapolation term is required", physics)
    pdep = fit.get("pdep_gev")
    if len(set(float(value) for value in pdep)) != len(pdep):
        return Issue(
            "pdep_gev",
            "must contain unique momenta",
            "Each requested diagnostic curve needs one distinct physical momentum.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_extrapolation_relations,)
