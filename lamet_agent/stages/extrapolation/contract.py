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
    Suggests,
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


_EXTRAPOLATION_TERMS = frozenset(
    {"a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"}
)


def _safe_systematics_id(value: str) -> bool:
    import re

    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))


def _systematics_terms(value: list[object]) -> bool:
    return len(value) == len(set(value)) and all(
        isinstance(term, str) and term in _EXTRAPOLATION_TERMS for term in value
    )


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Recommends("", "operation", physics="Selects either a fit whose intercept h0(x) is evaluated where the authored correction bases vanish, representing only the physical limits encoded by those bases, or an uncertainty budget assembled from completed central and variant distributions.", default="fit"),
    Value("operation", Literal["fit", "systematics_budget"], physics="fit applies the additive scaling ansatz to resampled matched distributions; systematics_budget performs no refit and publishes statistical, component, total-systematic, and combined errors."),
    Provides("", "fit", "operation", physics="For operation=fit, enables basis terms, x dependence, covariance, priors, physical pion mass, finite-P diagnostics, and posterior-informed resample controls used to determine h0(x)."),
    Provides("", "systematics_budget", "operation", physics="For operation=systematics_budget, enables the input-index grouping and envelope rule that convert a main result and authored variants into pointwise uncertainties."),
    Recommends("fit", "x_independent_terms", physics="Bases B_t entering h=h0(x)+sum_t c_t B_t with one coefficient c_t shared across all x; h0 remains pointwise, so only the magnitude of each named correction is assumed x independent.", default=[]),
    Recommends("fit", "x_dependent_terms", physics="Bases entering h=h0(x)+sum_t c_t(x) B_t with an independent coefficient at every x, allowing cutoff, momentum, pion-mass, or finite-volume corrections to distort the distribution shape.", default=[]),
    Recommends("fit", "priors", physics="Common numerical Gaussian mean and width for h0(x) and every correction coefficient in each parameter's native units during the sample-average fit; N(0,3^2) regularizes and can materially affect weakly constrained directions depending on basis scale.", default={"mean": 0.0, "sdev": 3.0}),
    Recommends("fit", "x_covariance", physics="false retains fixed-x covariance only among inputs with the same ensemble id and nonempty resample_id; true also retains cross-x covariance within those groups, while missing ids isolate inputs and distinct groups remain block diagonal.", default=False),
    Depends("fit", "pdep_gev", physics="Positive momenta used only for post-fit curves h0+c_inv_p2/P^2+c_inv_p4/P^4 when those terms exist; they add no observations and do not alter the P->infinity result."),
    Recommends("fit", "physical_pion_mass_gev", physics="Physical pion mass defining the chiral target: mpi2=m_pi^2-(m_pi^phys)^2 and mpi4_log_mpi2 is analogously physical-subtracted, so both bases vanish at this mass.", default=0.135),
    Depends("fit", "posterior_prior_error_scale", physics="Multiplier of sample-average posterior standard deviations used as priors for each bootstrap/jackknife fit; it stabilizes resample propagation and does not change the initial center-fit prior."),
    List("fit.x_independent_terms", "term", physics="Ordered unique bases with coefficients global in x; this list must be disjoint from x_dependent_terms so each correction appears in one coefficient class."),
    List("fit.x_dependent_terms", "term", physics="Ordered unique bases with pointwise coefficients c_t(x); together with x_independent_terms it must select at least one correction."),
    Value("fit.x_independent_terms.term", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Supported shared bases use a_s and L=L_s a_s in fm and P,m_pi in GeV: a_s, a_s^2, a_s^4; raw (a_s P)^2,(a_s P)^4 without hbar-c conversion; exp[-m_pi L/(hbar c)], exp[-sqrt(2)m_pi L/(hbar c)]; m_pi^2-m_phys^2; m_pi^4 log(m_pi^2)-m_phys^4 log(m_phys^2) using numerical GeV values; and P^-2,P^-4. Coefficients carry inverse basis units when h is dimensionless."),
    Value("fit.x_dependent_terms.term", Literal["a", "a2", "a4", "ap2", "ap4", "exp_mpi_L", "exp_sqrt2_mpi_L", "mpi2", "mpi4_log_mpi2", "inv_p2", "inv_p4"], physics="Uses the same implemented bases as the shared class, but fits c_t(x) independently at each x so the correction may change the distribution shape."),
    Value("fit.priors", dict, physics="Exactly one finite common numerical mean and positive width applied in each parameter's native units to initial priors for h0 and all correction coefficients; the same number therefore need not imply equal physical prior strength.", validator=_valid_priors),
    Value("fit.x_covariance", bool, physics="Whether to retain cross-x covariance inside groups sharing ensemble id and nonempty resample_id; fixed-x covariance remains when false, and x-independent coefficients still couple x points."),
    List("fit.pdep_gev", "momentum", physics="Nonempty ordered momenta for finite-P diagnostic curves only; fitted input momenta still come from the matched-distribution metadata.", validator=_nonempty),
    Value("fit.pdep_gev.momentum", (int, float), physics="Finite positive diagnostic momentum P in GeV as required by the contract; positivity also avoids the P=0 singularity when inv_p2 or inv_p4 is selected.", validator=_positive),
    Value("fit.physical_pion_mass_gev", (int, float), physics="Finite positive target pion mass in GeV that sets the zero of mpi2 and mpi4_log_mpi2 corrections when those bases are selected.", validator=_positive),
    Value("fit.posterior_prior_error_scale", (int, float), physics="Finite positive resample-prior multiplier: one reuses posterior widths, values above one weaken regularization, and values below one tighten it.", validator=_positive),
    Recommends("systematics_budget", "systematics_prescription", physics="Pointwise heuristic on variant central values: one variant gives |variant-main| and several give max-min after interpolation to the main x grid; four components are assumed mutually independent and combined in quadrature, and the resulting total systematic is also assumed independent of the main statistical error for their quadrature sum.", default="variant_envelope_quadrature"),
    Depends("systematics_budget", "systematics_groups", physics="Assigns every ordered input exactly once to main, zs, lambda_extrapolation, lamet_scale, or other_extrapolations; these authored labels define but do not verify each variant's physical provenance."),
    Value("systematics_budget.systematics_prescription", Literal["variant_envelope_quadrature"], physics="Uses only variant central values: |variant-main| for one, max-min for several, and zero for an empty group; mutually independent components combine in quadrature, then the total systematic and main statistical error are also assumed independent and combined, without variant statistical covariance."),
    Value("systematics_budget.systematics_groups", dict, physics="One main index plus authored index lists labeled as renormalization-switch, Fourier-tail, matching-scale, and other extrapolation variants; they must partition all inputs, but code does not infer or validate the labels' physics provenance.", validator=_valid_systematics_groups),
)

INPUT_RULES = (
    Depends("", "distributions", physics="In fit mode, matched finite-parameter distributions enter the selected-basis extrapolation for h0(x); in budget mode, ordered one-dimensional central and variant distributions enter an uncertainty assembly whose matching provenance is authored but not validated."),
    List("distributions", "distribution", physics="Nonempty ordered inputs; order identifies observations in fit mode and supplies the indices referenced by systematics_groups in budget mode.", validator=_nonempty),
    Source("distributions.distribution", physics="Each input may reference an upstream job or external file; all require x and resamples, while fit inputs additionally require ensemble kinematics and matching provenance. Fit propagation truncates to the smallest sample count and pairs equal sample indices, so compatible replica ordering is authored rather than validated."),
)

SYSTEMATICS_RULES = (
    Recommends("", "defaults", physics="Optional common fields inherited by extrapolation variants, reducing repetition without changing the authored central fit.", default={}),
    Recommends("", "variants", physics="Optional alternative scaling ansatze generated by adding or removing shared or pointwise correction bases from the central extrapolation fit.", default=[]),
    Value("defaults", dict, physics="Object of shared variant fields applied before variant-specific values, with explicit values in a variant taking precedence."),
    List("variants", "variant", physics="After propagated Fourier-tail and matching-scale fits, authored order fixes the relative order of ansatz-variation jobs and their indices in other_extrapolations, while each id determines its suffix."),
    Suggests("", "defaults", "variants.variant", physics="Missing fields in each extrapolation variant are filled from defaults before its modified-basis fit is generated."),
    Depends("variants.variant", "id", physics="Stable label for the alternative extrapolation ansatz in generated job IDs, artifacts, and uncertainty provenance."),
    Recommends("variants.variant", "append_x_independent_terms", physics="Correction bases added with one coefficient shared across all x, testing an additional shape-independent scaling effect.", default=[]),
    Recommends("variants.variant", "remove_x_independent_terms", physics="Shared-coefficient bases removed from the central ansatz to test sensitivity to that assumed global correction.", default=[]),
    Recommends("variants.variant", "append_x_dependent_terms", physics="Correction bases added with independent coefficients c_t(x), testing an additional x-dependent distortion.", default=[]),
    Recommends("variants.variant", "remove_x_dependent_terms", physics="Pointwise-coefficient bases removed from the central ansatz to test sensitivity to that x-dependent correction.", default=[]),
    Value("variants.variant.id", str, physics="Lowercase identifier beginning with a letter and containing only letters, digits, or underscores, suitable for reproducible job suffixes.", validator=_safe_systematics_id),
    Value("variants.variant.append_x_independent_terms", list, physics="Unique supported bases to add to the shared-coefficient class; each must be absent from the central class and remain disjoint from pointwise terms.", validator=_systematics_terms),
    Value("variants.variant.remove_x_independent_terms", list, physics="Unique supported bases to remove from the shared-coefficient class; each must exist in the central ansatz.", validator=_systematics_terms),
    Value("variants.variant.append_x_dependent_terms", list, physics="Unique supported bases to add to the pointwise-coefficient class; each must be absent from the central class and remain disjoint from shared terms.", validator=_systematics_terms),
    Value("variants.variant.remove_x_dependent_terms", list, physics="Unique supported bases to remove from the pointwise-coefficient class; each must exist in the central ansatz.", validator=_systematics_terms),
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


def check_systematics(context: CheckContext) -> list[Issue]:
    variants = context.params["variants"]
    labels = [variant["id"] for variant in variants]
    issues = []
    if len(set(labels)) != len(labels):
        issues.append(Issue("variants", "ids must be unique", "Every variation creates one job suffix."))
    for index, variant in enumerate(variants):
        controls = (
            "append_x_independent_terms",
            "remove_x_independent_terms",
            "append_x_dependent_terms",
            "remove_x_dependent_terms",
        )
        if not any(variant[key] for key in controls):
            issues.append(
                Issue(
                    f"variants[{index}]",
                    "must add or remove at least one extrapolation term",
                    "A systematic variant must differ from its central fit.",
                )
            )
        for dependence in ("independent", "dependent"):
            added = set(variant[f"append_x_{dependence}_terms"])
            removed = set(variant[f"remove_x_{dependence}_terms"])
            if added & removed:
                issues.append(
                    Issue(
                        f"variants[{index}].append_x_{dependence}_terms",
                        f"must be disjoint from remove_x_{dependence}_terms",
                        "One variation cannot add and remove the same term in one coefficient class.",
                    )
                )
    return issues


SYSTEMATICS_CHECKS = (check_systematics,)
