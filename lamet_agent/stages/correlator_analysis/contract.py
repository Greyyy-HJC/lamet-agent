"""Manifest contract for correlator analysis."""

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
from lamet_agent.stages.correlator_analysis.tools._joint_fit_recommendation import (
    pt2_windows as recommend_pt2_windows,
    pt3_windows as recommend_pt3_windows,
)


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _nonnegative(value: int) -> bool:
    return value >= 0


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _unit_interval(value: int | float) -> bool:
    return math.isfinite(value) and 0 <= value <= 1


def _unique(value: list[object]) -> bool:
    return len(set(value)) == len(value)


def _nonempty_unique(value: list[object]) -> bool:
    return _nonempty(value) and _unique(value)


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Recommends("", "analysis_method", physics="Ordinary jobs use the reference least-squares implementation unless Lanczos is selected explicitly.", default="lsqfit"),
    Value("analysis_method", Literal["lsqfit", "lanczos"], physics="The analysis method selector owns the supported implementation names."),
    Provides("", "lsqfit", "analysis_method", physics="Least-squares fitting owns its candidate methods, priors, windows, and model-selection controls."),
    Provides("", "lanczos", "analysis_method", physics="Lanczos reconstruction owns its Krylov-grid and nested-resampling controls."),
    Depends("", "component", physics="Real and imaginary channels are selected explicitly."),
    Depends("", "nstate", physics="Every authored state-count candidate is explicit."),
    Depends("", "correlator_ids", physics="Each job selects its correlator records explicitly."),
    Depends("lsqfit", "fit_scope", physics="The authored correlator observables define the allowed fit scopes."),
    Depends("lsqfit", "fit_strategy", physics="The authored candidates define how spectral information and covariance propagate."),
    Depends("lsqfit", "fitting_form", physics="The spectral decomposition is explicit."),
    Recommends("lsqfit", "prior_width", physics="The original correlator fit uses one unit prior-width candidate unless explicitly varied.", default=[1.0]),
    Depends("lsqfit", "model_average", physics="Candidate averaging is an explicit analysis choice."),
    Depends("lsqfit", "pt2_windows", physics="Two-point candidate windows are selected from the observed signal and uncertainty.", null_hook=recommend_pt2_windows),
    Depends("lsqfit", "pt3_windows", physics="Three-point and FH scopes require data-selected windows; scopes without three-point data use none.", null_hook=recommend_pt3_windows),
    Recommends("lsqfit", "svdcut", physics="The original correlator fit defaults to a 1e-12 relative covariance singular-value cut.", default=1e-12),
    Depends("lsqfit", "posterior_prior_error_scale", physics="Posterior-prior scaling is explicit."),
    Depends("lsqfit", "q_min", physics="The candidate quality threshold is explicit."),
    Recommends("lsqfit", "chi2_dof_tolerance", physics="The original data-window rule retains candidates within 0.25 chi2/dof of the best fit before maximizing information.", default=0.25),
    List("lsqfit.fit_scope", "scope", physics="Fit scopes are a nonempty candidate list.", validator=_nonempty),
    List("lsqfit.fit_strategy", "strategy", physics="Fit strategies are a nonempty candidate list.", validator=_nonempty),
    List("nstate", "state_count", physics="State-count candidates are a nonempty list.", validator=_nonempty),
    List("lsqfit.prior_width", "width", physics="Prior-width candidates are a nonempty list.", validator=_nonempty),
    List("correlator_ids", "correlator_id", physics="Selected correlator ids are a nonempty list.", validator=_nonempty),
    List("lsqfit.pt2_windows", "window", physics="At least one two-point window is required.", validator=_nonempty),
    Depends("lsqfit.pt2_windows.window", "tmin", physics="The lower two-point time is explicit."),
    Value("lsqfit.pt2_windows.window.tmin", int, physics="The lower two-point time is nonnegative.", validator=_nonnegative),
    Depends("lsqfit.pt2_windows.window", "tmax", physics="The upper two-point time is explicit."),
    Value("lsqfit.pt2_windows.window.tmax", int, physics="The upper two-point time is positive.", validator=_positive),
    List("lsqfit.pt3_windows", "window", physics="Three-point windows form an explicit candidate list."),
    Depends("lsqfit.pt3_windows.window", "tsep_ls", physics="Source-sink separations are explicit."),
    List("lsqfit.pt3_windows.window.tsep_ls", "tsep", physics="Source-sink separations are nonempty and unique.", validator=_nonempty_unique),
    Value("lsqfit.pt3_windows.window.tsep_ls.tsep", int, physics="Every source-sink separation is positive.", validator=_positive),
    Depends("lsqfit.pt3_windows.window", "tau_cut", physics="The insertion-time cut is explicit."),
    Value("lsqfit.pt3_windows.window.tau_cut", int, physics="The insertion-time cut is nonnegative.", validator=_nonnegative),
    Depends("lanczos", "scope", physics="Lanczos explicitly selects a two-point spectrum or three-point matrix."),
    Recommends("lanczos", "inner_samples", physics="The inner bootstrap distribution drives CW filtering and median aggregation.", default=200),
    Recommends("lanczos", "precision", physics="Lanczos recurrence construction uses NumPy double precision unless decimal precision is requested.", default=0),
    Depends("lanczos", "t0", physics="The nonnegative trimming offset fixes the first effective moment."),
    Depends("lanczos", "time_step", physics="The positive transfer-matrix power fixes sparse moment spacing."),
    Value("lsqfit.fit_scope.scope", Literal["spectrum", "3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"], physics="Every scope names one migrated correlator observable family."),
    Value("lsqfit.fit_strategy.strategy", Literal["joint", "chained", "independent"], physics="Every strategy names one migrated covariance-propagation path."),
    Value("component", Literal["re", "im", "both"], physics="Components are real, imaginary, or both."),
    Value("nstate.state_count", int, physics="Every state count is positive.", validator=_positive),
    Value("lsqfit.prior_width.width", float, physics="Every prior width is positive.", validator=_positive),
    Value("lsqfit.model_average", bool, physics="Model averaging is controlled by an explicit boolean."),
    Value("correlator_ids.correlator_id", str, physics="Correlator ids are descriptor record names."),
    Value("lsqfit.fitting_form", Literal["Breit", "NonBreit"], physics="Forward and non-forward fits use controlled spectral decompositions."),
    Value("lsqfit.svdcut", (int, float), physics="The covariance cutoff is finite and positive.", validator=_positive),
    Value("lsqfit.posterior_prior_error_scale", (int, float), physics="Posterior-prior scaling is finite and positive.", validator=_positive),
    Value("lsqfit.q_min", (int, float), physics="The candidate quality threshold is a probability.", validator=_unit_interval),
    Value("lsqfit.chi2_dof_tolerance", (int, float), physics="The data-window chi2/dof tolerance is finite and nonnegative.", validator=_nonnegative),
    Value("lanczos.scope", Literal["2pt_spectrum", "3pt_matrix"], physics="Lanczos scope is a controlled physical output."),
    Value("lanczos.inner_samples", int, physics="The inner bootstrap count is positive.", validator=_positive),
    Value("lanczos.precision", int, physics="Decimal precision is nonnegative; zero selects NumPy double precision.", validator=_nonnegative),
    Value("lanczos.t0", int, physics="The trimming offset is nonnegative.", validator=_nonnegative),
    Value("lanczos.time_step", int, physics="The transfer-matrix power is positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "correlators", physics="Correlator analysis consumes one descriptor source."),
    Source("correlators", physics="The correlator descriptor is one external file source.", allow_job=False),
)
# fmt: on
# ruff: enable[E501]


def check_method_family(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method", "lsqfit") != "lsqfit":
        return None
    settings = context.params
    scopes = set(settings["fit_scope"])
    strategies = set(settings["fit_strategy"])
    if "spectrum" in scopes:
        if scopes != {"spectrum"}:
            return Issue(
                "fit_scope",
                "spectrum must be the only scope in a spectrum job",
                "Spectrum and matrix-element jobs expose disjoint fit scopes.",
            )
        if strategies != {"independent"}:
            return Issue(
                "fit_strategy",
                "must contain only 'independent' for a spectrum job",
                "A direct two-point spectrum fit has no separate matrix-element covariance propagation.",
            )
    if "qda_ratio" in scopes and (len(scopes) != 1 or strategies != {"independent"}):
        return Issue(
            "fit_scope",
            "qda_ratio requires the exclusive fit_scope ['qda_ratio'] and fit_strategy ['independent']",
            "The migrated qDA path fits each nonlocal/local two-point ratio independently.",
        )
    return None


def check_lsqfit_windows(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method", "lsqfit") != "lsqfit":
        return None
    lsqfit = context.params
    scopes = set(lsqfit["fit_scope"])
    ordinary_scopes = scopes & {"3pt_ratio", "FH", "3pt_ratio+FH"}
    if ordinary_scopes and not lsqfit.get("pt3_windows"):
        return Issue(
            "pt3_windows",
            "is required for three-point and FH fit scopes",
            "The matrix-element fitter needs authored source-sink and insertion-time candidates.",
        )
    if lsqfit["fitting_form"] == "NonBreit" and ordinary_scopes != {"3pt_ratio"}:
        return Issue(
            "fit_scope",
            "must contain only '3pt_ratio' for NonBreit fitting",
            "The implemented non-forward model is the three-point ratio decomposition.",
        )
    for index, window in enumerate(lsqfit.get("pt2_windows") or []):
        if window["tmin"] >= window["tmax"]:
            return Issue(
                f"pt2_windows[{index}]",
                "must be an increasing nonnegative integer window",
                "Every two-point fit window contains physical lattice times.",
            )
    for index, window in enumerate(lsqfit.get("pt3_windows") or []):
        tseps = window["tsep_ls"]
        tau_cut = window["tau_cut"]
        if any(2 * tau_cut > value for value in tseps):
            return Issue(
                f"pt3_windows[{index}].tau_cut",
                "must leave at least one insertion point for every tsep",
                "The insertion cut cannot remove the complete three-point window.",
            )
    tau_cuts = [window["tau_cut"] for window in lsqfit.get("pt3_windows") or []]
    if len(set(tau_cuts)) != len(tau_cuts):
        return Issue(
            "pt3_windows",
            "tau_cut values must be unique",
            "The fit tool identifies each authored three-point window by its insertion cut.",
        )
    return None


def check_qda_scope(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method", "lsqfit") != "lsqfit":
        return None
    lsqfit = context.params
    if "qda_ratio" not in set(lsqfit["fit_scope"]):
        return None
    if lsqfit["fitting_form"] != "Breit":
        return Issue(
            "fitting_form",
            "must be 'Breit' for qda_ratio",
            "The implemented qDA ratio uses the forward one-state decomposition.",
        )
    if lsqfit.get("pt3_windows"):
        return Issue(
            "pt3_windows",
            "must be omitted for qda_ratio",
            "The qDA ratio consumes only nonlocal/local two-point correlators.",
        )
    return None


def check_candidate_policy(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method", "lsqfit") != "lsqfit":
        return None
    settings = context.params
    if settings.get("model_average") is True:
        return Issue(
            "model_average",
            "must be false until weighted candidate averaging is implemented",
            "Publishing one candidate and model averaging are distinct statistical procedures.",
        )
    if "qda_ratio" in set(settings.get("fit_scope", [])) and context.params.get("nstate") != [1]:
        return Issue(
            "nstate",
            "must be [1] for qDA ratio fitting",
            "The implemented qDA ratio model is a one-state constant fit.",
        )
    return None


def check_lanczos_branch(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method") != "lanczos":
        return None
    if len(context.params.get("nstate", [])) != 1:
        return Issue(
            "nstate",
            "must contain exactly one exported Ritz-state count",
            "Lanczos orders states internally and exports one authored count.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_method_family, check_lanczos_branch, check_lsqfit_windows, check_qda_scope, check_candidate_policy)
