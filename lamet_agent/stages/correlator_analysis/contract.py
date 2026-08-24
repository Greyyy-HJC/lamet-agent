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
    Value,
)
from lamet_agent.stages.correlator_analysis.hook import (
    recommend_pt2_windows,
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


PARAM_RULES = (
    Depends("", "observable", physics="The correlator job declares whether it produces a spectrum or matrix element."),
    Depends("", "analysis_method", physics="The job selects one complete correlator-analysis implementation."),
    Provides("", "lsqfit", "analysis_method", physics="Least-squares fitting owns its candidate methods, priors, windows, and model-selection controls."),
    Provides("", "lanczos", "analysis_method", physics="Lanczos reconstruction owns its Krylov-grid and nested-resampling controls."),
    Depends("", "resample_group", physics="Aligned correlators share a stable resampling plan by group."),
    Depends("", "component", physics="Real and imaginary channels are selected explicitly."),
    Depends("", "nstate", physics="Every authored state-count candidate is explicit."),
    Depends("", "correlator_ids", physics="Each job selects its correlator records explicitly."),
    Depends("lsqfit", "fit_scope", physics="The authored correlator observables define the allowed fit scopes."),
    Depends("lsqfit", "fit_strategy", physics="The authored candidates define how spectral information and covariance propagate."),
    Depends("lsqfit", "fitting_form", physics="The spectral decomposition is explicit."),
    Depends("lsqfit", "prior_width", physics="Every authored prior-width candidate is explicit."),
    Depends("lsqfit", "model_average", physics="Candidate averaging is an explicit analysis choice."),
    Depends("lsqfit", "time_range", physics="All fit decisions stay within the authored time range."),
    Depends("lsqfit", "pt2_windows", physics="Two-point candidate windows are selected from the observed signal and uncertainty.", null_hook=recommend_pt2_windows),
    Depends("lsqfit", "pt3_windows", physics="Three-point candidate windows are explicit for three-point/FH scopes.", required=False),
    Depends("lsqfit", "svdcut", physics="The covariance cutoff is explicit."),
    Depends("lsqfit", "posterior_prior_error_scale", physics="Posterior-prior scaling is explicit."),
    Depends("lsqfit", "q_min", physics="The candidate quality threshold is explicit."),
    Depends("lsqfit", "tune_z", physics="The representative tuning separation is explicit."),
    List("lsqfit.fit_scope", "scope", physics="Fit scopes are a nonempty candidate list.", validator=_nonempty),
    List("lsqfit.fit_strategy", "strategy", physics="Fit strategies are a nonempty candidate list.", validator=_nonempty),
    List("nstate", "state_count", physics="State-count candidates are a nonempty list.", validator=_nonempty),
    List("lsqfit.prior_width", "width", physics="Prior-width candidates are a nonempty list.", validator=_nonempty),
    List("correlator_ids", "correlator_id", physics="Selected correlator ids are a nonempty list.", validator=_nonempty),
    Depends("lsqfit.time_range", "min", physics="The lower fit-time bound is explicit."),
    Depends("lsqfit.time_range", "max", physics="The upper fit-time bound is explicit."),
    List("lsqfit.pt2_windows", "window", physics="At least one two-point window is required.", validator=_nonempty),
    Depends("lsqfit.pt2_windows.window", "tmin", physics="The lower two-point time is explicit."),
    Value("lsqfit.pt2_windows.window.tmin", int, physics="The lower two-point time is nonnegative.", validator=_nonnegative),
    Depends("lsqfit.pt2_windows.window", "tmax", physics="The upper two-point time is explicit."),
    Value("lsqfit.pt2_windows.window.tmax", int, physics="The upper two-point time is positive.", validator=_positive),
    List("lsqfit.pt3_windows", "window", physics="Three-point windows form an explicit candidate list.", validator=_nonempty),
    Depends("lsqfit.pt3_windows.window", "tsep_ls", physics="Source-sink separations are explicit."),
    List("lsqfit.pt3_windows.window.tsep_ls", "tsep", physics="Source-sink separations are nonempty and unique.", validator=_nonempty_unique),
    Value("lsqfit.pt3_windows.window.tsep_ls.tsep", int, physics="Every source-sink separation is positive.", validator=_positive),
    Depends("lsqfit.pt3_windows.window", "tau_cut", physics="The insertion-time cut is explicit."),
    Value("lsqfit.pt3_windows.window.tau_cut", int, physics="The insertion-time cut is nonnegative.", validator=_nonnegative),
    Depends("lanczos", "scope", physics="Lanczos explicitly selects a two-point spectrum or three-point matrix."),
    Recommends("lanczos", "inner_samples", physics="The inner bootstrap distribution drives CW filtering and median aggregation.", default=200),
    Recommends("lanczos", "precision", physics="Lanczos recurrence construction uses NumPy double precision unless decimal precision is requested.", default=0),
    Depends("lanczos", "t0", physics="An optional nonnegative trimming offset fixes the first effective moment.", required=False),
    Depends("lanczos", "time_step", physics="An optional positive transfer-matrix power fixes sparse moment spacing.", required=False),
    Value("observable", Literal["spectrum", "matrix_element"], physics="The observable is a controlled string."),
    Value("lsqfit.fit_scope.scope", Literal["spectrum", "3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"], physics="Every scope names one migrated correlator observable family."),
    Value("lsqfit.fit_strategy.strategy", Literal["joint", "chained", "independent"], physics="Every strategy names one migrated covariance-propagation path."),
    Value("resample_group", str, physics="The shared resample group is a stable id."),
    Value("component", Literal["re", "im", "both"], physics="Components are real, imaginary, or both."),
    Value("nstate.state_count", int, physics="Every state count is positive.", validator=_positive),
    Value("lsqfit.prior_width.width", float, physics="Every prior width is positive.", validator=_positive),
    Value("lsqfit.model_average", bool, physics="Model averaging is controlled by an explicit boolean."),
    Value("lsqfit.time_range.min", int, physics="The lower time bound is an integer.", validator=_nonnegative),
    Value("lsqfit.time_range.max", int, physics="The upper time bound is an integer.", validator=_positive),
    Value("correlator_ids.correlator_id", str, physics="Correlator ids are descriptor record names."),
    Value("lsqfit.fitting_form", Literal["Breit", "NonBreit"], physics="Forward and non-forward fits use controlled spectral decompositions."),
    Value("lsqfit.svdcut", (int, float), physics="The covariance cutoff is finite and positive.", validator=_positive),
    Value("lsqfit.posterior_prior_error_scale", (int, float), physics="Posterior-prior scaling is finite and positive.", validator=_positive),
    Value("lsqfit.q_min", (int, float), physics="The candidate quality threshold is a probability.", validator=_unit_interval),
    Value("lsqfit.tune_z", (int, float), physics="The representative tuning separation is finite.", validator=_finite),
    Value("lanczos.scope", Literal["2pt_spectrum", "3pt_matrix"], physics="Lanczos scope is a controlled physical output."),
    Value("lanczos.inner_samples", int, physics="The inner bootstrap count is positive.", validator=_positive),
    Value("lanczos.precision", int, physics="Decimal precision is nonnegative; zero selects NumPy double precision.", validator=_nonnegative),
    Value("lanczos.t0", int, physics="The trimming offset is nonnegative.", validator=_nonnegative),
    Value("lanczos.time_step", int, physics="The transfer-matrix power is positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "correlators", physics="Correlator analysis consumes one descriptor source."),
    Value("correlators", dict, physics="The descriptor source is exactly one source object."),
)


def check_time_range(context: CheckContext) -> Issue | None:
    settings = context.params.get("lsqfit")
    if not isinstance(settings, dict):
        return None
    time_range = settings.get("time_range")
    if time_range is None:
        return None
    if time_range["min"] >= time_range["max"]:
        return Issue("params.lsqfit.time_range.min", "must be smaller than time_range.max", "A fit range must contain increasing nonnegative lattice times.")
    return None


def check_method_family(context: CheckContext) -> Issue | None:
    settings = context.params.get("lsqfit")
    if not isinstance(settings, dict):
        return None
    scopes = set(settings["fit_scope"])
    strategies = set(settings["fit_strategy"])
    if context.params["observable"] == "spectrum":
        if scopes != {"spectrum"}:
            return Issue("params.lsqfit.fit_scope", "must contain only 'spectrum' for a spectrum job", "Spectrum and matrix-element jobs expose disjoint fit scopes.")
        if strategies != {"independent"}:
            return Issue("params.lsqfit.fit_strategy", "must contain only 'independent' for a spectrum job", "A direct two-point spectrum fit has no separate matrix-element covariance propagation.")
    elif "spectrum" in scopes:
        return Issue("params.lsqfit.fit_scope", "must not contain 'spectrum' for a matrix-element job", "Spectrum and matrix-element jobs expose disjoint fit scopes.")
    if "qda_ratio" in scopes and (len(scopes) != 1 or strategies != {"independent"}):
        return Issue("params.lsqfit", "qda_ratio requires the exclusive fit_scope ['qda_ratio'] and fit_strategy ['independent']", "The migrated qDA path fits each nonlocal/local two-point ratio independently.")
    return None


def check_lsqfit_windows(context: CheckContext) -> Issue | None:
    lsqfit = context.params.get("lsqfit")
    if not isinstance(lsqfit, dict):
        return None
    scopes = set(lsqfit["fit_scope"])
    ordinary_scopes = scopes & {"3pt_ratio", "FH", "3pt_ratio+FH"}
    if ordinary_scopes and "pt3_windows" not in lsqfit:
        return Issue("params.lsqfit.pt3_windows", "is required for three-point and FH fit scopes", "The matrix-element fitter needs authored source-sink and insertion-time candidates.")
    if lsqfit["fitting_form"] == "NonBreit" and ordinary_scopes != {"3pt_ratio"}:
        return Issue("params.lsqfit.fit_scope", "must contain only '3pt_ratio' for NonBreit fitting", "The implemented non-forward model is the three-point ratio decomposition.")
    for index, window in enumerate(lsqfit.get("pt2_windows") or []):
        if window["tmin"] >= window["tmax"]:
            return Issue(f"params.lsqfit.pt2_windows[{index}]", "must be an increasing nonnegative integer window", "Every two-point fit window contains physical lattice times.")
    for index, window in enumerate(lsqfit.get("pt3_windows") or []):
        tseps = window["tsep_ls"]
        tau_cut = window["tau_cut"]
        if any(2 * tau_cut > value for value in tseps):
            return Issue(f"params.lsqfit.pt3_windows[{index}].tau_cut", "must leave at least one insertion point for every tsep", "The insertion cut cannot remove the complete three-point window.")
    tau_cuts = [window["tau_cut"] for window in lsqfit.get("pt3_windows") or []]
    if len(set(tau_cuts)) != len(tau_cuts):
        return Issue("params.lsqfit.pt3_windows", "tau_cut values must be unique", "The fit tool identifies each authored three-point window by its insertion cut.")
    return None


def check_qda_scope(context: CheckContext) -> Issue | None:
    lsqfit = context.params.get("lsqfit")
    if not isinstance(lsqfit, dict):
        return None
    if "qda_ratio" not in set(lsqfit["fit_scope"]):
        return None
    if lsqfit["fitting_form"] != "Breit":
        return Issue("params.lsqfit.fitting_form", "must be 'Breit' for qda_ratio", "The implemented qDA ratio uses the forward one-state decomposition.")
    if "pt3_windows" in lsqfit:
        return Issue("params.lsqfit.pt3_windows", "must be omitted for qda_ratio", "The qDA ratio consumes only nonlocal/local two-point correlators.")
    return None


def check_candidate_policy(context: CheckContext) -> Issue | None:
    settings = context.params.get("lsqfit")
    if not isinstance(settings, dict):
        return None
    if settings.get("model_average") is True:
        return Issue("params.lsqfit.model_average", "must be false until weighted candidate averaging is implemented", "Publishing one candidate and model averaging are distinct statistical procedures.")
    if "qda_ratio" in set(settings.get("fit_scope", [])) and context.params.get("nstate") != [1]:
        return Issue("params.nstate", "must be [1] for qDA ratio fitting", "The implemented qDA ratio model is a one-state constant fit.")
    return None


def check_lanczos_branch(context: CheckContext) -> Issue | None:
    if context.params.get("analysis_method") != "lanczos":
        return None
    settings = context.params["lanczos"]
    if len(context.params.get("nstate", [])) != 1:
        return Issue("params.nstate", "must contain exactly one exported Ritz-state count", "Lanczos orders states internally and exports one authored count.")
    expected_observable = "spectrum" if settings.get("scope") == "2pt_spectrum" else "matrix_element"
    if context.params.get("observable") != expected_observable:
        return Issue("params.observable", f"must be {expected_observable!r} for lanczos.scope={settings.get('scope')!r}", "Lanczos scope determines whether the stage emits energies or a matrix element.")
    return None


CHECKS = (check_time_range, check_method_family, check_lanczos_branch, check_lsqfit_windows, check_qda_scope, check_candidate_policy)
