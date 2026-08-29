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
    stage_job_rules,
)
from lamet_agent.stages.correlator_analysis.ask import (
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


def _nonempty_string(value: str) -> bool:
    return bool(value)


def _json_path(value: str) -> bool:
    return value.lower().endswith(".json")


def _unique_record_ids(value: list[object]) -> bool:
    ids = [item.get("id") for item in value if isinstance(item, dict)]
    return len(ids) == len(value) and all(isinstance(item, str) and item for item in ids) and len(set(ids)) == len(ids)


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Recommends("", "analysis_method", physics="Selects whether the job extracts the result with direct nonlinear least-squares or the Lanczos algorithm.", default="lsqfit"),
    Value("analysis_method", Literal["lsqfit", "lanczos"], physics="'lsqfit' fits authored correlator models with sample-wise nonlinear least-squares; 'lanczos' extracts the spectrum or matrix element through the Lanczos algorithm."),
    Provides("", "lsqfit", "analysis_method", physics="The least-squares branch owns spectral and matrix-element candidate fitting."),
    Provides("", "lanczos", "analysis_method", physics="The Lanczos algorithm owns Krylov analysis and nested resampling."),
    Depends("", "component", physics="The fit needs an explicit real, imaginary, or complex channel selection."),
    Depends("", "nstate", physics="The fitting model needs candidate state counts, while Lanczos uses one authored exported Ritz-state count and infers its internal order."),
    Depends("lsqfit", "fit_scope", physics="The fit scope selects the observable-specific data and model function used by the least-squares fit."),
    List("lsqfit.fit_scope", "scope", physics="Multiple scopes allow the candidate scan to compare distinct observable models.", validator=_nonempty),
    Value("lsqfit.fit_scope.scope", Literal["spectrum", "3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"], physics="'spectrum' fits two-point energies; '3pt_ratio' fits the ratio of a three-point correlator to a two-point correlator; 'FH' extracts the matrix element from the slope of the summed ratio with respect to source-sink separation; '3pt_ratio+FH' combines both; 'qda_ratio' fits a nonlocal-to-local two-point ratio."),
    Depends("lsqfit", "fit_strategy", physics="Ordinary matrix-element fits need an explicit strategy for handling two-point information and propagating its uncertainty to the matrix element."),
    Depends("lsqfit", "fitting_form", physics="The matrix-element model needs a forward or non-forward spectral decomposition selected by the kinematics."),
    Recommends("lsqfit", "prior_width", physics="A default prior scale is needed to set the uncertainty of underconstrained spectral and matrix-element parameters.", default=[1.0]),
    Depends("lsqfit", "model_average", physics="Candidate selection needs an explicit choice between publishing one fit and averaging successful fits."),
    Depends("lsqfit", "pt2_windows", physics="Two-point spectrum information needs candidate time windows chosen from the observed signal and uncertainty.", null_hook=recommend_pt2_windows),
    Depends("lsqfit", "pt3_windows", physics="Three-point and Feynman-Hellmann observables need candidate source-sink and insertion-time windows.", null_hook=recommend_pt3_windows),
    Recommends("lsqfit", "svdcut", physics="Correlated fits need a relative covariance singular-value cutoff to suppress numerically unresolved directions.", default=1e-12),
    Depends("lsqfit", "posterior_prior_error_scale", physics="The fit needs a scale for propagating prior uncertainty; chained fits also use it to widen the preceding spectrum posterior."),
    Depends("lsqfit", "q_min", physics="Candidate comparison needs a minimum fit-quality probability for acceptance."),
    Recommends("lsqfit", "chi2_dof_tolerance", physics="The information-preserving window rule needs a tolerance for retaining fits near the best chi2/dof.", default=0.25),
    List("lsqfit.fit_strategy", "strategy", physics="Multiple strategies let the candidate scan compare alternative two-point/spectral uncertainty-propagation paths.", validator=_nonempty),
    List("nstate", "state_count", physics="Multiple state counts let the candidate scan compare spectral truncations.", validator=_nonempty),
    List("lsqfit.prior_width", "width", physics="Multiple prior widths let the candidate scan test prior sensitivity.", validator=_nonempty),
    List("lsqfit.pt2_windows", "window", physics="Multiple two-point windows let the candidate scan test fit-range stability.", validator=_nonempty),
    Depends("lsqfit.pt2_windows.window", "tmin", physics="A two-point fit window requires a lower endpoint."),
    Value("lsqfit.pt2_windows.window.tmin", int, physics="The starting Euclidean lattice-time coordinate of the two-point correlator fit; it must be a nonnegative integer.", validator=_nonnegative),
    Depends("lsqfit.pt2_windows.window", "tmax", physics="A two-point fit window requires an upper endpoint."),
    Value("lsqfit.pt2_windows.window.tmax", int, physics="The exclusive ending Euclidean lattice-time coordinate of the two-point correlator fit; it must be a positive integer.", validator=_positive),
    List("lsqfit.pt3_windows", "window", physics="Multiple three-point windows let the candidate scan test insertion-range stability."),
    Depends("lsqfit.pt3_windows.window", "tsep_ls", physics="A three-point window requires the source-sink separations it uses."),
    List("lsqfit.pt3_windows.window.tsep_ls", "tsep", physics="A list is needed because one candidate window may cover several source-sink separations, which must be unique.", validator=_nonempty_unique),
    Value("lsqfit.pt3_windows.window.tsep_ls.tsep", int, physics="The Euclidean lattice-time separation between the source and sink of a three-point correlator; it must be a positive integer.", validator=_positive),
    Depends("lsqfit.pt3_windows.window", "tau_cut", physics="A three-point window requires an insertion-time cut."),
    Value("lsqfit.pt3_windows.window.tau_cut", int, physics="The number of Euclidean lattice-time slices excluded near each insertion endpoint; it must be a nonnegative integer.", validator=_nonnegative),
    Depends("lanczos", "scope", physics="The Lanczos algorithm needs to know whether to analyze a two-point spectrum or a three-point matrix element."),
    Recommends("lanczos", "inner_samples", physics="Each outer sample needs an inner bootstrap ensemble for CW filtering and median aggregation.", default=200),
    Recommends("lanczos", "precision", physics="Lanczos recurrence arithmetic needs an explicit numeric precision; zero selects the normal NumPy double-precision path.", default=0),
    Value("lsqfit.fit_strategy.strategy", Literal["joint", "chained", "independent"], physics="'joint' propagates correlated two-point and matrix-element uncertainties in one fit; 'chained' transfers spectrum posterior uncertainties as widened priors; 'independent' omits a separate two-point model term."),
    Value("component", Literal["re", "im", "both"], physics="'re' selects the real channel, 'im' the imaginary channel, and 'both' fits both channels."),
    Value("nstate.state_count", int, physics="The number of retained spectral states in the correlator decomposition; it must be a positive integer.", validator=_positive),
    Value("lsqfit.prior_width.width", float, physics="The scale of Gaussian prior uncertainties for a fit candidate; it must be a positive floating-point value.", validator=_positive),
    Value("lsqfit.model_average", bool, physics="true statistically averages successful candidates; false publishes one selected candidate. The current policy requires false because weighted averaging is not implemented."),
    Value("lsqfit.fitting_form", Literal["Breit", "NonBreit"], physics="'Breit' is the equal-momentum forward decomposition; 'NonBreit' is the distinct source/sink momentum decomposition."),
    Value("lsqfit.svdcut", (int, float), physics="The relative covariance singular-value cutoff used to stabilize correlated fits; it must be finite and positive.", validator=_positive),
    Value("lsqfit.posterior_prior_error_scale", (int, float), physics="The factor used to widen propagated posterior or prior uncertainties; it must be finite and positive.", validator=_positive),
    Value("lsqfit.q_min", (int, float), physics="The minimum fit-quality probability Q for candidate acceptance; it must lie in [0, 1].", validator=_unit_interval),
    Value("lsqfit.chi2_dof_tolerance", (int, float), physics="The allowed increase in chi2/dof relative to the best candidate; it must be finite and nonnegative.", validator=_nonnegative),
    Value("lanczos.scope", Literal["2pt_spectrum", "3pt_matrix"], physics="'2pt_spectrum' extracts a two-point spectrum; '3pt_matrix' extracts a three-point matrix element with the Lanczos algorithm."),
    Value("lanczos.inner_samples", int, physics="The number of inner bootstrap replicas used for each outer sample; it must be a positive integer.", validator=_positive),
    Value("lanczos.precision", int, physics="The number of decimal digits used for Lanczos recurrence arithmetic; it must be a nonnegative integer, with zero selecting NumPy double precision.", validator=_nonnegative),
)

INPUT_RULES = (
    Depends("", "correlators", physics="Correlator analysis needs an explicit selection of descriptor records because one descriptor JSON can contain many correlators."),
    List("correlators", "correlator", physics="A list is needed to preserve the ordered set of correlator records used by one job.", validator=lambda value: _nonempty(value) and _unique_record_ids(value)),
    Depends("correlators.correlator", "json", physics="Each selected record needs the descriptor JSON that contains its data definition."),
    Depends("correlators.correlator", "id", physics="Each selected record needs the ID used to select one correlator definition from that descriptor."),
    Value("correlators.correlator.json", str, physics="The path to a project correlator descriptor JSON document; it must be a string ending in .json.", validator=_json_path),
    Value("correlators.correlator.id", str, physics="The identifier of one correlator record in the descriptor; it must be a nonempty string.", validator=_nonempty_string),
)
# fmt: on
# ruff: enable[E501]


def check_method_family(context: CheckContext) -> Issue | None:
    if context.params["analysis_method"] != "lsqfit":
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
    if context.params["analysis_method"] != "lsqfit":
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
    if context.params["analysis_method"] != "lsqfit":
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
    if context.params["analysis_method"] != "lsqfit":
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
