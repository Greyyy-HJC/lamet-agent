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
from lamet_agent.stages.correlator_analysis._scope import parse_fit_scope, valid_scope_stage


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
    List("lsqfit.fit_scope", "scope", physics="The ordered entries form chained fit stages; atoms joined with '+' inside one entry share a correlated joint likelihood.", validator=_nonempty),
    Value("lsqfit.fit_scope.scope", str, physics="Each list entry is one joint fit stage whose atoms are separated by '+'. List order denotes chained posterior propagation. Supported atoms are 2pt, 3pt, qda, FH, 3pt_ratio, and qda_ratio.", validator=valid_scope_stage),
    Depends("lsqfit", "fitting_form", physics="The matrix-element model needs a forward or non-forward spectral decomposition selected by the kinematics."),
    Recommends("lsqfit", "prior_width", physics="A default prior scale is needed to set the uncertainty of underconstrained spectral and matrix-element parameters.", default=[1.0]),
    Depends("lsqfit", "model_average", physics="At a fixed data window and fit-scope pipeline, false publishes the selected nstate/prior-width model; true forms per-resample, per-z logGBF-weighted means over those models."),
    Depends("lsqfit", "pt2_windows", physics="Two-point spectrum information needs candidate time windows chosen from the observed signal and uncertainty.", null_hook=recommend_pt2_windows),
    Depends("lsqfit", "pt3_windows", physics="Three-point and Feynman-Hellmann observables need candidate source-sink and insertion-time windows.", null_hook=recommend_pt3_windows),
    Recommends("lsqfit", "svdcut", physics="Correlated fits need a relative covariance singular-value cutoff to suppress numerically unresolved directions.", default=1e-12),
    Depends("lsqfit", "posterior_prior_error_scale", physics="The fit needs a scale for propagating prior uncertainty; chained fits also use it to widen the preceding spectrum posterior."),
    Depends("lsqfit", "q_min", physics="Candidate comparison needs a preferred fit-quality probability; after recommendation retries are exhausted, selection falls back across all retained numerical candidates."),
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
    Value("component", Literal["re", "im", "both"], physics="'re' selects the real channel, 'im' the imaginary channel, and 'both' fits both channels."),
    Value("nstate.state_count", int, physics="The number of retained spectral states in the correlator decomposition; it must be a positive integer.", validator=_positive),
    Value("lsqfit.prior_width.width", float, physics="The scale of Gaussian prior uncertainties for a fit candidate; it must be a positive floating-point value.", validator=_positive),
    Value("lsqfit.model_average", bool, physics="false publishes the window-selected nstate/prior-width model; true forms per-resample, per-z normalized exp(logGBF-max(logGBF)) means over nstate and prior_width at that frozen window, strategy, and scope, without Q filtering. Between-model spread of center values is recorded separately and is not mixed into the resampled samples."),
    Value("lsqfit.fitting_form", Literal["Breit", "NonBreit"], physics="'Breit' is the equal-momentum forward decomposition; 'NonBreit' is the distinct source/sink momentum decomposition."),
    Value("lsqfit.svdcut", (int, float), physics="The relative covariance singular-value cutoff used to stabilize correlated fits; it must be finite and positive.", validator=_positive),
    Value("lsqfit.posterior_prior_error_scale", (int, float), physics="The factor used to widen propagated posterior or prior uncertainties; it must be finite and positive.", validator=_positive),
    Value("lsqfit.q_min", (int, float), physics="The preferred fit-quality probability Q for acceptance; after recommendation retries are exhausted, a finite-Q candidate may be selected below it from all retained attempts.", validator=_unit_interval),
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
    try:
        parse_fit_scope(context.params["fit_scope"])
    except (TypeError, ValueError) as exc:
        return Issue(
            "fit_scope",
            str(exc),
            "The ordered scope pipeline must identify compatible joint and chained likelihoods.",
        )
    return None


def check_lsqfit_windows(context: CheckContext) -> Issue | None:
    if context.params["analysis_method"] != "lsqfit":
        return None
    lsqfit = context.params
    try:
        scope = parse_fit_scope(lsqfit["fit_scope"])
    except (TypeError, ValueError):
        return None
    if scope.needs_pt3_data and not lsqfit.get("pt3_windows"):
        return Issue(
            "pt3_windows",
            "is required for three-point and FH fit scopes",
            "The matrix-element fitter needs authored source-sink and insertion-time candidates.",
        )
    nonbreit_atoms = scope.atom_set & {"3pt", "3pt_ratio"}
    if lsqfit["fitting_form"] == "NonBreit" and (not nonbreit_atoms or scope.atom_set - {"2pt", "3pt", "3pt_ratio"}):
        return Issue(
            "fit_scope",
            "NonBreit requires a raw 3pt or 3pt_ratio path and permits only an accompanying 2pt atom",
            "qDA and FH models currently use the forward spectral decomposition.",
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
    try:
        scope = parse_fit_scope(lsqfit["fit_scope"])
    except (TypeError, ValueError):
        return None
    if not scope.is_qda:
        return None
    if lsqfit["fitting_form"] != "Breit":
        return Issue(
            "fitting_form",
            "must be 'Breit' for qDA fitting",
            "The implemented qDA correlator uses the forward spectral decomposition.",
        )
    if lsqfit.get("pt3_windows"):
        return Issue(
            "pt3_windows",
            "must be omitted for qDA fit scopes",
            "qDA fits consume only time-dependent local and nonlocal two-point correlators.",
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

CHECKS = (check_method_family, check_lanczos_branch, check_lsqfit_windows, check_qda_scope)
