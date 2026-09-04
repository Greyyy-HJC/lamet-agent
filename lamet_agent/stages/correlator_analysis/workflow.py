"""Correlator workflow with isolated typed fit-parameter recommendations."""

from __future__ import annotations

import copy
import math
from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.ui import warning
from lamet_agent.stages.correlator_analysis._fit_matrix import run as fit_matrix
from lamet_agent.stages.correlator_analysis._fit_qda import run as fit_qda
from lamet_agent.stages.correlator_analysis._fit_spectrum import run as fit_spectrum
from lamet_agent.stages.correlator_analysis._inspection import run as inspect
from lamet_agent.stages.correlator_analysis._lanczos import run as run_lanczos
from lamet_agent.stages.correlator_analysis._lanczos_inspection import run as inspect_lanczos
from lamet_agent.stages.correlator_analysis._publish import run as publish
from lamet_agent.stages.correlator_analysis.ask import initial, revise


def _candidate_attempts(context: ToolContext) -> dict[str, dict[str, Any]]:
    """Map every attempted authored combination to compact fit-quality diagnostics."""
    attempts = {}
    parameter_fields = (
        "method",
        "fit_strategy",
        "fit_scope",
        "window",
        "tsep_values",
        "nstate",
        "prior_width",
        "tune_z_values",
    )
    quality_fields = (
        "Q",
        "chi2",
        "dof",
        "chi2_dof",
        "min_Q",
        "worst_chi2_dof",
        "quality_passed",
        "numerical_failure",
        "feasible_at_all_tune_z",
        "failure_reasons",
    )
    for candidate in context.state.get("matrix_element_candidates", []):
        candidate_id = str(candidate["id"])
        attempt = {
            "parameters": {name: candidate[name] for name in parameter_fields if name in candidate},
            **{name: candidate[name] for name in quality_fields if name in candidate},
        }
        per_z = candidate.get("tune_z_diagnostics")
        if isinstance(per_z, dict):
            attempt["by_tune_z"] = {
                str(z_value): {
                    name: fit[name]
                    for name in ("Q", "chi2", "dof", "chi2_dof", "logGBF")
                    if isinstance(fit, dict) and name in fit
                }
                for z_value, fit in per_z.items()
            }
        attempts[candidate_id] = attempt
    return attempts


def _spectrum_attempt(
    suggestion: dict[str, Any], *, metrics: dict[str, Any] | None = None, error: str | None = None
) -> dict[str, dict[str, Any]]:
    attempt = {"parameters": suggestion}
    if metrics is not None:
        attempt.update({name: metrics[name] for name in ("Q", "chi2", "dof", "chi2_dof", "logGBF") if name in metrics})
    if error is not None:
        attempt.update({"numerical_failure": True, "error": error})
    return {"spectrum_001": attempt}


def _spectrum_parameters(suggestion: dict[str, Any]) -> dict[str, Any]:
    return {name: suggestion[name] for name in ("tmin", "tmax", "n_states", "prior_means", "prior_widths")}


def _apply_spectrum_suggestion(context: ToolContext, suggestion: dict[str, Any]) -> None:
    context.params["pt2_windows"] = [{"tmin": int(suggestion["tmin"]), "tmax": int(suggestion["tmax"])}]


def _qda_quality_is_low(context: ToolContext) -> bool:
    candidates = context.state.get("matrix_element_candidates", [])
    if not candidates:
        return False
    q_min = float(context.params["q_min"])
    return not any(
        candidate.get("feasible_at_all_tune_z", False)
        and not candidate.get("numerical_failure", False)
        and candidate.get("min_Q") is not None
        and float(candidate["min_Q"]) >= q_min
        for candidate in candidates
    )


def _can_revise(session: LlmSession) -> bool:
    return session.recommendation_calls < session.max_recommendation_calls


def _finite_quality(value: Any) -> float:
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return quality if math.isfinite(quality) else -math.inf


def _inverse_finite_quality(value: Any) -> float:
    quality = _finite_quality(value)
    return -quality if quality != -math.inf else -math.inf


def _apply_matrix_suggestion(context: ToolContext, scopes: set[str], suggestion: dict[str, Any]) -> list[Any]:
    context.params["pt2_windows"] = list(suggestion["pt2_windows"])
    if scopes != {"qda_ratio"}:
        context.params["pt3_windows"] = list(suggestion["pt3_windows"])
    return list(suggestion["tune_z_values"])


def run(context: ToolContext, session: LlmSession) -> None:
    """Scan authored candidates, then revise until quality passes or the job budget is spent."""
    if context.params["analysis_method"] == "lanczos":
        inspect_lanczos(context)
        run_lanczos(context)
        return

    inspect(context)
    scopes = set(context.params["fit_scope"])
    suggestion = initial(context, session)
    if scopes == {"spectrum"}:
        _apply_spectrum_suggestion(context, suggestion)
        q_min = float(context.params["q_min"])
        observation: dict[str, Any] | None = None
        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        while True:
            try:
                observation = fit_spectrum(context, **_spectrum_parameters(suggestion))
            except FitNumericalError as exc:
                if not _can_revise(session):
                    if best is None:
                        raise
                    _quality, observation, suggestion = best
                    _apply_spectrum_suggestion(context, suggestion)
                    break
                suggestion = revise(
                    context,
                    session,
                    _spectrum_attempt(suggestion, error=str(exc)),
                )
                _apply_spectrum_suggestion(context, suggestion)
                continue
            quality = _finite_quality(observation["metrics"].get("Q"))
            if quality != -math.inf and (best is None or quality >= best[0]):
                best = (quality, observation, copy.deepcopy(suggestion))
            if quality >= q_min:
                break
            if not _can_revise(session):
                if best is None:
                    raise FitNumericalError(
                        "no spectrum fit produced a publishable finite-Q result after the allowed attempts"
                    )
                _quality, observation, suggestion = best
                _apply_spectrum_suggestion(context, suggestion)
                break
            suggestion = revise(
                context,
                session,
                _spectrum_attempt(suggestion, metrics=dict(observation["metrics"])),
            )
            _apply_spectrum_suggestion(context, suggestion)
        candidate_id = str(observation["metrics"]["candidate_id"])
        final_low_quality = _finite_quality(observation["metrics"].get("Q")) < q_min
    else:
        fit = fit_qda if scopes == {"qda_ratio"} else fit_matrix
        tune_z_values = list(suggestion["tune_z_values"])
        observation = None
        best: (
            tuple[
                tuple[float, float],
                dict[str, Any],
                list[dict[str, Any]],
                dict[str, Any],
            ]
            | None
        ) = None
        while True:
            try:
                observation = fit(context, tune_z_values=tune_z_values)
            except FitNumericalError:
                if not _can_revise(session):
                    if best is None:
                        raise
                    _score, observation, candidates, parameters = best
                    context.state["matrix_element_candidates"] = candidates
                    context.params.update(parameters)
                    break
                suggestion = revise(context, session, _candidate_attempts(context))
                tune_z_values = _apply_matrix_suggestion(context, scopes, suggestion)
                continue
            candidate_id = str(observation["metrics"]["recommended_candidate_id"])
            candidates = list(context.state.get("matrix_element_candidates", []))
            selected = next(
                (candidate for candidate in candidates if str(candidate.get("id")) == candidate_id),
                observation["metrics"],
            )
            score = (
                _finite_quality(selected.get("min_Q" if scopes == {"qda_ratio"} else "Q")),
                _inverse_finite_quality(selected.get("worst_chi2_dof" if scopes == {"qda_ratio"} else "chi2_dof")),
            )
            if score[0] != -math.inf and (best is None or score >= best[0]):
                parameter_snapshot = {"pt2_windows": copy.deepcopy(context.params["pt2_windows"])}
                if scopes != {"qda_ratio"}:
                    parameter_snapshot["pt3_windows"] = copy.deepcopy(context.params["pt3_windows"])
                best = (
                    score,
                    observation,
                    copy.deepcopy(candidates),
                    parameter_snapshot,
                )
            low_quality = (
                _qda_quality_is_low(context)
                if scopes == {"qda_ratio"}
                else bool(observation["metrics"].get("fallback_no_q_passing", False))
            )
            low_quality = low_quality or score[0] == -math.inf
            if not low_quality:
                break
            if not _can_revise(session):
                if best is None:
                    raise FitNumericalError(
                        "no correlator fit produced a publishable finite-Q result after the allowed attempts"
                    )
                _score, observation, candidates, parameters = best
                context.state["matrix_element_candidates"] = candidates
                context.params.update(parameters)
                break
            suggestion = revise(context, session, _candidate_attempts(context))
            tune_z_values = _apply_matrix_suggestion(context, scopes, suggestion)
        candidate_id = str(observation["metrics"]["recommended_candidate_id"])
        final_low_quality = (
            _qda_quality_is_low(context)
            if scopes == {"qda_ratio"}
            else bool(observation["metrics"].get("fallback_no_q_passing", False))
        )
    context.state["fallback_no_q_passing"] = final_low_quality
    if final_low_quality:
        warning(
            "all correlator fit candidates remain below "
            f"q_min={context.params['q_min']} after the allowed attempts; continuing with {candidate_id}."
        )
    publish(context, candidate_id=candidate_id)


__all__ = ["run"]
