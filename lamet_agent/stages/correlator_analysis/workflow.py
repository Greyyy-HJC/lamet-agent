"""Correlator workflow with isolated typed fit-parameter recommendations."""

from __future__ import annotations

import copy
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
from lamet_agent.stages.correlator_analysis._selection import select_spectrum_candidate, select_tuned_candidate
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


def _can_revise(session: LlmSession) -> bool:
    return session.recommendation_calls < session.max_recommendation_calls


def _apply_matrix_suggestion(context: ToolContext, scopes: set[str], suggestion: dict[str, Any]) -> list[Any]:
    context.params["pt2_windows"] = list(suggestion["pt2_windows"])
    if scopes != {"qda_ratio"}:
        context.params["pt3_windows"] = list(suggestion["pt3_windows"])
    context.params["tune_z_values"] = list(suggestion["tune_z_values"])
    return list(context.params["tune_z_values"])


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
        last_error: Exception | None = None
        while True:
            try:
                observation = fit_spectrum(context, **_spectrum_parameters(suggestion))
            except (FitNumericalError, ValueError) as exc:
                last_error = exc
                previous = _spectrum_attempt(suggestion, error=f"{type(exc).__name__}: {exc}")
            else:
                last_error = None
                previous = _spectrum_attempt(suggestion, metrics=dict(observation["metrics"]))
                try:
                    _selected, fallback = select_spectrum_candidate(
                        list(context.state.get("spectrum_candidates", [])), q_min=q_min
                    )
                except ValueError:
                    fallback = True
                if not fallback:
                    break
            if not _can_revise(session):
                break
            suggestion = revise(context, session, previous)
            _apply_spectrum_suggestion(context, suggestion)
        try:
            selected, final_low_quality = select_spectrum_candidate(
                list(context.state.get("spectrum_candidates", [])), q_min=q_min
            )
        except ValueError as exc:
            raise FitNumericalError(
                "no spectrum fit produced a publishable finite-Q result after the allowed attempts"
            ) from (last_error or exc)
        candidate_id = str(selected["id"])
        window = dict(selected["window"])
        context.params["pt2_windows"] = [{"tmin": int(window["tmin"]), "tmax": int(window["tmax"])}]
    else:
        fit = fit_qda if scopes == {"qda_ratio"} else fit_matrix
        qda = scopes == {"qda_ratio"}
        q_min = float(context.params["q_min"])
        tolerance = float(context.params["chi2_dof_tolerance"])
        tune_z_values = list(suggestion["tune_z_values"])
        context.params["tune_z_values"] = list(tune_z_values)
        attempts: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
        last_error = None
        while True:
            try:
                fit(context, tune_z_values=tune_z_values)
            except (FitNumericalError, ValueError) as exc:
                last_error = exc
            else:
                candidates = list(context.state.get("matrix_element_candidates", []))
                try:
                    selected, fallback = select_tuned_candidate(
                        candidates, q_min=q_min, chi2_dof_tolerance=tolerance, qda=qda
                    )
                except ValueError as exc:
                    last_error = exc
                else:
                    copied = copy.deepcopy(candidates)
                    copied_selected = next(candidate for candidate in copied if candidate["id"] == selected["id"])
                    parameters = {
                        "pt2_windows": copy.deepcopy(context.params["pt2_windows"]),
                        "tune_z_values": copy.deepcopy(context.params["tune_z_values"]),
                    }
                    if not qda:
                        parameters["pt3_windows"] = copy.deepcopy(context.params["pt3_windows"])
                    attempts.append((copied_selected, copied, parameters))
                    if not fallback:
                        final_low_quality = False
                        chosen = attempts[-1]
                        break
            if not _can_revise(session):
                if not attempts:
                    raise FitNumericalError(
                        "no correlator fit produced a publishable finite-Q result after the allowed attempts"
                    ) from last_error
                retained_candidates = []
                for attempt_number, (_representative, candidates, _parameters) in enumerate(attempts, start=1):
                    for candidate in candidates:
                        if len(attempts) > 1:
                            candidate["id"] = f"attempt_{attempt_number:03d}_{candidate['id']}"
                        retained_candidates.append(candidate)
                selected, _fallback = select_tuned_candidate(
                    retained_candidates, q_min=q_min, chi2_dof_tolerance=tolerance, qda=qda
                )
                parameters = next(
                    parameters
                    for _representative, candidates, parameters in attempts
                    if any(candidate is selected for candidate in candidates)
                )
                chosen = (selected, retained_candidates, parameters)
                final_low_quality = True
                break
            suggestion = revise(context, session, _candidate_attempts(context))
            tune_z_values = _apply_matrix_suggestion(context, scopes, suggestion)
        selected, candidates, parameters = chosen
        context.state["matrix_element_candidates"] = candidates
        context.params.update(parameters)
        candidate_id = str(selected["id"])
    context.state["fallback_no_q_passing"] = final_low_quality
    if final_low_quality:
        warning(
            "all correlator fit candidates remain below "
            f"q_min={context.params['q_min']} after the allowed attempts; continuing with {candidate_id}."
        )
    publish(context, candidate_id=candidate_id)


__all__ = ["run"]
