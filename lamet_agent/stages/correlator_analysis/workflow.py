"""Correlator workflow with isolated typed fit-parameter recommendations."""

from __future__ import annotations

from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.stages.correlator_analysis._fit_matrix import run as fit_matrix
from lamet_agent.stages.correlator_analysis._fit_qda import run as fit_qda
from lamet_agent.stages.correlator_analysis._fit_spectrum import run as fit_spectrum
from lamet_agent.stages.correlator_analysis._inspection import run as inspect
from lamet_agent.stages.correlator_analysis._lanczos import run as run_lanczos
from lamet_agent.stages.correlator_analysis._lanczos_inspection import run as inspect_lanczos
from lamet_agent.stages.correlator_analysis._publish import run as publish
from lamet_agent.stages.correlator_analysis.tools._joint_fit_recommendation import initial, revise


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
    return {name: suggestion[name] for name in ("t_min", "t_max", "n_states", "prior_means", "prior_widths")}


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


def run(context: ToolContext, session: LlmSession) -> None:
    """Run deterministic analysis around one optional typed fit suggestion."""
    if context.params["analysis_method"] == "lanczos":
        inspect_lanczos(context)
        run_lanczos(context)
        return

    inspect(context)
    scopes = set(context.params["fit_scope"])
    suggestion = initial(context, session)
    if scopes == {"spectrum"}:
        retried = False
        try:
            observation = fit_spectrum(context, **_spectrum_parameters(suggestion))
        except FitNumericalError as exc:
            retried = True
            suggestion = revise(
                context,
                session,
                _spectrum_attempt(suggestion, error=str(exc)),
            )
            observation = fit_spectrum(context, **_spectrum_parameters(suggestion))
        if not retried and float(observation["metrics"].get("Q", 1.0)) < float(context.params["q_min"]):
            suggestion = revise(
                context,
                session,
                _spectrum_attempt(suggestion, metrics=dict(observation["metrics"])),
            )
            observation = fit_spectrum(context, **_spectrum_parameters(suggestion))
            retried = True
        if float(observation["metrics"].get("Q", 0.0)) < float(context.params["q_min"]):
            raise FitNumericalError("spectrum fit remains below q_min after the allowed recommendation attempts")
        candidate_id = observation["metrics"]["candidate_id"]
    else:
        fit = fit_qda if scopes == {"qda_ratio"} else fit_matrix
        tune_z_values = list(suggestion["tune_z_values"])
        retried = False
        try:
            observation = fit(context, tune_z_values=tune_z_values)
        except FitNumericalError:
            retried = True
            suggestion = revise(context, session, _candidate_attempts(context))
            context.params["pt2_windows"] = list(suggestion["pt2_windows"])
            if scopes != {"qda_ratio"}:
                context.params["pt3_windows"] = list(suggestion["pt3_windows"])
            tune_z_values = list(suggestion["tune_z_values"])
            observation = fit(context, tune_z_values=tune_z_values)
        low_quality = (
            _qda_quality_is_low(context)
            if scopes == {"qda_ratio"}
            else bool(observation["metrics"].get("fallback_no_q_passing", False))
        )
        if low_quality and not retried:
            suggestion = revise(context, session, _candidate_attempts(context))
            context.params["pt2_windows"] = list(suggestion["pt2_windows"])
            if scopes != {"qda_ratio"}:
                context.params["pt3_windows"] = list(suggestion["pt3_windows"])
            tune_z_values = list(suggestion["tune_z_values"])
            observation = fit(context, tune_z_values=tune_z_values)
            retried = True
        final_low_quality = (
            _qda_quality_is_low(context)
            if scopes == {"qda_ratio"}
            else bool(observation["metrics"].get("fallback_no_q_passing", False))
        )
        if final_low_quality:
            raise FitNumericalError("all correlator fit candidates remain below q_min after the allowed attempts")
        candidate_id = observation["metrics"]["recommended_candidate_id"]
    publish(context, candidate_id=str(candidate_id))


__all__ = ["run"]
