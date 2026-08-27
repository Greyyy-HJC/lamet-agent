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
from lamet_agent.stages.correlator_analysis.tools.recommend_matrix_tune_z.recommendation import (
    recommend as recommend_matrix,
)
from lamet_agent.stages.correlator_analysis.tools.recommend_qda_tune_z.recommendation import (
    recommend as recommend_qda,
)
from lamet_agent.stages.correlator_analysis.tools.recommend_spectrum_fit.recommendation import (
    recommend as recommend_spectrum,
)


def _candidate_diagnostics(context: ToolContext) -> list[dict[str, Any]]:
    """Return the compact fit-quality evidence used for one recommendation retry."""
    fields = (
        "id",
        "Q",
        "chi2_dof",
        "min_Q",
        "worst_chi2_dof",
        "quality_passed",
        "numerical_failure",
        "feasible_at_all_tune_z",
        "failure_reasons",
        "tune_z_values",
    )
    return [
        {name: candidate[name] for name in fields if name in candidate}
        for candidate in context.state.get("matrix_element_candidates", [])
    ]


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
    if scopes == {"spectrum"}:
        suggestion = recommend_spectrum(context, session)
        retried = False
        try:
            observation = fit_spectrum(context, **suggestion)
        except FitNumericalError as exc:
            retried = True
            suggestion = recommend_spectrum(context, session, diagnostics={"error": str(exc)})
            observation = fit_spectrum(context, **suggestion)
        if not retried and float(observation["metrics"].get("Q", 1.0)) < float(context.params["q_min"]):
            suggestion = recommend_spectrum(context, session, diagnostics=dict(observation["metrics"]))
            observation = fit_spectrum(context, **suggestion)
        candidate_id = observation["metrics"]["candidate_id"]
    else:
        recommend = recommend_qda if scopes == {"qda_ratio"} else recommend_matrix
        fit = fit_qda if scopes == {"qda_ratio"} else fit_matrix
        tune_z_values = recommend(context, session)
        retried = False
        try:
            observation = fit(context, tune_z_values=tune_z_values)
        except FitNumericalError:
            retried = True
            tune_z_values = recommend(context, session, diagnostics=_candidate_diagnostics(context))
            observation = fit(context, tune_z_values=tune_z_values)
        low_quality = (
            _qda_quality_is_low(context)
            if scopes == {"qda_ratio"}
            else bool(observation["metrics"].get("fallback_no_q_passing", False))
        )
        if low_quality and not retried:
            tune_z_values = recommend(context, session, diagnostics=_candidate_diagnostics(context))
            observation = fit(context, tune_z_values=tune_z_values)
        candidate_id = observation["metrics"]["recommended_candidate_id"]
    publish(context, candidate_id=str(candidate_id))


__all__ = ["run"]
