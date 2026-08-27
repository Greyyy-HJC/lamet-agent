"""Deterministic Fourier scan with bounded tail-range recommendations."""

from __future__ import annotations

import math
from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.stages.fourier_transform._inspection import run as inspect
from lamet_agent.stages.fourier_transform._scan import attempt, publish
from lamet_agent.stages.fourier_transform.recommendation import revise


def _attempts(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    fields = (
        "z_min_fm",
        "z_max_fm",
        "order",
        "prior_width",
        "model_id",
        "smoothing_method",
        "smoothing_width_fm",
    )
    quality = ("Q", "chi2", "dof", "chi2_dof", "logGBF", "fit_success", "error")
    candidates = [*result.get("range_candidates", []), *result.get("model_candidates", [])]
    for index, candidate in enumerate(candidates, start=1):
        label = str(candidate.get("label", candidate.get("model_id", f"fourier_{index:03d}")))
        key = label if label not in records else f"{label}_{index:03d}"
        records[key] = {
            "parameters": {name: candidate[name] for name in fields if name in candidate},
            **{name: candidate[name] for name in quality if name in candidate},
        }
    return records


def _accepted(result: dict[str, Any], q_min: float) -> bool:
    return any(
        candidate.get("error") is None
        and candidate.get("Q") is not None
        and math.isfinite(float(candidate["Q"]))
        and float(candidate["Q"]) >= q_min
        for candidate in result.get("model_candidates", [])
    )


def run(context: ToolContext, session: LlmSession) -> None:
    """Try authored/recommended ranges, then at most the job recommendation budget."""
    inspect(context)
    q_min = float(context.params["scheme_scan"]["q_min"])
    history = []
    while True:
        try:
            result = attempt(context)
            attempts = _attempts(result)
        except (FitNumericalError, ValueError) as exc:
            result = None
            attempts = {
                f"attempt_{len(history) + 1:02d}": {
                    "parameters": {
                        "zmin_fm": list(context.params["zmin_fm"]),
                        "zmax_fm": list(context.params["zmax_fm"]),
                    },
                    "numerical_failure": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            }
        history.append(attempts)
        if result is not None and _accepted(result, q_min):
            context.state["fourier_parameter_attempts"] = history
            publish(context, result)
            return
        if session.recommendation_calls >= session.max_recommendation_calls:
            context.state["fourier_parameter_attempts"] = history
            raise FitNumericalError("all Fourier candidates remain below q_min after the allowed attempts")
        suggestion = revise(context, session, attempts)
        context.params["zmin_fm"] = list(suggestion["zmin_fm"])
        context.params["zmax_fm"] = list(suggestion["zmax_fm"])
        inspect(context)


__all__ = ["run"]
