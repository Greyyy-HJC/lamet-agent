"""Deterministic Fourier scan with bounded tail-range recommendations."""

from __future__ import annotations

import math
from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.ui import warning
from lamet_agent.stages.fourier_transform._inspection import run as inspect
from lamet_agent.stages.fourier_transform._scan import attempt, publish
from lamet_agent.stages.fourier_transform.ask import revise


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


def _selected_quality(result: dict[str, Any]) -> float | None:
    candidate = result.get("selected_candidate")
    if not isinstance(candidate, dict) or candidate.get("error") is not None or candidate.get("Q") is None:
        return None
    try:
        quality = float(candidate["Q"])
    except (TypeError, ValueError):
        return None
    return quality if math.isfinite(quality) else None


def run(context: ToolContext, session: LlmSession) -> None:
    """Try authored/recommended ranges, then at most the job recommendation budget."""
    inspect(context)
    q_min = float(context.params["scheme_scan"]["q_min"])
    history = []
    best_result: dict[str, Any] | None = None
    best_quality: float | None = None
    best_parameters: tuple[list[Any], list[Any]] | None = None
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
        quality = None
        if result is not None:
            quality = _selected_quality(result)
            if quality is not None and (best_quality is None or quality > best_quality):
                best_result = result
                best_quality = quality
                best_parameters = (list(context.params["zmin_fm"]), list(context.params["zmax_fm"]))
        if result is not None and quality is not None and quality >= q_min:
            context.state["fourier_parameter_attempts"] = history
            context.state["fallback_no_q_passing"] = False
            publish(context, result)
            return
        if session.recommendation_calls >= session.max_recommendation_calls:
            context.state["fourier_parameter_attempts"] = history
            if best_result is None:
                raise FitNumericalError(
                    "no Fourier scan produced a publishable numerical result after the allowed attempts"
                )
            if best_parameters is None:
                raise RuntimeError("a retained Fourier result must include its effective range parameters")
            context.params["zmin_fm"], context.params["zmax_fm"] = best_parameters
            context.state["fallback_no_q_passing"] = True
            warning(
                "all Fourier candidates remain below "
                f"q_min={q_min} after the allowed attempts; continuing with the best available scan."
            )
            publish(context, best_result)
            return
        suggestion = revise(context, session, attempts)
        context.params["zmin_fm"] = list(suggestion["zmin_fm"])
        context.params["zmax_fm"] = list(suggestion["zmax_fm"])
        inspect(context)


__all__ = ["run"]
