"""One cached, joint correlator fit-parameter recommendation per attempt."""

from __future__ import annotations

from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools.recommend_matrix_tune_z.recommendation import (
    recommend as recommend_matrix,
)
from lamet_agent.stages.correlator_analysis.tools.recommend_qda_tune_z.recommendation import (
    recommend as recommend_qda,
)
from lamet_agent.stages.correlator_analysis.tools.recommend_spectrum_fit.recommendation import (
    recommend as recommend_spectrum,
)


_CACHE_KEY = "joint_fit_parameter_suggestion"


def _scopes(context: ToolContext) -> set[str]:
    return set(context.params["fit_scope"])


def initial(context: ToolContext, session: LlmSession) -> dict[str, Any]:
    """Return the cached initial joint suggestion, requesting it at most once."""
    cached = context.state.get(_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    scopes = _scopes(context)
    pt2 = context.params.get("pt2_windows")
    if scopes == {"spectrum"}:
        fixed = {"pt2_windows": pt2} if pt2 else {}
        result = dict(recommend_spectrum(context, session, fixed_parameters=fixed))
        if pt2 and not any(
            int(window["tmin"]) == int(result["t_min"]) and int(window["tmax"]) == int(result["t_max"])
            for window in pt2
        ):
            raise ValueError("initial spectrum recommendation must select an authored pt2 window")
        result["pt2_windows"] = [{"tmin": int(result["t_min"]), "tmax": int(result["t_max"])}]
    else:
        ordinary = scopes != {"qda_ratio"}
        requested = {"tune_z_values"}
        fixed = {}
        if pt2:
            fixed["pt2_windows"] = pt2
        else:
            requested.add("pt2_windows")
        pt3 = context.params.get("pt3_windows")
        if ordinary:
            if pt3:
                fixed["pt3_windows"] = pt3
            else:
                requested.add("pt3_windows")
            result = dict(
                recommend_matrix(
                    context,
                    session,
                    requested_fields=requested,
                    fixed_parameters=fixed,
                )
            )
        else:
            result = dict(
                recommend_qda(
                    context,
                    session,
                    requested_fields=requested,
                    fixed_parameters=fixed,
                )
            )
    context.state[_CACHE_KEY] = result
    context.state.setdefault("fit_parameter_provenance", []).append(
        {"attempt": "initial", "fixed": fixed, "suggested": dict(result)}
    )
    return result


def revise(
    context: ToolContext,
    session: LlmSession,
    previous_attempts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Request one runtime override after an unsuccessful complete scan."""
    scopes = _scopes(context)
    if scopes == {"spectrum"}:
        result = dict(recommend_spectrum(context, session, previous_attempts=previous_attempts))
        result["pt2_windows"] = [{"tmin": int(result["t_min"]), "tmax": int(result["t_max"])}]
    elif scopes == {"qda_ratio"}:
        result = dict(
            recommend_qda(
                context,
                session,
                requested_fields={"pt2_windows", "tune_z_values"},
                previous_attempts=previous_attempts,
            )
        )
    else:
        result = dict(
            recommend_matrix(
                context,
                session,
                requested_fields={"pt2_windows", "pt3_windows", "tune_z_values"},
                previous_attempts=previous_attempts,
            )
        )
    context.state[_CACHE_KEY] = result
    context.state.setdefault("fit_parameter_provenance", []).append(
        {"attempt": "retry", "previous_attempts": previous_attempts, "suggested": dict(result)}
    )
    return result


def pt2_windows(context: ToolContext, session: LlmSession) -> list[dict[str, int]]:
    """Null-hook accessor for the joint pt2 recommendation."""
    return list(initial(context, session)["pt2_windows"])


def pt3_windows(context: ToolContext, session: LlmSession) -> list[dict[str, Any]]:
    """Null-hook accessor; scopes without three-point data deterministically use no windows."""
    if _scopes(context) in ({"spectrum"}, {"qda_ratio"}):
        return []
    return list(initial(context, session)["pt3_windows"])


__all__ = ["initial", "pt2_windows", "pt3_windows", "revise"]
