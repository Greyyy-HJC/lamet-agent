"""Lazy shared context and structured asks for correlator fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.data import format_gvar
from lamet_agent.stages.correlator_analysis._input import ensure_correlators


_CONTEXT_KEY = "correlator_fit_data"
_CACHE_KEY = "joint_fit_parameter_suggestion"


def _context(context: ToolContext) -> dict[str, Any]:
    """Return coordinates plus compact central values and errors for one job."""
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    correlators = {}
    for name, data in ensure_correlators(context).items():
        components = {}
        selected_components = (
            ("real", "imag")
            if np.iscomplexobj(data.values) and context.params["component"] == "both"
            else ("imag",)
            if np.iscomplexobj(data.values) and context.params["component"] == "im"
            else ("real",)
        )
        for component in selected_components:
            selected = data.imag if component == "imag" else data.real
            components[component] = format_gvar(selected.average(sample_error_mode))
        correlators[name] = {
            "dims": data.dims,
            "coords": data.coords,
            "components": components,
        }
    return {
        "inspection": context.state.get("inspection", {}),
        "correlators": correlators,
        "params": context.params,
    }


def ensure(context: ToolContext, session: LlmSession) -> None:
    """Initialize this job's correlator evidence exactly once."""
    if not session.has_context(_CONTEXT_KEY):
        session.add_context(_CONTEXT_KEY, _context(context))


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
        from .ask_for_spectrum_fit import recommend

        fixed = {"pt2_windows": pt2} if pt2 else {}
        result = dict(recommend(context, session, fixed_parameters=fixed))
        if pt2 and not any(
            int(window["tmin"]) == int(result["tmin"]) and int(window["tmax"]) == int(result["tmax"]) for window in pt2
        ):
            raise ValueError("initial spectrum recommendation must select an authored pt2 window")
        result["pt2_windows"] = [{"tmin": int(result["tmin"]), "tmax": int(result["tmax"])}]
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
            from .ask_for_matrix_tune_z import recommend

            if pt3:
                fixed["pt3_windows"] = pt3
            else:
                requested.add("pt3_windows")
        else:
            from .ask_for_qda_tune_z import recommend

        result = dict(
            recommend(
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
        from .ask_for_spectrum_fit import recommend

        result = dict(recommend(context, session, previous_attempts=previous_attempts))
        result["pt2_windows"] = [{"tmin": int(result["tmin"]), "tmax": int(result["tmax"])}]
    elif scopes == {"qda_ratio"}:
        from .ask_for_qda_tune_z import recommend

        result = dict(
            recommend(
                context,
                session,
                requested_fields={"pt2_windows", "tune_z_values"},
                previous_attempts=previous_attempts,
            )
        )
    else:
        from .ask_for_matrix_tune_z import recommend

        result = dict(
            recommend(
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
    """Return no three-point windows for scopes without three-point data."""
    if _scopes(context) in ({"spectrum"}, {"qda_ratio"}):
        return []
    return list(initial(context, session)["pt3_windows"])


__all__ = ["ensure", "initial", "pt2_windows", "pt3_windows", "revise"]
