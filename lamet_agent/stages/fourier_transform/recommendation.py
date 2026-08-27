"""Shared Fourier tail-range null hooks and bounded runtime revision."""

from __future__ import annotations

from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.fourier_transform.tools.recommend_tail_ranges.recommendation import recommend


_CACHE_KEY = "fourier_tail_range_suggestion"


def initial(context: ToolContext, session: LlmSession) -> dict[str, Any]:
    cached = context.state.get(_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    requested = {name for name in ("zmin_fm", "zmax_fm") if not context.params.get(name)}
    fixed = {name: context.params[name] for name in ("zmin_fm", "zmax_fm") if context.params.get(name)}
    if not requested:
        result: dict[str, Any] = {}
    else:
        result = dict(recommend(context, session, requested_fields=requested, fixed_parameters=fixed))
        context.state.setdefault("tail_range_provenance", []).append(
            {"attempt": "initial", "fixed": fixed, "suggested": dict(result)}
        )
    context.state[_CACHE_KEY] = result
    return result


def revise(context: ToolContext, session: LlmSession, previous_attempts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(
        recommend(
            context,
            session,
            requested_fields={"zmin_fm", "zmax_fm"},
            previous_attempts=previous_attempts,
        )
    )
    context.state[_CACHE_KEY] = result
    context.state.setdefault("tail_range_provenance", []).append(
        {"attempt": "retry", "previous_attempts": previous_attempts, "suggested": dict(result)}
    )
    return result


def zmin_fm(context: ToolContext, session: LlmSession) -> list[float]:
    return list(initial(context, session)["zmin_fm"])


def zmax_fm(context: ToolContext, session: LlmSession) -> list[float]:
    return list(initial(context, session)["zmax_fm"])


__all__ = ["initial", "revise", "zmax_fm", "zmin_fm"]
