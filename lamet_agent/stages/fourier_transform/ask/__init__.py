"""Lazy shared context and structured asks for Fourier tail ranges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.data import format_gvar
from lamet_agent.stages.fourier_transform._inspection import prepare


_PROMPT_KEY = "fourier_transform_ask_prompt"
_CONTEXT_KEY = "fourier_tail_fit_data"
_CACHE_KEY = "fourier_tail_range_suggestion"


def ensure(context: ToolContext, session: LlmSession) -> None:
    """Initialize this job's base prompt and Fourier evidence exactly once."""
    if not session.has_system_prompt(_PROMPT_KEY):
        prompt = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
        session.add_system_prompt(_PROMPT_KEY, prompt)
    if session.has_context(_CONTEXT_KEY):
        return
    data, z_grid_step = prepare(context)
    mode = str(context.manifest["metadata"]["sample_error_mode"])
    components = {
        name: format_gvar(selected.average(mode)) for name, selected in (("real", data.real), ("imag", data.imag))
    }
    session.add_context(
        _CONTEXT_KEY,
        {
            "z_fm": [float(value) for value in data.coords["z"]],
            "z_grid_step_fm": z_grid_step,
            "lattice_spacing_fm": data.ensemble.a_s,
            "momentum_gev": data.attrs.get("momentum_gev"),
            "components": components,
        },
    )


def initial(context: ToolContext, session: LlmSession) -> dict[str, Any]:
    """Return one cached initial suggestion for missing tail ranges."""
    cached = context.state.get(_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    requested = {name for name in ("zmin_fm", "zmax_fm") if not context.params.get(name)}
    fixed = {name: context.params[name] for name in ("zmin_fm", "zmax_fm") if context.params.get(name)}
    if not requested:
        result: dict[str, Any] = {}
    else:
        from .ask_for_tail_ranges import recommend

        result = dict(recommend(context, session, requested_fields=requested, fixed_parameters=fixed))
        context.state.setdefault("tail_range_provenance", []).append(
            {"attempt": "initial", "fixed": fixed, "suggested": dict(result)}
        )
    context.state[_CACHE_KEY] = result
    return result


def revise(context: ToolContext, session: LlmSession, previous_attempts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Request one runtime tail-range override after an unsuccessful scan."""
    from .ask_for_tail_ranges import recommend

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


__all__ = ["ensure", "initial", "revise", "zmax_fm", "zmin_fm"]
