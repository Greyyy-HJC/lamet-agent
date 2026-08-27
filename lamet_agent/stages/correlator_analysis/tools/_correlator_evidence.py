"""Shared conversion of prepared correlators to LLM-safe fit evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis._input import ensure_correlators


_CONTEXT_KEY = "correlator_fit_data"


def prepare(context: ToolContext) -> dict[str, Any]:
    """Return coordinates plus central values and errors from EnsembleData averages."""
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
            average = selected.average(sample_error_mode)
            components[component] = str(average)
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


def ensure_context(context: ToolContext, session: LlmSession) -> None:
    """Queue the job's complete correlator fit evidence exactly once."""
    if not session.has_context(_CONTEXT_KEY):
        session.add_context(_CONTEXT_KEY, prepare(context))


__all__ = ["ensure_context", "prepare"]
