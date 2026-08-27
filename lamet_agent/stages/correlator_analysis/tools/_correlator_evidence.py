"""Shared conversion of prepared correlators to LLM-safe fit evidence."""

from __future__ import annotations

from typing import Any

import gvar
import numpy as np

from lamet_agent.agent import ToolContext


def prepare(context: ToolContext) -> dict[str, Any]:
    """Return coordinates plus central values and errors from EnsembleData averages."""
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    correlators = {}
    for name, data in context.state.get("correlators", {}).items():
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
            components[component] = {
                "mean": np.asarray(gvar.mean(average), dtype=float).tolist(),
                "sdev": np.asarray(gvar.sdev(average), dtype=float).tolist(),
            }
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


__all__ = ["prepare"]
