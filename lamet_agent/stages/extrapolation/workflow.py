"""Deterministic reference extrapolation workflows."""

from __future__ import annotations

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.extrapolation._fit import run as fit
from lamet_agent.stages.extrapolation._inspection import run as inspect
from lamet_agent.stages.extrapolation._publish import run as publish
from lamet_agent.stages.extrapolation._systematics_budget import run as publish_budget
from lamet_agent.stages.extrapolation.selection import select_single_candidate


def run(context: ToolContext, _session: LlmSession) -> None:
    """Execute a fixed reference model or authored systematics budget."""
    if context.params["operation"] == "systematics_budget":
        publish_budget(context)
        return
    inspect(context)
    fit(context, excluded_ensembles=[])
    selected, comparison = select_single_candidate(context.state.get("extrapolation_candidates", []))
    context.state["extrapolation_selected_data"] = selected
    context.state["extrapolation_comparison"] = comparison
    publish(context)


__all__ = ["run"]
