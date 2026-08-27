"""Deterministic reference extrapolation workflows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.extrapolation._fit import run as fit
from lamet_agent.stages.extrapolation._inspection import run as inspect
from lamet_agent.stages.extrapolation._publish import run as publish
from lamet_agent.stages.extrapolation._systematics_budget import run as publish_budget


def select_single_candidate(candidates: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Return the sole fitted candidate and its unit-weight comparison record."""
    if len(candidates) != 1:
        raise ValueError("the reference extrapolation requires exactly one fitted candidate")
    candidate = candidates[0]
    spread = np.zeros_like(np.asarray(candidate["data"].mean, dtype=float))
    comparison = {
        "candidate_ids": [candidate["id"]],
        "criterion": "single_authored_model",
        "weights": [1.0],
        "between_model_sdev": spread.tolist(),
        "statistical_source": "reference sample-level fit",
        "stability_sigma": 0.0,
        "candidates": [
            {
                "candidate_id": candidate["id"],
                "terms": candidate["terms"],
                "excluded_ensembles": candidate["excluded_ensembles"],
                "chi2": candidate["chi2"],
                "dof": candidate["dof"],
                "chi2_dof": candidate["chi2_dof"],
                "Q": candidate["Q"],
                "aic": candidate["aic"],
                "parameter_mean": candidate["parameter_mean"],
                "parameter_sdev": candidate["parameter_sdev"],
                "momentum_dependence": candidate["momentum_dependence"],
                "weight": 1.0,
            }
        ],
    }
    return candidate["data"], comparison


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


__all__ = ["run", "select_single_candidate"]
