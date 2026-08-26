"""Compare declared candidates and build a compact error budget."""

from __future__ import annotations

import numpy as np

from lamet_agent.agent import ToolContext


def run(context: ToolContext, *, candidate_ids: list[str]) -> dict[str, object]:
    """Store normalized candidate weights and between-model spread."""
    if context.params["operation"] != "fit":
        raise ValueError("compare_extrapolations is only available for operation='fit'")
    candidates = {candidate["id"]: candidate for candidate in context.state.get("extrapolation_candidates", [])}
    if len(candidates) != 1:
        raise ValueError("the reference extrapolation requires exactly one fitted candidate")
    candidate = next(iter(candidates.values()))
    if candidate_ids != [candidate["id"]]:
        raise ValueError("candidate_ids must contain the single fitted candidate")
    spread = np.zeros_like(np.asarray(candidate["data"].mean, dtype=float))
    comparison = {"candidate_ids": candidate_ids, "criterion": "single_authored_model", "weights": [1.0], "between_model_sdev": spread.tolist(), "statistical_source": "reference sample-level fit", "stability_sigma": 0.0, "candidates": [{"candidate_id": candidate["id"], "terms": candidate["terms"], "excluded_ensembles": candidate["excluded_ensembles"], "chi2": candidate["chi2"], "dof": candidate["dof"], "chi2_dof": candidate["chi2_dof"], "Q": candidate["Q"], "aic": candidate["aic"], "parameter_mean": candidate["parameter_mean"], "parameter_sdev": candidate["parameter_sdev"], "momentum_dependence": candidate["momentum_dependence"], "weight": 1.0}]}
    context.state["extrapolation_comparison"] = comparison
    context.state["extrapolation_selected_data"] = candidate["data"]
    return {"summary": "selected the single authored extrapolation candidate", "metrics": {"candidate_ids": candidate_ids, "weights": [1.0], "max_between_model_sdev": 0.0}, "state_keys": ["extrapolation_comparison", "extrapolation_selected_data"], "artifacts": []}
