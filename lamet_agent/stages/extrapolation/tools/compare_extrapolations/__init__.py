"""Compare declared candidates and build a compact error budget."""

from __future__ import annotations

import numpy as np
import gvar as gv

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData


def run(context: ToolContext, *, candidate_ids: list[str]) -> dict[str, object]:
    """Store normalized candidate weights and between-model spread."""
    candidates = {candidate["id"]: candidate for candidate in context.state.get("extrapolation_candidates", [])}
    if not candidate_ids or any(candidate_id not in candidates for candidate_id in candidate_ids):
        raise ValueError("candidate_ids must name existing extrapolation candidates")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    selected = [candidates[candidate_id] for candidate_id in candidate_ids]
    minimum_q = float(context.params["model_selection"]["min_Q"])
    eligible_ids = {candidate_id for candidate_id, candidate in candidates.items() if float(candidate["Q"]) >= minimum_q}
    if set(candidate_ids) != eligible_ids:
        raise ValueError("candidate_ids must contain every fitted candidate that passes model_selection.min_Q")
    rejected = [candidate["id"] for candidate in selected if float(candidate["Q"]) < minimum_q]
    if rejected:
        raise ValueError(f"selected candidates fail model_selection.min_Q: {rejected}")
    aic = np.asarray([float(candidate["aic"]) for candidate in selected])
    weights = np.exp(-0.5 * (aic - np.min(aic)))
    weights = weights / np.sum(weights)
    candidate_gvars = [np.asarray(candidate["data"].gvar, dtype=object) for candidate in selected]
    candidate_means = np.stack([np.asarray(gv.mean(value), dtype=float) for value in candidate_gvars])
    candidate_sdevs = np.stack([np.asarray(gv.sdev(value), dtype=float) for value in candidate_gvars])
    stability = 0.0
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            denominator = np.sqrt(candidate_sdevs[left] ** 2 + candidate_sdevs[right] ** 2)
            ratio = np.divide(np.abs(candidate_means[left] - candidate_means[right]), denominator, out=np.full_like(denominator, np.inf), where=denominator > 0)
            stability = max(stability, float(np.max(ratio)))
    if stability > float(context.params["model_selection"]["stability_sigma"]):
        raise ValueError(f"selected candidates differ by {stability:.3g} sigma, above stability_sigma")
    mean = np.tensordot(weights, candidate_means, axes=(0, 0))
    flat_mean = mean.reshape(-1)
    covariance = np.zeros((flat_mean.size, flat_mean.size), dtype=float)
    for weight, value, candidate_mean in zip(weights, candidate_gvars, candidate_means):
        difference = candidate_mean.reshape(-1) - flat_mean
        covariance += weight * (gv.evalcov(value.reshape(-1)) + np.outer(difference, difference))
    averaged_values = gv.gvar(flat_mean, covariance).reshape(mean.shape)
    between = np.tensordot(weights, (candidate_means - mean[None, ...]) ** 2, axes=(0, 0))
    spread = np.sqrt(between)
    template = selected[0]["data"]
    attrs = template.attrs
    attrs.update({"model_selection_criterion": "aic", "model_candidate_ids": ",".join(candidate_ids), "between_model_sdev_max": float(np.max(spread))})
    averaged = EnsembleData(None, "gvar", averaged_values, template.dims, template.coords, attrs=attrs, name="physical_distribution")
    comparison = {"candidate_ids": candidate_ids, "criterion": "aic", "weights": weights.tolist(), "between_model_sdev": spread.tolist(), "statistical_source": "within-model covariance plus between-model covariance", "stability_sigma": stability, "candidates": [{"candidate_id": candidate["id"], "terms": candidate["terms"], "excluded_ensembles": candidate["excluded_ensembles"], "chi2": candidate["chi2"], "dof": candidate["dof"], "chi2_dof": candidate["chi2_dof"], "Q": candidate["Q"], "aic": candidate["aic"], "weight": float(weight)} for candidate, weight in zip(selected, weights)]}
    context.state["extrapolation_comparison"] = comparison
    context.state["extrapolation_selected_data"] = averaged
    return {"summary": f"compared {len(selected)} extrapolation candidates", "metrics": {"candidate_ids": candidate_ids, "weights": weights.tolist(), "max_between_model_sdev": float(np.max(spread))}, "state_keys": ["extrapolation_comparison", "extrapolation_selected_data"], "artifacts": []}
