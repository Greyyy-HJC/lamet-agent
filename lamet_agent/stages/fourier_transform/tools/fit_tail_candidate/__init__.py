"""Fit and store one connected long-distance tail candidate."""

from __future__ import annotations

from typing import Literal

from lamet_agent.agent import ToolContext
from lamet_agent.stages.fourier_transform.physics import extend_tail, fit_tail_parameters


def run(context: ToolContext, *, model_id: str, z_min_fm: float, z_max_fm: float, smoothing_method: Literal["linear", "cosine"], smoothing_width_fm: float, prior_means: dict[str, float], prior_widths: dict[str, float]) -> dict[str, object]:
    """Store a tail candidate for later terminal transformation."""
    data = context.state.get("fourier_input")
    if data is None:
        raise RuntimeError("inspect_long_distance must run before fit_tail_candidate")
    if model_id not in context.params["tail_models"]:
        raise ValueError(f"tail model '{model_id}' is not allowed")
    if z_min_fm not in context.params["zmin_fm"] or z_max_fm not in context.params["zmax_fm"]:
        raise ValueError("tail range values must be selected from the authored candidates")
    if z_min_fm >= z_max_fm or smoothing_width_fm <= 0 or z_min_fm + smoothing_width_fm > z_max_fm:
        raise ValueError("tail range and smoothing overlap are invalid")
    if smoothing_method not in context.params["smoothing"]["smooth"] or smoothing_width_fm not in context.params["smoothing"]["widths_fm"]:
        raise ValueError("smoothing choices must be selected from the authored candidates")
    if z_max_fm > max(abs(float(value)) for value in data.coords["z"]):
        raise ValueError("tail upper range is outside the input coordinate coverage")
    parameters, fit = fit_tail_parameters(data, model_id=model_id, z_min_fm=z_min_fm, z_max_fm=z_max_fm, prior_means=prior_means, prior_widths=prior_widths, workers=context.workers)
    candidate = extend_tail(data, z_max_fm=float(context.params["zmax_ext_fm"]), z_min_fm=z_min_fm, smoothing_method=smoothing_method, smoothing_width_fm=smoothing_width_fm, model_id=model_id, tail_parameters=parameters)
    candidates = context.state.setdefault("tail_candidates", [])
    candidate_id = f"tail_{len(candidates) + 1:03d}"
    candidates.append({"id": candidate_id, "model_id": model_id, "z_min_fm": z_min_fm, "z_max_fm": z_max_fm, "zmax_ext_fm": context.params["zmax_ext_fm"], "smoothing_method": smoothing_method, "data": candidate, "parameters": parameters, "prior_means": dict(prior_means), "prior_widths": dict(prior_widths), **fit})
    return {"summary": f"stored tail candidate {candidate_id}", "metrics": {"candidate_id": candidate_id, "model_id": model_id, "z_count": len(candidate.coords["z"]), **fit}, "state_keys": ["tail_candidates"], "artifacts": []}
