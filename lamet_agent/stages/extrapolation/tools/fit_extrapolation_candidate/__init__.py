"""Fit one authored extrapolation term set."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.extrapolation.physics import fit_candidate


def run(context: ToolContext, *, terms: list[str], excluded_ensembles: list[str]) -> dict[str, object]:
    """Append a sample-bearing physical-point candidate."""
    allowed = set(context.params["allowed_terms"])
    required = set(context.params["required_terms"])
    if len(set(terms)) != len(terms) or not required.issubset(terms) or not set(terms).issubset(required | allowed) or len(terms) > context.params["max_terms"]:
        raise ValueError("terms must contain required terms, use only allowed terms, and respect max_terms")
    data = context.state.get("scaling_data")
    if not data:
        raise RuntimeError("inspect_scaling must run before fit_extrapolation_candidate")
    allowed_exclusions = set(context.params["fit_ranges"]["allowed_exclusions"])
    excluded = set(excluded_ensembles)
    if not excluded.issubset(allowed_exclusions):
        raise ValueError("excluded_ensembles must be selected from fit_ranges.allowed_exclusions")
    selected = [item for item in data if str(item.attrs.get("ensemble_id", item.attrs.get("ensemble"))) not in excluded]
    actual_ids = {str(item.attrs.get("ensemble_id", item.attrs.get("ensemble"))) for item in data}
    if not excluded.issubset(actual_ids):
        raise ValueError("excluded_ensembles contains an id absent from the inspected inputs")
    if len(selected) <= len(terms) + 1:
        raise ValueError("candidate needs more inputs than its intercept and correction coefficients")
    x_range = tuple(float(value) for value in context.params["fit_ranges"]["x"])
    priors = {term: context.params["priors"][term] for term in terms}
    x_dependence = {term: context.params["x_dependence"][term] for term in terms}
    result, fit = fit_candidate(selected, terms, float(context.params["physical_pion_mass_gev"]), priors, x_range=x_range, x_dependence=x_dependence)
    candidates = context.state.setdefault("extrapolation_candidates", [])
    candidate_id = f"extrapolation_{len(candidates) + 1:03d}"
    candidates.append({"id": candidate_id, "terms": list(terms), "x_dependence": x_dependence, "excluded_ensembles": list(excluded_ensembles), "data": result, **fit})
    return {"summary": f"stored extrapolation candidate {candidate_id}", "metrics": {"candidate_id": candidate_id, "term_count": len(terms), "input_count": len(selected), **fit}, "state_keys": ["extrapolation_candidates"], "artifacts": []}
