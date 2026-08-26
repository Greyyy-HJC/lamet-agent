"""Fit one authored extrapolation term set."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.extrapolation.physics import fit_candidate


def run(context: ToolContext, *, terms: list[str], excluded_ensembles: list[str]) -> dict[str, object]:
    """Append a sample-bearing physical-point candidate."""
    if context.params["operation"] != "fit":
        raise ValueError("fit_extrapolation_candidate is only available for operation='fit'")
    params = context.params["fit"]
    allowed = set(params["allowed_terms"])
    required = set(params["required_terms"])
    if (
        len(set(terms)) != len(terms)
        or not required.issubset(terms)
        or not set(terms).issubset(required | allowed)
        or len(terms) > params["max_terms"]
    ):
        raise ValueError("terms must contain required terms, use only allowed terms, and respect max_terms")
    data = context.state.get("scaling_data")
    if not data:
        raise RuntimeError("inspect_scaling must run before fit_extrapolation_candidate")
    excluded = set(excluded_ensembles)
    if excluded:
        raise ValueError("the reference extrapolation uses every authored input")
    selected = list(data)
    if len(selected) <= len(terms) + 1:
        raise ValueError("candidate needs more inputs than its intercept and correction coefficients")
    x_coordinates = [float(value) for value in selected[0].coords["x"]]
    x_range = (min(x_coordinates), max(x_coordinates))
    priors = params["priors"]
    x_dependence = {term: params["x_dependence"][term] for term in terms}
    physical_mass = params.get("physical_pion_mass_gev")
    result, fit = fit_candidate(
        selected,
        terms,
        None if physical_mass is None else float(physical_mass),
        priors,
        x_range=x_range,
        x_dependence=x_dependence,
        pdep_gev=[float(value) for value in params["pdep_gev"]],
        posterior_prior_error_scale=float(params["posterior_prior_error_scale"]),
        workers=context.workers,
        _parallel=context._parallel,
    )
    candidates = context.state.setdefault("extrapolation_candidates", [])
    candidate_id = f"extrapolation_{len(candidates) + 1:03d}"
    candidates.append(
        {
            "id": candidate_id,
            "terms": list(terms),
            "x_dependence": x_dependence,
            "excluded_ensembles": list(excluded_ensembles),
            "data": result,
            **fit,
        }
    )
    return {
        "summary": f"stored extrapolation candidate {candidate_id}",
        "metrics": {"candidate_id": candidate_id, "term_count": len(terms), "input_count": len(selected), **fit},
        "state_keys": ["extrapolation_candidates"],
        "artifacts": [],
    }
