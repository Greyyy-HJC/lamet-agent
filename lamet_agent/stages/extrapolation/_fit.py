"""Fit one authored extrapolation term set."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.extrapolation.physics import fit_candidate


def run(context: ToolContext, *, excluded_ensembles: list[str]) -> dict[str, object]:
    """Append a sample-bearing physical-point candidate."""
    if context.params["operation"] != "fit":
        raise ValueError("fit_extrapolation_candidate is only available for operation='fit'")
    params = context.params
    independent_terms = list(params["x_independent_terms"])
    dependent_terms = list(params["x_dependent_terms"])
    terms = [*independent_terms, *dependent_terms]
    data = context.state.get("scaling_data")
    if not data:
        raise RuntimeError("inspect_scaling must run before fit_extrapolation_candidate")
    excluded = set(excluded_ensembles)
    if excluded:
        raise ValueError("the reference extrapolation uses every authored input")
    selected = list(data)
    x_coordinates = [float(value) for value in selected[0].coords["x"]]
    x_range = (min(x_coordinates), max(x_coordinates))
    priors = params["priors"]
    physical_mass = params["physical_pion_mass_gev"] if set(terms) & {"mpi2", "mpi4_log_mpi2"} else None
    result, fit = fit_candidate(
        selected,
        terms,
        None if physical_mass is None else float(physical_mass),
        priors,
        x_range=x_range,
        x_independent_terms=independent_terms,
        x_covariance=bool(params["x_covariance"]),
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
            "x_independent_terms": independent_terms,
            "x_dependent_terms": dependent_terms,
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
