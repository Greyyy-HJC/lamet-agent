"""Fit and store one connected long-distance tail candidate."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.fourier_transform.physics import extend_tail, fit_tail_parameters


def run(
    context: ToolContext,
    *,
    model_id: str,
    z_min_fm: float,
    z_max_fm: float,
    prior_means: dict[str, float],
    prior_widths: dict[str, float],
) -> dict[str, object]:
    """Store one reference-compatible tail candidate for later transformation."""
    data = context.state.get("fourier_input")
    if data is None:
        raise RuntimeError("inspect_long_distance must run before fit_tail_candidate")
    conventions = context.state.get("fourier_conventions")
    if not isinstance(conventions, dict):
        raise RuntimeError("inspect_long_distance did not derive Fourier conventions")
    if model_id not in conventions["tail_models"]:
        raise ValueError(f"tail model '{model_id}' is not allowed")
    if z_min_fm not in context.params["zmin_fm"] or z_max_fm not in context.params["zmax_fm"]:
        raise ValueError("tail range values must be selected from the authored candidates")
    if z_min_fm >= z_max_fm:
        raise ValueError("tail range must be increasing")
    effective_z_min = float(z_min_fm)
    if z_max_fm > max(abs(float(value)) for value in data.coords["z"]):
        raise ValueError("tail upper range is outside the input coordinate coverage")
    scan = context.params["scheme_scan"]
    order = str(scan["order"][0])
    observable = context.manifest["metadata"]["target_observable"].upper()
    sector = str(scan["sector"])
    hadron = str(data.attrs.get("hadron", ""))
    is_da = observable.lower() == "da"
    da = context.params["da"] if is_da else None
    psi1_flavor_class = da["psi1_flavor_class"] if da is not None else "heavy"
    psi2_flavor_class = da["psi2_flavor_class"] if da is not None else "heavy"
    parameters, fit = fit_tail_parameters(
        data,
        model_id=model_id,
        z_min_fm=effective_z_min,
        z_max_fm=z_max_fm,
        prior_means=prior_means,
        prior_widths=prior_widths,
        order=order,
        component=str(conventions["component"]),
        lambda0_gev=float(scan["Lambda0_gev"]),
        observable=observable,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        sector=sector,
        hadron=hadron,
        workers=context.workers,
    )
    candidate = extend_tail(
        data,
        z_max_fm=float(context.params["zmax_ext_fm"]),
        z_min_fm=effective_z_min,
        smoothing_method=str(context.params["smooth"]),
        smoothing_width_fm=z_max_fm - effective_z_min,
        model_id=model_id,
        tail_parameters=parameters,
        order=order,
        observable=observable,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        sector=sector,
        hadron=hadron,
    )
    candidates = context.state.setdefault("tail_candidates", [])
    candidate_id = f"tail_{len(candidates) + 1:03d}"
    candidates.append(
        {
            "id": candidate_id,
            "model_id": model_id,
            "z_min_fm": effective_z_min,
            "z_max_fm": z_max_fm,
            "zmax_ext_fm": context.params["zmax_ext_fm"],
            "smoothing_method": context.params["smooth"],
            "data": candidate,
            "parameters": parameters,
            "prior_means": dict(prior_means),
            "prior_widths": dict(prior_widths),
            **fit,
        }
    )
    return {
        "summary": f"stored tail candidate {candidate_id}",
        "metrics": {"candidate_id": candidate_id, "model_id": model_id, "z_count": len(candidate.coords["z"]), **fit},
        "state_keys": ["tail_candidates"],
        "artifacts": [],
    }
