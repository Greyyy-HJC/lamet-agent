"""Tune the complete authored qDA candidate grid on sample averages."""

from __future__ import annotations

from itertools import product

from lamet_agent.agent import ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.stages.correlator_analysis.physics import (
    matrix_element_samples,
)
from lamet_agent.stages.correlator_analysis._selection import (
    select_tuned_candidate,
)


def run(context: ToolContext, *, tune_z_values: list[float]) -> dict[str, object]:
    """Evaluate every authored qDA window/model candidate."""
    correlators = context.state.get("correlators")
    if not isinstance(correlators, dict):
        raise RuntimeError("correlator inspection must run before qDA tuning")
    settings = context.params
    if settings["fit_scope"] != ["qda_ratio"]:
        raise ValueError("qDA tuning requires fit_scope=['qda_ratio']")
    sources = [value for value in correlators.values() if value.attrs.get("correlator_type") == "qda"]
    if len(sources) != 1:
        raise ValueError("qDA tuning requires exactly one selected qDA correlator")
    available_z = [float(value) for value in sources[0].coords["z"]]
    if not tune_z_values or len(set(tune_z_values)) != len(tune_z_values):
        raise ValueError("tune_z_values must be a nonempty unique list")
    tune_z_values = [float(value) for value in tune_z_values]
    if any(not any(abs(value - available) <= 1e-12 for available in available_z) for value in tune_z_values):
        raise ValueError("every tune_z_values entry must name an available qDA z coordinate")
    candidates: list[dict[str, object]] = []
    authored = sorted(
        product(
            settings["fit_strategy"],
            context.params["nstate"],
            settings["prior_width"],
            settings["pt2_windows"],
        ),
        key=lambda item: (
            str(item[0]),
            int(item[1]),
            float(item[2]),
            int(item[3]["tmin"]),
            int(item[3]["tmax"]),
        ),
    )
    for strategy, nstate, prior_width, window in authored:
        candidate_id = f"matrix_{len(candidates) + 1:03d}"
        metadata = {
            "id": candidate_id,
            "method": "qda",
            "fit_strategy": str(strategy),
            "fit_scope": "qda_ratio",
            "observable": "matrix_element",
            "window": {
                "tmin": int(window["tmin"]),
                "tmax": int(window["tmax"]),
                "tau_min": None,
            },
            "component": context.params["component"],
            "nstate": int(nstate),
            "prior_width": float(prior_width),
        }
        per_z: dict[str, dict[str, object]] = {}
        failures: dict[str, str] = {}
        for tune_z in tune_z_values:
            try:
                values, _coordinates, fit = matrix_element_samples(
                    correlators,
                    method="qda",
                    tmin=int(window["tmin"]),
                    tmax=int(window["tmax"]),
                    tau_min=None,
                    lsqfit=settings,
                    sample_error_mode=str(context.manifest["metadata"]["sample_error_mode"]),
                    workers=context.workers,
                    tune_z=tune_z,
                    fit_samples=False,
                    n_states=int(nstate),
                    prior_width=float(prior_width),
                    _parallel=context._parallel,
                )
            except FitNumericalError as exc:
                failures[str(tune_z)] = str(exc)
                continue
            if values is not None:
                raise RuntimeError("qDA candidate tuning must not produce full sample values")
            per_z[str(tune_z)] = fit
        primary = per_z.get(str(tune_z_values[0]))
        usable = list(per_z.values())
        candidate = {
            **metadata,
            "tune_z_values": tune_z_values,
            "tune_z_diagnostics": per_z,
            "feasible_at_all_tune_z": not failures and len(per_z) == len(tune_z_values),
            "failure_reasons": failures,
            "numerical_failure": bool(failures),
            "min_Q": min(float(fit["Q"]) for fit in usable) if usable else None,
            "worst_chi2_dof": max(float(fit["chi2_dof"]) for fit in usable) if usable else None,
        }
        if primary is not None:
            candidate.update(primary)
            candidate["quality_passed"] = float(primary["Q"]) >= float(settings["q_min"])
        else:
            candidate["quality_passed"] = False
        candidates.append(candidate)
    context.state["matrix_element_candidates"] = candidates
    try:
        recommended, fallback = select_tuned_candidate(
            candidates,
            q_min=float(settings["q_min"]),
            chi2_dof_tolerance=float(settings["chi2_dof_tolerance"]),
            qda=True,
        )
    except ValueError as exc:
        raise FitNumericalError("no qDA candidate is feasible across tune_z_values") from exc
    return {
        "summary": (f"tuned {len(candidates)} authored qDA candidates; recommended {recommended['id']}"),
        "metrics": {
            "candidate_count": len(candidates),
            "tune_z_values": tune_z_values,
            "recommended_candidate_id": recommended["id"],
            "fallback_no_q_passing": fallback,
        },
        "state_keys": ["matrix_element_candidates"],
        "artifacts": [],
    }
