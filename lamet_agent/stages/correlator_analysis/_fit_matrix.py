"""Tune the complete authored spectral-model grid on sample averages."""

from __future__ import annotations

from itertools import product

from lamet_agent.agent import ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.stages.correlator_analysis.physics import (
    fit_matrix_element_samples,
)
from lamet_agent.stages.correlator_analysis._selection import (
    select_tuned_candidate,
)


def run(context: ToolContext, *, tune_z_values: list[float]) -> dict[str, object]:
    """Evaluate every authored ordinary matrix-element candidate."""
    settings = context.params
    correlators = context.state.get("correlators")
    if not isinstance(correlators, dict):
        raise RuntimeError("correlator inspection must run before ordinary matrix-element tuning")
    three_points = [value for value in correlators.values() if value.attrs.get("correlator_type") == "three_point"]
    if len(three_points) != 1:
        raise ValueError("matrix-element model fitting requires exactly one selected three-point correlator")
    three_point = three_points[0]
    available_tseps = {int(value) for value in three_point.coords["tsep"]}
    if not tune_z_values or len(set(tune_z_values)) != len(tune_z_values):
        raise ValueError("tune_z_values must be a nonempty unique list")
    tune_z_values = [float(value) for value in tune_z_values]
    available_z = [float(value) for value in three_point.coords["z"]]
    if any(not any(abs(value - available) <= 1e-12 for available in available_z) for value in tune_z_values):
        raise ValueError("every tune_z_values entry must name an available z coordinate")
    correlator_rescale = context.state.get("correlator_rescale")
    if not isinstance(correlator_rescale, float):
        raise RuntimeError("inspect_correlators must determine correlator_rescale before spectral fitting")
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    component = {
        "re": "real",
        "im": "imag",
        "both": "both",
    }[context.params["component"]]
    candidates: list[dict[str, object]] = []
    ordinary_scopes = [scope for scope in settings["fit_scope"] if scope in {"3pt_ratio", "FH", "3pt_ratio+FH"}]
    authored = sorted(
        product(
            settings["fit_strategy"],
            ordinary_scopes,
            context.params["nstate"],
            settings["prior_width"],
            settings["pt2_windows"],
            settings["pt3_windows"],
        ),
        key=lambda item: (
            str(item[0]),
            str(item[1]),
            int(item[2]),
            float(item[3]),
            int(item[4]["tmin"]),
            int(item[4]["tmax"]),
            tuple(int(value) for value in item[5]["tsep_ls"]),
            int(item[5]["tau_cut"]),
        ),
    )
    for (
        strategy,
        fit_scope,
        nstate,
        prior_width,
        pt2_window,
        pt3_window,
    ) in authored:
        tseps = [int(value) for value in pt3_window["tsep_ls"]]
        if not set(tseps).issubset(available_tseps):
            raise ValueError("an authored three-point window is not covered by the input")
        candidate_id = f"matrix_{len(candidates) + 1:03d}"
        metadata = {
            "id": candidate_id,
            "method": str(strategy),
            "fit_scope": str(fit_scope),
            "observable": "matrix_element",
            "window": {
                "tmin": int(pt2_window["tmin"]),
                "tmax": int(pt2_window["tmax"]),
                "tau_min": int(pt3_window["tau_cut"]),
            },
            "tsep_values": tseps,
            "nstate": int(nstate),
            "prior_width": float(prior_width),
            "component": context.params["component"],
            "correlator_rescale": correlator_rescale,
        }
        per_z: dict[str, dict[str, object]] = {}
        failures: dict[str, str] = {}
        for tune_z in tune_z_values:
            try:
                data, fit = fit_matrix_element_samples(
                    correlators,
                    strategy=str(strategy),
                    fitting_form=str(settings["fitting_form"]),
                    fit_scope=str(fit_scope),
                    components=component,
                    tmin=int(pt2_window["tmin"]),
                    tmax=int(pt2_window["tmax"]),
                    tsep_values=tseps,
                    tau_min=int(pt3_window["tau_cut"]),
                    n_states=int(nstate),
                    prior_width=float(prior_width),
                    correlator_rescale=correlator_rescale,
                    svdcut=float(settings["svdcut"]),
                    posterior_prior_error_scale=float(settings["posterior_prior_error_scale"]),
                    sample_error_mode=sample_error_mode,
                    workers=context.workers,
                    tune_z=tune_z,
                    fit_samples=False,
                    _parallel=context._parallel,
                )
            except FitNumericalError as exc:
                failures[str(tune_z)] = str(exc)
                continue
            if data is not None:
                raise RuntimeError("candidate tuning must not produce a full sample result")
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
            qda=False,
        )
    except ValueError as exc:
        raise FitNumericalError("no ordinary matrix-fit candidate is feasible across tune_z_values") from exc
    return {
        "summary": (f"tuned {len(candidates)} authored matrix-element candidates; recommended {recommended['id']}"),
        "metrics": {
            "candidate_count": len(candidates),
            "tune_z_values": tune_z_values,
            "recommended_candidate_id": recommended["id"],
            "fallback_no_q_passing": fallback,
        },
        "state_keys": ["matrix_element_candidates"],
        "artifacts": [],
    }
