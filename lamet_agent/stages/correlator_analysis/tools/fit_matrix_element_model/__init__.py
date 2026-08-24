"""Fit matrix elements with native neo spectral models."""

from __future__ import annotations

from typing import Literal

from lamet_agent.agent import ToolContext
from lamet_agent.parallel import FitNumericalError
from lamet_agent.stages.correlator_analysis.physics import fit_matrix_element_samples


def run(
    context: ToolContext,
    *,
    strategy: Literal["joint", "chained", "independent"],
    fit_scope: Literal["3pt_ratio", "FH", "3pt_ratio+FH"],
    t_min: int,
    t_max: int,
    tau_min: int,
    nstate: int,
    prior_width: float,
) -> dict[str, object]:
    """Tune one authored spectral-model candidate on one sample average."""
    lsqfit = context.params["lsqfit"]
    if strategy not in lsqfit["fit_strategy"]:
        raise ValueError(f"{strategy} fitting is not allowed for this job")
    if fit_scope not in lsqfit["fit_scope"]:
        raise ValueError(f"{fit_scope} is not an authored fit_scope")
    settings = lsqfit
    pt2_window = {"tmin": t_min, "tmax": t_max}
    correlators = context.state.get("correlators")
    if not isinstance(correlators, dict):
        raise RuntimeError("inspect_correlators must run before fit_matrix_element_model")
    three_points = [value for value in correlators.values() if value.attrs.get("correlator_type") == "three_point"]
    if len(three_points) != 1:
        raise ValueError("matrix-element model fitting requires exactly one selected three-point correlator")
    matching_pt3_windows = [
        window for window in settings["pt3_windows"] if int(window["tau_cut"]) == tau_min
    ]
    tsep_candidates = {
        tuple(int(value) for value in window["tsep_ls"]) for window in matching_pt3_windows
    }
    if len(tsep_candidates) != 1:
        raise ValueError("tau_min must identify exactly one authored tsep selection")
    tseps = list(next(iter(tsep_candidates)))
    available_tseps = {int(value) for value in three_points[0].coords["tsep"]}
    if not set(tseps).issubset(available_tseps):
        raise ValueError("the authored three-point window is not covered by the input")
    pt3_window = {"tsep_ls": tseps, "tau_cut": tau_min}
    if pt2_window not in settings["pt2_windows"] or pt3_window not in settings["pt3_windows"]:
        raise ValueError("the selected fit window is not an authored candidate")
    if nstate not in context.params["nstate"] or prior_width not in lsqfit["prior_width"]:
        raise ValueError("nstate and prior_width must be selected from the authored candidate lists")
    correlator_rescale = context.state.get("correlator_rescale")
    if not isinstance(correlator_rescale, float):
        raise RuntimeError("inspect_correlators must determine correlator_rescale before spectral fitting")
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    candidates = context.state.setdefault("matrix_element_candidates", [])
    candidate_id = f"matrix_{len(candidates) + 1:03d}"
    component = {"re": "real", "im": "imag", "both": "both"}[context.params["component"]]
    candidate_metadata = {"id": candidate_id, "method": strategy, "fit_scope": fit_scope, "observable": "matrix_element", "window": {"t_min": t_min, "t_max": t_max, "tau_min": tau_min}, "tsep_values": tseps, "nstate": nstate, "prior_width": prior_width, "component": context.params["component"], "correlator_rescale": correlator_rescale}
    try:
        data, fit = fit_matrix_element_samples(
            correlators,
            strategy=strategy,
            fitting_form=str(settings["fitting_form"]),
            fit_scope=fit_scope,
            components=component,
            t_min=t_min,
            t_max=t_max,
            tsep_values=tseps,
            tau_min=tau_min,
            n_states=nstate,
            prior_width=prior_width,
            correlator_rescale=correlator_rescale,
            svdcut=float(settings["svdcut"]),
            posterior_prior_error_scale=float(settings["posterior_prior_error_scale"]),
            sample_error_mode=sample_error_mode,
            workers=context.workers,
            tune_z=settings["tune_z"],
            fit_samples=False,
            _parallel=context._parallel,
        )
    except FitNumericalError as exc:
        candidates.append({**candidate_metadata, "quality_passed": False, "numerical_failure": True, "error": str(exc)})
        return {"summary": f"rejected numerically unusable matrix-element candidate {candidate_id}", "metrics": {"candidate_id": candidate_id, "quality_passed": False, "numerical_failure": True, "error": str(exc)}, "state_keys": ["matrix_element_candidates"], "artifacts": []}
    fit["q_min"] = float(settings["q_min"])
    if data is not None:
        raise RuntimeError("candidate tuning must not produce a full sample result")
    fit["quality_passed"] = fit["Q"] >= fit["q_min"]
    candidates.append({**candidate_metadata, "numerical_failure": False, **fit})
    return {"summary": f"tuned {strategy} matrix-element candidate {candidate_id} at z={fit['tune_z']}", "metrics": {"candidate_id": candidate_id, "tune_z": fit["tune_z"], "fit_scope": fit["fit_scope"], "Q": fit["Q"], "chi2_dof": fit["chi2_dof"], "quality_passed": fit["quality_passed"]}, "state_keys": ["matrix_element_candidates"], "artifacts": []}
