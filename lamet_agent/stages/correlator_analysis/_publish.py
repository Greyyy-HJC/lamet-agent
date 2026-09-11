"""Publish a selected correlator candidate as the stage terminal result."""

from __future__ import annotations

import json
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError
from lamet_agent.plotting import configure_plot, errorline, save_figure, start_plot
from lamet_agent.ui import log
from lamet_agent.stages.correlator_analysis._diagnostics import write_fit_artifacts
from lamet_agent.stages.correlator_analysis.physics import (
    fit_matrix_element_samples,
    fit_qda_samples,
)
from lamet_agent.stages.correlator_analysis._scope import nstate_combinations, nstate_key, parse_fit_scope
from lamet_agent.stages.correlator_analysis._model_average import combine_matrix_samples
from lamet_agent.stages.correlator_analysis._selection import (
    models_on_dataset,
    select_spectrum_candidate,
    select_tuned_candidate,
)


def _json_ready(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _apply_qda_candidate(
    context: ToolContext,
    candidate: dict[str, object],
    *,
    settings: dict[str, object],
    correlators: dict[str, object],
) -> tuple[EnsembleData, None, dict[str, object]]:
    log(f"Running: full qDA sample fits for matrix candidate {candidate['id']}...")
    data, application_fit = fit_qda_samples(
        correlators,
        fit_scope=list(settings["fit_scope"]),
        components={"re": "real", "im": "imag", "both": "both"}[str(context.params["component"])],
        tmin=int(candidate["window"]["tmin"]),
        tmax=int(candidate["window"]["tmax"]),
        n_states=candidate["nstate"],
        prior_width=float(candidate["prior_width"]),
        svdcut=float(settings["svdcut"]),
        posterior_prior_error_scale=float(settings["posterior_prior_error_scale"]),
        sample_error_mode=str(context.manifest["metadata"]["sample_error_mode"]),
        workers=context.workers,
        fit_samples=True,
        show_progress=bool(context.state.get("show_job_progress", False)),
        _parallel=context._parallel,
    )
    if data is None:
        raise RuntimeError("full qDA fitting produced no sample values")
    data.array.attrs.update(
        {
            "observable": "matrix_element",
            "prior_width": float(candidate["prior_width"]),
            "units": '{"values":"dimensionless","z":"lattice"}',
        }
    )
    return data, None, application_fit


def _apply_ordinary_candidate(
    context: ToolContext,
    candidate: dict[str, object],
    *,
    settings: dict[str, object],
    correlators: dict[str, object],
) -> tuple[EnsembleData, dict[str, object] | None, dict[str, object]]:
    application_kwargs = {
        "fitting_form": str(settings["fitting_form"]),
        "fit_scope": list(candidate["fit_scope"]),
        "components": {"re": "real", "im": "imag", "both": "both"}[str(context.params["component"])],
        "tmin": int(candidate["window"]["tmin"]),
        "tmax": int(candidate["window"]["tmax"]),
        "tsep_values": [int(value) for value in candidate["tsep_values"]],
        "tau_min": int(candidate["window"]["tau_min"]),
        "n_states": candidate["nstate"],
        "prior_width": float(candidate["prior_width"]),
        "correlator_rescale": float(candidate["correlator_rescale"]),
        "svdcut": float(settings["svdcut"]),
        "posterior_prior_error_scale": float(settings["posterior_prior_error_scale"]),
        "sample_error_mode": str(context.manifest["metadata"]["sample_error_mode"]),
        "workers": context.workers,
        "show_progress": bool(context.state.get("show_job_progress", False)),
        "_parallel": context._parallel,
    }
    log(f"Preflighting matrix candidate {candidate['id']} on the full z grid...")
    preflight_data, preflight_fit = fit_matrix_element_samples(
        correlators,
        **application_kwargs,
        tune_z=None,
        fit_samples=False,
    )
    if preflight_data is not None:
        raise RuntimeError("full-grid center preflight unexpectedly produced sample data")
    log(f"Running: full sample fits for matrix candidate {candidate['id']}...")
    data, application_fit = fit_matrix_element_samples(correlators, **application_kwargs)
    return data, preflight_fit, application_fit


def _usable_average_models(candidates: list[dict[str, object]], selected: dict[str, object]) -> list[dict[str, object]]:
    siblings = [
        candidate
        for candidate in models_on_dataset(candidates, selected)
        if not candidate.get("numerical_failure", False) and candidate.get("error") is None
    ]
    return siblings or [selected]


def run(context: ToolContext, *, candidate_id: str) -> dict[str, object]:
    """Select one candidate, write ``output.nc``, and finish the job."""
    lsqfit = context.params if context.params["analysis_method"] == "lsqfit" else None
    if not isinstance(lsqfit, dict):
        raise ValueError("publish_correlator_result is only available for lsqfit jobs")
    scope = parse_fit_scope(lsqfit["fit_scope"])
    candidates = [*context.state.get("spectrum_candidates", []), *context.state.get("matrix_element_candidates", [])]
    if scope.needs_pt3_data:
        expected = {
            (
                int(pt2["tmin"]),
                int(pt2["tmax"]),
                int(pt3["tau_cut"]),
                tuple(int(value) for value in pt3["tsep_ls"]),
                nstate_key(nstate),
                float(width),
            )
            for pt2 in lsqfit["pt2_windows"]
            for pt3 in lsqfit["pt3_windows"]
            for nstate in nstate_combinations(context.params["nstate"], lsqfit["fit_scope"])
            for width in lsqfit["prior_width"]
        }
        observed = {
            (
                candidate["window"]["tmin"],
                candidate["window"]["tmax"],
                candidate["window"]["tau_min"],
                tuple(int(value) for value in candidate.get("tsep_values", [])),
                nstate_key(candidate["nstate"]) if candidate.get("nstate") is not None else None,
                candidate.get("prior_width"),
            )
            for candidate in candidates
            if candidate.get("observable") == "matrix_element"
        }
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(
                f"all authored matrix-fit candidates must be evaluated before publishing; missing {missing[:3]}"
            )
    if scope.is_qda:
        expected_qda = {
            (
                nstate_key(nstate),
                float(width),
                int(window["tmin"]),
                int(window["tmax"]),
            )
            for nstate in nstate_combinations(context.params["nstate"], lsqfit["fit_scope"])
            for width in lsqfit["prior_width"]
            for window in lsqfit["pt2_windows"]
        }
        observed_qda = {
            (
                nstate_key(candidate["nstate"])
                if candidate.get("nstate") is not None
                else nstate_key(nstate_combinations(context.params["nstate"], lsqfit["fit_scope"])[0]),
                float(candidate.get("prior_width", lsqfit["prior_width"][0])),
                int(candidate["window"]["tmin"]),
                int(candidate["window"]["tmax"]),
            )
            for candidate in candidates
            if candidate.get("observable") == "matrix_element"
        }
        missing_qda = sorted(expected_qda - observed_qda)
        if missing_qda:
            raise ValueError(
                f"all authored qDA candidates must be evaluated before publishing; missing {missing_qda[:3]}"
            )
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if candidate_id not in by_id:
        raise ValueError("candidate_id must name an existing candidate")
    matrix_candidates = [candidate for candidate in candidates if candidate.get("observable") == "matrix_element"]
    if matrix_candidates:
        deterministic, fallback = select_tuned_candidate(
            matrix_candidates,
            q_min=float(lsqfit["q_min"]),
        )
        selection_rule = f"robust_rule(min_Q_then_worst_chi2_dof, fallback_no_q_passing={fallback})"
    else:
        deterministic, fallback = select_spectrum_candidate(candidates, q_min=float(lsqfit["q_min"]))
        selection_rule = "highest_quality_then_lowest_chi2_dof_then_id"
    if candidate_id != deterministic["id"]:
        raise ValueError(f"candidate_id must be the deterministic best acceptable candidate '{deterministic['id']}'")
    selected = deterministic
    application_rejections: list[dict[str, object]] = []
    correlators = context.state.get("correlators")
    settings = lsqfit
    model_average = bool(lsqfit.get("model_average"))
    targets = _usable_average_models(candidates, selected) if model_average else [selected]
    artifact_source = selected
    combined_result: dict[str, object] | None = None
    needs_application = selected.get("observable") == "matrix_element"
    if needs_application:
        missing_data = any(not isinstance(candidate.get("data"), EnsembleData) for candidate in targets)
        if missing_data:
            if not isinstance(correlators, dict):
                raise RuntimeError("inspect_correlators must run before publishing a matrix-element model")
        for candidate in targets:
            if isinstance(candidate.get("data"), EnsembleData):
                continue
            try:
                if scope.is_qda:
                    data, preflight_fit, application_fit = _apply_qda_candidate(
                        context, candidate, settings=settings, correlators=correlators
                    )
                else:
                    data, preflight_fit, application_fit = _apply_ordinary_candidate(
                        context, candidate, settings=settings, correlators=correlators
                    )
            except FitNumericalError as exc:
                error = str(exc)
                candidate.update({"quality_passed": False, "numerical_failure": True, "error": error})
                application_rejections.append({"candidate_id": str(candidate["id"]), "error": error})
                if not model_average:
                    raise FitNumericalError(
                        f"selected candidate {candidate['id']} failed full-grid application: {error}"
                    ) from exc
                continue
            if not model_average and application_fit is not None and int(application_fit.get("n_failed_samples", 0)):
                error = f"{application_fit['n_failed_samples']} sample fit(s) failed numerically"
                candidate.update(
                    {
                        "quality_passed": False,
                        "numerical_failure": True,
                        "error": error,
                        "application_fit": application_fit,
                    }
                )
                application_rejections.append({"candidate_id": str(candidate["id"]), "error": error})
                raise FitNumericalError(f"selected candidate {candidate['id']} failed full-grid application: {error}")
            if not isinstance(data, EnsembleData) or application_fit is None:
                raise RuntimeError("full-grid matrix-element fitting produced no sample result")
            candidate["data"] = data
            candidate["preflight_fit"] = preflight_fit
            candidate["application_fit"] = application_fit
    applied = [candidate for candidate in targets if isinstance(candidate.get("data"), EnsembleData)]
    data = selected.get("data")
    if model_average and len(applied) > 1:
        try:
            combined_result = combine_matrix_samples(applied)
        except ValueError:
            combined_result = None
        if combined_result is not None:
            data = combined_result["data"]
            artifact_source = combined_result["primary_model"]
    elif not isinstance(data, EnsembleData) and applied:
        data = applied[0]["data"]
    if combined_result is None and isinstance(data, EnsembleData):
        data.array.attrs["model_average"] = "true" if model_average else "false"
        data.array.attrs["selected_models"] = json.dumps([str(selected["id"])])
        data.array.attrs["model_weights"] = json.dumps([1.0])
    candidate_id = str(selected["id"])
    if not isinstance(data, EnsembleData):
        if model_average and needs_application:
            raise FitNumericalError("all averaged models failed full-grid application")
        raise TypeError("selected candidate has no EnsembleData result")
    weight_by_id = {str(selected["id"]): 1.0}
    averaged_ids = [str(selected["id"])]
    mean_weights: list[float] = [1.0]
    if combined_result is not None:
        mean_weights = [float(weight) for weight in combined_result["mean_weights"]]
        weight_by_id = {str(candidate["id"]): float(weight) for candidate, weight in zip(applied, mean_weights)}
        averaged_ids = [str(value) for value in combined_result["selected_models"]]
        primary_quality = combined_result["center_quality"][int(combined_result["primary_index"])]
        for key in ("Q", "chi2_dof", "logGBF"):
            if primary_quality.get(key) is not None:
                selected[key] = primary_quality[key]
    context.state["correlator_result"] = data
    data.to_netcdf(context.artifact_directory / "output.nc")
    candidate_table = [
        {
            "candidate_id": candidate["id"],
            "method": candidate.get("method"),
            "fit_scope": candidate.get("fit_scope"),
            "window": candidate.get("window"),
            "tsep_values": candidate.get("tsep_values"),
            "nstate": candidate.get("nstate"),
            "prior_width": candidate.get("prior_width"),
            "correlator_rescale": candidate.get("correlator_rescale"),
            "quality_passed": candidate.get("quality_passed", True),
            "numerical_failure": candidate.get("numerical_failure", False),
            "model_weight": weight_by_id.get(str(candidate["id"]), 0.0),
            "averaged": str(candidate["id"]) in averaged_ids,
            **{
                key: candidate[key]
                for key in (
                    "error",
                    "failure_reasons",
                    "feasible_at_all_tune_z",
                    "tune_z_values",
                    "tune_z_diagnostics",
                    "n_failed_samples",
                    "n_data",
                    "n_params",
                    "chi2",
                    "dof",
                    "chi2_dof",
                    "Q",
                    "min_Q",
                    "worst_chi2_dof",
                    "max_chi2_dof",
                    "logGBF",
                    "aic",
                )
                if key in candidate
            },
        }
        for candidate in sorted(candidates, key=lambda item: str(item["id"]))
    ]
    fit_artifacts: list[str] = []
    sample_fit_quality: dict[str, object] = {}
    dispersion_energy: dict[str, object] = {}
    application_fit = artifact_source.get("application_fit")
    if isinstance(application_fit, dict):
        fit_result = write_fit_artifacts(
            job_id=context.job_id,
            selected=artifact_source,
            candidates=candidates,
            preflight_fit=artifact_source.get("preflight_fit"),
            application_fit=application_fit,
            application_rejections=application_rejections,
            artifact_directory=context.artifact_directory,
            component=str(context.params["component"]),
            q_min=float(settings["q_min"]),
            model_average=model_average,
            fit_model_weights=mean_weights,
            averaged_ids=averaged_ids,
        )
        fit_artifacts = list(fit_result.artifacts)
        sample_fit_quality = fit_result.sample_fit_quality
        dispersion_energy = fit_result.dispersion_energy
        artifact_source["application_fit"] = fit_result.application_fit
        if artifact_source is selected:
            selected["application_fit"] = fit_result.application_fit
    fallback_no_q_passing = bool(context.state.get("fallback_no_q_passing", fallback))
    diagnostics = {
        "candidate_id": candidate_id,
        "method": selected.get("method"),
        "selection_rule": selection_rule,
        "fallback_no_q_passing": fallback_no_q_passing,
        "model_average": model_average,
        "selected_models": averaged_ids,
        "fit_model_weights": mean_weights,
        "real_sys_sdev": combined_result.get("real_sys_sdev") if combined_result else None,
        "imag_sys_sdev": combined_result.get("imag_sys_sdev") if combined_result else None,
        "recommended_defaults": context.state.get("recommended_defaults", {}),
        "correlator_scale_inspection": context.state.get("correlator_scale_inspection", {}),
        "selected_preflight_fit": artifact_source.get("preflight_fit"),
        "selected_application_fit": artifact_source.get("application_fit"),
        "candidates": candidate_table,
        "sample_fit_quality": sample_fit_quality,
        "dispersion_energy": dispersion_energy,
        **{key: selected[key] for key in ("chi2", "dof", "chi2_dof", "Q", "aic") if key in selected},
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "candidates.json").write_text(
        json.dumps(_json_ready(diagnostics), indent=2), encoding="utf-8"
    )
    plot_dim = "z" if "z" in data.dims else "state" if "state" in data.dims else data.dims[0]
    plot_samples = np.asarray(data.real.values if np.iscomplexobj(data.values) else data.values)
    plot_axis = data.array.dims.index(plot_dim)
    plot_samples = (
        np.moveaxis(plot_samples, plot_axis, 1).reshape(data.n_sample, len(data.coords[plot_dim]), -1).mean(axis=2)
    )
    plot_data = EnsembleData(
        data.ensemble,
        data.resample,
        list(plot_samples),
        [plot_dim],
        {plot_dim: data.coords[plot_dim]},
        attrs=data.attrs,
    )
    plot_artifact = selected.get("plot_artifact")
    if plot_artifact is None:
        sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
        start_plot()
        errorline(data.coords[plot_dim], plot_data.average(sample_error_mode))
        xlabel = r"$z~/~a$" if plot_dim == "z" else plot_dim
        configure_plot(xlabel=xlabel, ylabel=str(data.name or "result").replace("_", " "))
        save_figure(context.artifact_directory / "plots" / "result.pdf")
        plot_artifact = "plots/result.pdf"
    artifacts = (
        ["output.nc", "diagnostics/candidates.json"] + ([plot_artifact] if plot_artifact else []) + fit_artifacts
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": str(data.name or "correlator_result"),
        "decisions": {
            "candidate_id": candidate_id,
            "method": selected.get("method"),
            "fit_scope": selected.get("fit_scope"),
            "model_average": model_average,
            "fallback_no_q_passing": fallback_no_q_passing,
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    context.finish(data, summary)
    return {
        "summary": f"published {data.name or 'correlator result'}",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
