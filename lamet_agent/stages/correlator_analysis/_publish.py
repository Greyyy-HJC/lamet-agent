"""Publish a selected correlator candidate as the stage terminal result."""

from __future__ import annotations

import json
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError
from lamet_agent.plotting import configure_plot, errorline, save_figure, start_plot
from lamet_agent.stages.correlator_analysis.diagnostics import write_fit_artifacts
from lamet_agent.stages.correlator_analysis.physics import (
    fit_matrix_element_samples,
    matrix_element_samples,
)
from lamet_agent.stages.correlator_analysis.selection import (
    select_tuned_candidate,
)


def run(context: ToolContext, *, candidate_id: str) -> dict[str, object]:
    """Select one candidate, write ``output.nc``, and finish the job."""
    lsqfit = context.params if context.params.get("analysis_method", "lsqfit") == "lsqfit" else None
    if not isinstance(lsqfit, dict):
        raise ValueError("publish_correlator_result is only available for lsqfit jobs")
    candidates = [*context.state.get("spectrum_candidates", []), *context.state.get("matrix_element_candidates", [])]
    ordinary_scopes = [scope for scope in lsqfit["fit_scope"] if scope in {"3pt_ratio", "FH", "3pt_ratio+FH"}]
    spectral_methods = list(lsqfit["fit_strategy"]) if ordinary_scopes else []
    if spectral_methods:
        expected = {
            (method, scope, int(pt2["tmin"]), int(pt2["tmax"]), int(pt3["tau_cut"]), int(nstate), float(width))
            for method in spectral_methods
            for scope in ordinary_scopes
            for pt2 in lsqfit["pt2_windows"]
            for pt3 in lsqfit["pt3_windows"]
            for nstate in context.params["nstate"]
            for width in lsqfit["prior_width"]
        }
        observed = {
            (
                candidate["method"],
                candidate.get("fit_scope"),
                candidate["window"]["t_min"],
                candidate["window"]["t_max"],
                candidate["window"]["tau_min"],
                candidate.get("nstate"),
                candidate.get("prior_width"),
            )
            for candidate in candidates
            if candidate.get("method") in spectral_methods
        }
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(
                f"all authored matrix-fit candidates must be evaluated before publishing; missing {missing[:3]}"
            )
    if "qda_ratio" in lsqfit["fit_scope"]:
        expected_qda = {
            (
                str(strategy),
                int(nstate),
                float(width),
                int(window["tmin"]),
                int(window["tmax"]),
            )
            for strategy in lsqfit["fit_strategy"]
            for nstate in context.params["nstate"]
            for width in lsqfit["prior_width"]
            for window in lsqfit["pt2_windows"]
        }
        observed_qda = {
            (
                str(candidate.get("fit_strategy")),
                int(candidate.get("nstate", context.params["nstate"][0])),
                float(candidate.get("prior_width", lsqfit["prior_width"][0])),
                int(candidate["window"]["t_min"]),
                int(candidate["window"]["t_max"]),
            )
            for candidate in candidates
            if candidate.get("method") == "qda"
        }
        missing_qda = sorted(expected_qda - observed_qda)
        if missing_qda:
            raise ValueError(
                f"all authored qDA candidates must be evaluated before publishing; missing {missing_qda[:3]}"
            )
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if candidate_id not in by_id:
        raise ValueError("candidate_id must name an existing candidate")
    matrix_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("method") in spectral_methods or candidate.get("method") == "qda"
    ]
    if matrix_candidates:
        is_qda = bool(matrix_candidates) and all(candidate.get("method") == "qda" for candidate in matrix_candidates)
        deterministic, fallback = select_tuned_candidate(
            matrix_candidates,
            q_min=float(lsqfit["q_min"]),
            chi2_dof_tolerance=float(lsqfit["chi2_dof_tolerance"]),
            qda=is_qda,
        )
        selection_rule = (
            "original_qda_robust_rule(min_Q_then_worst_chi2_dof)"
            if is_qda
            else f"original_data_window_rule(fallback_no_q_passing={fallback})"
        )
    else:
        acceptable = [candidate for candidate in candidates if candidate.get("quality_passed", True)]
        if not acceptable:
            raise ValueError("no candidate passes the authored quality threshold")

        def rank(candidate: dict[str, object]) -> tuple[float, float, str]:
            quality = float(candidate.get("Q", candidate.get("min_Q", 1.0)))
            chi2_dof = float(candidate.get("max_chi2_dof", candidate.get("chi2_dof", 0.0)))
            return (-quality, chi2_dof, str(candidate["id"]))

        deterministic = min(acceptable, key=rank)
        selection_rule = "highest_quality_then_lowest_chi2_dof_then_id"
    if candidate_id != deterministic["id"]:
        raise ValueError(f"candidate_id must be the deterministic best acceptable candidate '{deterministic['id']}'")
    selected = deterministic
    application_rejections: list[dict[str, object]] = []
    correlators = context.state.get("correlators")
    settings = lsqfit
    data = selected.get("data")
    selected_method = selected.get("method")
    if not isinstance(data, EnsembleData) and (selected_method in spectral_methods or selected_method == "qda"):
        if not isinstance(correlators, dict):
            raise RuntimeError("inspect_correlators must run before publishing a matrix-element model")
        application_fit = None
        preflight_fit = None
        try:
            if selected_method == "qda":
                print(
                    f"Running full qDA sample fits for matrix candidate {selected['id']}...",
                    flush=True,
                )
                values, z_coordinates, application_fit = matrix_element_samples(
                    correlators,
                    method="qda",
                    t_min=int(selected["window"]["t_min"]),
                    t_max=int(selected["window"]["t_max"]),
                    tau_min=None,
                    lsqfit=settings,
                    sample_error_mode=str(context.manifest["metadata"]["sample_error_mode"]),
                    workers=context.workers,
                    fit_samples=True,
                    show_progress=bool(context.state.get("show_job_progress", False)),
                    n_states=int(selected["nstate"]),
                    prior_width=float(selected["prior_width"]),
                    _parallel=context._parallel,
                )
                if values is None:
                    raise RuntimeError("full qDA fitting produced no sample values")
                component = str(context.params["component"])
                if component == "re":
                    values = values.real
                elif component == "im":
                    values = values.imag
                source = next(value for value in correlators.values() if value.attrs.get("correlator_type") == "qda")
                attrs = dict(source.attrs)
                attrs.update(
                    {
                        "observable": "matrix_element",
                        "method": "qda",
                        "n_states": int(selected["nstate"]),
                        "prior_width": float(selected["prior_width"]),
                        "sample_error_mode": context.manifest["metadata"]["sample_error_mode"],
                        "units": '{"values":"dimensionless","z":"lattice"}',
                    }
                )
                data = EnsembleData(
                    source.ensemble,
                    source.resample,
                    [sample for sample in values],
                    ["z"],
                    {"z": z_coordinates},
                    attrs=attrs,
                    name="bare_matrix_element",
                )
            else:
                application_kwargs = {
                    "strategy": str(selected_method),
                    "fitting_form": str(settings["fitting_form"]),
                    "fit_scope": str(selected["fit_scope"]),
                    "components": {"re": "real", "im": "imag", "both": "both"}[str(context.params["component"])],
                    "t_min": int(selected["window"]["t_min"]),
                    "t_max": int(selected["window"]["t_max"]),
                    "tsep_values": [int(value) for value in selected["tsep_values"]],
                    "tau_min": int(selected["window"]["tau_min"]),
                    "n_states": int(selected["nstate"]),
                    "prior_width": float(selected["prior_width"]),
                    "correlator_rescale": float(selected["correlator_rescale"]),
                    "svdcut": float(settings["svdcut"]),
                    "posterior_prior_error_scale": float(settings["posterior_prior_error_scale"]),
                    "sample_error_mode": str(context.manifest["metadata"]["sample_error_mode"]),
                    "workers": context.workers,
                    "show_progress": bool(context.state.get("show_job_progress", False)),
                    "_parallel": context._parallel,
                }
                print(f"Preflighting matrix candidate {selected['id']} on the full z grid...", flush=True)
                preflight_data, preflight_fit = fit_matrix_element_samples(
                    correlators,
                    **application_kwargs,
                    tune_z=None,
                    fit_samples=False,
                )
                if preflight_data is not None:
                    raise RuntimeError("full-grid center preflight unexpectedly produced sample data")
                print(f"Running full sample fits for matrix candidate {selected['id']}...", flush=True)
                data, application_fit = fit_matrix_element_samples(correlators, **application_kwargs)
        except FitNumericalError as exc:
            error = str(exc)
            selected.update({"quality_passed": False, "numerical_failure": True, "error": error})
            application_rejections.append({"candidate_id": str(selected["id"]), "error": error})
            raise FitNumericalError(
                f"selected candidate {selected['id']} failed full-grid application: {error}"
            ) from exc
        if application_fit is not None and int(application_fit.get("n_failed_samples", 0)):
            error = f"{application_fit['n_failed_samples']} sample fit(s) failed numerically"
            selected.update(
                {
                    "quality_passed": False,
                    "numerical_failure": True,
                    "error": error,
                    "application_fit": application_fit,
                }
            )
            application_rejections.append({"candidate_id": str(selected["id"]), "error": error})
            raise FitNumericalError(f"selected candidate {selected['id']} failed full-grid application: {error}")
        if not isinstance(data, EnsembleData) or application_fit is None:
            raise RuntimeError("full-grid matrix-element fitting produced no sample result")
        selected["data"] = data
        selected["preflight_fit"] = preflight_fit
        selected["application_fit"] = application_fit
    candidate_id = str(selected["id"])
    if not isinstance(data, EnsembleData):
        raise TypeError("selected candidate has no EnsembleData result")
    context.state["correlator_result"] = data
    data.to_netcdf(context.artifact_directory / "output.nc")
    candidate_table = [
        {
            "candidate_id": candidate["id"],
            "method": candidate.get("method"),
            "window": candidate.get("window"),
            "nstate": candidate.get("nstate"),
            "prior_width": candidate.get("prior_width"),
            "correlator_rescale": candidate.get("correlator_rescale"),
            "quality_passed": candidate.get("quality_passed", True),
            "numerical_failure": candidate.get("numerical_failure", False),
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
    application_fit = selected.get("application_fit")
    if isinstance(application_fit, dict):
        fit_result = write_fit_artifacts(
            job_id=context.job_id,
            selected=selected,
            candidates=candidates,
            preflight_fit=selected.get("preflight_fit"),
            application_fit=application_fit,
            application_rejections=application_rejections,
            artifact_directory=context.artifact_directory,
            component=str(context.params["component"]),
            q_min=float(settings["q_min"]),
        )
        fit_artifacts = list(fit_result.artifacts)
        sample_fit_quality = fit_result.sample_fit_quality
        dispersion_energy = fit_result.dispersion_energy
        selected["application_fit"] = fit_result.application_fit
    diagnostics = {
        "candidate_id": candidate_id,
        "method": selected.get("method"),
        "selection_rule": selection_rule,
        "recommended_defaults": context.state.get("recommended_defaults", {}),
        "correlator_scale_inspection": context.state.get("correlator_scale_inspection", {}),
        "selected_preflight_fit": selected.get("preflight_fit"),
        "selected_application_fit": selected.get("application_fit"),
        "candidates": candidate_table,
        "sample_fit_quality": sample_fit_quality,
        "dispersion_energy": dispersion_energy,
        **{key: selected[key] for key in ("chi2", "dof", "chi2_dof", "Q", "aic") if key in selected},
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "candidates.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
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
        sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
        start_plot()
        errorline(data.coords[plot_dim], plot_data.average(sample_error_mode))
        xlabel = r"$z~/~a$" if plot_dim == "z" else plot_dim
        configure_plot(xlabel=xlabel, ylabel=str(data.name or "result").replace("_", " "))
        save_figure(context.artifact_directory / "plots" / "result.pdf")
        plot_artifact = "plots/result.pdf"
    report = f"# Correlator result\n\nSelected candidate: `{candidate_id}`.\nMethod: `{selected.get('method')}`.\n"
    (context.artifact_directory / "report.md").write_text(report, encoding="utf-8")
    artifacts = (
        ["output.nc", "diagnostics/candidates.json", "report.md"]
        + ([plot_artifact] if plot_artifact else [])
        + fit_artifacts
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": str(data.name or "correlator_result"),
        "decisions": {"candidate_id": candidate_id, "method": selected.get("method")},
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
