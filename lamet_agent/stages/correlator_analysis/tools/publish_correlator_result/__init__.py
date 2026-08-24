"""Publish a selected correlator candidate as the stage terminal result."""

from __future__ import annotations

import json
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError
from lamet_agent.plotting import configure_plot, errorbar, save_figure, start_plot
from lamet_agent.stages.correlator_analysis.physics import fit_matrix_element_samples


_CHI2_DOF_TOLERANCE = 0.25


def _select_data_window(candidates: list[dict[str, object]], *, q_min: float) -> tuple[dict[str, object], bool]:
    """Apply the original information-preserving window-selection rule."""
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.get("numerical_failure", False)
        and int(candidate.get("n_data", 0)) > int(candidate.get("n_params", 0))
        and np.isfinite(float(candidate.get("chi2_dof", np.inf)))
    ]
    if not eligible:
        raise ValueError("no overdetermined matrix-fit candidate is available")
    passing = [candidate for candidate in eligible if float(candidate.get("Q", 0.0)) >= q_min]
    pool = passing or eligible
    best_chi2_dof = min(float(candidate["chi2_dof"]) for candidate in pool)
    comparable = [
        candidate
        for candidate in pool
        if float(candidate["chi2_dof"]) <= best_chi2_dof + _CHI2_DOF_TOLERANCE
    ]
    selected = max(
        comparable,
        key=lambda candidate: (
            int(candidate["n_data"]),
            -float(candidate["chi2_dof"]),
            float(candidate.get("Q", 0.0)),
        ),
    )
    return selected, not bool(passing)


def run(context: ToolContext, *, candidate_id: str) -> dict[str, object]:
    """Select one candidate, write ``output.nc``, and finish the job."""
    lsqfit = context.params.get("lsqfit")
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
            (candidate["method"], candidate.get("fit_scope"), candidate["window"]["t_min"], candidate["window"]["t_max"], candidate["window"]["tau_min"], candidate.get("nstate"), candidate.get("prior_width"))
            for candidate in candidates
            if candidate.get("method") in spectral_methods
        }
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(f"all authored matrix-fit candidates must be evaluated before publishing; missing {missing[:3]}")
    if "qda_ratio" in lsqfit["fit_scope"]:
        expected_windows = {(int(window["tmin"]), int(window["tmax"])) for window in lsqfit["pt2_windows"]}
        observed_windows = {
            (candidate["window"]["t_min"], candidate["window"]["t_max"])
            for candidate in candidates
            if candidate.get("method") == "qda"
        }
        missing_windows = sorted(expected_windows - observed_windows)
        if missing_windows:
            raise ValueError(f"all authored qDA windows must be evaluated before publishing; missing {missing_windows}")
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if candidate_id not in by_id:
        raise ValueError("candidate_id must name an existing candidate")
    matrix_candidates = [candidate for candidate in candidates if candidate.get("method") in spectral_methods]
    if matrix_candidates:
        deterministic, fallback = _select_data_window(
            matrix_candidates,
            q_min=float(lsqfit["q_min"]),
        )
        selection_rule = f"original_data_window_rule(fallback_no_q_passing={fallback})"
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
    requested = context.params["observable"]
    application_rejections: list[dict[str, object]] = []
    correlators = context.state.get("correlators")
    settings = lsqfit
    while True:
        if selected.get("observable", selected.get("data").attrs.get("observable") if isinstance(selected.get("data"), EnsembleData) else None) != requested:
            raise ValueError("selected candidate does not match the requested observable")
        data = selected.get("data")
        if isinstance(data, EnsembleData) or selected.get("method") not in spectral_methods:
            break
        if not isinstance(correlators, dict):
            raise RuntimeError("inspect_correlators must run before publishing a matrix-element model")
        application_kwargs = {
            "strategy": str(selected["method"]),
            "fitting_form": str(settings["fitting_form"]),
            "fit_scope": str(selected["fit_scope"]),
            "component": str(context.params["component"]),
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
            "_parallel": context._parallel,
        }
        application_fit = None
        error = None
        try:
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
        if error is None and application_fit is not None and application_fit["n_failed_samples"]:
            error = f"{application_fit['n_failed_samples']} sample fit(s) failed numerically"
        if error is None:
            if not isinstance(data, EnsembleData) or application_fit is None:
                raise RuntimeError("full-grid matrix-element fitting produced no sample result")
            selected["data"] = data
            selected["preflight_fit"] = preflight_fit
            selected["application_fit"] = application_fit
            break
        rejected_id = str(selected["id"])
        selected.update({"quality_passed": False, "numerical_failure": True, "error": error})
        if application_fit is not None:
            selected["application_fit"] = application_fit
        application_rejections.append({"candidate_id": rejected_id, "error": error})
        print(f"Rejected matrix candidate {rejected_id}: {error}", flush=True)
        try:
            selected, fallback = _select_data_window(
                matrix_candidates,
                q_min=float(settings["q_min"]),
            )
        except ValueError as exc:
            raise FitNumericalError(f"all matrix-fit candidates failed full-grid application: {application_rejections}") from exc
        print(f"Retrying publication with matrix candidate {selected['id']}...", flush=True)
        selection_rule = f"original_data_window_rule(fallback_no_q_passing={fallback})"
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
            **{key: candidate[key] for key in ("error", "n_failed_samples", "n_data", "n_params", "chi2", "dof", "chi2_dof", "Q", "min_Q", "max_chi2_dof", "logGBF", "aic") if key in candidate},
        }
        for candidate in sorted(candidates, key=lambda item: str(item["id"]))
    ]
    diagnostics = {"candidate_id": candidate_id, "method": selected.get("method"), "selection_rule": selection_rule, "application_rejections": application_rejections, "candidates": candidate_table, **{key: selected[key] for key in ("chi2", "dof", "chi2_dof", "Q", "aic") if key in selected}}
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "candidates.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot_dim = "z" if "z" in data.dims else "state" if "state" in data.dims else data.dims[0]
    plot_samples = np.asarray(data.real.values if np.iscomplexobj(data.values) else data.values)
    plot_axis = data.array.dims.index(plot_dim)
    plot_samples = np.moveaxis(plot_samples, plot_axis, 1).reshape(data.n_sample, len(data.coords[plot_dim]), -1).mean(axis=2)
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
        errorbar(data.coords[plot_dim], plot_data.average(sample_error_mode))
        configure_plot(xlabel=plot_dim, ylabel=str(data.name or "result"))
        save_figure(context.artifact_directory / "plots" / "result.pdf")
        plot_artifact = "plots/result.pdf"
    report = f"# Correlator result\n\nSelected candidate: `{candidate_id}`.\nMethod: `{selected.get('method')}`.\n"
    (context.artifact_directory / "report.md").write_text(report, encoding="utf-8")
    artifacts = ["output.nc", "diagnostics/candidates.json", "report.md"] + ([plot_artifact] if plot_artifact else [])
    summary = {"stage_id": context.stage_id, "job_id": context.job_id, "result": str(data.name or "correlator_result"), "decisions": {"candidate_id": candidate_id, "method": selected.get("method")}, "diagnostics": diagnostics, "artifacts": artifacts}
    context.finish(data, summary)
    return {"summary": f"published {data.name or 'correlator result'}", "metrics": diagnostics, "state_keys": [], "artifacts": summary["artifacts"]}
