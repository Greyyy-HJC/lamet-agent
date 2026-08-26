"""Run the complete native LA/NLA Fourier candidate scan."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.plotting import (
    X_LABEL,
    configure_plot,
    errorband,
    momentum_label,
    quasi_distribution_label,
    save_figure,
    series_color,
    start_plot,
    vline,
)
from lamet_agent.stages.fourier_transform.physics import scan_fourier_transform


def run(context: ToolContext) -> dict[str, object]:
    """Evaluate the authored scan and publish the selected quasi-distribution."""
    if "tail_inspection" not in context.state:
        raise RuntimeError("inspect_long_distance must run before run_fourier_scan")
    scheme_scan = context.params["scheme_scan"]
    conventions = context.state.get("fourier_conventions")
    if not isinstance(conventions, dict):
        raise RuntimeError("inspect_long_distance did not derive Fourier conventions")
    scan = {
        "orders": scheme_scan["order"],
        "sector": scheme_scan["sector"],
        "lambda0_gev": scheme_scan["Lambda0_gev"],
        "prior_widths": scheme_scan["posterior_prior_error_scale"],
        "model_average": scheme_scan["model_average"],
        "max_schemes": scheme_scan["max_schemes"],
        "component": conventions["component"],
        "output_scale": conventions["output_scale"],
        "q_min": scheme_scan["q_min"],
    }
    source = context.state.get("fourier_input")
    if not isinstance(source, EnsembleData):
        raise RuntimeError("inspect_long_distance did not retain an EnsembleData input")
    grid = context.params["quasi_y_ls"]
    if isinstance(grid, dict):
        grid = np.linspace(float(grid["start"]), float(grid["stop"]), int(grid["num"])).tolist()
    is_da = str(context.manifest["metadata"]["target_observable"]).lower() == "da"
    da = context.params["da"] if is_da else None
    result = scan_fourier_transform(
        source,
        grid,
        transform=conventions["transform"],
        tail={
            "models": conventions["tail_models"],
            "z_min_fm": context.params["zmin_fm"],
            "z_max_fm": context.params["zmax_fm"],
            "extent_fm": context.params["zmax_ext_fm"],
            "smoothing_method": context.params["smooth"],
        },
        scan=scan,
        observable=context.manifest["metadata"]["target_observable"].upper(),
        phase_transfer_da=da["phase_transfer_da"] if da is not None else False,
        psi1_flavor_class=da["psi1_flavor_class"] if da is not None else "heavy",
        psi2_flavor_class=da["psi2_flavor_class"] if da is not None else "heavy",
        workers=context.workers,
        _parallel=context._parallel,
    )
    scanned = result["data"]
    output_attrs = dict(scanned.attrs)
    output_attrs.update(
        {
            "target_observable": context.manifest["metadata"]["target_observable"],
            "parton": conventions["parton"],
            "gfix": conventions["gfix"],
        }
    )
    output = EnsembleData(
        scanned.ensemble,
        scanned.resample,
        [np.asarray(sample) for sample in scanned.values],
        scanned.dims,
        scanned.coords,
        attrs=output_attrs,
        name="quasi_distribution",
    )
    output.to_netcdf(context.artifact_directory / "output.nc")
    selected_range = result["selected_range"]
    selected_range_label = f"zmin_{selected_range['z_min_fm']:g}_zmax_{selected_range['z_max_fm']:g}".replace(".", "p")
    model_labels = [candidate["label"] for candidate in result["model_candidates"]]
    diagnostics = {
        "selected_range_label": selected_range_label,
        "fit_model_labels": model_labels,
        "fit_model_weights": result["weights"],
        "selected_fit_model_labels": result["selected_labels"],
        "selected_Q": result["selected_candidate"]["Q"],
        "selected_chi2_dof": result["selected_candidate"]["chi2_dof"],
        "range_candidate_count": len(result["range_candidates"]),
        "model_candidate_count": len(result["model_candidates"]),
        "sample_count": output.n_sample,
        "x_count": len(output.coords["x"]),
        "workers": result["workers"],
    }
    model_weights = dict(zip(model_labels, result["weights"]))
    selected_labels = set(result["selected_labels"])
    candidate_table = [
        {
            key: candidate[key]
            for key in (
                "label",
                "model_id",
                "z_min_fm",
                "z_max_fm",
                "order",
                "prior_width",
                "smoothing_method",
                "smoothing_width_fm",
                "parameter_mean",
                "parameter_sdev",
                "chi2",
                "dof",
                "chi2_dof",
                "Q",
                "logGBF",
                "n_failed_samples",
                "sample_failures",
            )
            if key in candidate
        }
        | {"selected": candidate["label"] in selected_labels, "model_weight": model_weights[candidate["label"]]}
        for candidate in result["model_candidates"]
    ]
    range_table = [
        {
            key: candidate[key]
            for key in (
                "model_id",
                "z_min_fm",
                "z_max_fm",
                "order",
                "prior_width",
                "fit_success",
                "fit_parameters",
                "chi2",
                "dof",
                "chi2_dof",
                "Q",
                "logGBF",
                "error",
            )
            if key in candidate
        }
        | {
            "selected": candidate["model_id"] == selected_range["model_id"]
            and float(candidate["z_min_fm"]) == float(selected_range["z_min_fm"])
            and float(candidate["z_max_fm"]) == float(selected_range["z_max_fm"])
        }
        for candidate in result["range_candidates"]
    ]
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "candidates.json").write_text(
        json.dumps(candidate_table, indent=2), encoding="utf-8"
    )
    (context.artifact_directory / "diagnostics" / "ranges.json").write_text(
        json.dumps(range_table, indent=2), encoding="utf-8"
    )
    artifacts = [
        "output.nc",
        "diagnostics/candidates.json",
        "diagnostics/ranges.json",
        "output_xdep.pdf",
        "output_re.pdf",
        "output_im.pdf",
    ]
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    momentum = float(source.attrs["momentum_gev"])
    pz_label = momentum_label(momentum)
    component = str(scan["component"])
    component_series = {
        "re": (("real", pz_label, 0),),
        "im": (("imag", pz_label, 1),),
        "both": (
            ("real", "Re", 0),
            ("imag", "Im", 1),
        ),
    }[component]
    start_plot()
    for data_component, label, color_index in component_series:
        errorband(
            output.coords["x"],
            getattr(output, data_component).average(sample_error_mode),
            color=series_color(color_index),
            label=label,
        )
    configure_plot(
        xlabel=X_LABEL,
        ylabel=quasi_distribution_label(component),
        legend=True,
    )
    save_figure(context.artifact_directory / "output_xdep.pdf")
    extended = result["selected_candidate"]["extended"]
    z_min_fm = float(selected_range["z_min_fm"])
    z_max_fm = float(selected_range["z_max_fm"])
    extension_z = np.asarray(extended.coords["z"], dtype=float)
    extension_coords = extension_z[extension_z >= z_min_fm - 1e-12].tolist()
    if not extension_coords:
        raise RuntimeError("selected Fourier extension does not reach z_min_fm")
    extension_segment = extended.at("z", extension_coords)
    ioffe_time_scale = float(source.attrs["momentum_gev"]) / HBAR_C_GEV_FM
    input_lambda = np.asarray(source.coords["z"], dtype=float) * ioffe_time_scale
    extension_lambda = np.asarray(extension_segment.coords["z"], dtype=float) * ioffe_time_scale
    lambda_min = z_min_fm * ioffe_time_scale
    lambda_max = z_max_fm * ioffe_time_scale
    for component, filename in (("real", "output_re.pdf"), ("imag", "output_im.pdf")):
        start_plot()
        errorband(
            input_lambda, getattr(source, component).average(sample_error_mode), color=series_color(0), label="input"
        )
        errorband(
            extension_lambda,
            getattr(extension_segment, component).average(sample_error_mode),
            color=series_color(1),
            label="extrapolation",
        )
        vline(lambda_min, color="black", linestyle="dashed")
        vline(lambda_max, color="black", linestyle="dashed")
        configure_plot(
            xlabel=r"$\lambda = zP^z$",
            ylabel=r"Re $h(\lambda)$" if component == "real" else r"Im $h(\lambda)$",
            legend=True,
        )
        save_figure(context.artifact_directory / filename)
    (context.artifact_directory / "report.md").write_text(
        f"# Fourier transform\n\nSelected range: `{selected_range_label}`.\n\nSelected models: {', '.join(result['selected_labels'])}.\n",
        encoding="utf-8",
    )
    artifacts.append("report.md")
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "quasi_distribution",
        "decisions": {"selected_range_label": selected_range_label, "fit_model_labels": result["selected_labels"]},
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    context.finish(output, summary)
    return {
        "summary": "published scanned quasi distribution",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": artifacts,
    }
