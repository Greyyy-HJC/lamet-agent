"""Apply the inspected kernel matrix and publish the matched distribution."""

from __future__ import annotations

import json

import gvar
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import (
    COLOR_CYCLE,
    configure_plot,
    errorband,
    hline,
    save_figure,
    start_plot,
)
from lamet_agent.stages.perturbative_matching.physics import apply_matrix


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("matching accepts one quasi source")
        return value[0]
    return value


def run(context: ToolContext) -> dict[str, object]:
    """Build one matching matrix, apply it sample-wise, and finish."""
    kernel = context.state.get("kernel")
    data = context.state.get("quasi")
    if kernel is None or data is None:
        raise RuntimeError("inspect_kernel must run before apply_matching")
    x_in = list(data.coords["x"])
    lc_x_ls = context.params["lc_x_ls"]
    x_out = (
        list(lc_x_ls)
        if isinstance(lc_x_ls, list)
        else [value for value in x_in if float(lc_x_ls["start"]) <= float(value) <= float(lc_x_ls["stop"])]
    )
    if isinstance(lc_x_ls, dict):
        if not x_out:
            raise ValueError("lc_x_ls selects no quasi-grid points")
    momentum = data.attrs.get("momentum_gev")
    if not isinstance(momentum, (int, float)) or isinstance(momentum, bool):
        raise ValueError("quasi input requires momentum_gev")
    kernel_parameters = dict(context.params["kernel_parameters"])
    if "rgr_kappa" in kernel_parameters:
        kernel_parameters["kappa"] = kernel_parameters.pop("rgr_kappa")
    if "rgr_mu_min_gev" in kernel_parameters:
        kernel_parameters["mu_min_gev"] = kernel_parameters.pop("rgr_mu_min_gev")
    if "hybrid" in context.params:
        kernel_parameters["zs_fm"] = float(context.params["hybrid"]["zs_fm"])
    matrix = kernel(
        np.asarray(x_out, dtype=float),
        np.asarray(x_in, dtype=float),
        momentum_gev=float(momentum),
        scale_gev=float(context.params["mu"]),
        **kernel_parameters,
    )
    matrix = np.asarray(matrix)
    if matrix.shape != (len(x_out), len(x_in)) or not np.all(np.isfinite(matrix)):
        raise ValueError("kernel returned an invalid matching matrix shape or value")
    result = apply_matrix(data, matrix, x_out)
    attrs = result.attrs
    attrs.update(
        {
            "kernel_id": context.params["kernel_id"],
            "mu": float(context.params["mu"]),
            "kernel_parameters": json.dumps(context.params["kernel_parameters"], sort_keys=True),
            "units": '{"values":"dimensionless","x":"dimensionless"}',
        }
    )
    result = EnsembleData(
        result.ensemble,
        result.resample,
        [sample for sample in result.values],
        result.dims,
        result.coords,
        attrs=attrs,
        name=result.name,
    )
    context.state["matching_result"] = {"data": result, "matrix": matrix, "x_in": x_in, "x_out": x_out}
    result.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {
        "kernel_id": context.params["kernel_id"],
        "matrix_shape": list(matrix.shape),
        "x_in_count": len(x_in),
        "x_out_count": len(x_out),
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "matching.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    start_plot()
    hline(0.0, color="black")
    plot_min = np.inf
    plot_max = -np.inf
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    for values, x_values, label, color in (
        (data, x_in, "quasi", COLOR_CYCLE[0]),
        (result, x_out, "light-cone", COLOR_CYCLE[1]),
    ):
        plotted = values.real if np.iscomplexobj(values.values) else values
        average = plotted.average(sample_error_mode)
        center = np.asarray(gvar.mean(average), dtype=float)
        error = np.asarray(gvar.sdev(average), dtype=float)
        plot_min = min(plot_min, float(np.min(center - error)))
        plot_max = max(plot_max, float(np.max(center + error)))
        errorband(x_values, average, color=color, label=label)
    all_x = np.concatenate([np.asarray(x_in, dtype=float), np.asarray(x_out, dtype=float)])
    configure_plot(
        xlabel="x",
        ylabel="distribution",
        xlim=(float(np.min(all_x)) - 0.01, float(np.max(all_x)) + 0.01),
        ylim=(plot_min - 0.2, plot_max + 1.0),
        legend=True,
    )
    plot_pdf = context.artifact_directory / "plots" / "result.pdf"
    plot_svg = context.artifact_directory / "plots" / "result.svg"
    save_figure(plot_pdf, plot_svg)
    document = str(context.state.get("kernel_inspection", {}).get("document", "")).strip()
    report = f"# Perturbative matching\n\nKernel: `{context.params['kernel_id']}`.\nScheme: `{context.params['scheme']}`.\n\nPlot: [PDF](plots/result.pdf) ([SVG](plots/result.svg)).\n\n{document}\n"
    (context.artifact_directory / "report.md").write_text(report, encoding="utf-8")
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "matched_distribution",
        "decisions": {
            "kernel_id": context.params["kernel_id"],
            "scheme": context.params["scheme"],
            "mu": context.params["mu"],
        },
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/matching.json", "plots/result.pdf", "plots/result.svg", "report.md"],
    }
    context.finish(result, summary)
    return {
        "summary": "published matched distribution",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
