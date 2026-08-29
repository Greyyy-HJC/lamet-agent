"""Apply the inspected kernel matrix and publish the matched distribution."""

from __future__ import annotations

import json

import gvar
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import (
    configure_plot,
    errorband,
    hline,
    save_figure,
    series_color,
    start_plot,
)
from lamet_agent.stages.perturbative_matching.physics import apply_matrix, is_even_about_zero


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("matching accepts one quasi source")
        return value[0]
    return value


def _nonnegative_x(values: EnsembleData, x_values: list[float]) -> tuple[EnsembleData, list[float]]:
    selected = [value for value in x_values if float(value) >= -1e-12]
    if not selected:
        raise RuntimeError("matching plot has no nonnegative x coordinates")
    return values.at("x", selected), selected


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
    kernel_arguments = {
        "x_out": np.asarray(x_out, dtype=float),
        "x_in": np.asarray(x_in, dtype=float),
        "momentum_gev": float(momentum),
        "scale_gev": float(context.params["mu"]),
    }
    if context.params["scheme"] == "hybrid":
        kernel_arguments["zs_fm"] = float(context.params["zs_fm"])
    kernel_arguments.update(kernel_parameters)
    matrix = kernel(**kernel_arguments)
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
        "kernel_parameters": kernel_parameters,
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "matching.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    start_plot()
    hline(0.0, color="black")
    plot_min = np.inf
    plot_max = -np.inf
    plotted_x: list[float] = []
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    quasi_label = rf"quasi, $P_z={round(float(momentum), 2):g}\,\mathrm{{GeV}}$"
    series = (
        (data, x_in, quasi_label, series_color(0)),
        (result, x_out, "light-cone", series_color(1)),
    )
    if is_even_about_zero(data):
        series = tuple(
            (*_nonnegative_x(values, list(x_values)), label, color) for values, x_values, label, color in series
        )
    for values, x_values, label, color in series:
        plotted = values.real if np.iscomplexobj(values.values) else values
        average = plotted.average(sample_error_mode)
        center = np.asarray(gvar.mean(average), dtype=float)
        error = np.asarray(gvar.sdev(average), dtype=float)
        plot_min = min(plot_min, float(np.min(center - error)))
        plot_max = max(plot_max, float(np.max(center + error)))
        plotted_x.extend(float(value) for value in np.asarray(x_values, dtype=float))
        errorband(x_values, average, color=color, label=label)
    configure_plot(
        xlabel=r"$x$",
        ylabel="distribution",
        xlim=(float(np.min(plotted_x)) - 0.01, float(np.max(plotted_x)) + 0.01),
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
