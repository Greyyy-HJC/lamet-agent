"""Publish the selected physical-point extrapolation."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import gvar
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.agent import ToolContext
from lamet_agent.plotting import configure_plot, errorband, save_figure, series_color, start_plot


def _spacing_slug(spacing: float) -> str:
    """Return a stable, filesystem-safe label for one lattice spacing."""
    return f"{spacing:.6g}".replace("-", "m").replace(".", "p")


def _inputs_by_spacing(inputs: object) -> list[tuple[float, list[EnsembleData]]]:
    """Group the fitted input distributions by their lattice spacing."""
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError("extrapolation inputs are unavailable for lattice-spacing plots")
    grouped: defaultdict[float, list[EnsembleData]] = defaultdict(list)
    for item in inputs:
        if not isinstance(item, EnsembleData) or item.ensemble is None:
            raise RuntimeError("extrapolation inputs must retain ensemble metadata for lattice-spacing plots")
        spacing = float(item.ensemble.a_s)
        if not math.isfinite(spacing) or spacing <= 0:
            raise RuntimeError("extrapolation input lattice spacings must be finite and positive")
        grouped[spacing].append(item)
    return sorted(
        (spacing, sorted(values, key=lambda item: float(item.attrs["momentum_gev"])))
        for spacing, values in grouped.items()
    )


def run(context: ToolContext) -> dict[str, object]:
    """Write the physical distribution and finish the stage."""
    if context.params["operation"] != "fit":
        raise ValueError("publish_extrapolation is only available for operation='fit'")
    data = context.state.get("extrapolation_selected_data")
    comparison = context.state.get("extrapolation_comparison")
    if data is None or comparison is None:
        raise RuntimeError("extrapolation selection must run before publish_extrapolation")
    context.state["extrapolation_result"] = data
    data.to_netcdf(context.artifact_directory / "output.nc")
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "extrapolation.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    plot_data = data.at("component", "real") if "component" in data.dims else data
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    candidate = comparison["candidates"][0]
    momentum_dependence = candidate.get("momentum_dependence")
    params = context.params
    if not isinstance(momentum_dependence, dict) or set(momentum_dependence) != {
        f"{float(value):g}" for value in params["pdep_gev"]
    }:
        raise RuntimeError("selected extrapolation candidate is missing the authored pdep_gev diagnostics")
    start_plot()
    for record in momentum_dependence.values():
        errorband(
            data.coords["x"],
            gvar.gvar(record["mean"], record["sdev"]),
            label=rf"$P_z={round(float(record['momentum_gev']), 2):g}\,\mathrm{{GeV}}$",
        )
    errorband(data.coords["x"], plot_data.average(sample_error_mode), label=r"$P_z\to\infty$")
    configure_plot(xlabel=r"$x$", ylabel="physical distribution", legend=True)
    save_figure(context.artifact_directory / "plots" / "momentum_dependence.pdf")
    spacing_plot_artifacts: list[str] = []
    for spacing, inputs in _inputs_by_spacing(context.state.get("scaling_data")):
        start_plot()
        for index, input_data in enumerate(inputs):
            plotted = input_data.real if np.iscomplexobj(input_data.values) else input_data
            errorband(
                plotted.coords["x"],
                plotted.average(sample_error_mode),
                color=series_color(index),
                label=rf"$P_z={round(float(plotted.attrs['momentum_gev']), 2):g}\,\mathrm{{GeV}}$",
            )
        errorband(
            data.coords["x"],
            plot_data.average(sample_error_mode),
            color="black",
            label=r"$a\to0,\;P_z\to\infty$",
        )
        configure_plot(
            xlabel=r"$x$",
            ylabel="physical distribution",
            legend=True,
            title=rf"$a={spacing:.4g}\,\mathrm{{fm}}$",
        )
        stem = f"distribution_a_{_spacing_slug(spacing)}"
        pdf = context.artifact_directory / "plots" / f"{stem}.pdf"
        save_figure(pdf)
        spacing_plot_artifacts.append(f"plots/{pdf.name}")
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "physical_distribution",
        "decisions": {"terms": data.attrs.get("extrapolation_terms"), "pdep_gev": params["pdep_gev"]},
        "diagnostics": comparison,
        "artifacts": [
            "output.nc",
            "diagnostics/extrapolation.json",
            "plots/momentum_dependence.pdf",
            *spacing_plot_artifacts,
        ],
    }
    context.finish(data, summary)
    return {
        "summary": "published physical distribution",
        "metrics": {"candidate_count": len(comparison["candidate_ids"])},
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
