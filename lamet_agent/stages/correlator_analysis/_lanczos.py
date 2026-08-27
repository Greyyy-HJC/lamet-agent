"""Run the terminal nested-resampled Lanczos correlator analysis."""

from __future__ import annotations

import json

import gvar
import numpy as np
import xarray as xr

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import configure_plot, errorline, save_figure, start_plot
from lamet_agent.parallel.lanczos import (
    analyze_prepared_lanczos,
)


def _plot_values(samples: np.ndarray) -> np.ndarray:
    """Return nan-aware gvars for a Lanczos diagnostic curve."""
    with np.errstate(invalid="ignore", divide="ignore"):
        center = np.nanmedian(samples, axis=0)
        error = np.nanstd(samples, axis=0, ddof=1)
    return np.asarray(gvar.gvar(center, error), dtype=object)


def run(context: ToolContext) -> dict[str, object]:
    """Run Lanczos, publish its artifacts, and finish the correlator job."""
    if context.params.get("analysis_method") != "lanczos":
        raise ValueError("run_lanczos_analysis is only available for Lanczos jobs")
    prepared = context.state.get("lanczos_prepared")
    if not isinstance(prepared, dict):
        raise RuntimeError("inspect_lanczos_inputs must run before run_lanczos_analysis")
    settings = context.params
    max_states = int(context.params["nstate"][0])
    component = {"re": "real", "im": "imag", "both": "both"}[context.params["component"]]
    metadata = context.manifest["metadata"]
    resampling = metadata["resample_mode"]
    result = analyze_prepared_lanczos(
        prepared,
        components=component,
        max_states=max_states,
        resampling=resampling,
        bootstrap_samples=metadata.get("samples"),
        bin_size=int(metadata["bin_size"]),
        inner_samples=int(settings["inner_samples"]),
        precision=int(settings["precision"]),
        seed=int(context.manifest["metadata"]["random_seed"]),
        workers=context.workers,
        _parallel=context._parallel,
    )
    inspection = prepared["inspection"]
    source_data = prepared["source_data"]
    attrs = {
        **source_data.attrs,
        "analysis_method": "lanczos",
        "lanczos_scope": str(settings["scope"]),
        "lanczos_iterations": int(inspection["iterations"]),
        "lanczos_inner_samples": int(settings["inner_samples"]),
        "lanczos_precision": int(settings["precision"]),
        "lanczos_t0": int(inspection["lanczos_t0"]),
        "lanczos_time_step": int(inspection["lanczos_time_step"]),
        "sample_error_mode": str(context.manifest["metadata"]["sample_error_mode"]),
    }
    artifacts = ["output.nc", "plots/result.pdf", "plots/result.svg", "diagnostics/lanczos.json", "report.md"]
    start_plot()
    if settings["scope"] == "2pt_spectrum":
        values = np.asarray(result["values"], dtype=float)
        output = EnsembleData(
            source_data.ensemble,
            resampling,
            [sample for sample in values],
            ["channel", "iteration", "state"],
            {
                "channel": result["channels"],
                "iteration": list(range(1, int(inspection["iterations"]) + 1)),
                "state": list(range(max_states)),
            },
            attrs=attrs,
            name="lanczos_energy",
        )
        for channel, label in enumerate(result["channels"]):
            errorline(
                output.coords["iteration"],
                _plot_values(values[:, channel, :, 0]),
                label=f"{label} ground state",
            )
        configure_plot(
            xlabel="Lanczos iteration m",
            ylabel="Energy (lattice units)",
            legend=True,
        )
        decisions = {"method": "lanczos", "scope": "2pt_spectrum"}
    else:
        values = np.asarray(result["values"])
        z_values = list(prepared["z_values"])
        output = EnsembleData(
            source_data.ensemble,
            resampling,
            [sample for sample in values],
            ["z"],
            {"z": z_values},
            attrs={**attrs, "component": str(context.params["component"])},
            name="bare_matrix_element",
        )
        if context.params["component"] in {"re", "both"}:
            errorline(z_values, _plot_values(np.real(values)), label="Re")
        if context.params["component"] in {"im", "both"}:
            errorline(z_values, _plot_values(np.imag(values)), marker="s", label="Im")
        configure_plot(xlabel=r"$z~/~a$", ylabel="bare matrix element", legend=True)
        details_path = context.artifact_directory / "diagnostics" / "state_matrices.nc"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        xr.DataArray(
            result["matrices"],
            dims=("sample", "z", "component", "final_state", "initial_state"),
            coords={
                "sample": list(range(int(result["outer_samples"]))),
                "z": z_values,
                "component": result["components"],
                "final_state": list(range(max_states)),
                "initial_state": list(range(max_states)),
            },
            name="lanczos_matrix_element",
            attrs=attrs,
        ).to_netcdf(details_path)
        artifacts.append("diagnostics/state_matrices.nc")
        decisions = {"method": "lanczos", "scope": "3pt_matrix"}
        context.state["lanczos_state_matrices"] = result["matrices"]
    save_figure(
        context.artifact_directory / "plots" / "result.pdf",
        context.artifact_directory / "plots" / "result.svg",
    )
    output.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {
        "outer_samples": int(result["outer_samples"]),
        "states": max_states,
        "inspection": inspection,
    }
    diagnostics_path = context.artifact_directory / "diagnostics" / "lanczos.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (context.artifact_directory / "report.md").write_text(
        "# Lanczos correlator result\n\n"
        f"Scope: `{settings['scope']}`.\n\n"
        f"Iterations: {inspection['iterations']}.\n\n"
        + (f"{inspection.get('point_usage_warning')}\n" if inspection.get("point_usage_warning") else ""),
        encoding="utf-8",
    )
    context.state["correlator_result"] = output
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": str(output.name),
        "decisions": decisions,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    context.finish(output, summary)
    return {
        "summary": f"published {output.name}",
        "metrics": diagnostics,
        "state_keys": ["correlator_result"],
        "artifacts": artifacts,
    }
