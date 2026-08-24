"""Publish the selected physical-point extrapolation."""

from __future__ import annotations

import json

from lamet_agent.agent import ToolContext
from lamet_agent.plotting import configure_plot, errorband, save_figure, start_plot


def run(context: ToolContext) -> dict[str, object]:
    """Write the physical distribution and finish the stage."""
    data = context.state.get("extrapolation_selected_data")
    comparison = context.state.get("extrapolation_comparison")
    if data is None or comparison is None:
        raise RuntimeError("compare_extrapolations must run before publish_extrapolation")
    context.state["extrapolation_result"] = data
    data.to_netcdf(context.artifact_directory / "output.nc")
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "extrapolation.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    start_plot()
    plot_data = data.at("component", "real")
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorband(data.coords["x"], plot_data.average(sample_error_mode))
    configure_plot(xlabel="x", ylabel="physical distribution")
    save_figure(context.artifact_directory / "plots" / "distribution.pdf")
    (context.artifact_directory / "report.md").write_text("# Extrapolated physical distribution\n\nThe selected candidates were evaluated at the authored physical pion mass and fixed continuum point.\n", encoding="utf-8")
    summary = {"stage_id": context.stage_id, "job_id": context.job_id, "result": "physical_distribution", "decisions": {"physical_pion_mass_gev": context.params["physical_pion_mass_gev"]}, "diagnostics": comparison, "artifacts": ["output.nc", "diagnostics/extrapolation.json", "plots/distribution.pdf", "report.md"]}
    context.finish(data, summary)
    return {"summary": "published physical distribution", "metrics": {"candidate_count": len(comparison["candidate_ids"])}, "state_keys": [], "artifacts": summary["artifacts"]}
