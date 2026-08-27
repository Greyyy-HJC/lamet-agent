"""Publish the selected physical-point extrapolation."""

from __future__ import annotations

import json

import gvar

from lamet_agent.agent import ToolContext
from lamet_agent.plotting import X_LABEL, configure_plot, errorband, momentum_label, save_figure, start_plot


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
    start_plot()
    plot_data = data.at("component", "real") if "component" in data.dims else data
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorband(data.coords["x"], plot_data.average(sample_error_mode))
    configure_plot(xlabel=X_LABEL, ylabel="physical distribution")
    save_figure(context.artifact_directory / "plots" / "distribution.pdf")
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
            label=momentum_label(record["momentum_gev"]),
        )
    errorband(data.coords["x"], plot_data.average(sample_error_mode), label="Pz→∞")
    configure_plot(xlabel=X_LABEL, ylabel="physical distribution", legend=True)
    save_figure(
        context.artifact_directory / "plots" / "momentum_dependence.pdf",
        context.artifact_directory / "plots" / "momentum_dependence.svg",
    )
    mass_text = (
        f" and physical pion mass {float(data.attrs['physical_pion_mass_gev']):g} GeV"
        if "physical_pion_mass_gev" in data.attrs
        else ""
    )
    (context.artifact_directory / "report.md").write_text(
        "# Extrapolated physical distribution\n\n"
        f"The selected model was evaluated at the continuum, infinite-momentum point{mass_text}.\n\n"
        f"Momentum-dependence diagnostics were evaluated at Pz={params['pdep_gev']} GeV.\n",
        encoding="utf-8",
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "physical_distribution",
        "decisions": {"terms": data.attrs.get("extrapolation_terms"), "pdep_gev": params["pdep_gev"]},
        "diagnostics": comparison,
        "artifacts": [
            "output.nc",
            "diagnostics/extrapolation.json",
            "plots/distribution.pdf",
            "plots/momentum_dependence.pdf",
            "plots/momentum_dependence.svg",
            "report.md",
        ],
    }
    context.finish(data, summary)
    return {
        "summary": "published physical distribution",
        "metrics": {"candidate_count": len(comparison["candidate_ids"])},
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
