"""Fit and publish a reusable self-renormalization factor."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import configure_plot, errorbar, save_figure, start_plot
from lamet_agent.stages.renormalization.physics import fit_factor, normalize_at_origin


def run(context: ToolContext, *, short_distance_max_fm: float) -> dict[str, object]:
    """Publish a sample-bearing ``(a,z)`` renormalization factor."""
    if context.params.get("operation") != "fit_factor":
        raise ValueError("fit_self_renormalization is available only for operation='fit_factor'")
    allowed_range = context.params["self_renormalization"]["short_distance_range_fm"]
    if short_distance_max_fm < float(allowed_range["min"]) or short_distance_max_fm > float(allowed_range["max"]):
        raise ValueError("short_distance_max_fm must lie within the authored short-distance range")
    aligned = context.state.get("aligned_inputs")
    if not isinstance(aligned, dict) or "reference" not in aligned:
        raise RuntimeError("inspect_renormalization must run before fitting")
    source = aligned["reference"]
    if isinstance(source, list):
        prepared = [normalize_at_origin(item) if context.params.get("normalization") else item for item in source]
    else:
        prepared = normalize_at_origin(source) if context.params.get("normalization") else source
    settings = context.params["self_renormalization"]
    spacing_range = settings["lattice_spacing_range_fm"]
    factor = fit_factor(
        prepared,
        short_distance_max_fm=short_distance_max_fm,
        short_distance_min_fm=float(allowed_range["min"]),
        k=float(settings["k"]),
        lambda_qcd_gev=float(settings["LambdaQCD_gev"]),
        d=float(settings["d"]),
        n_f=int(settings["n_f"]),
        scale_gev=float(context.params["mu"]),
        zms_model=str(settings["zms_model"]),
        lattice_spacing_range_fm=(float(spacing_range["min"]), float(spacing_range["max"])),
    )
    attrs = factor.attrs
    attrs.update({"scale_gev": float(context.params["mu"]), "zms_model": context.params["self_renormalization"]["zms_model"], "short_distance_range_fm": json.dumps(context.params["self_renormalization"]["short_distance_range_fm"], sort_keys=True), "lattice_spacing_range_fm": json.dumps(context.params["self_renormalization"]["lattice_spacing_range_fm"], sort_keys=True)})
    factor = EnsembleData(factor.ensemble, factor.resample, [sample for sample in factor.values], factor.dims, factor.coords, attrs=attrs, name=factor.name)
    context.state["self_renormalization"] = {"factor": factor, "m0_gev": factor.attrs.get("m0_gev"), "formula": factor.attrs.get("formula")}
    factor.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {"short_distance_max_fm": short_distance_max_fm, "formula": factor.attrs["formula"], "factor_dims": factor.dims}
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "self_renormalization.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    start_plot()
    plot_data = factor.real if np.iscomplexobj(factor.values) else factor
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorbar(factor.coords["a"], np.mean(plot_data.average(sample_error_mode), axis=-1))
    configure_plot(xlabel="a [fm]", ylabel="Z_R")
    save_figure(context.artifact_directory / "plots" / "factor.pdf")
    (context.artifact_directory / "report.md").write_text("# Self-renormalization factor\n\nA reusable sample-bearing factor was fitted on the authored reference grid.\n", encoding="utf-8")
    summary = {"stage_id": context.stage_id, "job_id": context.job_id, "result": "renormalization_factor", "decisions": {"short_distance_max_fm": short_distance_max_fm}, "diagnostics": diagnostics, "artifacts": ["output.nc", "diagnostics/self_renormalization.json", "plots/factor.pdf", "report.md"]}
    context.finish(factor, summary)
    return {"summary": "published self-renormalization factor", "metrics": diagnostics, "state_keys": [], "artifacts": summary["artifacts"]}
