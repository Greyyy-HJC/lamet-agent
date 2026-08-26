"""Fit and publish a reusable self-renormalization factor."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import configure_plot, errorbar, save_figure, start_plot
from lamet_agent.stages.renormalization.physics import fit_factor, normalize_at_origin
from lamet_agent.stages.renormalization.parameters import effective_params


_REFERENCE_K = 0.6551255749279999
_REFERENCE_N_F = 3
_REFERENCE_ZMS_MODEL = "pdf_nlo"


def run(context: ToolContext) -> dict[str, object]:
    """Publish a sample-bearing ``(a,z)`` renormalization factor."""
    if set(context.inputs) != {"reference"}:
        raise ValueError("fit_self_renormalization requires exactly the reference input")
    aligned = context.state.get("aligned_inputs")
    if not isinstance(aligned, dict) or "reference" not in aligned:
        raise RuntimeError("inspect_renormalization must run before fitting")
    source = aligned["reference"]
    params = effective_params(context.params)
    if isinstance(source, list):
        prepared = [normalize_at_origin(item) if params["normalization"] else item for item in source]
    else:
        prepared = normalize_at_origin(source) if params["normalization"] else source
    items = prepared if isinstance(prepared, list) else [prepared]
    positive_z = sorted({float(value) for item in items for value in item.coords["z"] if float(value) > 0})
    if len(positive_z) < 3:
        raise ValueError("self-renormalization reference requires at least three positive z coordinates")
    short_distance_min_fm = positive_z[0]
    short_distance_max_fm = positive_z[2]
    spacings = sorted(
        {
            float(value)
            for item in items
            for value in (item.coords["a"] if "a" in item.dims else [item.attrs.get("lattice_spacing_fm")])
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    if not spacings:
        raise ValueError("self-renormalization reference has no lattice-spacing coordinates")
    factor = fit_factor(
        prepared,
        short_distance_max_fm=short_distance_max_fm,
        short_distance_min_fm=short_distance_min_fm,
        k=_REFERENCE_K,
        lambda_qcd_gev=float(params["LambdaQCD_gev"]),
        d=float(params["d"]),
        n_f=_REFERENCE_N_F,
        scale_gev=float(params["mu"]),
        zms_model=_REFERENCE_ZMS_MODEL,
        svdcut=float(params["svdcut"]),
        lattice_spacing_range_fm=(spacings[0], spacings[-1]),
    )
    attrs = factor.attrs
    attrs.update(
        {
            "scale_gev": float(params["mu"]),
            "zms_model": _REFERENCE_ZMS_MODEL,
            "short_distance_range_fm": json.dumps(
                {"min": short_distance_min_fm, "max": short_distance_max_fm}, sort_keys=True
            ),
            "lattice_spacing_range_fm": json.dumps({"min": spacings[0], "max": spacings[-1]}, sort_keys=True),
        }
    )
    factor = EnsembleData(
        factor.ensemble,
        factor.resample,
        [sample for sample in factor.values],
        factor.dims,
        factor.coords,
        attrs=attrs,
        name=factor.name,
    )
    context.state["self_renormalization"] = {
        "factor": factor,
        "m0_gev": factor.attrs.get("m0_gev"),
        "formula": factor.attrs.get("formula"),
    }
    factor.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {
        "short_distance_min_fm": short_distance_min_fm,
        "short_distance_max_fm": short_distance_max_fm,
        "lattice_spacing_range_fm": [spacings[0], spacings[-1]],
        "z_range_fm": [float(min(factor.coords["z"])), float(max(factor.coords["z"]))],
        "m0_gev": factor.attrs.get("m0_gev"),
        "d": factor.attrs.get("d"),
        "k": factor.attrs.get("k"),
        "n_f": factor.attrs.get("n_f"),
        "scale_gev": factor.attrs.get("scale_gev"),
        "LambdaQCD_gev": factor.attrs.get("LambdaQCD_gev"),
        "zms_model": factor.attrs.get("zms_model"),
        "formula": factor.attrs["formula"],
        "factor_dims": factor.dims,
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "self_renormalization.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    start_plot()
    plot_data = factor.real if np.iscomplexobj(factor.values) else factor
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorbar(factor.coords["a"], np.mean(plot_data.average(sample_error_mode), axis=-1))
    configure_plot(xlabel="a [fm]", ylabel="Z_R")
    save_figure(context.artifact_directory / "plots" / "factor.pdf")
    (context.artifact_directory / "report.md").write_text(
        "# Self-renormalization factor\n\nA reusable sample-bearing factor was fitted on the authored reference grid.\n",
        encoding="utf-8",
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "renormalization_factor",
        "decisions": {"short_distance_max_fm": short_distance_max_fm},
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/self_renormalization.json", "plots/factor.pdf", "report.md"],
    }
    context.finish(factor, summary)
    return {
        "summary": "published self-renormalization factor",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
