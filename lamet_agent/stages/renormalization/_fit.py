"""Fit and publish a reusable self-renormalization factor."""

from __future__ import annotations

import json

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.stages.renormalization._plotting import render_fit_diagnostics
from lamet_agent.stages.renormalization.physics import _fit_factor_result, normalize_at_origin
from lamet_agent.stages.renormalization.parameters import authored_kernel_parameters, effective_params


_REFERENCE_K = 0.6551255749279999
_REFERENCE_N_F = 3


def run(context: ToolContext) -> dict[str, object]:
    """Publish a sample-bearing ``(a,z)`` renormalization factor."""
    if set(context.inputs) != {"reference"}:
        raise ValueError("fit_self_renormalization requires exactly the reference input")
    aligned = context.state.get("aligned_inputs")
    if not isinstance(aligned, dict) or "reference" not in aligned:
        raise RuntimeError("inspect_renormalization must run before fitting")
    source = aligned["reference"]
    params = effective_params(context.params)
    kernel_id = str(params["kernel_id"])
    zms_kernel = load_renormalization_kernel(kernel_id)
    kernel_parameters = authored_kernel_parameters(params)
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
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    fit_result = _fit_factor_result(
        prepared,
        short_distance_max_fm=short_distance_max_fm,
        short_distance_min_fm=short_distance_min_fm,
        k=_REFERENCE_K,
        lambda_qcd_gev=float(params["LambdaQCD_gev"]),
        d=float(params["d"]),
        n_f=_REFERENCE_N_F,
        scale_gev=float(params["mu"]),
        zms_kernel=zms_kernel,
        kernel_id=kernel_id,
        kernel_parameters=kernel_parameters,
        svdcut=float(params["svdcut"]),
        lattice_spacing_range_fm=(spacings[0], spacings[-1]),
        sample_error_mode=sample_error_mode,
    )
    factor = fit_result.factor
    attrs = factor.attrs
    attrs.update(
        {
            "scale_gev": float(params["mu"]),
            "kernel_id": kernel_id,
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
        "LambdaQCD_gev": float(params["LambdaQCD_gev"]),
        "kernel_id": kernel_id,
        "kernel_parameters": kernel_parameters,
        "formula": factor.attrs["formula"],
        "factor_dims": factor.dims,
    }
    diagnostic_payload = {**diagnostics, "plot_data": fit_result.plot_data}
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "self_renormalization.json").write_text(
        json.dumps(diagnostic_payload, indent=2), encoding="utf-8"
    )
    rendered = render_fit_diagnostics(
        fit_result.plot_data,
        directory=context.artifact_directory / "plots",
        formats=("pdf",),
    )
    (context.artifact_directory / "report.md").write_text(
        "# Self-renormalization factor\n\n"
        "A reusable sample-bearing factor was fitted on the authored reference grid.\n",
        encoding="utf-8",
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "renormalization_factor",
        "decisions": {"type": "fit", "kernel_id": kernel_id, "short_distance_max_fm": short_distance_max_fm},
        "diagnostics": diagnostics,
        "artifacts": [
            "output.nc",
            "diagnostics/self_renormalization.json",
            *[f"plots/{stem}.pdf" for stem, _caption in rendered],
            "report.md",
        ],
    }
    context.finish(factor, summary)
    return {
        "summary": "published self-renormalization factor",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
