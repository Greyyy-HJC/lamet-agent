"""Publish the reference-compatible extrapolation systematics budget."""

from __future__ import annotations

import json

import gvar as gv
import numpy as np
import xarray as xr

from lamet_agent.agent import ToolContext
from lamet_agent.plotting import band, bar, configure_plot, line, save_figure, start_plot
from lamet_agent.stages.extrapolation.physics import load_data


_COMPONENTS = ("zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations")


def _aligned_mean(data, x: np.ndarray, sample_error_mode: str) -> np.ndarray:
    variant_x = np.asarray(data.coords["x"], dtype=float)
    mean = np.asarray(gv.mean(data.average(sample_error_mode)), dtype=float)
    if np.array_equal(variant_x, x):
        return mean
    if variant_x.ndim != 1 or np.any(np.diff(variant_x) <= 0):
        raise ValueError("systematics variants require a strictly increasing one-dimensional x grid")
    if x[0] < variant_x[0] or x[-1] > variant_x[-1]:
        raise ValueError("systematics variant x coverage does not contain the main result grid")
    return np.interp(x, variant_x, mean)


def run(context: ToolContext) -> dict[str, object]:
    """Combine authored extrapolation branches using envelope and quadrature rules."""
    if context.params["operation"] != "systematics_budget":
        raise ValueError("publish_systematics_budget requires operation='systematics_budget'")
    params = context.params
    if params["systematics_prescription"] != "variant_envelope_quadrature":
        raise ValueError("unsupported extrapolation systematics prescription")
    sources = context.inputs["distributions"]
    groups = params["systematics_groups"]
    data = [load_data(source) for source in sources]
    main = data[groups["main"]]
    if main.dims != ["x"]:
        raise ValueError("the main systematics input must be a one-dimensional x distribution")
    x = np.asarray(main.coords["x"], dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(np.diff(x) <= 0):
        raise ValueError("the main systematics input requires a strictly increasing x grid")
    sample_error_mode = str(
        main.attrs.get("sample_error_mode", context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    )
    variant_modes = {
        str(
            item.attrs.get(
                "sample_error_mode", context.manifest.get("metadata", {}).get("sample_error_mode", "covariance")
            )
        )
        for item in data
    }
    if variant_modes != {sample_error_mode}:
        raise ValueError("all systematics inputs must share sample_error_mode")
    main_average = main.average(sample_error_mode)
    central = np.asarray(gv.mean(main_average), dtype=float)
    stat_sdev = np.asarray(gv.sdev(main_average), dtype=float)
    components: dict[str, np.ndarray] = {}
    for name in _COMPONENTS:
        indices = groups[name]
        if not indices:
            components[name] = np.zeros_like(central)
            continue
        variants = np.stack([_aligned_mean(data[index], x, sample_error_mode) for index in indices])
        components[name] = (
            np.abs(variants[0] - central) if len(indices) == 1 else np.max(variants, axis=0) - np.min(variants, axis=0)
        )
    total_systematic = np.sqrt(sum(components[name] ** 2 for name in _COMPONENTS))
    total_error = np.sqrt(stat_sdev**2 + total_systematic**2)

    dataset = xr.Dataset(
        {
            "central": (("x",), central),
            "stat_sdev": (("x",), stat_sdev),
            **{name: (("x",), components[name]) for name in _COMPONENTS},
            "total_systematic_error": (("x",), total_systematic),
            "total_error": (("x",), total_error),
        },
        coords={"x": x},
        attrs={"operation": "systematics_budget", "prescription": params["systematics_prescription"]},
    )
    dataset.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {name: np.asarray(dataset[name], dtype=float).tolist() for name in dataset.data_vars}
    diagnostics["x"] = x.tolist()
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "systematics_budget.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    start_plot()
    bar(x, total_error, width=0.078, color="0.82", label="total error")
    bar(x, total_systematic, width=0.06, color="0.65", label="total systematic")
    configure_plot(xlabel="x", ylabel="distribution uncertainty", legend=True)
    save_figure(context.artifact_directory / "plots" / "systematics_budget.pdf")

    start_plot()
    band(x, central - total_error, central + total_error, color="0.88")
    band(x, central - stat_sdev, central + stat_sdev, color="0.70")
    line(x, central, color="0.35")
    configure_plot(xlabel="x", ylabel="physical distribution")
    save_figure(context.artifact_directory / "plots" / "distribution_with_systematics.pdf")

    (context.artifact_directory / "report.md").write_text(
        "# Extrapolation systematics budget\n\nVariant envelopes are combined in quadrature with the statistical uncertainty.\n",
        encoding="utf-8",
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "systematics_budget",
        "decisions": {"systematics_groups": groups, "systematics_prescription": params["systematics_prescription"]},
        "diagnostics": {"point_count": int(x.size), "sources": list(_COMPONENTS)},
        "artifacts": [
            "output.nc",
            "diagnostics/systematics_budget.json",
            "plots/systematics_budget.pdf",
            "plots/distribution_with_systematics.pdf",
            "report.md",
        ],
    }
    context.state["systematics_budget"] = dataset
    context.finish(main, summary)
    return {
        "summary": "published extrapolation systematics budget",
        "metrics": summary["diagnostics"],
        "state_keys": ["systematics_budget"],
        "artifacts": summary["artifacts"],
    }
