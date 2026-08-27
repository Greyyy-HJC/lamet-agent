"""Apply external or reusable renormalization and publish the result."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.plotting import configure_plot, errorline, save_figure, start_plot
from lamet_agent.stages.renormalization.physics import (
    divide_by_constant,
    hybrid_ratio,
    log_m,
    normalize_at_origin,
    ratio,
    zmsbar_log,
)
from lamet_agent.stages.renormalization.parameters import effective_params


def run(context: ToolContext) -> dict[str, object]:
    """Apply the declared renormalization mode and call ``finish``."""
    aligned = context.state.get("aligned_inputs")
    if not isinstance(aligned, dict) or "target" not in aligned:
        raise RuntimeError("inspect_renormalization must run before application")
    target = aligned["target"]
    if isinstance(target, list):
        raise ValueError("target must be one source")
    params = effective_params(context.params)
    strategy = params["strategy"]
    scheme = params["scheme"]
    normalize_inputs = bool(params["normalization"])
    if normalize_inputs and strategy != "self_renormalization":
        target = normalize_at_origin(target)
    if strategy == "self_renormalization":
        kernel_id = str(params["kernel_id"])
        zms_kernel = load_renormalization_kernel(kernel_id)
        factor = aligned.get("zR")
        if not isinstance(factor, EnsembleData):
            raise ValueError("zR must be one numerical source")
        spacing = target.attrs.get("lattice_spacing_fm")
        if not isinstance(spacing, (int, float)) or isinstance(spacing, bool):
            raise ValueError("self-renormalization target requires lattice_spacing_fm")
        if "a" in factor.dims:
            matches = [
                index for index, value in enumerate(factor.coords["a"]) if abs(float(value) - float(spacing)) <= 1e-12
            ]
            if len(matches) != 1:
                raise ValueError("target lattice spacing must match exactly one factor a coordinate")
            factor = factor.at("a", factor.coords["a"][matches[0]])
        if (
            factor.attrs.get("scale_gev") is not None
            and abs(float(factor.attrs["scale_gev"]) - float(params.get("mu", factor.attrs["scale_gev"]))) > 1e-12
        ):
            raise ValueError("self-renormalization factor scale does not match the apply job")
        if params["normalization"]:
            target = normalize_at_origin(target)
            factor = normalize_at_origin(factor)
        d_from = factor.attrs.get("d")
        m0_from = factor.attrs.get("m0_gev")
        if (
            not isinstance(d_from, (int, float))
            or isinstance(d_from, bool)
            or not isinstance(m0_from, (int, float))
            or isinstance(m0_from, bool)
        ):
            raise ValueError("self-renormalization factor is missing numeric d and m0_gev provenance")
        z_factor = np.asarray(factor.coords["z"], dtype=float)
        z_target = np.asarray(target.coords["z"], dtype=float)
        nonzero = np.abs(z_target) > 1e-12
        if np.any(np.diff(z_factor) <= 0) or np.any(z_factor <= 0):
            raise ValueError("self-renormalization factor z coordinates must be strictly increasing and positive")
        if np.any(np.abs(z_target[nonzero]) < z_factor[0] - 1e-12):
            raise ValueError("target nonzero z grid starts below the reusable factor range")
        mean_factor = np.mean(np.real(np.asarray(factor.values)), axis=0)
        lambda_qcd = float(params["LambdaQCD_gev"])
        d_to = float(params.get("d", d_from))
        m0_to = float(params.get("m0_gev", m0_from))
        log_a_lambda = np.log(float(spacing) * lambda_qcd / HBAR_C_GEV_FM)
        remapped_factor = (
            mean_factor
            * (1.0 + d_to / log_a_lambda)
            / (1.0 + float(d_from) / log_a_lambda)
            * np.exp((m0_to - float(m0_from)) * z_factor)
        )
        denominator_values = np.ones_like(z_target)
        target_nonzero = np.abs(z_target[nonzero])
        denominator_values[nonzero] = np.interp(np.minimum(target_nonzero, z_factor[-1]), z_factor, remapped_factor)
        long_distance = target_nonzero > z_factor[-1] + 1e-12
        if np.any(long_distance):
            k = float(factor.attrs["k"])
            n_f = int(factor.attrs["n_f"])
            scale_gev = float(params["mu"])
            baseline = (
                log_m(z_factor, float(spacing), k=k, lambda_qcd_gev=lambda_qcd, d=d_to, n_f=n_f, scale_gev=scale_gev)
                + m0_to * z_factor
            )
            finite_term = (np.log(remapped_factor) - baseline) / float(spacing)
            tail = z_factor >= 0.4 * z_factor[-1] - 1e-12
            if np.count_nonzero(tail) < 3:
                raise ValueError("long-distance factor completion requires at least three tail coordinates")
            coefficients = np.polyfit(z_factor[tail], finite_term[tail], 2)
            z_long = target_nonzero[long_distance]
            completed_log = (
                log_m(z_long, float(spacing), k=k, lambda_qcd_gev=lambda_qcd, d=d_to, n_f=n_f, scale_gev=scale_gev)
                + m0_to * z_long
                + np.polyval(coefficients, z_long) * float(spacing)
            )
            denominator_values[np.flatnonzero(nonzero)[long_distance]] = np.exp(completed_log)
        if scheme == "ratio":
            denominator_values[nonzero] *= np.exp(
                zmsbar_log(
                    zms_kernel,
                    np.abs(z_target[nonzero]),
                    scale_gev=float(params["mu"]),
                )
            )
        denominator = EnsembleData(None, "raw", [denominator_values], ["z"], {"z": z_target.tolist()})
        if scheme == "hybrid":
            denominator = aligned.get("denominator")
            if not isinstance(denominator, EnsembleData):
                raise ValueError("hybrid denominator must be one numerical source")
            switch = float(params["zs_fm"])
            z = np.asarray(target.coords["z"], dtype=float)
            matches = np.flatnonzero(np.isclose(z, switch, rtol=0.0, atol=1e-12))
            if len(matches) != 1:
                raise ValueError("hybrid switch must be an exact positive z coordinate")
            switch_coord = switch
            short = ratio(target, denominator)
            factor_switch = factor.near("z", switch_coord, tolerance=1e-12)
            denominator_switch = denominator.near("z", switch_coord, tolerance=1e-12)
            transfer = denominator_switch.div(factor_switch)
            long = target.div(factor).div(transfer)
            mask = np.abs(z) <= abs(switch_coord)
            values = [
                np.where(mask, np.asarray(short_sample), np.asarray(long_sample))
                for short_sample, long_sample in zip(short.values, long.values)
            ]
            attrs = short.attrs
            attrs.update({"hybrid_switch_coord_fm": switch_coord, "strategy": strategy})
            result = EnsembleData(
                target.ensemble,
                target.resample,
                values,
                target.dims,
                target.coords,
                attrs=attrs,
                name="renormalized_matrix_element",
            )
        else:
            result = target.div(denominator)
    else:
        denominator_source = context.inputs["denominator"]
        if isinstance(denominator_source, (int, float)) and not isinstance(denominator_source, bool):
            result = divide_by_constant(target, float(denominator_source))
        else:
            denominator = aligned.get("denominator")
            if not isinstance(denominator, EnsembleData):
                raise ValueError("denominator must be one numerical source")
            if normalize_inputs:
                denominator = normalize_at_origin(denominator)
            if scheme == "hybrid":
                result = hybrid_ratio(
                    target,
                    denominator,
                    zs_fm=float(params["zs_fm"]),
                    delta_m_gev=float(params["delta_m_gev"]),
                    m0_gev=float(params["m0_gev"]),
                )
            else:
                result = ratio(target, denominator)
    attrs = result.attrs
    attrs.update(
        {
            "operation": "apply",
            "renormalization_scheme": scheme,
            "strategy": strategy,
            "type": params["type"],
            "coord_unit": "fm",
            "input_coord_unit": attrs.get("input_coord_unit", "fm"),
            "units": json.dumps({"values": "dimensionless", **{dim: "fm" for dim in result.dims}}),
        }
    )
    if strategy == "self_renormalization":
        attrs["kernel_id"] = params["kernel_id"]
    result = EnsembleData(
        result.ensemble,
        result.resample,
        [np.asarray(sample) for sample in result.values],
        result.dims,
        result.coords,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    context.state["renormalized"] = {"data": result, "scheme": scheme, "strategy": strategy}
    result.to_netcdf(context.artifact_directory / "output.nc")
    z_values = np.asarray(result.coords["z"], dtype=float)
    diagnostics = {
        "scheme": scheme,
        "strategy": strategy,
        "type": params["type"],
        "kernel_id": params.get("kernel_id"),
        "sample_count": result.n_sample,
        "dims": result.dims,
        "z_range_fm": [float(np.min(z_values)), float(np.max(z_values))],
        "z_count": int(z_values.size),
        "normalization": normalize_inputs,
        "zs_fm": params.get("zs_fm"),
        "m0_gev": params.get("m0_gev"),
        "delta_m_gev": params.get("delta_m_gev"),
        "d": params.get("d"),
        "mu": params.get("mu"),
        "LambdaQCD_gev": params.get("LambdaQCD_gev"),
        "target_samples": target.n_sample,
        "denominator_kind": "constant"
        if isinstance(context.inputs.get("denominator"), (int, float))
        and not isinstance(context.inputs.get("denominator"), bool)
        else "matrix"
        if "denominator" in context.inputs
        else "self_factor",
        "input_z_ranges_fm": {
            role: [float(np.min(value.coords["z"])), float(np.max(value.coords["z"]))]
            for role, value in aligned.items()
            if isinstance(value, EnsembleData) and "z" in value.coords
        },
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "renormalization.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    start_plot()
    plot_data = result.real if np.iscomplexobj(result.values) else result
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorline(result.coords["z"], plot_data.average(sample_error_mode))
    configure_plot(xlabel="z [fm]", ylabel="renormalized matrix element")
    save_figure(context.artifact_directory / "plots" / "result.pdf")
    report = f"# Renormalized matrix element\n\nScheme: `{scheme}`.\nStrategy: `{strategy}`.\n"
    (context.artifact_directory / "report.md").write_text(report, encoding="utf-8")
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "renormalized_matrix_element",
        "decisions": {
            "type": params["type"],
            "scheme": scheme,
            "strategy": strategy,
            "kernel_id": params.get("kernel_id"),
            "normalization": bool(params["normalization"]),
        },
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/renormalization.json", "plots/result.pdf", "report.md"],
    }
    context.finish(result, summary)
    return {
        "summary": "published renormalized matrix element",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
