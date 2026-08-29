"""Apply external or reusable renormalization and publish the result."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.renormalization._plotting import render_result
from lamet_agent.stages.renormalization.physics import (
    divide_by_constant,
    hybrid_ratio,
    log_m,
    normalize_at_origin,
    ratio,
    zmsbar_log,
)


def _coverage_mask(z_target: np.ndarray, z_factor: np.ndarray, policy: str) -> np.ndarray:
    """Return the original self-renormalization target-coverage selection."""
    if policy not in {"strict", "intersection", "extrapolate"}:
        raise ValueError("z_coverage_policy must be 'strict', 'intersection', or 'extrapolate'")
    tolerance = 1e-12
    covered = (z_target >= z_factor[0] - tolerance) & (z_target <= z_factor[-1] + tolerance)
    if policy == "strict" and not np.all(covered):
        raise ValueError(
            "target z grid lies outside the fitted zR range: "
            f"target=[{float(np.min(z_target))}, {float(np.max(z_target))}], "
            f"zR=[{float(z_factor[0])}, {float(z_factor[-1])}]"
        )
    if not np.any(covered):
        raise ValueError("target and zR grids have no overlapping z range")
    if policy == "extrapolate":
        if np.any(z_target < z_factor[0] - tolerance):
            raise ValueError("zR extrapolation only supports the long-distance upper end")
        return np.ones_like(z_target, dtype=bool)
    return covered


def _complete_long_distance_factor(
    z_target: np.ndarray,
    z_factor: np.ndarray,
    factor_values: np.ndarray,
    *,
    spacing_fm: float,
    k: float,
    n_f: int,
    lambda_qcd_gev: float,
    d: float,
    m0: float,
    scale_gev: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Complete zR above its fitted range with the original quadratic f1 tail."""
    if np.any(factor_values <= 0):
        raise ValueError("zR extrapolation requires positive fitted zR values")
    zmax = float(z_factor[-1])
    extrapolated = z_target > zmax + 1e-12
    result = np.interp(np.minimum(z_target, zmax), z_factor, factor_values)
    if not np.any(extrapolated):
        return result, {
            "n_z_extrapolated": 0,
            "z_extrapolation_method": "none",
            "f1_tail_zmin_fm": None,
        }
    baseline = (
        log_m(
            z_factor,
            spacing_fm,
            k=k,
            lambda_qcd_gev=lambda_qcd_gev,
            d=d,
            n_f=n_f,
            scale_gev=scale_gev,
        )
        + m0 * z_factor
    )
    finite_term = (np.log(factor_values) - baseline) / spacing_fm
    tail = z_factor >= 0.4 * zmax - 1e-12
    if np.count_nonzero(tail) < 3:
        tail = np.zeros_like(z_factor, dtype=bool)
        tail[-min(3, len(z_factor)) :] = True
    if np.count_nonzero(tail) < 3:
        raise ValueError("zR extrapolation requires at least three fitted z points")
    coefficients = np.polyfit(z_factor[tail], finite_term[tail], 2)
    z_extra = z_target[extrapolated]
    completed_log = (
        log_m(
            z_extra,
            spacing_fm,
            k=k,
            lambda_qcd_gev=lambda_qcd_gev,
            d=d,
            n_f=n_f,
            scale_gev=scale_gev,
        )
        + m0 * z_extra
        + np.polyval(coefficients, z_extra) * spacing_fm
    )
    result[extrapolated] = np.exp(completed_log)
    return result, {
        "n_z_extrapolated": int(np.count_nonzero(extrapolated)),
        "z_extrapolation_method": "quadratic_f1_tail",
        "f1_tail_zmin_fm": float(np.min(z_factor[tail])),
    }


def run(context: ToolContext) -> dict[str, object]:
    """Apply the declared renormalization mode and call ``finish``."""
    aligned = context.state.get("aligned_inputs")
    if not isinstance(aligned, dict) or "target" not in aligned:
        raise RuntimeError("inspect_renormalization must run before application")
    target = aligned["target"]
    if isinstance(target, list):
        raise ValueError("target must be one source")
    params = context.params
    strategy = params["strategy"]
    scheme = params["scheme"]
    normalize_inputs = bool(params["normalization"])
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    apply_plot_data: dict[str, object] | None = None
    coverage_diagnostics: dict[str, object] = {}
    if normalize_inputs and strategy != "self_renormalization":
        target = normalize_at_origin(target)
    if strategy == "self_renormalization":
        kernel_id = str(params["kernel_id"])
        zms_kernel = load_renormalization_kernel(kernel_id)
        kernel_parameters = dict(params["kernel_parameters"])
        factor = aligned.get("zR")
        if not isinstance(factor, EnsembleData):
            raise ValueError("zR must be one numerical source")
        spacing = float(target.ensemble.a_s)
        if "a" in factor.dims:
            matches = [
                index for index, value in enumerate(factor.coords["a"]) if abs(float(value) - float(spacing)) <= 1e-12
            ]
            if len(matches) != 1:
                raise ValueError("target lattice spacing must match exactly one factor a coordinate")
            factor = factor.at("a", factor.coords["a"][matches[0]])
        if (
            factor.attrs.get("scale_gev") is not None
            and abs(float(factor.attrs["scale_gev"]) - float(params["mu"])) > 1e-12
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
        mean_factor = np.mean(np.real(np.asarray(factor.values)), axis=0)
        lambda_qcd = float(params["LambdaQCD_gev"])
        d_to = float(params["d"])
        m0_to = float(params["m0_gev"])
        log_a_lambda = np.log(float(spacing) * lambda_qcd / HBAR_C_GEV_FM)
        remapped_factor = (
            mean_factor
            * (1.0 + d_to / log_a_lambda)
            / (1.0 + float(d_from) / log_a_lambda)
            * np.exp((m0_to - float(m0_from)) * z_factor)
        )
        policy = str(params["z_coverage_policy"])
        nonzero_indices = np.flatnonzero(nonzero)
        target_nonzero = np.abs(z_target[nonzero])
        selected = _coverage_mask(target_nonzero, z_factor, policy)
        target_indices = nonzero_indices[selected]
        zero_indices = np.flatnonzero(~nonzero)
        output_indices = np.sort(np.concatenate((zero_indices, target_indices)))
        z_output = z_target[output_indices]
        selected_z = np.abs(z_target[target_indices])
        if policy == "extrapolate":
            factor_on_target, extrapolation = _complete_long_distance_factor(
                selected_z,
                z_factor,
                remapped_factor,
                spacing_fm=float(spacing),
                k=float(factor.attrs["k"]),
                n_f=int(factor.attrs["n_f"]),
                lambda_qcd_gev=lambda_qcd,
                d=d_to,
                m0=m0_to,
                scale_gev=float(params["mu"]),
            )
        else:
            factor_on_target = np.interp(selected_z, z_factor, remapped_factor)
            extrapolation = {
                "n_z_extrapolated": 0,
                "z_extrapolation_method": "none",
                "f1_tail_zmin_fm": None,
            }
        target_values = np.asarray(target.values)[:, output_indices]
        if scheme == "hybrid":
            denominator = aligned.get("denominator")
            if not isinstance(denominator, EnsembleData):
                raise ValueError("hybrid denominator must be one numerical source")
            if (
                denominator.resample != target.resample
                or denominator.coords.get("z") != target.coords.get("z")
                or np.asarray(denominator.values).shape != np.asarray(target.values).shape
            ):
                raise ValueError("hybrid target and denominator must have matching samples and z coordinates")
            denominator_values = np.asarray(denominator.values)[:, output_indices]
            switch = float(params["zs_fm"])
            matches = np.flatnonzero(np.isclose(z_output, switch, rtol=0.0, atol=1e-12))
            if len(matches) != 1:
                raise ValueError("hybrid switch must be an exact positive z coordinate")
            switch_index = int(matches[0])
            factor_values = np.ones_like(z_output, dtype=float)
            factor_values[np.isin(output_indices, target_indices)] = factor_on_target
            transfer = denominator_values[:, switch_index] / factor_values[switch_index]
            if np.any(np.isclose(np.abs(transfer), 0.0, rtol=0.0, atol=1e-30)):
                raise ValueError("hybrid self-renormalization produced zero transfer at the switch point")
            short_values = target_values / denominator_values
            long_values = target_values / (factor_values[None, :] * transfer[:, None])
            values = np.where((np.abs(z_output) <= switch)[None, :], short_values, long_values)
            result_attrs = target.attrs
            result_attrs.update({"hybrid_switch_coord_fm": switch, "strategy": strategy})
        else:
            factor_values = np.ones_like(z_output, dtype=float)
            nonzero_output = np.abs(z_output) > 1e-12
            factor_values[nonzero_output] = factor_on_target
            if scheme == "ratio":
                zmsbar_values = np.exp(
                    zmsbar_log(
                        zms_kernel,
                        np.abs(z_output[nonzero_output]),
                        scale_gev=float(params["mu"]),
                        kernel_parameters=kernel_parameters,
                    )
                )
                h_over_zr = EnsembleData(
                    target.ensemble,
                    target.resample,
                    [sample[nonzero_output] / factor_values[nonzero_output] for sample in target_values],
                    ["z"],
                    {"z": z_output[nonzero_output].tolist()},
                    attrs={"sample_error_mode": sample_error_mode},
                    name="bare_over_self_renormalization_factor",
                )
                h_over_zr_average = h_over_zr.real.average(sample_error_mode)
                apply_plot_data = {
                    "kind": "apply",
                    "z_fm": z_output[nonzero_output].tolist(),
                    "h_over_zR_real_mean": np.asarray(
                        [float(value.mean) for value in h_over_zr_average], dtype=float
                    ).tolist(),
                    "h_over_zR_real_sdev": np.asarray(
                        [float(value.sdev) for value in h_over_zr_average], dtype=float
                    ).tolist(),
                    "zmsbar": np.asarray(zmsbar_values, dtype=float).tolist(),
                }
                factor_values[nonzero_output] *= zmsbar_values
            values = target_values / factor_values[None, :]
            result_attrs = target.attrs
        result = EnsembleData(
            target.ensemble,
            target.resample,
            [np.asarray(sample) for sample in values],
            ["z"],
            {"z": z_output.tolist()},
            attrs=result_attrs,
            name="renormalized_matrix_element",
        )
        coverage_diagnostics = {
            "z_coverage_policy": policy,
            "n_z_input": int(z_target.size),
            "n_z_dropped": int(z_target.size - z_output.size),
            "n_z_coverage_dropped": int(np.count_nonzero(nonzero) - len(target_indices)),
            "n_z_zero_passthrough": int(len(zero_indices)),
            "z_input_range_fm": [float(np.min(z_target)), float(np.max(z_target))],
            "z_output_range_fm": [float(np.min(z_output)), float(np.max(z_output))],
            "zR_input_range_fm": [float(z_factor[0]), float(z_factor[-1])],
            **extrapolation,
        }
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
        attrs["kernel_parameters"] = json.dumps(kernel_parameters, sort_keys=True)
        attrs.update(
            {
                "z_coverage_policy": coverage_diagnostics["z_coverage_policy"],
                "n_z_dropped": coverage_diagnostics["n_z_dropped"],
                "n_z_coverage_dropped": coverage_diagnostics["n_z_coverage_dropped"],
                "n_z_extrapolated": coverage_diagnostics["n_z_extrapolated"],
                "z_extrapolation_method": coverage_diagnostics["z_extrapolation_method"],
                "f1_tail_zmin_fm": ""
                if coverage_diagnostics["f1_tail_zmin_fm"] is None
                else coverage_diagnostics["f1_tail_zmin_fm"],
            }
        )
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
        "kernel_parameters": params.get("kernel_parameters") if strategy == "self_renormalization" else None,
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
        **coverage_diagnostics,
    }
    diagnostic_payload = dict(diagnostics)
    if apply_plot_data is not None:
        diagnostic_payload["plot_data"] = apply_plot_data
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "renormalization.json").write_text(
        json.dumps(diagnostic_payload, indent=2), encoding="utf-8"
    )
    rendered = [
        render_result(
            result,
            directory=context.artifact_directory / "plots",
            stem="result",
            formats=("pdf",),
            sample_error_mode=sample_error_mode,
        )
    ]
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
            "z_coverage_policy": params.get("z_coverage_policy") if strategy == "self_renormalization" else None,
        },
        "diagnostics": diagnostics,
        "artifacts": [
            "output.nc",
            "diagnostics/renormalization.json",
            *[f"plots/{stem}.pdf" for stem, _caption in rendered],
        ],
    }
    context.finish(result, summary)
    return {
        "summary": "published renormalized matrix element",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
