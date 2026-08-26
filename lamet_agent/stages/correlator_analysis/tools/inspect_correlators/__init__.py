"""Inspect descriptor correlators without returning raw arrays to the model."""

from __future__ import annotations

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.stages.correlator_analysis.hook import ensure_correlators


_RESCALED_TYPICAL_ABS_RANGE = (1.0e-4, 1.0e-2)


def _automatic_correlator_rescale(
    correlators: dict[str, object],
    windows: list[dict[str, int]],
) -> dict[str, object]:
    """Choose a data-driven power-of-ten scale for spectral two-point fits."""
    window_statistics: list[dict[str, object]] = []
    typical_values: list[float] = []
    for correlator_id, data in correlators.items():
        if data.attrs.get("correlator_type") != "two_point" or "t" not in data.dims:
            continue
        times = np.asarray(data.coords["t"])
        time_axis = data.array.dims.index("t")
        values = np.moveaxis(np.asarray(data.values), time_axis, 1)
        for window in windows:
            mask = (times >= int(window["tmin"])) & (times < int(window["tmax"]))
            absolute = np.abs(values[:, mask, ...]).reshape(-1)
            finite_nonzero = absolute[np.isfinite(absolute) & (absolute > 0.0)]
            if not finite_nonzero.size:
                raise ValueError(
                    f"two-point correlator '{correlator_id}' has no finite nonzero "
                    f"values in window [{window['tmin']}, {window['tmax']})"
                )
            typical = float(np.median(finite_nonzero))
            typical_values.append(typical)
            window_statistics.append(
                {
                    "correlator_id": correlator_id,
                    "tmin": int(window["tmin"]),
                    "tmax": int(window["tmax"]),
                    "median_abs": typical,
                    "max_abs": float(np.max(finite_nonzero)),
                    "min_abs_nonzero": float(np.min(finite_nonzero)),
                }
            )
    if not typical_values:
        raise ValueError("automatic correlator rescaling requires a selected two-point correlator")
    typical_abs = float(np.median(typical_values))
    target_min = _RESCALED_TYPICAL_ABS_RANGE[0]
    exponent = int(np.ceil(np.log10(target_min / typical_abs)))
    correlator_rescale = float(10.0**exponent)
    if not np.isfinite(correlator_rescale) or correlator_rescale <= 0.0:
        raise ValueError("the automatic correlator rescale is outside the finite float range")
    return {
        "correlator_rescale": correlator_rescale,
        "power_of_ten": exponent,
        "typical_abs": typical_abs,
        "rescaled_typical_abs": typical_abs * correlator_rescale,
        "target_typical_abs_range": list(_RESCALED_TYPICAL_ABS_RANGE),
        "windows": window_statistics,
    }


def run(context: ToolContext, *, correlator_ids: list[str] | None = None) -> dict[str, object]:
    """Load, resample, and summarize selected correlators."""
    resampled = ensure_correlators(context, correlator_ids)
    lsqfit = context.params.get("lsqfit")
    fit_scopes = set(lsqfit.get("fit_scope", [])) if isinstance(lsqfit, dict) else set()
    scale_inspection = None
    if fit_scopes & {"3pt_ratio", "FH", "3pt_ratio+FH"}:
        scale_inspection = _automatic_correlator_rescale(
            resampled,
            lsqfit["pt2_windows"],
        )
        context.state["correlator_scale_inspection"] = scale_inspection
        context.state["correlator_rescale"] = scale_inspection["correlator_rescale"]
    inspection = {}
    for key, value in resampled.items():
        metrics = {
            "dims": value.dims,
            "shape": list(value.values.shape),
            "n_sample": value.n_sample,
            "coord_ranges": {dim: [min(coords), max(coords)] for dim, coords in value.coords.items()},
        }
        if "t" in value.dims:
            times = np.asarray(value.coords["t"], dtype=float)
            samples = np.asarray(value.real.values if np.iscomplexobj(value.values) else value.values)
            time_axis = value.array.dims.index("t")
            samples = np.moveaxis(samples, time_axis, 1)
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1).mean(axis=2)
            central = np.mean(samples, axis=0)
            usable = np.isfinite(central) & (central > 0)
            effective_mass = (
                -np.diff(np.log(central[usable])) / np.diff(times[usable])
                if np.count_nonzero(usable) > 1
                else np.asarray([], dtype=float)
            )
            relative_noise = (
                np.std(samples, axis=0, ddof=1) / np.maximum(np.abs(central), np.finfo(float).eps)
                if samples.shape[0] > 1
                else np.zeros_like(central)
            )
            metrics.update(
                {
                    "effective_mass": effective_mass.tolist(),
                    "relative_noise": relative_noise.tolist(),
                    "usable_time_count": int(np.count_nonzero(usable)),
                }
            )
        inspection[key] = metrics
    context.state["inspection"] = inspection
    metrics = {
        "correlator_count": len(resampled),
        "resample_id": next(iter(resampled.values())).attrs.get("resample_id") if resampled else None,
        "diagnostics": {
            key: {name: value for name, value in item.items() if name not in {"effective_mass", "relative_noise"}}
            for key, item in inspection.items()
        },
    }
    state_keys = ["correlators", "inspection"]
    if scale_inspection is not None:
        metrics["correlator_scale"] = scale_inspection
        state_keys.extend(["correlator_scale_inspection", "correlator_rescale"])
    return {
        "summary": f"inspected {len(resampled)} correlators with shared resample plan",
        "metrics": metrics,
        "state_keys": state_keys,
        "artifacts": [],
    }
