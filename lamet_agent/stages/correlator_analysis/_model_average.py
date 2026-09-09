"""Per-resample logGBF model averaging for correlator candidates."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError


def loggbf_weights(log_gbf: np.ndarray) -> np.ndarray:
    """Return normalized ``exp(logGBF - max(logGBF))`` weights."""
    values = np.asarray(log_gbf, dtype=float)
    shifted = np.exp(values - np.max(values))
    return shifted / np.sum(shifted)


def _finite_loggbf(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _usable_value(value: object) -> bool:
    try:
        real = float(np.real(value))
        imag = float(np.imag(value))
    except (TypeError, ValueError):
        return False
    return math.isfinite(real) and math.isfinite(imag)


def _sample_loggbf_map(fit: dict[str, Any]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for record in fit.get("sample_diagnostics") or []:
        if not isinstance(record, dict) or record.get("sample") is None:
            continue
        log_gbf = _finite_loggbf(record.get("logGBF"))
        if not math.isfinite(log_gbf):
            continue
        mapping[int(record["sample"])] = log_gbf
    return mapping


def _weighted_model_sdev(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    wgt = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(wgt) & (wgt > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    vals = vals[mask]
    wgt = wgt[mask]
    wgt = wgt / np.sum(wgt)
    average = float(np.sum(wgt * vals))
    return float(np.sqrt(max(float(np.sum(wgt * (vals - average) ** 2)), 0.0)))


def _fit_lookup(application_fit: dict[str, Any] | None, coords: list[object]) -> list[dict[str, Any] | None]:
    fits = list((application_fit or {}).get("fits") or [])
    by_z: dict[float, dict[str, Any]] = {}
    for fit in fits:
        if isinstance(fit, dict) and fit.get("z") is not None:
            by_z[float(fit["z"])] = fit
    lookup: list[dict[str, Any] | None] = []
    for coord in coords:
        try:
            match = by_z.get(float(coord))
        except (TypeError, ValueError):
            match = None
        if match is None and len(fits) == 1 and isinstance(fits[0], dict):
            match = fits[0]
        lookup.append(match)
    return lookup


def combine_matrix_samples(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine sibling full-sample results with per-coordinate logGBF weights.

    Each model dict needs ``id`` and ``data``. Optional ``application_fit`` supplies
    per-sample, per-z ``logGBF``; otherwise the model's center ``logGBF`` is used.
    """
    if not models:
        raise ValueError("model averaging requires at least one applied model")
    datasets = [model["data"] for model in models]
    reference = datasets[0]
    if not isinstance(reference, EnsembleData):
        raise TypeError("model averaging requires EnsembleData results")
    dim = reference.dims[0]
    coords = list(reference.coords[dim])
    n_sample = reference.n_sample
    n_coord = len(coords)
    stacked = []
    log_gbf = np.full((len(models), n_sample, n_coord), np.nan, dtype=float)
    center_quality: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        data = model["data"]
        if not isinstance(data, EnsembleData):
            raise TypeError("model averaging requires EnsembleData results")
        if data.n_sample != n_sample or list(data.dims) != list(reference.dims):
            raise ValueError("averaged models must share sample count and dimensions")
        if list(data.coords[dim]) != coords:
            raise ValueError("averaged models must share coordinates")
        stacked.append(np.asarray(data.values))
        application_fit = model.get("application_fit")
        fits = _fit_lookup(application_fit if isinstance(application_fit, dict) else None, coords)
        fallback = _finite_loggbf(model.get("logGBF"))
        quality = {
            "id": str(model["id"]),
            "Q": model.get("Q"),
            "chi2_dof": model.get("chi2_dof"),
            "logGBF": model.get("logGBF"),
        }
        for coord_index, fit in enumerate(fits):
            center = fallback
            if isinstance(fit, dict):
                center = _finite_loggbf(fit.get("logGBF"))
                if coord_index == 0:
                    quality["Q"] = fit.get("Q", quality["Q"])
                    quality["chi2_dof"] = fit.get("chi2_dof", quality["chi2_dof"])
                    quality["logGBF"] = fit.get("logGBF", quality["logGBF"])
                for sample_index, value in _sample_loggbf_map(fit).items():
                    if 0 <= sample_index < n_sample:
                        log_gbf[model_index, sample_index, coord_index] = value
            if not np.isfinite(log_gbf[model_index, :, coord_index]).any() and math.isfinite(center):
                log_gbf[model_index, :, coord_index] = center
        center_quality.append(quality)
    stacked_values = np.stack(stacked, axis=0)
    weights = np.zeros((len(models), n_sample, n_coord), dtype=float)
    combined = np.zeros((n_sample, n_coord), dtype=stacked_values.dtype)
    for sample_index in range(n_sample):
        for coord_index in range(n_coord):
            values = stacked_values[:, sample_index, coord_index]
            valid = np.asarray(
                [
                    _usable_value(value) and math.isfinite(float(log_gbf[model_index, sample_index, coord_index]))
                    for model_index, value in enumerate(values)
                ],
                dtype=bool,
            )
            if not np.any(valid):
                raise FitNumericalError(
                    f"all averaged models failed at sample={sample_index} {dim}={coords[coord_index]}"
                )
            sample_weights = np.zeros(len(models), dtype=float)
            sample_weights[valid] = loggbf_weights(log_gbf[valid, sample_index, coord_index])
            weights[:, sample_index, coord_index] = sample_weights
            combined[sample_index, coord_index] = np.sum(sample_weights * values)
    mean_weights = np.mean(weights, axis=(1, 2))
    primary_index = int(np.argmax(mean_weights))
    real_means = np.asarray([np.nanmean(np.real(values), axis=0) for values in stacked], dtype=float)
    imag_means = np.asarray([np.nanmean(np.imag(values), axis=0) for values in stacked], dtype=float)
    coord_weights = np.mean(weights, axis=1)
    real_sys = [_weighted_model_sdev(real_means[:, index], coord_weights[:, index]) for index in range(n_coord)]
    imag_sys = [_weighted_model_sdev(imag_means[:, index], coord_weights[:, index]) for index in range(n_coord)]
    selected_models = [
        str(model["id"]) for model, weight in zip(models, mean_weights) if float(weight) > 0.0
    ]
    attrs = dict(reference.attrs)
    attrs.update(
        {
            "model_average": "true",
            "selected_models": json.dumps(selected_models),
            "model_weights": json.dumps([float(weight) for weight in mean_weights]),
            "real_sys_sdev": json.dumps([float(value) for value in real_sys]),
            "imag_sys_sdev": json.dumps([float(value) for value in imag_sys]),
        }
    )
    data = EnsembleData(
        reference.ensemble,
        reference.resample,
        list(combined),
        list(reference.dims),
        {dim: coords},
        attrs=attrs,
        name=reference.name,
    )
    primary = models[primary_index]
    return {
        "data": data,
        "mean_weights": [float(weight) for weight in mean_weights],
        "primary_index": primary_index,
        "primary_id": str(primary["id"]),
        "primary_model": primary,
        "selected_models": selected_models,
        "real_sys_sdev": [float(value) for value in real_sys],
        "imag_sys_sdev": [float(value) for value in imag_sys],
        "center_quality": center_quality,
    }
