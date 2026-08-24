"""Renormalization equations kept local to the renormalization stage."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM


def load_data(value: Any) -> EnsembleData:
    """Load one explicit sample-bearing NetCDF source."""
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, Path):
        if value.suffix.lower() != ".nc":
            raise ValueError(f"renormalization input must be a .nc artifact: {value}")
        return EnsembleData.from_netcdf(value)
    raise TypeError("renormalization input is neither EnsembleData nor a NetCDF Path")


def physical_z_coordinates(data: EnsembleData) -> EnsembleData:
    """Convert a lattice-unit ``z`` coordinate to fm exactly once."""
    if "z" not in data.dims:
        return data
    attrs = data.attrs
    coord_unit = attrs.get("coord_unit")
    if coord_unit is None:
        coord_unit = attrs.get("input_coord_unit")
    if coord_unit not in {"fm", "lattice"}:
        raise ValueError("renormalization input requires coord_unit='fm' or 'lattice'")
    if coord_unit == "fm":
        return data
    spacing = attrs.get("lattice_spacing_fm")
    if not isinstance(spacing, (int, float)) or isinstance(spacing, bool) or not math.isfinite(float(spacing)) or float(spacing) <= 0:
        raise ValueError("lattice-coordinate renormalization input requires positive lattice_spacing_fm")
    coords = data.coords
    coords["z"] = [float(value) * float(spacing) for value in coords["z"]]
    attrs.update({"coord_unit": "fm", "input_coord_unit": "lattice"})
    return EnsembleData(data.ensemble, data.resample, [np.asarray(sample) for sample in data.values], data.dims, coords, attrs=attrs, name=data.name)


def divide_by_constant(target: EnsembleData, denominator: float) -> EnsembleData:
    """Apply ``h_R(z)=h_target(z)/C`` sample by sample."""
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("constant denominator must be finite and nonzero")
    values = [sample / denominator for sample in target.values]
    attrs = target.attrs
    attrs["denominator_kind"] = "constant"
    return EnsembleData(target.ensemble, target.resample, values, target.dims, target.coords, attrs=attrs, name="renormalized_matrix_element")


def ratio(target: EnsembleData, denominator: EnsembleData) -> EnsembleData:
    """Apply the pointwise complex ratio on the complete physical grid."""
    if target.attrs.get("resample_id") and denominator.attrs.get("resample_id") and target.attrs["resample_id"] != denominator.attrs["resample_id"]:
        raise ValueError("target and denominator use different resample_id plans")
    if target.resample != denominator.resample or target.n_sample != denominator.n_sample:
        raise ValueError("target and denominator must use the same resampling mode and sample count")
    return target.div(denominator).copy()


def normalize_at_origin(data: EnsembleData) -> EnsembleData:
    """Divide each stored sample by its own value at the unique ``z=0`` point."""
    if "z" not in data.dims:
        raise ValueError("origin normalization requires a z=0 coordinate")
    matches = [value for value in data.coords["z"] if abs(float(value)) <= 1e-12]
    if len(matches) != 1:
        raise ValueError("origin normalization requires one unique z=0 coordinate")
    origin = data.at("z", matches[0])
    result = data.div(origin)
    attrs = result.attrs
    attrs["normalized_at_origin"] = 1
    return EnsembleData(result.ensemble, result.resample, [np.asarray(sample) for sample in result.values], result.dims, result.coords, attrs=attrs, name=result.name)


def hybrid_ratio(target: EnsembleData, denominator: EnsembleData, *, zs_fm: float, delta_m_gev: float, m0_gev: float = 0.0) -> EnsembleData:
    """Use a short ratio and a continuous long-distance denominator anchor.

    The exponent is dimensionless because ``z`` is in fm and ``delta_m`` is in
    GeV. Above the nearest switching coordinate the fixed denominator value is
    multiplied by ``exp((delta_m+m0)(|z|-z_s)/(hbar*c))``.  The
    authored switch must be present on the coordinate grid exactly.
    """
    if "z" not in target.dims:
        raise ValueError("hybrid renormalization requires a z dimension")
    base = ratio(target, denominator)
    z = np.asarray(base.coords["z"], dtype=float)
    switch = float(zs_fm)
    positive = np.flatnonzero(np.isclose(z, switch, rtol=0.0, atol=1e-12))
    negative = np.flatnonzero(np.isclose(z, -switch, rtol=0.0, atol=1e-12))
    if len(positive) != 1 or (np.any(z < -switch) and len(negative) != 1):
        raise ValueError("hybrid switch requires exact signed z coordinates on every long-distance branch")
    z_axis = target.dims.index("z")
    short_mask = np.abs(z) <= switch + 1e-12
    exponent = np.exp((float(delta_m_gev) + float(m0_gev)) * (np.abs(z) - switch) / HBAR_C_GEV_FM)
    shape = [1] * len(target.dims)
    shape[z_axis] = len(z)
    long_weight = exponent.reshape(shape)
    values = []
    for target_sample, denominator_sample, short_sample in zip(target.values, denominator.values, base.values):
        positive_anchor = np.take(denominator_sample, int(positive[0]), axis=z_axis)
        positive_long = target_sample / np.expand_dims(positive_anchor, axis=z_axis)
        if len(negative) == 1:
            negative_anchor = np.take(denominator_sample, int(negative[0]), axis=z_axis)
            negative_long = target_sample / np.expand_dims(negative_anchor, axis=z_axis)
            long = np.where((z < 0).reshape(shape), negative_long, positive_long)
        else:
            long = positive_long
        values.append(np.where(short_mask.reshape(shape), short_sample, long * long_weight))
    attrs = base.attrs
    attrs.update({"zs_fm": switch, "delta_m_gev": float(delta_m_gev), "m0_gev": float(m0_gev), "strategy": "hybrid", "hybrid_switch_coord_fm": switch})
    return EnsembleData(base.ensemble, base.resample, values, base.dims, base.coords, attrs=attrs, name="renormalized_matrix_element")


def _perturbative_log(a_fm: float, *, lambda_qcd_gev: float, scale_gev: float, d: float, n_f: int) -> float:
    """Evaluate the locked one-loop finite term in ``log M(z,a)``."""
    a_lambda = float(a_fm) * float(lambda_qcd_gev) / HBAR_C_GEV_FM
    if a_lambda <= 0 or scale_gev <= lambda_qcd_gev:
        raise ValueError("self-renormalization requires a*Lambda>0 and scale>Lambda")
    log_a_lambda = math.log(a_lambda)
    log_inverse = math.log(1.0 / a_lambda)
    log_scale_ratio = math.log(float(scale_gev) / float(lambda_qcd_gev))
    b0 = 11.0 - 2.0 * int(n_f) / 3.0
    if b0 <= 0 or log_inverse <= 0 or log_scale_ratio <= 0 or 1.0 + float(d) / log_a_lambda <= 0:
        raise ValueError("self-renormalization logarithms are outside their physical domain")
    c_f = 4.0 / 3.0
    return (3.0 * c_f / b0) * math.log(log_inverse / log_scale_ratio) + math.log(1.0 + float(d) / log_a_lambda)


def zmsbar_pdf_log(z_fm: np.ndarray, *, scale_gev: float, model: str = "pdf_nlo") -> np.ndarray:
    """Return ``log Z_MSbar`` for the one-loop PDF or DA conversion factor."""
    if model not in {"pdf_nlo", "da_nlo"}:
        raise ValueError(f"unsupported zms model '{model}'")
    if not math.isfinite(float(scale_gev)) or float(scale_gev) <= 0:
        raise ValueError("scale_gev must be finite and positive")
    z = np.asarray(z_fm, dtype=float)
    b0 = 11.0 - 2.0 * 3.0 / 3.0
    alpha_reference = 0.293 / (4.0 * np.pi)
    running = 1.0 + alpha_reference * b0 * np.log((float(scale_gev) / 2.0) ** 2)
    alpha_s = alpha_reference * 4.0 * np.pi / running
    nonzero = np.abs(z) > 1e-14
    conversion = np.ones_like(z)
    log_term = np.log(float(scale_gev) ** 2 * (z[nonzero] / HBAR_C_GEV_FM) ** 2 * np.exp(2.0 * np.euler_gamma) / 4.0)
    offset = 2.5 if model == "pdf_nlo" else 3.5
    conversion[nonzero] = 1.0 + alpha_s * (4.0 / 3.0) / (2.0 * np.pi) * (1.5 * log_term + offset)
    if np.any(conversion[nonzero] <= 0):
        raise ValueError("pdf_nlo Z_MSbar is nonpositive on the short-distance grid")
    return np.log(conversion)


def log_m(
    z_fm: np.ndarray,
    a_fm: float,
    *,
    m0_gev: float = 0.0,
    delta_m_gev: float = 0.0,
    k: float | None = None,
    lambda_qcd_gev: float = 0.2,
    d: float = 0.0,
    n_f: int = 3,
    scale_gev: float = 2.0,
) -> np.ndarray:
    """Evaluate the locked self-renormalization logarithmic model.

    With the full constants, ``log M`` is
    ``k*z/(a*log(a*Lambda)) + g(z) + f(z)*a + perturbative(a)``;
    this helper evaluates the known divergence and finite terms.  The legacy
    ``m0_gev``/``delta_m_gev`` form remains an explicit small equation used by
    hybrid continuity diagnostics.
    """
    z = np.asarray(z_fm, dtype=float)
    if not np.all(np.isfinite(z)) or not math.isfinite(float(a_fm)) or float(a_fm) <= 0:
        raise ValueError("z and lattice spacing must be finite, with a>0")
    if k is None:
        return float(m0_gev) * z / HBAR_C_GEV_FM + float(delta_m_gev) * float(a_fm) / HBAR_C_GEV_FM
    a_lambda = float(a_fm) * float(lambda_qcd_gev) / HBAR_C_GEV_FM
    if a_lambda <= 0:
        raise ValueError("a*Lambda must be positive")
    divergence = float(k) * (z / float(a_fm)) / math.log(a_lambda)
    return divergence + _perturbative_log(float(a_fm), lambda_qcd_gev=float(lambda_qcd_gev), scale_gev=float(scale_gev), d=float(d), n_f=int(n_f))


def fit_factor(
    reference: EnsembleData | Sequence[EnsembleData],
    *,
    short_distance_max_fm: float,
    k: float,
    lambda_qcd_gev: float,
    d: float,
    n_f: int,
    scale_gev: float,
    zms_model: str = "pdf_nlo",
    short_distance_min_fm: float = 0.0,
    lattice_spacing_range_fm: tuple[float, float] | None = None,
) -> EnsembleData:
    """Fit a sample-bearing ``Z_R(a,z)`` factor over the declared ``(a,z)`` grid.

    For each resample and coordinate, the known divergence and perturbative
    term are subtracted from ``log M``.  A linear fit in ``a`` yields ``g(z)``;
    the short-distance fit of ``g(z)-log Z_MS`` through the origin yields
    ``m0`` and hence ``M_R``.  The returned factor is ``M/M_R`` and keeps the
    authored source ordering and sample index intact.
    """
    if isinstance(reference, EnsembleData) and reference.dims == ["a", "z"]:
        references = []
        for spacing in reference.coords["a"]:
            item = reference.at("a", spacing)
            attrs = item.attrs
            attrs["lattice_spacing_fm"] = float(spacing)
            references.append(
                EnsembleData(
                    item.ensemble,
                    item.resample,
                    [sample for sample in item.values],
                    item.dims,
                    item.coords,
                    attrs=attrs,
                    name=item.name,
                )
            )
    else:
        references = [reference] if isinstance(reference, EnsembleData) else list(reference)
    if len(references) < 2 or any("z" not in item.dims or item.dims != ["z"] for item in references) or short_distance_min_fm < 0 or short_distance_max_fm <= short_distance_min_fm:
        raise ValueError("self-renormalization requires at least two one-dimensional z references")
    first = references[0]
    z = np.asarray(first.coords["z"], dtype=float)
    if z.size == 0 or not np.all(np.isfinite(z)) or any(not np.allclose(item.coords["z"], z, rtol=0.0, atol=1e-12) for item in references[1:]):
        raise ValueError("self-renormalization references must share one finite z grid")
    if any(item.resample != first.resample or item.n_sample != first.n_sample for item in references[1:]):
        raise ValueError("self-renormalization references must share resampling mode and sample count")
    spacings: list[float] = []
    for item in references:
        spacing = item.attrs.get("lattice_spacing_fm")
        if spacing is None and item.ensemble is not None:
            spacing = item.ensemble.a_s
        if not isinstance(spacing, (int, float)) or isinstance(spacing, bool) or not math.isfinite(float(spacing)) or float(spacing) <= 0:
            raise ValueError("self-renormalization references require positive lattice_spacing_fm")
        spacings.append(float(spacing))
    if len(set(spacings)) != len(spacings):
        raise ValueError("self-renormalization reference lattice spacings must be unique")
    if lattice_spacing_range_fm is not None and any(value < lattice_spacing_range_fm[0] - 1e-12 or value > lattice_spacing_range_fm[1] + 1e-12 for value in spacings):
        raise ValueError("reference lattice spacings are outside the authored fit range")
    design = np.column_stack([np.ones(len(spacings)), np.asarray(spacings, dtype=float)])
    known = np.asarray([log_m(z, spacing, k=k, lambda_qcd_gev=lambda_qcd_gev, d=d, n_f=n_f, scale_gev=scale_gev) for spacing in spacings])
    values = np.stack([np.asarray(item.values) for item in references], axis=0)
    if np.any(~np.isfinite(values)) or np.any(values == 0):
        raise ValueError("self-renormalization references must be finite and nonzero")
    factor_samples: list[np.ndarray] = []
    m0_samples: list[float] = []
    m_r_samples: list[np.ndarray] = []
    zms = zmsbar_pdf_log(z, scale_gev=scale_gev, model=zms_model)
    short_mask = (np.abs(z) >= float(short_distance_min_fm) - 1e-12) & (np.abs(z) <= float(short_distance_max_fm) + 1e-12) & (np.abs(z) > 1e-12)
    if not np.any(short_mask):
        raise ValueError("short-distance range contains no nonzero z coordinate")
    for sample_index in range(first.n_sample):
        log_values = np.log(np.abs(values[:, sample_index, :]))
        adjusted = log_values - known
        coefficients = np.linalg.lstsq(design, adjusted, rcond=None)[0]
        g = coefficients[0]
        z_dimensionless = z[short_mask] / HBAR_C_GEV_FM
        remainder = np.real(g[short_mask] - zms[short_mask])
        m0 = float(np.linalg.lstsq(np.column_stack([z_dimensionless, np.ones_like(z_dimensionless)]), remainder, rcond=None)[0][0])
        m_r = np.exp(g - m0 * z / HBAR_C_GEV_FM)
        predicted_log = known + g[None, :] + coefficients[1][None, :] * np.asarray(spacings)[:, None]
        factor_samples.append(np.exp(predicted_log - np.log(m_r)[None, :]))
        m0_samples.append(m0)
        m_r_samples.append(m_r)
    attrs = first.attrs
    source_ids = [str(item.attrs.get("resample_id", item.attrs.get("ensemble_id", index))) for index, item in enumerate(references)]
    digest_payload = json.dumps({"source_resample_ids": source_ids, "spacings": spacings, "short_distance_min_fm": float(short_distance_min_fm), "short_distance_max_fm": float(short_distance_max_fm), "k": float(k), "lambda_qcd_gev": float(lambda_qcd_gev), "d": float(d), "n_f": int(n_f), "scale_gev": float(scale_gev), "zms_model": zms_model}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    attrs.update({
        "operation": "fit_factor",
        "renormalization_scheme": "msbar",
        "strategy": "self_renormalization",
        "short_distance_min_fm": float(short_distance_min_fm),
        "short_distance_max_fm": float(short_distance_max_fm),
        "formula": "log M=k*z/(a*log(a*Lambda))+g(z)+f(z)*a+perturbative(a); Z_R=M/M_R",
        "source_resample_ids": json.dumps(source_ids),
        "resample_id": hashlib.sha256(digest_payload).hexdigest(),
        "k": float(k),
        "lambda_qcd_gev": float(lambda_qcd_gev),
        "d": float(d),
        "n_f": int(n_f),
        "scale_gev": float(scale_gev),
        "zms_model": zms_model,
        "m0_gev": float(np.mean(m0_samples)),
        "units": '{"values":"dimensionless","a":"fm","z":"fm"}',
    })
    return EnsembleData(None, first.resample, factor_samples, ["a", "z"], {"a": spacings, "z": list(first.coords["z"])}, attrs=attrs, name="renormalization_factor")
