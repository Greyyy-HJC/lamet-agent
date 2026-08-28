"""Renormalization equations kept local to the renormalization stage."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel import nonlinear_fit


@dataclass(frozen=True)
class _FactorFitResult:
    factor: EnsembleData
    plot_data: dict[str, Any]


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
    spacing = float(data.ensemble.a_s)
    coords = data.coords
    coords["z"] = [float(value) * float(spacing) for value in coords["z"]]
    attrs.update({"coord_unit": "fm", "input_coord_unit": "lattice"})
    return EnsembleData(
        data.ensemble,
        data.resample,
        [np.asarray(sample) for sample in data.values],
        data.dims,
        coords,
        attrs=attrs,
        name=data.name,
    )


def divide_by_constant(target: EnsembleData, denominator: float) -> EnsembleData:
    """Apply ``h_R(z)=h_target(z)/C`` sample by sample."""
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("constant denominator must be finite and nonzero")
    values = [sample / denominator for sample in target.values]
    attrs = target.attrs
    attrs["denominator_kind"] = "constant"
    return EnsembleData(
        target.ensemble,
        target.resample,
        values,
        target.dims,
        target.coords,
        attrs=attrs,
        name="renormalized_matrix_element",
    )


def ratio(target: EnsembleData, denominator: EnsembleData) -> EnsembleData:
    """Apply the pointwise complex ratio on the complete physical grid."""
    if (
        target.attrs.get("resample_id")
        and denominator.attrs.get("resample_id")
        and target.attrs["resample_id"] != denominator.attrs["resample_id"]
    ):
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
    return EnsembleData(
        result.ensemble,
        result.resample,
        [np.asarray(sample) for sample in result.values],
        result.dims,
        result.coords,
        attrs=attrs,
        name=result.name,
    )


def hybrid_ratio(
    target: EnsembleData, denominator: EnsembleData, *, zs_fm: float, delta_m_gev: float, m0_gev: float = 0.0
) -> EnsembleData:
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
    attrs.update(
        {
            "zs_fm": switch,
            "delta_m_gev": float(delta_m_gev),
            "m0_gev": float(m0_gev),
            "strategy": "hybrid",
            "hybrid_switch_coord_fm": switch,
        }
    )
    return EnsembleData(
        base.ensemble, base.resample, values, base.dims, base.coords, attrs=attrs, name="renormalized_matrix_element"
    )


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


def zmsbar_log(
    kernel: Callable[..., Any],
    z_fm: np.ndarray,
    *,
    scale_gev: float,
    kernel_parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate one selected coordinate-space conversion kernel as ``log Z_MSbar``."""
    if not math.isfinite(float(scale_gev)) or float(scale_gev) <= 0:
        raise ValueError("scale_gev must be finite and positive")
    z = np.asarray(z_fm, dtype=float)
    nonzero = np.abs(z) > 1e-14
    conversion = np.ones_like(z)
    arguments = {"z_fm": z[nonzero], "mu": float(scale_gev), **dict(kernel_parameters or {})}
    evaluated = np.asarray(kernel(**arguments), dtype=float)
    try:
        evaluated = np.broadcast_to(evaluated, z[nonzero].shape)
    except ValueError as exc:
        raise ValueError("renormalization kernel output does not match the physical z grid") from exc
    if np.any(~np.isfinite(evaluated)) or np.any(evaluated <= 0):
        raise ValueError("renormalization kernel returned an invalid nonpositive conversion factor")
    conversion[nonzero] = evaluated
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
    return divergence + _perturbative_log(
        float(a_fm), lambda_qcd_gev=float(lambda_qcd_gev), scale_gev=float(scale_gev), d=float(d), n_f=int(n_f)
    )


def _fit_factor_result(
    reference: EnsembleData | Sequence[EnsembleData],
    *,
    short_distance_max_fm: float,
    k: float,
    lambda_qcd_gev: float,
    d: float,
    n_f: int,
    scale_gev: float,
    zms_kernel: Callable[..., Any],
    kernel_id: str,
    svdcut: float,
    kernel_parameters: Mapping[str, Any] | None = None,
    short_distance_min_fm: float = 0.0,
    lattice_spacing_range_fm: tuple[float, float] | None = None,
    sample_error_mode: str = "covariance",
) -> _FactorFitResult:
    """Fit the reference workflow's correlated mean self-renormalization factor."""
    if isinstance(reference, EnsembleData) and reference.dims == ["a", "z"]:
        source = reference
        spacings = [float(value) for value in source.coords["a"]]
        z = np.asarray(source.coords["z"], dtype=float)
        values = np.asarray(source.values)
    else:
        references = [reference] if isinstance(reference, EnsembleData) else list(reference)
        if len(references) < 2 or any(item.dims != ["z"] for item in references):
            raise ValueError("self-renormalization requires an (a,z) source or at least two z sources")
        source = references[0]
        spacings = [float(item.ensemble.a_s) for item in references]
        z = np.asarray(source.coords["z"], dtype=float)
        if any(not np.allclose(item.coords["z"], z, rtol=0.0, atol=1e-12) for item in references[1:]):
            raise ValueError("self-renormalization references must share one z grid")
        values = np.stack([np.asarray(item.values) for item in references], axis=1)
    if (
        values.ndim != 3
        or values.shape[1:] != (len(spacings), len(z))
        or np.any(~np.isfinite(values))
        or np.any(values == 0)
    ):
        raise ValueError("self-renormalization reference values must be finite, nonzero (sample,a,z) data")
    if lattice_spacing_range_fm is not None and any(
        value < lattice_spacing_range_fm[0] - 1e-12 or value > lattice_spacing_range_fm[1] + 1e-12 for value in spacings
    ):
        raise ValueError("reference lattice spacings are outside the authored fit range")
    log_data = EnsembleData(
        None,
        source.resample,
        [np.log(np.abs(sample)) for sample in values],
        ["a", "z"],
        {"a": spacings, "z": z.tolist()},
    )
    prior = gv.BufferDict()
    for coordinate in z:
        prior[f"g{coordinate}"] = gv.gvar(0.0, 20.0)
        prior[f"f1{coordinate}"] = gv.gvar(0.0, 5.0)
    fit_x = {"z": [], "a": []}
    for spacing in spacings:
        for coordinate in z:
            fit_x["z"].append(float(coordinate))
            fit_x["a"].append(float(spacing))
    fit_data = EnsembleData(
        None,
        source.resample,
        [np.asarray(sample).reshape(-1) for sample in log_data.values],
        ["point"],
        {"point": list(range(len(fit_x["z"])))},
    )

    def model(x, parameters):
        return [
            log_m(
                np.asarray([coordinate]),
                spacing,
                k=k,
                lambda_qcd_gev=lambda_qcd_gev,
                d=d,
                n_f=n_f,
                scale_gev=scale_gev,
            )[0]
            + parameters[f"g{coordinate}"]
            + parameters[f"f1{coordinate}"] * spacing / HBAR_C_GEV_FM
            for coordinate, spacing in zip(x["z"], x["a"])
        ]

    fit = nonlinear_fit(
        data=(fit_x, fit_data),
        prior=prior,
        fcn=model,
        mode="center",
        maxit=10000,
        svdcut=svdcut,
    )
    short = (z >= short_distance_min_fm - 1e-12) & (z <= short_distance_max_fm + 1e-12)
    if np.count_nonzero(short) < 3:
        raise ValueError("short-distance range must contain at least three coordinates")
    short_z = z[short]
    short_g = np.asarray([fit.p[f"g{coordinate}"] for coordinate in z], dtype=object)[short]
    short_g_data = EnsembleData(
        None,
        "gvar",
        short_g,
        ["z"],
        {"z": short_z.tolist()},
    )
    zms = zmsbar_log(
        zms_kernel,
        short_z,
        scale_gev=scale_gev,
        kernel_parameters=kernel_parameters,
    )
    m0_prior = gv.BufferDict()
    m0_prior["m0"] = gv.gvar(0.0, 20.0)
    m0_prior["b"] = gv.gvar(0.0, 100.0)
    m0_fit = nonlinear_fit(
        data=(short_z, short_g_data),
        prior=m0_prior,
        fcn=lambda coordinates, parameters: zms + parameters["m0"] * coordinates + parameters["b"],
        mode="center",
        maxit=10000,
        svdcut=svdcut,
    )
    m0 = m0_fit.p["m0"]
    g_parameters = np.asarray([fit.p[f"g{coordinate}"] for coordinate in z], dtype=object)
    finite = np.asarray([fit.p[f"f1{coordinate}"] for coordinate in z], dtype=object)
    factor_gvar = np.empty((len(spacings), len(z)), dtype=object)
    for spacing_index, spacing in enumerate(spacings):
        known = log_m(
            z,
            spacing,
            k=k,
            lambda_qcd_gev=lambda_qcd_gev,
            d=d,
            n_f=n_f,
            scale_gev=scale_gev,
        )
        factor_gvar[spacing_index] = gv.exp(known + finite * spacing / HBAR_C_GEV_FM + m0 * z)
    factor = np.asarray(gv.mean(factor_gvar), dtype=float)
    source_ids = [
        str(source.attrs.get("resample_id", source.ensemble.id if source.ensemble is not None else "reference"))
    ]
    attrs = dict(source.attrs)
    attrs.update(
        {
            "operation": "fit_factor",
            "type": "fit",
            "renormalization_scheme": "msbar",
            "strategy": "self_renormalization",
            "short_distance_min_fm": float(short_distance_min_fm),
            "short_distance_max_fm": float(short_distance_max_fm),
            "formula": "reference_correlated_mean_self_renormalization",
            "source_resample_ids": json.dumps(source_ids),
            "resample_id": hashlib.sha256(
                json.dumps(
                    {
                        "source_resample_ids": source_ids,
                        "spacings": spacings,
                        "short_distance_min_fm": float(short_distance_min_fm),
                        "short_distance_max_fm": float(short_distance_max_fm),
                        "k": float(k),
                        "lambda_qcd_gev": float(lambda_qcd_gev),
                        "d": float(d),
                        "n_f": int(n_f),
                        "scale_gev": float(scale_gev),
                        "kernel_id": kernel_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "k": float(k),
            "lambda_qcd_gev": float(lambda_qcd_gev),
            "d": float(d),
            "n_f": int(n_f),
            "scale_gev": float(scale_gev),
            "kernel_id": kernel_id,
            "kernel_parameters": json.dumps(dict(kernel_parameters or {}), sort_keys=True),
            "m0_gev": float(gv.mean(m0)),
            "m0_convention": "reference_inverse_fm",
            "units": '{"values":"dimensionless","a":"fm","z":"fm"}',
        }
    )
    factor_data = EnsembleData(
        None,
        source.resample,
        [factor],
        ["a", "z"],
        {"a": spacings, "z": z.tolist()},
        attrs=attrs,
        name="renormalization_factor",
    )
    lnm = log_data.average(sample_error_mode)
    m_r = gv.exp(g_parameters - m0 * z)
    zmsbar = np.full(len(z), np.nan, dtype=float)
    finite_z = np.abs(z) > 1e-12
    if np.any(finite_z):
        zmsbar[finite_z] = np.exp(zmsbar_log(zms_kernel, np.abs(z[finite_z]), scale_gev=scale_gev))
    m_r_ratio_mean = np.full(len(z), np.nan, dtype=float)
    m_r_ratio_sdev = np.full(len(z), np.nan, dtype=float)
    if np.any(finite_z):
        ratio_values = m_r[finite_z] / zmsbar[finite_z]
        m_r_ratio_mean[finite_z] = np.asarray(gv.mean(ratio_values), dtype=float)
        m_r_ratio_sdev[finite_z] = np.asarray(gv.sdev(ratio_values), dtype=float)
    m_over_zr = gv.exp(lnm) / factor_gvar
    plot_data = {
        "kind": "fit",
        "z_fm": z.tolist(),
        "a_fm": [float(value) for value in spacings],
        "inverse_a_gev": (HBAR_C_GEV_FM / np.asarray(spacings, dtype=float)).tolist(),
        "lnm_mean": np.asarray(gv.mean(lnm), dtype=float).tolist(),
        "lnm_sdev": np.asarray(gv.sdev(lnm), dtype=float).tolist(),
        "factor_mean": factor.tolist(),
        "factor_sdev": np.asarray(gv.sdev(factor_gvar), dtype=float).tolist(),
        "g_mean": np.asarray(gv.mean(g_parameters), dtype=float).tolist(),
        "g_sdev": np.asarray(gv.sdev(g_parameters), dtype=float).tolist(),
        "f1_mean": np.asarray(gv.mean(finite), dtype=float).tolist(),
        "f1_sdev": np.asarray(gv.sdev(finite), dtype=float).tolist(),
        "m0_mean": float(gv.mean(m0)),
        "m0_sdev": float(gv.sdev(m0)),
        "mR_mean": np.asarray(gv.mean(m_r), dtype=float).tolist(),
        "mR_sdev": np.asarray(gv.sdev(m_r), dtype=float).tolist(),
        "zmsbar": [float(value) if np.isfinite(value) else None for value in zmsbar],
        "mR_over_zmsbar_mean": [float(value) if np.isfinite(value) else None for value in m_r_ratio_mean],
        "mR_over_zmsbar_sdev": [float(value) if np.isfinite(value) else None for value in m_r_ratio_sdev],
        "m_over_zR_mean": np.asarray(gv.mean(m_over_zr), dtype=float).tolist(),
        "m_over_zR_sdev": np.asarray(gv.sdev(m_over_zr), dtype=float).tolist(),
    }
    return _FactorFitResult(factor=factor_data, plot_data=plot_data)


def fit_factor(
    reference: EnsembleData | Sequence[EnsembleData],
    *,
    short_distance_max_fm: float,
    k: float,
    lambda_qcd_gev: float,
    d: float,
    n_f: int,
    scale_gev: float,
    zms_kernel: Callable[..., Any],
    kernel_id: str,
    svdcut: float,
    short_distance_min_fm: float = 0.0,
    lattice_spacing_range_fm: tuple[float, float] | None = None,
    sample_error_mode: str = "covariance",
) -> EnsembleData:
    """Fit and return the reusable factor while keeping diagnostics internal."""
    return _fit_factor_result(
        reference,
        short_distance_max_fm=short_distance_max_fm,
        k=k,
        lambda_qcd_gev=lambda_qcd_gev,
        d=d,
        n_f=n_f,
        scale_gev=scale_gev,
        zms_kernel=zms_kernel,
        kernel_id=kernel_id,
        svdcut=svdcut,
        short_distance_min_fm=short_distance_min_fm,
        lattice_spacing_range_fm=lattice_spacing_range_fm,
        sample_error_mode=sample_error_mode,
    ).factor
