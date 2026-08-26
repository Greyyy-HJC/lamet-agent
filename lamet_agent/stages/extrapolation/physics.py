"""Additive extrapolation basis and sample-level linear fits."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM


def load_data(value: Any) -> EnsembleData:
    """Load one explicitly listed matched-distribution artifact."""
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, Path):
        if value.suffix.lower() != ".nc":
            raise ValueError(f"extrapolation input must be a .nc artifact: {value}")
        return EnsembleData.from_netcdf(value)
    raise TypeError("extrapolation input is neither EnsembleData nor a NetCDF Path")


def basis_terms(attrs: dict[str, object], terms: list[str], physical_mass: float | None) -> list[float]:
    """Evaluate the authored dimensionless correction basis at one input."""
    a = float(attrs["lattice_spacing_fm"])
    length = float(attrs["L_s"]) * a
    pion_mass = float(attrs["m_pi"])
    momentum = float(attrs["momentum_gev"])
    if a <= 0 or length <= 0 or pion_mass <= 0 or momentum <= 0:
        raise ValueError("extrapolation kinematics must be finite and positive")
    r = pion_mass * pion_mass
    physical_r = None if physical_mass is None else physical_mass * physical_mass
    values = []
    for term in terms:
        if term == "a":
            values.append(a)
        elif term == "a2":
            values.append(a**2)
        elif term == "a4":
            values.append(a**4)
        elif term == "ap2":
            values.append((a * momentum) ** 2)
        elif term == "ap4":
            values.append((a * momentum) ** 4)
        elif term == "exp_mpi_L":
            values.append(math.exp(-pion_mass * length / HBAR_C_GEV_FM))
        elif term == "exp_sqrt2_mpi_L":
            values.append(math.exp(-math.sqrt(2.0) * pion_mass * length / HBAR_C_GEV_FM))
        elif term == "mpi2":
            if physical_r is None:
                raise ValueError("mpi2 requires physical_pion_mass_gev")
            values.append(r - physical_r)
        elif term == "mpi4_log_mpi2":
            if physical_r is None:
                raise ValueError("mpi4_log_mpi2 requires physical_pion_mass_gev")
            values.append(r * r * math.log(r) - physical_r * physical_r * math.log(physical_r))
        elif term == "inv_p2":
            values.append(1.0 / momentum**2)
        elif term == "inv_p4":
            values.append(1.0 / momentum**4)
        else:
            raise ValueError(f"unsupported extrapolation term '{term}'")
    return values


def extrapolation_fcn(x: dict[str, object], parameters: dict[str, object]) -> np.ndarray:
    """Evaluate real and imaginary ensemble differences jointly."""
    design = np.asarray(x["design"], dtype=float)
    terms = list(x["terms"])
    x_dependence = dict(x["x_dependence"])
    output_size = int(x["output_size"])
    predicted_re = np.zeros((design.shape[0], output_size), dtype=object)
    predicted_im = np.zeros((design.shape[0], output_size), dtype=object)
    for term_index, term in enumerate(terms):
        basis = design[:, term_index, None]
        coefficient_re = np.asarray(parameters[f"{term}_re"])
        coefficient_im = np.asarray(parameters[f"{term}_im"])
        if not x_dependence[term]:
            coefficient_re = np.full(output_size, coefficient_re.item(), dtype=object)
            coefficient_im = np.full(output_size, coefficient_im.item(), dtype=object)
        predicted_re += basis * coefficient_re[None, :]
        predicted_im += basis * coefficient_im[None, :]
    return np.concatenate([predicted_re.T, predicted_im.T], axis=1).reshape(-1)


def _reference_extrapolation_fcn(x: dict[str, object], parameters: dict[str, object]) -> np.ndarray:
    design = np.asarray(x["design"], dtype=float)
    terms = list(x["terms"])
    x_dependence = dict(x["x_dependence"])
    n_x = int(x["n_x"])
    prediction = np.tile(np.asarray(parameters["h0"], dtype=object), (design.shape[0], 1))
    for term_index, term in enumerate(terms):
        coefficient = np.asarray(parameters[term], dtype=object)
        if not x_dependence[term]:
            coefficient = np.full(n_x, coefficient.item(), dtype=object)
        prediction = prediction + design[:, term_index, None] * coefficient[None, :]
    return prediction.reshape(-1)


def fit_candidate(
    data: list[EnsembleData],
    terms: list[str],
    physical_mass: float | None,
    priors: dict[str, float],
    *,
    x_range: tuple[float, float],
    x_dependence: dict[str, bool] | None = None,
    pdep_gev: list[float] | None = None,
    posterior_prior_error_scale: float = 3.0,
    workers: int = 1,
    _parallel=None,
) -> tuple[EnsembleData, dict[str, float]]:
    """Run the reference joint-x, sample-bearing extrapolation fit."""
    import gvar as gv

    from lamet_agent.parallel import nonlinear_fit

    if not data or any(item.dims != ["x"] for item in data):
        raise ValueError("extrapolation requires one-dimensional x-space inputs")
    x_all = np.asarray(data[0].coords["x"], dtype=float)
    if any(not np.allclose(item.coords["x"], x_all, rtol=0.0, atol=1e-12) for item in data[1:]):
        raise ValueError("all extrapolation grids must match by coordinate value")
    mask = (x_all >= float(x_range[0]) - 1e-12) & (x_all <= float(x_range[1]) + 1e-12)
    if not np.any(mask):
        raise ValueError("fit range selects no x coordinate")
    x = x_all[mask]
    if set(priors) != {"mean", "sdev"}:
        raise ValueError("candidate priors must contain exactly mean and sdev")
    prior_center = float(priors["mean"])
    prior_width = float(priors["sdev"])
    if (
        not math.isfinite(prior_center)
        or not math.isfinite(prior_width)
        or prior_width <= 0
    ):
        raise ValueError("candidate prior mean and sdev must be finite with positive sdev")
    x_dependence = {term: True for term in terms} if x_dependence is None else dict(x_dependence)
    if set(x_dependence) != set(terms):
        raise ValueError("x_dependence must contain every selected term")
    if not math.isfinite(posterior_prior_error_scale) or posterior_prior_error_scale <= 0:
        raise ValueError("posterior_prior_error_scale must be finite and positive")
    n_sample = min(item.n_sample for item in data)
    if n_sample < 2 or any(item.resample != data[0].resample for item in data):
        raise ValueError("extrapolation inputs require one shared bootstrap/jackknife mode")
    values = np.stack([np.asarray(item.values)[:n_sample, mask] for item in data], axis=0)
    if np.iscomplexobj(values):
        if not np.allclose(np.imag(values), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("reference extrapolation consumes the real matching channel")
        values = np.real(values)
    design = np.asarray(
        [basis_terms(item.attrs, terms, physical_mass) for item in data],
        dtype=float,
    )
    flattened = np.moveaxis(values, 1, 0).reshape(n_sample, -1)
    fit_data = EnsembleData(
        None,
        data[0].resample,
        list(flattened),
        ["observation"],
        {"observation": list(range(flattened.shape[1]))},
    )
    sample_error_modes = {str(item.attrs.get("sample_error_mode", "covariance")) for item in data}
    if len(sample_error_modes) != 1:
        raise ValueError("all extrapolation inputs must share sample_error_mode")
    sample_error_mode = sample_error_modes.pop()
    covariance = np.asarray(gv.evalcov(fit_data.average(sample_error_mode)), dtype=float)
    prior = gv.BufferDict()
    prior["h0"] = gv.gvar(
        np.full(len(x), prior_center), np.full(len(x), prior_width)
    )
    for term in terms:
        prior[term] = (
            gv.gvar(
                np.full(len(x), prior_center), np.full(len(x), prior_width)
            )
            if x_dependence[term]
            else gv.gvar(prior_center, prior_width)
        )
    fit_x = {"design": design, "terms": terms, "x_dependence": x_dependence, "n_x": len(x)}
    fit = nonlinear_fit(
        (fit_x, fit_data),
        _reference_extrapolation_fcn,
        prior,
        workers=workers,
        sample_prior_scale=posterior_prior_error_scale,
        covariance=covariance,
        sample_error_mode=sample_error_mode,
        tolerate_sample_failures=True,
        _parallel=_parallel,
        maxit=2000,
        svdcut=1e-12,
    )
    fitted_parameters = [parameters or fit.pmean for parameters in fit.samples]
    samples = [np.asarray(parameters["h0"], dtype=float) for parameters in fitted_parameters]
    parameter_mean: dict[str, object] = {}
    parameter_sdev: dict[str, object] = {}
    for name in ["h0", *terms]:
        values = np.asarray([np.asarray(parameters[name], dtype=float) for parameters in fitted_parameters])
        mean = np.mean(values, axis=0)
        sdev = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        parameter_mean[name] = float(mean) if mean.ndim == 0 else mean.tolist()
        parameter_sdev[name] = float(sdev) if sdev.ndim == 0 else sdev.tolist()
    momentum_dependence: dict[str, dict[str, object]] = {}
    for momentum in pdep_gev or []:
        momentum = float(momentum)
        if not math.isfinite(momentum) or momentum <= 0:
            raise ValueError("pdep_gev values must be finite and positive")
        predictions = []
        for parameters in fitted_parameters:
            prediction = np.asarray(parameters["h0"], dtype=float).copy()
            if "inv_p2" in terms:
                prediction = prediction + np.asarray(parameters["inv_p2"], dtype=float) / momentum**2
            if "inv_p4" in terms:
                prediction = prediction + np.asarray(parameters["inv_p4"], dtype=float) / momentum**4
            predictions.append(prediction)
        values = np.asarray(predictions, dtype=float)
        momentum_dependence[f"{momentum:g}"] = {
            "momentum_gev": momentum,
            "mean": np.mean(values, axis=0).tolist(),
            "sdev": (np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1], dtype=float)).tolist(),
        }
    attrs = dict(data[0].attrs)
    attrs.update(
        {
            "ensemble": None,
            "extrapolation_terms": ",".join(terms),
            "x_dependence": json.dumps(x_dependence, sort_keys=True),
            "physical_point": "continuum,infinite_momentum",
            "sample_error_mode": sample_error_mode,
            "posterior_prior_error_scale": float(posterior_prior_error_scale),
            "initial_prior_mean": prior_center,
            "initial_prior_sdev": prior_width,
            "units": '{"values":"dimensionless","x":"dimensionless"}',
        }
    )
    if physical_mass is not None:
        attrs["physical_pion_mass_gev"] = float(physical_mass)
    result = EnsembleData(
        None,
        data[0].resample,
        samples,
        ["x"],
        {"x": x.tolist()},
        attrs=attrs,
        name="extrapolated_distribution",
    )
    parameter_count = len(x) + sum(len(x) if x_dependence[term] else 1 for term in terms)
    return result, {
        "chi2": float(fit.chi2),
        "dof": float(fit.dof),
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "aic": float(fit.chi2 + 2.0 * parameter_count),
        "n_failed_samples": float(fit.n_failed_samples),
        "parameter_mean": parameter_mean,
        "parameter_sdev": parameter_sdev,
        "momentum_dependence": momentum_dependence,
    }
