"""Helpers for qPDF/qTMDPDF asymptotic extrapolation and Fourier transforms."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lamet_agent.constants import GEV_FM
from lamet_agent.errors import OptionalDependencyError
from lamet_agent.extensions.statistics import gv, lsqfit


def _require_fit_dependencies() -> None:
    if gv is None:
        raise OptionalDependencyError("gvar is required for qPDF extrapolation.")
    if lsqfit is None:
        raise OptionalDependencyError("lsqfit is required for qPDF extrapolation.")


def _normalized_quark_sector(hadron: str, quark_sector: str | None) -> str:
    """Return a normalized quark-sector name for asymptotic-form dispatch."""

    hadron_name = str(hadron).lower()
    if hadron_name != "pion":
        return "generic"
    sector = "valence" if quark_sector is None else str(quark_sector).lower()
    if sector not in {"valence", "sea", "full"}:
        raise ValueError(f"Unsupported pion quark_sector {quark_sector!r}.")
    return sector


def _merge_prior_overrides(
    joint_prior_overrides: dict[str, Sequence[float]] | None,
    real_prior_overrides: dict[str, Sequence[float]] | None,
    imag_prior_overrides: dict[str, Sequence[float]] | None,
) -> dict[str, Sequence[float]]:
    """Merge optional prior overrides for joint fits and reject conflicting values."""

    merged: dict[str, Sequence[float]] = dict(joint_prior_overrides or {})
    for overrides in (real_prior_overrides or {}, imag_prior_overrides or {}):
        for key, value in overrides.items():
            if key in merged and list(merged[key]) != list(value):
                raise ValueError(f"Conflicting prior overrides for joint fit parameter {key!r}.")
            merged[key] = value
    return merged


def build_lambda_axis(
    z_axis: Sequence[float],
    *,
    lattice_spacing_fm: float,
    spatial_extent: int,
    momentum_vector: Sequence[float],
    coordinate_direction: Sequence[float],
    coordinate_step_multiplier: float = 1.0,
) -> np.ndarray:
    """Convert discrete z values to dimensionless Ioffe-time lambda."""

    z_array = np.asarray(z_axis, dtype=float)
    momentum = np.asarray(momentum_vector, dtype=float)
    direction = np.asarray(coordinate_direction, dtype=float)
    if momentum.shape != (3,) or direction.shape != (3,):
        raise ValueError("momentum_vector and coordinate_direction must both have length 3.")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("coordinate_direction must have non-zero norm.")
    direction = direction / norm
    momentum_gev = momentum * (2.0 * np.pi * GEV_FM / (float(spatial_extent) * float(lattice_spacing_fm)))
    lambda_per_z = (
        float(coordinate_step_multiplier)
        * float(lattice_spacing_fm)
        * float(np.dot(momentum_gev, direction))
        / GEV_FM
    )
    return z_array * lambda_per_z


def build_x_grid(config: dict[str, Any]) -> np.ndarray:
    """Build an x grid from either explicit values or start/stop/num settings."""

    if "values" in config:
        return np.asarray(config["values"], dtype=float)
    start = float(config.get("start", -2.0))
    stop = float(config.get("stop", 2.0))
    num = int(config.get("num", 4000))
    return np.linspace(start, stop, num=num, endpoint=bool(config.get("endpoint", False)))


def exp_decay_prior(
    *,
    hadron: str,
    gauge_type: str,
    quark_sector: str = "valence",
    prior_overrides: dict[str, Sequence[float]] | None = None,
):
    """Return the default asymptotic-form prior used for large-lambda fits."""

    _require_fit_dependencies()
    priors = gv.BufferDict()
    hadron_name = str(hadron).lower()
    gauge_name = str(gauge_type).lower()
    if hadron_name == "proton":
        priors["b"] = gv.gvar(0, 10)
        priors["c"] = gv.gvar(0, 10)
        priors["d"] = gv.gvar(0, 10)
        priors["e"] = gv.gvar(0, 10)
        priors["log(m)"] = gv.gvar(-2, 2)
        if gauge_name == "cg":
            priors["log(n)"] = gv.gvar(0.7, 1)
    elif hadron_name == "pion":
        sector = _normalized_quark_sector(hadron_name, quark_sector)
        if sector == "valence":
            for name in ("b1", "b2", "d1", "d2"):
                priors[name] = gv.gvar(0, 10)
            for name in ("c1", "e1"):
                priors[name] = gv.gvar(0, 10)
        elif sector == "sea":
            for name in ("b2", "d2"):
                priors[name] = gv.gvar(0, 10)
            for name in ("c2", "e2"):
                priors[name] = gv.gvar(0, 10)
        else:
            for name in ("b1", "b2", "b3", "d1", "d2", "d3"):
                priors[name] = gv.gvar(0, 10)
            for name in ("c1", "c2", "c3", "e1", "e2", "e3"):
                priors[name] = gv.gvar(0, 10)
        priors["log(m)"] = gv.gvar(-2, 2)
        if gauge_name == "cg":
            priors["log(n)"] = gv.gvar(0.7, 1)
    else:
        raise ValueError(f"Unsupported asymptotic hadron {hadron!r}.")
    for key, value in (prior_overrides or {}).items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Prior override for {key!r} must be a [mean, sigma] pair.")
        priors[key] = gv.gvar(float(value[0]), float(value[1]))
    return priors


def _tail_power(lambda_values: np.ndarray, params, *, gauge_type: str, m0: float) -> np.ndarray:
    """Return the common exponential and CG power-law suppression factor."""

    lam = np.asarray(lambda_values, dtype=float)
    tail = np.exp(-lam * (params["m"] + m0))
    if str(gauge_type).lower() == "cg":
        tail = tail / (lam**params["n"])
    return tail


def _complex_asymptotic_components(
    lambda_values: Sequence[float],
    params,
    *,
    hadron: str,
    gauge_type: str,
    quark_sector: str = "valence",
    m0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the real and imaginary asymptotic components for one hadron/sector."""

    lam = np.asarray(lambda_values, dtype=float)
    hadron_name = str(hadron).lower()
    gauge_name = str(gauge_type).lower()

    if hadron_name == "proton":
        real = params["b"] * np.cos(params["c"]) + params["d"] * np.cos(params["e"]) / np.abs(lam)
        imag = params["b"] * np.sin(params["c"]) + params["d"] * np.sin(params["e"]) / np.abs(lam)
        tail = _tail_power(lam, params, gauge_type=gauge_name, m0=m0)
        return real * tail, imag * tail

    if hadron_name == "pion":
        sector = _normalized_quark_sector(hadron_name, quark_sector)
        if sector == "valence":
            real = (
                params["b2"]
                + 2.0 * params["b1"] * np.cos(params["c1"] - lam)
                + (params["d2"] + 2.0 * params["d1"] * np.cos(params["e1"] - lam)) / np.abs(lam)
            )
            imag = np.zeros_like(lam, dtype=float)
        elif sector == "sea":
            real = params["b2"] * np.cos(params["c2"]) + params["d2"] * np.cos(params["e2"]) / np.abs(lam)
            imag = params["b2"] * np.sin(params["c2"]) + params["d2"] * np.sin(params["e2"]) / np.abs(lam)
        else:
            real = (
                params["b2"] * np.cos(params["c2"])
                + params["b1"] * np.cos(params["c1"] - lam)
                + params["b3"] * np.cos(params["c3"] + lam)
                + (
                    params["d2"] * np.cos(params["e2"])
                    + params["d1"] * np.cos(params["e1"] - lam)
                    + params["d3"] * np.cos(params["e3"] + lam)
                )
                / np.abs(lam)
            )
            imag = (
                params["b2"] * np.sin(params["c2"])
                + params["b1"] * np.sin(params["c1"] - lam)
                + params["b3"] * np.sin(params["c3"] + lam)
                + (
                    params["d2"] * np.sin(params["e2"])
                    + params["d1"] * np.sin(params["e1"] - lam)
                    + params["d3"] * np.sin(params["e3"] + lam)
                )
                / np.abs(lam)
            )
        tail = _tail_power(lam, params, gauge_type=gauge_name, m0=m0)
        return real * tail, imag * tail

    raise ValueError(f"Unsupported asymptotic hadron {hadron!r}.")


def asymptotic_joint_function(*, hadron: str, gauge_type: str, quark_sector: str = "valence", m0: float = 0.0):
    """Return the joint real/imaginary asymptotic fit function."""

    def fcn(lambda_values, params):
        real, imag = _complex_asymptotic_components(
            lambda_values,
            params,
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )
        return {"real": real, "imag": imag}

    return fcn


def asymptotic_real_function(*, hadron: str, gauge_type: str, quark_sector: str = "valence", m0: float = 0.0):
    """Return the asymptotic real-part fit function."""

    def fcn(lambda_values, params):
        real, _ = _complex_asymptotic_components(
            lambda_values,
            params,
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )
        return real

    return fcn


def asymptotic_imag_function(*, hadron: str, gauge_type: str, quark_sector: str = "valence", m0: float = 0.0):
    """Return the asymptotic imaginary-part fit function."""

    def fcn(lambda_values, params):
        _, imag = _complex_asymptotic_components(
            lambda_values,
            params,
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )
        return imag

    return fcn


def extrapolate_asymptotic_qpdf(
    lambda_axis: Sequence[float],
    real_values: Sequence[float],
    imag_values: Sequence[float],
    real_errors: Sequence[float],
    imag_errors: Sequence[float],
    *,
    fit_idx_range: Sequence[int],
    extrapolated_length: float,
    weight_ini: float = 0.0,
    m0: float = 0.0,
    hadron: str = "proton",
    gauge_type: str = "cg",
    quark_sector: str = "valence",
    joint_re_im_fit: bool = True,
    joint_prior_overrides: dict[str, Sequence[float]] | None = None,
    real_prior_overrides: dict[str, Sequence[float]] | None = None,
    imag_prior_overrides: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Fit the large-lambda tail and splice it onto the measured positive-lambda data."""

    _require_fit_dependencies()
    lambda_array = np.asarray(lambda_axis, dtype=float)
    real_array = np.asarray(real_values, dtype=float)
    imag_array = np.asarray(imag_values, dtype=float)
    real_sigma = np.asarray(real_errors, dtype=float)
    imag_sigma = np.asarray(imag_errors, dtype=float)
    if len(lambda_array) < 2:
        raise ValueError("qPDF extrapolation requires at least two lambda points.")
    start = int(fit_idx_range[0])
    stop = int(fit_idx_range[1])
    if start < 1 or stop <= start or stop > len(lambda_array):
        raise ValueError("fit_idx_range must satisfy 1 <= start < stop <= len(lambda_axis).")

    lam_gap = float(abs(lambda_array[1] - lambda_array[0]))
    fit_lambda = lambda_array[start:stop]
    fit_real = gv.gvar(real_array[start:stop], np.clip(real_sigma[start:stop], 1.0e-12, None))
    fit_imag = gv.gvar(imag_array[start:stop], np.clip(imag_sigma[start:stop], 1.0e-12, None))

    fit_result_joint = None
    if joint_re_im_fit:
        merged_prior_overrides = _merge_prior_overrides(
            joint_prior_overrides,
            real_prior_overrides,
            imag_prior_overrides,
        )
        fit_result_re = lsqfit.nonlinear_fit(
            data=(fit_lambda, {"real": fit_real, "imag": fit_imag}),
            prior=exp_decay_prior(
                hadron=hadron,
                gauge_type=gauge_type,
                quark_sector=quark_sector,
                prior_overrides=merged_prior_overrides,
            ),
            fcn=asymptotic_joint_function(
                hadron=hadron,
                gauge_type=gauge_type,
                quark_sector=quark_sector,
                m0=m0,
            ),
            maxit=10_000,
            svdcut=1e-100,
            fitter="scipy_least_squares",
        )
        fit_result_im = fit_result_re
        fit_result_joint = fit_result_re
    else:
        fit_result_re = lsqfit.nonlinear_fit(
            data=(fit_lambda, fit_real),
            prior=exp_decay_prior(
                hadron=hadron,
                gauge_type=gauge_type,
                quark_sector=quark_sector,
                prior_overrides=real_prior_overrides,
            ),
            fcn=asymptotic_real_function(hadron=hadron, gauge_type=gauge_type, quark_sector=quark_sector, m0=m0),
            maxit=10_000,
            svdcut=1e-100,
            fitter="scipy_least_squares",
        )
        fit_result_im = lsqfit.nonlinear_fit(
            data=(fit_lambda, fit_imag),
            prior=exp_decay_prior(
                hadron=hadron,
                gauge_type=gauge_type,
                quark_sector=quark_sector,
                prior_overrides=imag_prior_overrides,
            ),
            fcn=asymptotic_imag_function(hadron=hadron, gauge_type=gauge_type, quark_sector=quark_sector, m0=m0),
            maxit=10_000,
            svdcut=1e-100,
            fitter="scipy_least_squares",
        )

    lambda_tail = np.arange(lambda_array[start], float(extrapolated_length), lam_gap)
    if joint_re_im_fit:
        joint_tail = asymptotic_joint_function(
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )(lambda_tail, fit_result_joint.p)
        real_tail = joint_tail["real"]
        imag_tail = joint_tail["imag"]
    else:
        real_tail = asymptotic_real_function(
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )(lambda_tail, fit_result_re.p)
        imag_tail = asymptotic_imag_function(
            hadron=hadron,
            gauge_type=gauge_type,
            quark_sector=quark_sector,
            m0=m0,
        )(lambda_tail, fit_result_im.p)

    extrapolated_lambda = list(lambda_array[:start]) + list(lambda_tail)
    extrapolated_real = list(real_array[:start]) + list(np.asarray(gv.mean(real_tail), dtype=float))
    extrapolated_imag = list(imag_array[:start]) + list(np.asarray(gv.mean(imag_tail), dtype=float))

    num_gradual_points = min(stop - start, len(lambda_tail))
    weights = np.linspace(float(weight_ini), 1.0, num_gradual_points) if num_gradual_points else np.asarray([], dtype=float)
    for index, weight in enumerate(weights):
        extrapolated_real[start + index] = weight * extrapolated_real[start + index] + (1.0 - weight) * real_array[start + index]
        extrapolated_imag[start + index] = weight * extrapolated_imag[start + index] + (1.0 - weight) * imag_array[start + index]

    return {
        "lambda_axis": np.asarray(extrapolated_lambda, dtype=float),
        "real": np.asarray(extrapolated_real, dtype=float),
        "imag": np.asarray(extrapolated_imag, dtype=float),
        "fit_result_real": fit_result_re,
        "fit_result_imag": fit_result_im,
        "fit_result_joint": fit_result_joint,
        "joint_re_im_fit": bool(joint_re_im_fit),
    }


def mirror_qpdf_coordinate_space(
    lambda_axis: Sequence[float],
    real_values: Sequence[float],
    imag_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror positive-lambda qPDF data to negative lambda using even/odd symmetry."""

    lambda_array = np.asarray(lambda_axis, dtype=float)
    real_array = np.asarray(real_values, dtype=float)
    imag_array = np.asarray(imag_values, dtype=float)
    negative_lambda = -lambda_array[1:][::-1]
    mirrored_lambda = np.concatenate([negative_lambda, lambda_array])
    mirrored_real = np.concatenate([real_array[1:][::-1], real_array])
    mirrored_imag = np.concatenate([-imag_array[1:][::-1], imag_array])
    return mirrored_lambda, mirrored_real, mirrored_imag


def mirror_qpdf_coordinate_space_samples(
    lambda_axis: Sequence[float],
    real_samples: np.ndarray,
    imag_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror positive-lambda qPDF samples to negative lambda using even/odd symmetry."""

    lambda_array = np.asarray(lambda_axis, dtype=float)
    real_array = np.asarray(real_samples, dtype=float)
    imag_array = np.asarray(imag_samples, dtype=float)
    negative_lambda = -lambda_array[1:][::-1]
    mirrored_lambda = np.concatenate([negative_lambda, lambda_array])
    mirrored_real = np.concatenate([real_array[:, 1:][:, ::-1], real_array], axis=1)
    mirrored_imag = np.concatenate([-imag_array[:, 1:][:, ::-1], imag_array], axis=1)
    return mirrored_lambda, mirrored_real, mirrored_imag


def build_fourier_kernel(lambda_axis: Sequence[float], x_grid: Sequence[float]) -> dict[str, np.ndarray]:
    """Precompute cosine/sine kernels for repeated qPDF Fourier transforms."""

    lambda_array = np.asarray(lambda_axis, dtype=float)
    x_array = np.asarray(x_grid, dtype=float)
    delta_lambda = float(abs(lambda_array[1] - lambda_array[0])) if len(lambda_array) > 1 else 1.0
    phase = np.outer(x_array, lambda_array)
    return {
        "lambda_axis": lambda_array,
        "x_grid": x_array,
        "delta_lambda": np.asarray(delta_lambda, dtype=float),
        "cos": np.cos(phase),
        "sin": np.sin(phase),
    }


def fourier_transform_qpdf(
    lambda_axis: Sequence[float],
    real_values: Sequence[float],
    imag_values: Sequence[float],
    x_grid: Sequence[float],
    *,
    separate_re_im: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete Fourier transform using the Peskin-sign convention."""

    lambda_array = np.asarray(lambda_axis, dtype=float)
    real_array = np.asarray(real_values, dtype=float)
    imag_array = np.asarray(imag_values, dtype=float)
    x_array = np.asarray(x_grid, dtype=float)
    delta_lambda = float(abs(lambda_array[1] - lambda_array[0])) if len(lambda_array) > 1 else 1.0
    cos_term = np.cos(np.outer(x_array, lambda_array))
    sin_term = np.sin(np.outer(x_array, lambda_array))
    if separate_re_im:
        real_out = delta_lambda / (2.0 * np.pi) * (cos_term @ real_array)
        imag_out = delta_lambda / (2.0 * np.pi) * (-sin_term @ imag_array)
    else:
        real_out = delta_lambda / (2.0 * np.pi) * (cos_term @ real_array - sin_term @ imag_array)
        imag_out = delta_lambda / (2.0 * np.pi) * (sin_term @ real_array + cos_term @ imag_array)
    return real_out, imag_out


def batch_fourier_transform_qpdf(
    kernel: dict[str, np.ndarray],
    real_samples: np.ndarray,
    imag_samples: np.ndarray,
    *,
    separate_re_im: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the discrete qPDF Fourier transform to many samples at once."""

    real_array = np.asarray(real_samples, dtype=float)
    imag_array = np.asarray(imag_samples, dtype=float)
    factor = float(np.asarray(kernel["delta_lambda"])) / (2.0 * np.pi)
    cos_term = np.asarray(kernel["cos"], dtype=float)
    sin_term = np.asarray(kernel["sin"], dtype=float)
    if separate_re_im:
        real_out = factor * (real_array @ cos_term.T)
        imag_out = factor * (-imag_array @ sin_term.T)
    else:
        real_out = factor * (real_array @ cos_term.T - imag_array @ sin_term.T)
        imag_out = factor * (real_array @ sin_term.T + imag_array @ cos_term.T)
    return real_out, imag_out
