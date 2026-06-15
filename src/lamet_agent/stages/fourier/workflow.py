"""Sample-preserving extrapolation and Fourier-transform workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gvar as gv
import lsqfit
import numpy as np

from lamet_agent.core.resampling import bs_ls_avg, jk_ls_avg

FM_TO_GEV_INV = 5.067731237

OBSERVABLE_ALIASES = {
    "pion_quark_quasi_pdf": "pion_quark_quasi_pdf",
    "pion_pdf": "pion_quark_quasi_pdf",
    "nucleon_quark_unpolarized_quasi_pdf": "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_unpolarized_pdf": "nucleon_quark_unpolarized_quasi_pdf",
    "unpolarized_pdf": "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_quark_transversity_quasi_pdf": "nucleon_quark_transversity_quasi_pdf",
    "nucleon_transversity_pdf": "nucleon_quark_transversity_quasi_pdf",
    "transversity_pdf": "nucleon_quark_transversity_quasi_pdf",
    "pion_gluon_quasi_pdf": "pion_gluon_quasi_pdf",
    "pion_gluon_pdf": "pion_gluon_quasi_pdf",
    "nucleon_gluon_quasi_pdf": "nucleon_gluon_quasi_pdf",
    "nucleon_gluon_pdf": "nucleon_gluon_quasi_pdf",
    "meson_quasi_da": "meson_quasi_da",
    "quasi_da": "meson_quasi_da",
    "pion_quark_quasi_gpd": "pion_quark_quasi_gpd",
    "pion_gpd": "pion_quark_quasi_gpd",
    "nucleon_quark_quasi_gpd": "nucleon_quark_quasi_gpd",
    "nucleon_gpd": "nucleon_quark_quasi_gpd",
}


def _normalise_resample_mode(value: str | None) -> str:
    mode = "bootstrap" if value is None else str(value).strip().lower()
    aliases = {
        "bs": "bootstrap",
        "boot": "bootstrap",
        "bootstrap": "bootstrap",
        "jk": "jackknife",
        "jackknife": "jackknife",
        "raw": "raw",
    }
    if mode not in aliases:
        raise ValueError("resample_mode must be 'bs'/'bootstrap', 'jk'/'jackknife', or 'raw'")
    return aliases[mode]


def _sample_gvar(samples, *, resample_mode: str = "bootstrap") -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        return gv.gvar(arr, np.zeros_like(arr, dtype=float))
    if arr.shape[0] < 2:
        mean = np.mean(arr, axis=0)
        return gv.gvar(mean, np.zeros_like(mean, dtype=float))

    mode = _normalise_resample_mode(resample_mode)
    trailing_shape = arr.shape[1:]
    if mode in {"bootstrap", "jackknife"} and int(np.prod(trailing_shape or (1,))) == 1:
        flat = arr.reshape(arr.shape[0], 1)
        duplicated = np.repeat(flat, 2, axis=1)
        values = jk_ls_avg(duplicated) if mode == "jackknife" else bs_ls_avg(duplicated)
        value = values.reshape(-1)[0]
        mean = float(gv.mean(value))
        sdev = float(gv.sdev(value))
        if trailing_shape == ():
            return gv.gvar(mean, sdev)
        return gv.gvar(np.full(trailing_shape, mean), np.full(trailing_shape, sdev))
    if mode == "jackknife":
        return jk_ls_avg(arr)
    if mode == "bootstrap":
        return bs_ls_avg(arr)

    mean = np.mean(arr, axis=0)
    sdev = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return gv.gvar(mean, sdev)


def _sample_sdev(samples, *, resample_mode: str = "bootstrap") -> np.ndarray:
    return np.asarray(gv.sdev(_sample_gvar(samples, resample_mode=resample_mode)), dtype=float)


def _normalise_fit_error_mode(value: str | None) -> str:
    mode = "diagonal" if value is None else str(value).strip().lower()
    aliases = {
        "diag": "diagonal",
        "diagonal": "diagonal",
        "sdev": "diagonal",
        "std": "diagonal",
        "cov": "covariance",
        "covariance": "covariance",
        "full": "covariance",
    }
    if mode not in aliases:
        raise ValueError("fit_error_mode must be 'diagonal' or 'covariance'")
    return aliases[mode]


def _normalise_part(value: str | None) -> str:
    part = "both" if value is None else str(value).strip().lower()
    aliases = {
        "both": "both",
        "re": "re",
        "real": "re",
        "im": "im",
        "imag": "im",
        "imaginary": "im",
    }
    if part not in aliases:
        raise ValueError("part must be 'both', 're', or 'im'")
    return aliases[part]


def _uses_re(part: str) -> bool:
    return _normalise_part(part) in {"both", "re"}


def _uses_im(part: str) -> bool:
    return _normalise_part(part) in {"both", "im"}


def _n_fit_channels(part: str) -> int:
    return int(_uses_re(part)) + int(_uses_im(part))


def _fit_y_data(
    re_fit: np.ndarray,
    im_fit: np.ndarray,
    *,
    fit_error_mode: str,
    resample_mode: str,
    part: str = "both",
    re_fit_samples: np.ndarray | None = None,
    im_fit_samples: np.ndarray | None = None,
    sigma_re: np.ndarray | None = None,
    sigma_im: np.ndarray | None = None,
) -> np.ndarray:
    mode = _normalise_fit_error_mode(fit_error_mode)
    part = _normalise_part(part)
    if mode == "covariance":
        blocks = []
        centers = []
        if _uses_re(part):
            if re_fit_samples is None:
                raise ValueError("fit_error_mode='covariance' requires re_fit_samples for part='re' or 'both'")
            blocks.append(np.asarray(re_fit_samples, dtype=float))
            centers.append(np.asarray(re_fit, dtype=float))
        if _uses_im(part):
            if im_fit_samples is None:
                raise ValueError("fit_error_mode='covariance' requires im_fit_samples for part='im' or 'both'")
            blocks.append(np.asarray(im_fit_samples, dtype=float))
            centers.append(np.asarray(im_fit, dtype=float))
        sample_matrix = np.concatenate(blocks, axis=1)
        covariance_data = _sample_gvar(sample_matrix, resample_mode=resample_mode)
        center = np.concatenate(centers)
        return gv.gvar(center, gv.evalcov(covariance_data))

    re_scale = np.ones_like(re_fit, dtype=float) if sigma_re is None else np.asarray(sigma_re, dtype=float)
    im_scale = np.ones_like(im_fit, dtype=float) if sigma_im is None else np.asarray(sigma_im, dtype=float)
    values = []
    errors = []
    if _uses_re(part):
        values.append(np.asarray(re_fit, dtype=float))
        errors.append(re_scale)
    if _uses_im(part):
        values.append(np.asarray(im_fit, dtype=float))
        errors.append(im_scale)
    sigma = np.maximum(np.concatenate(errors), 1e-12)
    return gv.gvar(np.concatenate(values), sigma)


def _select_fit_prediction(pred_re: np.ndarray, pred_im: np.ndarray, part: str) -> np.ndarray:
    part = _normalise_part(part)
    if part == "re":
        return pred_re
    if part == "im":
        return pred_im
    return np.concatenate([pred_re, pred_im])


def _zero_inactive_channel(re_values: np.ndarray, im_values: np.ndarray, part: str) -> tuple[np.ndarray, np.ndarray]:
    part = _normalise_part(part)
    if part == "re":
        return re_values, np.zeros_like(im_values, dtype=float)
    if part == "im":
        return np.zeros_like(re_values, dtype=float), im_values
    return re_values, im_values


def sum_ft_re_im(x_ls, fx_re_ls, fx_im_ls, output_k):
    """Forward transform with separated real and imaginary input parts."""
    x = np.asarray(x_ls)
    fx_re = np.asarray(fx_re_ls)
    fx_im = np.asarray(fx_im_ls)
    delta_x = abs(x[1] - x[0])
    k = np.asarray(output_k)
    pref = delta_x / (2 * np.pi)

    if k.ndim == 0:
        phase = x * k
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        val_re = pref * np.sum(cos_phase * fx_re) - pref * np.sum(sin_phase * fx_im)
        val_im = pref * np.sum(sin_phase * fx_re) + pref * np.sum(cos_phase * fx_im)
        return val_re, val_im

    phase = np.multiply.outer(x, k)
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    val_re = pref * np.sum(cos_phase * fx_re[:, None], axis=0) - pref * np.sum(
        sin_phase * fx_im[:, None], axis=0
    )
    val_im = pref * np.sum(sin_phase * fx_re[:, None], axis=0) + pref * np.sum(
        cos_phase * fx_im[:, None], axis=0
    )
    return val_re, val_im


def complete_z_negative(lam_ls, re_ls, im_ls, *, im_flip_for_ft=False):
    """Complete the negative-z branch using Re even and Im odd symmetry."""
    lam = np.asarray(lam_ls)
    re = np.asarray(re_ls)
    im = np.asarray(im_ls)

    if im_flip_for_ft:
        im = -im

    lam_full = np.concatenate([-lam[::-1][:-1], lam])
    re_full = np.concatenate([re[::-1][:-1], re])
    im_full = np.concatenate([-im[::-1][:-1], im])
    return lam_full, re_full, im_full


def _as_sample_matrix(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array shaped (n_sample,n_z)")
    return arr


def _uniform_step(coord: np.ndarray) -> float:
    if coord.ndim != 1 or len(coord) < 2:
        raise ValueError("coordinate grid must be a 1D array with at least two points")
    diffs = np.diff(coord)
    if np.any(diffs <= 0):
        raise ValueError("coordinate grid must be strictly increasing")
    if not np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        raise ValueError("coordinate grid must be uniform for this workflow")
    return float(diffs[0])


def _coord_scale(coord_unit: str, *, pz_gev: float | None, a_fm: float | None) -> tuple[float, float]:
    """Return ``(fit_scale, ft_scale)`` from input coordinates."""
    unit = coord_unit.lower()
    if unit == "lambda":
        return 1.0, 1.0
    if unit == "gev_inv":
        if pz_gev is None:
            raise ValueError("pz_gev is required when coord_unit='gev_inv'")
        return 1.0, float(pz_gev)
    if unit == "fm":
        if pz_gev is None:
            raise ValueError("pz_gev is required when coord_unit='fm'")
        return FM_TO_GEV_INV, FM_TO_GEV_INV * float(pz_gev)
    if unit == "lattice":
        if pz_gev is None or a_fm is None:
            raise ValueError("pz_gev and a_fm are required when coord_unit='lattice'")
        return float(a_fm) * FM_TO_GEV_INV, float(a_fm) * FM_TO_GEV_INV * float(pz_gev)
    raise ValueError("coord_unit must be 'lambda', 'gev_inv', 'fm', or 'lattice'")


def _convert_scheme_value(value: float, fit_scale: float) -> float:
    return float(value) * fit_scale


def _canonical_observable(observable: str) -> str:
    key = observable.lower().replace("-", "_").replace(" ", "_")
    if key not in OBSERVABLE_ALIASES:
        allowed = ", ".join(sorted(set(OBSERVABLE_ALIASES.values())))
        raise ValueError(f"observable must be one of: {allowed}")
    return OBSERVABLE_ALIASES[key]


def _observable_term_names(observable: str) -> list[str]:
    observable = _canonical_observable(observable)
    if observable == "pion_quark_quasi_pdf":
        return ["2", "1", "3"]
    if observable in {"nucleon_quark_unpolarized_quasi_pdf", "nucleon_quark_transversity_quasi_pdf"}:
        return ["2"]
    if observable == "meson_quasi_da":
        return ["1", "2"]
    if observable == "pion_quark_quasi_gpd":
        return ["1", "3", "2", "t2"]
    if observable == "nucleon_quark_quasi_gpd":
        return ["2", "t2"]
    raise ValueError(f"unsupported observable {observable!r}")


def _phase_scales(
    *,
    coord_unit: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    ft_scale_over_fit_scale: float,
) -> tuple[float, float | None]:
    if coord_unit.lower() == "lambda":
        phase_scale = ft_scale_over_fit_scale
        if pz_prime_gev is None:
            return phase_scale, None
        if pz_gev is None:
            raise ValueError("pz_gev is required with pz_prime_gev when coord_unit='lambda'")
        return phase_scale, float(pz_prime_gev) / float(pz_gev)
    return float(pz_gev or 0.0), None if pz_prime_gev is None else float(pz_prime_gev)


def _term_phase_scales(
    observable: str,
    *,
    phase_scale: float,
    phase_prime_scale: float | None,
) -> list[float]:
    observable = _canonical_observable(observable)
    pzp = phase_scale if phase_prime_scale is None else phase_prime_scale
    if observable == "pion_quark_quasi_pdf":
        return [0.0, -phase_scale, phase_scale]
    if observable in {"nucleon_quark_unpolarized_quasi_pdf", "nucleon_quark_transversity_quasi_pdf"}:
        return [0.0]
    if observable == "meson_quasi_da":
        return [-phase_scale, 0.0]
    if observable == "pion_quark_quasi_gpd":
        return [-phase_scale, pzp, 0.0, -(phase_scale - pzp)]
    if observable == "nucleon_quark_quasi_gpd":
        return [0.0, -(phase_scale - pzp)]
    raise ValueError(f"unsupported observable {observable!r}")


def _append_tail_parameter_bounds(
    p0: list[float],
    lower: list[float],
    upper: list[float],
    *,
    method: str,
    lambda_lower: float,
) -> None:
    p0.append(max(0.5, lambda_lower + 0.05))
    lower.append(lambda_lower)
    upper.append(max(3.0, lambda_lower + 1.0))

    if method == "CG":
        p0.append(0.5)
        lower.append(-2.0)
        upper.append(4.0)


def _param_template(
    method: str,
    order: str,
    observable: str,
    *,
    Lambda0: float = 0.1,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    method = method.upper()
    order = order.upper()
    lambda_lower = float(Lambda0)
    if method not in {"GI", "CG"}:
        raise ValueError("method must be 'GI' or 'CG'")
    if order not in {"LA", "NLA"}:
        raise ValueError("order must be 'LA' or 'NLA'")

    observable = _canonical_observable(observable)
    if observable == "nucleon_gluon_quasi_pdf":
        p0 = [1.0]
        lower = [-np.inf]
        upper = [np.inf]
        if order == "NLA":
            p0.append(0.1)
            lower.append(-np.inf)
            upper.append(np.inf)
        _append_tail_parameter_bounds(p0, lower, upper, method=method, lambda_lower=lambda_lower)
        return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

    if observable == "pion_gluon_quasi_pdf":
        p0 = [1.0]
        lower = [-np.inf]
        upper = [np.inf]
        if order == "NLA":
            p0.extend([0.1, 0.1, 0.0])
            lower.extend([-np.inf, -np.inf, -np.pi])
            upper.extend([np.inf, np.inf, np.pi])
        _append_tail_parameter_bounds(p0, lower, upper, method=method, lambda_lower=lambda_lower)
        return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

    term_names = _observable_term_names(observable)
    p0 = []
    lower = []
    upper = []
    for idx, _name in enumerate(term_names):
        p0.extend([1.0 if idx == 0 else 0.1, 0.0])
        lower.extend([-np.inf, -np.pi])
        upper.extend([np.inf, np.pi])

    if order == "NLA":
        for _name in term_names:
            p0.extend([0.1, 0.0])
            lower.extend([-np.inf, -np.pi])
            upper.extend([np.inf, np.pi])

    _append_tail_parameter_bounds(p0, lower, upper, method=method, lambda_lower=lambda_lower)

    return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _n_fit_parameters(method: str, order: str, observable: str) -> int:
    return len(_param_labels(method, order, observable))


def _param_labels(method: str, order: str, observable: str) -> list[str]:
    observable = _canonical_observable(observable)
    if observable == "nucleon_gluon_quasi_pdf":
        labels = ["A"]
        if order.upper() == "NLA":
            labels.append("Ap")
        labels.append("Lambda")
        if method.upper() == "CG":
            labels.append("n")
        return labels
    if observable == "pion_gluon_quasi_pdf":
        labels = ["A2"]
        if order.upper() == "NLA":
            labels.extend(["A2p", "A1", "phi"])
        labels.append("Lambda")
        if method.upper() == "CG":
            labels.append("n")
        return labels

    term_names = _observable_term_names(observable)
    labels = []
    for name in term_names:
        labels.extend([f"A{name}", f"phi{name}"])
    if order.upper() == "NLA":
        for name in term_names:
            labels.extend([f"A{name}p", f"phi{name}p"])
    labels.append("Lambda")
    if method.upper() == "CG":
        labels.append("n")
    return labels


def _tail_factor(z: np.ndarray, params: np.ndarray, *, cursor: int, method: str) -> Any:
    tail = gv.exp(-params[cursor] * z)
    if method.upper() == "CG":
        tail = tail * gv.exp(-params[-1] * np.log(z))
    return tail


def _asymptotic_values(
    z: np.ndarray,
    params: np.ndarray,
    *,
    method: str,
    order: str,
    observable: str,
    phase_scale: float,
    phase_prime_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("asymptotic form requires positive coordinates")

    observable = _canonical_observable(observable)
    if observable == "nucleon_gluon_quasi_pdf":
        re = params[0] * z
        im = np.zeros_like(z, dtype=object)
        cursor = 1
        if order.upper() == "NLA":
            re = re + params[cursor]
            cursor += 1
        tail = _tail_factor(z, params, cursor=cursor, method=method)
        return re * tail, im * tail

    if observable == "pion_gluon_quasi_pdf":
        re = params[0] * z
        im = np.zeros_like(z, dtype=object)
        cursor = 1
        if order.upper() == "NLA":
            re = re + params[cursor]
            cursor += 1
            re = re + 2.0 * params[cursor] * gv.cos(params[cursor + 1] - phase_scale * z)
            cursor += 2
        tail = _tail_factor(z, params, cursor=cursor, method=method)
        return re * tail, im * tail

    term_names = _observable_term_names(observable)
    phase_scales = _term_phase_scales(
        observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
    )
    re = np.zeros_like(z, dtype=object)
    im = np.zeros_like(z, dtype=object)
    cursor = 0
    for phase in phase_scales:
        arg = params[cursor + 1] + phase * z
        re = re + params[cursor] * gv.cos(arg)
        im = im + params[cursor] * gv.sin(arg)
        cursor += 2

    if order.upper() == "NLA":
        for phase in phase_scales:
            arg = params[cursor + 1] + phase * z
            re = re + params[cursor] * gv.cos(arg) / z
            im = im + params[cursor] * gv.sin(arg) / z
            cursor += 2

    tail = _tail_factor(z, params, cursor=cursor, method=method)

    return re * tail, im * tail


def _bounded_to_internal(value: float, lower: float, upper: float) -> float:
    if np.isfinite(lower) and np.isfinite(upper):
        width = upper - lower
        if width <= 0:
            raise ValueError("parameter upper bound must be larger than lower bound")
        clipped = min(max(float(value), lower + 1e-8 * width), upper - 1e-8 * width)
        ratio = (clipped - lower) / (upper - lower)
        return float(np.log(ratio / (1.0 - ratio)))
    if np.isfinite(lower):
        return float(np.log(max(float(value) - lower, 1e-8)))
    if np.isfinite(upper):
        return float(np.log(max(upper - float(value), 1e-8)))
    return float(value)


def _internal_to_bounded(value: Any, lower: float, upper: float) -> Any:
    if np.isfinite(lower) and np.isfinite(upper):
        width = upper - lower
        return lower + width / (1.0 + gv.exp(-value))
    if np.isfinite(lower):
        return lower + gv.exp(value)
    if np.isfinite(upper):
        return upper - gv.exp(value)
    return value


def _internal_p0(params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> gv.BufferDict:
    lower, upper = bounds
    p0 = gv.BufferDict()
    for idx, value in enumerate(np.asarray(params, dtype=float)):
        p0[f"u{idx}"] = _bounded_to_internal(float(value), float(lower[idx]), float(upper[idx]))
    return p0


def _physical_params(p: gv.BufferDict, bounds: tuple[np.ndarray, np.ndarray]) -> list[Any]:
    lower, upper = bounds
    return [
        _internal_to_bounded(p[f"u{idx}"], float(lower[idx]), float(upper[idx]))
        for idx in range(len(lower))
    ]


def _scaled_internal_prior(pmean: gv.BufferDict, psdev: gv.BufferDict, scale: float) -> gv.BufferDict:
    prior = gv.BufferDict()
    width_scale = max(float(scale), 0.0)
    for key in pmean:
        width = float(psdev[key]) * width_scale
        prior[key] = gv.gvar(float(pmean[key]), max(width, 1e-8))
    return prior


def _fit_one_sample(
    z_fit: np.ndarray,
    *,
    y_data: np.ndarray,
    method: str,
    order: str,
    observable: str,
    part: str,
    phase_scale: float,
    phase_prime_scale: float | None = None,
    p0: np.ndarray | None = None,
    prior: gv.BufferDict | None = None,
    Lambda0: float = 0.1,
) -> tuple[np.ndarray, gv.BufferDict | None, gv.BufferDict | None, bool, float, int, float]:
    default_p0, bounds = _param_template(method, order, observable, Lambda0=Lambda0)
    start = default_p0 if p0 is None else np.asarray(p0, dtype=float)

    def fcn(z: np.ndarray, p: gv.BufferDict) -> np.ndarray:
        params = _physical_params(p, bounds)
        pred_re, pred_im = _asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            observable=observable,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
        )
        return _select_fit_prediction(pred_re, pred_im, part)

    dof = max(1, _n_fit_channels(part) * len(z_fit) - len(default_p0))
    try:
        fit_args = {
            "data": (z_fit, y_data),
            "fcn": fcn,
            "p0": _internal_p0(start, bounds),
            "maxit": 2000,
            "svdcut": 1e-12,
            "fitter": "scipy_least_squares",
        }
        if prior is not None:
            fit_args["prior"] = prior
        fit = lsqfit.nonlinear_fit(**fit_args)
        physical = _physical_params(fit.pmean, bounds)
        params = np.asarray([float(item) for item in physical], dtype=float)
    except (FloatingPointError, RuntimeError, ValueError, OverflowError, AssertionError):
        return default_p0, None, None, False, float("inf"), dof, 0.0

    return params, fit.pmean, fit.psdev, bool(np.isfinite(fit.chi2)), float(fit.chi2), int(fit.dof), float(fit.Q)


def fit_tail_quality_for_mean(
    coord: Sequence[float],
    re_samples,
    im_samples,
    *,
    zmin: float,
    zmax: float,
    method: str,
    order: str,
    observable: str,
    coord_unit: str,
    pz_gev: float | None = None,
    pz_prime_gev: float | None = None,
    a_fm: float | None = None,
    resample_mode: str = "bootstrap",
    Lambda0: float = 0.1,
    min_fit_points: int | None = None,
    fit_error_mode: str = "diagonal",
    part: str = "both",
) -> dict[str, Any]:
    """Fit the mean matrix element on one range and return quality diagnostics."""
    coord_arr = np.asarray(coord, dtype=float)
    re_mat = np.asarray(re_samples, dtype=float)
    im_mat = np.asarray(im_samples, dtype=float)
    if re_mat.ndim != 2 or im_mat.ndim != 2 or re_mat.shape != im_mat.shape:
        raise ValueError("re_samples and im_samples must be matching (n_sample,n_z) arrays")
    if re_mat.shape[1] != len(coord_arr):
        raise ValueError("sample arrays must have one value per coordinate point")

    observable = _canonical_observable(observable)
    fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    fit_coord = coord_arr * fit_scale
    ft_scale_over_fit_scale = ft_scale / fit_scale
    phase_scale, phase_prime_scale = _phase_scales(
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        ft_scale_over_fit_scale=ft_scale_over_fit_scale,
    )
    zmin_fit = _convert_scheme_value(zmin, fit_scale)
    zmax_fit = _convert_scheme_value(zmax, fit_scale)
    fit_mask = (fit_coord >= zmin_fit) & (fit_coord <= zmax_fit) & (fit_coord > 0)
    n_points = int(np.count_nonzero(fit_mask))
    n_params = _n_fit_parameters(method, order, observable)
    required_points = max(n_params, int(min_fit_points or 0), 2)
    if n_points < required_points:
        dof = max(1, _n_fit_channels(part) * n_points - n_params)
        return {
            "ok": False,
            "chi2": float("inf"),
            "dof": int(dof),
            "chi2_dof": float("inf"),
            "q_value": 0.0,
            "n_points": n_points,
            "min_fit_points": required_points,
        }

    z_fit = fit_coord[fit_mask]
    mean_re = np.mean(re_mat[:, fit_mask], axis=0)
    mean_im = np.mean(im_mat[:, fit_mask], axis=0)
    sigma_re = _sample_sdev(re_mat[:, fit_mask], resample_mode=resample_mode)
    sigma_im = _sample_sdev(im_mat[:, fit_mask], resample_mode=resample_mode)
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)
    y_data = _fit_y_data(
        mean_re,
        mean_im,
        fit_error_mode=fit_error_mode,
        resample_mode=resample_mode,
        part=part,
        re_fit_samples=re_mat[:, fit_mask],
        im_fit_samples=im_mat[:, fit_mask],
        sigma_re=sigma_re,
        sigma_im=sigma_im,
    )

    _params, _pmean, _psdev, ok, chi2, dof, q_value = _fit_one_sample(
        z_fit,
        y_data=y_data,
        method=method,
        order=order,
        observable=observable,
        part=part,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        Lambda0=Lambda0,
    )
    return {
        "ok": bool(ok),
        "chi2": float(chi2),
        "dof": int(dof),
        "chi2_dof": float(chi2 / max(dof, 1)),
        "q_value": float(q_value),
        "n_points": n_points,
        "min_fit_points": required_points,
    }


def _linear_fit_weight(z: np.ndarray, blend_start: float, trusted_stop: float) -> np.ndarray:
    weights = np.zeros_like(z, dtype=float)
    if trusted_stop <= blend_start:
        weights[z > trusted_stop] = 1.0
        return weights
    mask = (z >= blend_start) & (z <= trusted_stop)
    weights[mask] = (z[mask] - blend_start) / (trusted_stop - blend_start)
    weights[z > trusted_stop] = 1.0
    return weights


def _interp_samples(x: np.ndarray, y_samples: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    out = np.empty((y_samples.shape[0], len(x_new)), dtype=float)
    for i, row in enumerate(y_samples):
        out[i] = np.interp(x_new, x, row)
    return out


def _scheme_ranges(scheme: dict[str, Any], coord: np.ndarray) -> tuple[float, float, float]:
    zmin = float(scheme.get("zmin", coord[1]))
    zmax = float(scheme.get("zmax", coord[-1]))
    z_ext_max = float(scheme.get("z_ext_max", zmax))
    return zmin, zmax, z_ext_max


def _run_one_scheme(
    *,
    coord: np.ndarray,
    fit_coord: np.ndarray,
    ft_scale_over_fit_scale: float,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    k_grid: np.ndarray,
    scheme: dict[str, Any],
    method: str,
    order: str,
    observable: str,
    fit_scale: float,
    im_flip_for_ft: bool,
    phase_scale: float,
    phase_prime_scale: float | None,
    resample_mode: str,
    Lambda0: float,
    min_fit_points: int | None,
    posterior_prior_error_scale: float,
    fit_error_mode: str,
    part: str,
) -> dict[str, Any]:
    zmin, zmax, z_ext_max = _scheme_ranges(scheme, coord)
    zmin_fit = _convert_scheme_value(zmin, fit_scale)
    zmax_fit = _convert_scheme_value(zmax, fit_scale)
    z_ext_fit_max = _convert_scheme_value(z_ext_max, fit_scale)

    if zmin_fit <= 0:
        raise ValueError("zmin must be positive; asymptotic forms are singular at zero")
    if zmax_fit <= zmin_fit:
        raise ValueError("zmax must be larger than zmin")
    if z_ext_fit_max < zmax_fit:
        raise ValueError("z_ext_max must be >= zmax")

    fit_mask = (fit_coord >= zmin_fit) & (fit_coord <= zmax_fit) & (fit_coord > 0)
    n_params = _n_fit_parameters(method, order, observable)
    required_points = max(n_params, int(min_fit_points or 0), 2)
    if np.count_nonzero(fit_mask) < required_points:
        raise ValueError("fit range has too few points for the selected asymptotic form")

    z_fit = fit_coord[fit_mask]
    mean_re = np.mean(re_samples, axis=0)[fit_mask]
    mean_im = np.mean(im_samples, axis=0)[fit_mask]
    sigma_re = _sample_sdev(re_samples[:, fit_mask], resample_mode=resample_mode)
    sigma_im = _sample_sdev(im_samples[:, fit_mask], resample_mode=resample_mode)
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)
    mean_y_data = _fit_y_data(
        mean_re,
        mean_im,
        fit_error_mode=fit_error_mode,
        resample_mode=resample_mode,
        part=part,
        re_fit_samples=re_samples[:, fit_mask],
        im_fit_samples=im_samples[:, fit_mask],
        sigma_re=sigma_re,
        sigma_im=sigma_im,
    )
    mean_params, mean_pmean, mean_psdev, mean_ok, mean_chi2, mean_dof, mean_q = _fit_one_sample(
        z_fit,
        y_data=mean_y_data,
        method=method,
        order=order,
        observable=observable,
        part=part,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        Lambda0=Lambda0,
    )
    sample_prior = None
    if mean_ok and mean_pmean is not None and mean_psdev is not None:
        sample_prior = _scaled_internal_prior(mean_pmean, mean_psdev, posterior_prior_error_scale)

    dz = _uniform_step(fit_coord)
    z_ext = np.arange(0.0, z_ext_fit_max + 0.5 * dz, dz)
    lambda_ext = z_ext * ft_scale_over_fit_scale

    data_re = _interp_samples(fit_coord, re_samples, z_ext)
    data_im = _interp_samples(fit_coord, im_samples, z_ext)

    trusted_stop = min(zmax_fit, fit_coord[-1])
    smooth = str(scheme.get("smooth", "linear")).lower()
    if smooth == "none":
        fit_weight = np.zeros_like(z_ext)
        fit_weight[z_ext > trusted_stop] = 1.0
    elif smooth == "linear":
        fit_weight = _linear_fit_weight(z_ext, zmin_fit, trusted_stop)
    else:
        raise ValueError("smooth must be 'linear' or 'none'")
    fit_weight[z_ext <= 0] = 0.0

    n_samples = re_samples.shape[0]
    ext_re = np.empty((n_samples, len(z_ext)), dtype=float)
    ext_im = np.empty_like(ext_re)
    fit_re_samples = np.empty_like(ext_re)
    fit_im_samples = np.empty_like(ext_re)
    ft_re = np.empty((n_samples, len(k_grid)), dtype=float)
    ft_im = np.empty_like(ft_re)
    fit_params = np.empty((n_samples, n_params), dtype=float)
    fit_chi2 = np.empty(n_samples, dtype=float)
    fit_dof = np.empty(n_samples, dtype=int)
    fit_q = np.empty(n_samples, dtype=float)
    failures = 0

    positive = z_ext > 0
    for sample in range(n_samples):
        sample_y_data = _fit_y_data(
            re_samples[sample, fit_mask],
            im_samples[sample, fit_mask],
            fit_error_mode=fit_error_mode,
            resample_mode=resample_mode,
            part=part,
            re_fit_samples=re_samples[:, fit_mask],
            im_fit_samples=im_samples[:, fit_mask],
            sigma_re=sigma_re,
            sigma_im=sigma_im,
        )
        params, _sample_pmean, _sample_psdev, ok, chi2, dof, q_value = _fit_one_sample(
            z_fit,
            y_data=sample_y_data,
            method=method,
            order=order,
            observable=observable,
            part=part,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            p0=mean_params,
            prior=sample_prior,
            Lambda0=Lambda0,
        )
        if not ok:
            failures += 1
            params = mean_params
            chi2 = mean_chi2
            dof = mean_dof
            q_value = mean_q
        fit_params[sample] = params
        fit_chi2[sample] = chi2
        fit_dof[sample] = dof
        fit_q[sample] = q_value

        fit_re = np.zeros_like(z_ext)
        fit_im = np.zeros_like(z_ext)
        fit_re[positive], fit_im[positive] = _asymptotic_values(
            z_ext[positive],
            params,
            method=method,
            order=order,
            observable=observable,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
        )

        fit_re, fit_im = _zero_inactive_channel(fit_re, fit_im, part)
        fit_re_samples[sample] = fit_re
        fit_im_samples[sample] = fit_im
        ext_re_sample = fit_weight * fit_re + (1.0 - fit_weight) * data_re[sample]
        ext_im_sample = fit_weight * fit_im + (1.0 - fit_weight) * data_im[sample]
        ext_re[sample], ext_im[sample] = _zero_inactive_channel(ext_re_sample, ext_im_sample, part)

        lam_full, re_full, im_full = complete_z_negative(
            lambda_ext,
            ext_re[sample],
            ext_im[sample],
            im_flip_for_ft=im_flip_for_ft,
        )
        ft_re[sample], ft_im[sample] = sum_ft_re_im(lam_full, re_full, im_full, k_grid)

    return {
        "label": str(scheme.get("label", f"{method}_{order}_{zmin}_{zmax}")),
        "z_ext": z_ext,
        "lambda_ext": lambda_ext,
        "fit_weight": fit_weight,
        "fit_re_samples": fit_re_samples,
        "fit_im_samples": fit_im_samples,
        "fit_params": fit_params,
        "fit_param_labels": _param_labels(method, order, observable),
        "fit_chi2": fit_chi2,
        "fit_dof": fit_dof,
        "fit_q": fit_q,
        "mean_fit_params": mean_params,
        "mean_fit_chi2": mean_chi2,
        "mean_fit_dof": mean_dof,
        "mean_fit_q": mean_q,
        "extended_re_samples": ext_re,
        "extended_im_samples": ext_im,
        "ft_re_samples": ft_re,
        "ft_im_samples": ft_im,
        "fit_failures": failures,
        "fit_range": (zmin, zmax),
        "z_ext_max": z_ext_max,
        "smooth": smooth,
    }


def run_fourier_workflow(
    coord: Sequence[float],
    re_samples,
    im_samples,
    k_grid: Sequence[float],
    *,
    schemes: list[dict[str, Any]] | None = None,
    method: str = "GI",
    order: str = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_prime_gev: float | None = None,
    a_fm: float | None = None,
    im_flip_for_ft: bool = False,
    resample_mode: str = "bootstrap",
    Lambda0: float = 0.1,
    min_fit_points: int | None = None,
    posterior_prior_error_scale: float = 3.0,
    fit_error_mode: str = "diagonal",
    part: str = "both",
) -> dict[str, Any]:
    """Run asymptotic extension and Fourier transform for resampled data.

    ``schemes`` values are in the same unit as ``coord``. The output keeps
    sample information as arrays shaped ``(scheme, sample, k)``.
    """
    coord_arr = np.asarray(coord, dtype=float)
    resample_mode = _normalise_resample_mode(resample_mode)
    fit_error_mode = _normalise_fit_error_mode(fit_error_mode)
    part = _normalise_part(part)
    _uniform_step(coord_arr)
    if not np.isclose(coord_arr[0], 0.0):
        raise ValueError("coordinate grid must start at zero")

    re_mat = _as_sample_matrix("re_samples", re_samples)
    im_mat = _as_sample_matrix("im_samples", im_samples)
    if re_mat.shape != im_mat.shape:
        raise ValueError("re_samples and im_samples must have the same shape")
    if re_mat.shape[1] != len(coord_arr):
        raise ValueError("sample arrays must have one value per coordinate point")

    observable = _canonical_observable(observable)
    fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    fit_coord = coord_arr * fit_scale
    ft_scale_over_fit_scale = ft_scale / fit_scale
    phase_scale, phase_prime_scale = _phase_scales(
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        ft_scale_over_fit_scale=ft_scale_over_fit_scale,
    )
    k_arr = np.asarray(k_grid, dtype=float)
    if k_arr.ndim != 1:
        raise ValueError("k_grid must be one-dimensional")

    if schemes is None:
        schemes = [
            {
                "label": "default",
                "zmin": coord_arr[1],
                "zmax": coord_arr[-1],
                "z_ext_max": coord_arr[-1] + 8.0 / ft_scale,
            }
        ]

    scheme_results = [
        _run_one_scheme(
            coord=coord_arr,
            fit_coord=fit_coord,
            ft_scale_over_fit_scale=ft_scale_over_fit_scale,
            re_samples=re_mat,
            im_samples=im_mat,
            k_grid=k_arr,
            scheme=scheme,
            method=method,
            order=order,
            observable=observable,
            fit_scale=fit_scale,
            im_flip_for_ft=im_flip_for_ft,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            resample_mode=resample_mode,
            Lambda0=Lambda0,
            min_fit_points=min_fit_points,
            posterior_prior_error_scale=posterior_prior_error_scale,
            fit_error_mode=fit_error_mode,
            part=part,
        )
        for scheme in schemes
    ]

    ft_re = np.asarray([item["ft_re_samples"] for item in scheme_results])
    ft_im = np.asarray([item["ft_im_samples"] for item in scheme_results])
    re_mean_by_scheme = np.mean(ft_re, axis=1)
    im_mean_by_scheme = np.mean(ft_im, axis=1)
    re_stat_by_scheme = np.asarray(
        [_sample_sdev(item["ft_re_samples"], resample_mode=resample_mode) for item in scheme_results]
    )
    im_stat_by_scheme = np.asarray(
        [_sample_sdev(item["ft_im_samples"], resample_mode=resample_mode) for item in scheme_results]
    )

    re_mean = np.mean(re_mean_by_scheme, axis=0)
    im_mean = np.mean(im_mean_by_scheme, axis=0)
    re_stat = np.sqrt(np.mean(re_stat_by_scheme**2, axis=0))
    im_stat = np.sqrt(np.mean(im_stat_by_scheme**2, axis=0))
    re_sys = np.std(re_mean_by_scheme, axis=0, ddof=0)
    im_sys = np.std(im_mean_by_scheme, axis=0, ddof=0)

    return {
        "k_grid": k_arr,
        "ft_re_samples": ft_re,
        "ft_im_samples": ft_im,
        "ft_re_mean": re_mean,
        "ft_im_mean": im_mean,
        "ft_re_stat_sdev": re_stat,
        "ft_im_stat_sdev": im_stat,
        "ft_re_sys_sdev": re_sys,
        "ft_im_sys_sdev": im_sys,
        "scheme_results": scheme_results,
        "scheme_labels": [item["label"] for item in scheme_results],
        "fit_failures": [item["fit_failures"] for item in scheme_results],
        "method": method.upper(),
        "order": order.upper(),
        "observable": observable,
        "coord_unit": coord_unit,
        "fit_coord_unit": "lambda" if coord_unit.lower() == "lambda" else "gev_inv",
        "resample_mode": resample_mode,
        "Lambda0": float(Lambda0),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "fit_error_mode": fit_error_mode,
        "part": part,
    }
