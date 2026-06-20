"""Fourier-transform stage tools.

Purpose:
- load coordinate-space real/imaginary samples
- call the local sample-preserving extrapolation and Fourier workflow
- keep large arrays in the stage store and write `.nc` EnsembleData artifacts

Expected inputs:
- an `.nc` EnsembleData file, an `.npz` file with `coord`, `re_samples`, and `im_samples`, or
  an HDF5 file with group datasets such as `Pz=4/z_ary`, `Pz=4/Re`, `Pz=4/Im`
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- Fourier samples stored in the per-stage store
- summary arrays written under `artifacts/`
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.plotting import plot_fourier_artifact, plot_fourier_extension_quality
from lamet_agent.core.resampling import bs_ls_avg, jk_ls_avg
from lamet_agent.stages.fourier.reporting import write_fourier_report

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


@dataclass(frozen=True)
class _TailParameter:
    label: str
    p0: float
    lower: float = -np.inf
    upper: float = np.inf


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


def _minimum_fit_points_for_parameters(n_params: int, part: str) -> int:
    """Minimum coordinate points needed to provide at least n_params data values."""
    channel_count = max(_n_fit_channels(part), 1)
    from_parameters = int(np.ceil(float(n_params) / float(channel_count)))
    return max(from_parameters, 2)


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


QUARK_LIKE_TERMS = {
    "pion_quark_quasi_pdf": ("2", "1", "3"),
    "nucleon_quark_unpolarized_quasi_pdf": ("2",),
    "nucleon_quark_transversity_quasi_pdf": ("2",),
    "meson_quasi_da": ("1", "2"),
    "pion_quark_quasi_gpd": ("1", "3", "2", "t2"),
    "nucleon_quark_quasi_gpd": ("2", "t2"),
}


def _quark_like_term_names(observable: str) -> tuple[str, ...]:
    observable = _canonical_observable(observable)
    if observable not in QUARK_LIKE_TERMS:
        raise ValueError(f"unsupported quark-like observable {observable!r}")
    return QUARK_LIKE_TERMS[observable]


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


def _quark_like_phase_scales(
    observable: str,
    *,
    phase_scale: float,
    phase_prime_scale: float | None,
) -> tuple[float, ...]:
    observable = _canonical_observable(observable)
    pzp = phase_scale if phase_prime_scale is None else phase_prime_scale
    if observable == "pion_quark_quasi_pdf":
        return (0.0, -phase_scale, phase_scale)
    if observable in {"nucleon_quark_unpolarized_quasi_pdf", "nucleon_quark_transversity_quasi_pdf"}:
        return (0.0,)
    if observable == "meson_quasi_da":
        return (-phase_scale, 0.0)
    if observable == "pion_quark_quasi_gpd":
        return (-phase_scale, pzp, 0.0, -(phase_scale - pzp))
    if observable == "nucleon_quark_quasi_gpd":
        return (0.0, -(phase_scale - pzp))
    raise ValueError(f"unsupported observable {observable!r}")


def _with_method_tail_parameters(
    parameters: list[_TailParameter],
    *,
    method: str,
    lambda_lower: float,
) -> list[_TailParameter]:
    parameters = [
        *parameters,
        _TailParameter(
            "Lambda",
            max(0.5, lambda_lower + 0.05),
            lambda_lower,
            max(3.0, lambda_lower + 1.0),
        ),
    ]
    if method.upper() == "CG":
        parameters.append(_TailParameter("n", 0.5, -2.0, 4.0))
    return parameters


def _quark_like_parameters(order: str, observable: str) -> list[_TailParameter]:
    term_names = _quark_like_term_names(observable)
    parameters = []
    for idx, name in enumerate(term_names):
        parameters.extend(
            [
                _TailParameter(f"A{name}", 1.0 if idx == 0 else 0.1),
                _TailParameter(f"phi{name}", 0.0, -np.pi, np.pi),
            ]
        )
    if order.upper() == "NLA":
        for name in term_names:
            parameters.extend(
                [
                    _TailParameter(f"A{name}p", 0.1),
                    _TailParameter(f"phi{name}p", 0.0, -np.pi, np.pi),
                ]
            )
    return parameters


def _nucleon_gluon_parameters(order: str) -> list[_TailParameter]:
    parameters = [_TailParameter("A", 1.0)]
    if order.upper() == "NLA":
        parameters.append(_TailParameter("Ap", 0.1))
    return parameters


def _pion_gluon_parameters(order: str) -> list[_TailParameter]:
    parameters = [_TailParameter("A2", 1.0)]
    if order.upper() == "NLA":
        parameters.extend(
            [
                _TailParameter("A2p", 0.1),
                _TailParameter("A1", 0.1),
                _TailParameter("phi", 0.0, -np.pi, np.pi),
            ]
        )
    return parameters


def _observable_parameters(order: str, observable: str) -> list[_TailParameter]:
    observable = _canonical_observable(observable)
    if observable == "nucleon_gluon_quasi_pdf":
        return _nucleon_gluon_parameters(order)
    if observable == "pion_gluon_quasi_pdf":
        return _pion_gluon_parameters(order)
    return _quark_like_parameters(order, observable)


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
    parameters = _with_method_tail_parameters(
        _observable_parameters(order, observable),
        method=method,
        lambda_lower=lambda_lower,
    )
    p0 = [item.p0 for item in parameters]
    lower = [item.lower for item in parameters]
    upper = [item.upper for item in parameters]
    return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _param_labels(method: str, order: str, observable: str) -> list[str]:
    method = method.upper()
    if method not in {"GI", "CG"}:
        raise ValueError("method must be 'GI' or 'CG'")
    order = order.upper()
    if order not in {"LA", "NLA"}:
        raise ValueError("order must be 'LA' or 'NLA'")
    observable = _canonical_observable(observable)
    parameters = _with_method_tail_parameters(
        _observable_parameters(order, observable),
        method=method,
        lambda_lower=0.1,
    )
    return [item.label for item in parameters]


def _decay_tail(z: np.ndarray, params: Sequence[Any], *, lambda_index: int, method: str) -> Any:
    tail = gv.exp(-params[lambda_index] * z)
    if method.upper() == "CG":
        tail = tail * gv.exp(-params[lambda_index + 1] * np.log(z))
    return tail


def _quark_like_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    observable: str,
    phase_scale: float,
    phase_prime_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quark-like LA/NLA forms: oscillatory terms times a GI or CG decay tail."""
    phase_scales = _quark_like_phase_scales(
        observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
    )
    re = np.zeros_like(z, dtype=object)
    im = np.zeros_like(z, dtype=object)
    cursor = 0

    # LA: sum_j A_j exp(i(phi_j + omega_j z)) exp(-Lambda z)
    for phase in phase_scales:
        arg = params[cursor + 1] + phase * z
        re = re + params[cursor] * gv.cos(arg)
        im = im + params[cursor] * gv.sin(arg)
        cursor += 2

    # NLA: add sum_j A'_j/z exp(i(phi'_j + omega_j z)) before the common tail.
    if order.upper() == "NLA":
        for phase in phase_scales:
            arg = params[cursor + 1] + phase * z
            re = re + params[cursor] * gv.cos(arg) / z
            im = im + params[cursor] * gv.sin(arg) / z
            cursor += 2

    tail = _decay_tail(z, params, lambda_index=cursor, method=method)
    return re * tail, im * tail


def _nucleon_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F nucleon gluon form: (A z [+ A']) exp(-Lambda z) with zero Im."""
    # LA:  A z exp(-Lambda z)
    # NLA: (A z + A') exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1]
        lambda_index = 2
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method)
    im = np.zeros_like(z, dtype=object)
    return re * tail, im * tail


def _pion_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    phase_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F pion gluon form: A2 z plus optional constant and cosine term."""
    # LA:  A2 z exp(-Lambda z)
    # NLA: (A2 z + A2' + 2 A1 cos(phi - Pz z)) exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1] + 2.0 * params[2] * gv.cos(params[3] - phase_scale * z)
        lambda_index = 4
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method)
    im = np.zeros_like(z, dtype=object)
    return re * tail, im * tail


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
        return _nucleon_gluon_asymptotic_values(
            z,
            params,
            method=method,
            order=order,
        )

    if observable == "pion_gluon_quasi_pdf":
        return _pion_gluon_asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            phase_scale=phase_scale,
        )

    return _quark_like_asymptotic_values(
        z,
        params,
        method=method,
        order=order,
        observable=observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
    )


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
    n_params = len(_param_labels(method, order, observable))
    required_points = _minimum_fit_points_for_parameters(n_params, part)
    if n_points < required_points:
        dof = max(1, _n_fit_channels(part) * n_points - n_params)
        return {
            "ok": False,
            "chi2": float("inf"),
            "dof": int(dof),
            "chi2_dof": float("inf"),
            "q_value": 0.0,
            "n_points": n_points,
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
    n_params = len(_param_labels(method, order, observable))
    required_points = _minimum_fit_points_for_parameters(n_params, part)
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



def _samples_axis_zero(values: np.ndarray, coord: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix-element samples must be a 2D array")
    n_coord = len(coord)
    if arr.shape[1] == n_coord:
        return arr
    if arr.shape[0] == n_coord:
        return np.moveaxis(arr, 1, 0)
    raise ValueError("matrix-element samples must have one axis matching the coordinate length")


def matrix_element_to_ensemble_data(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    resample: str = "bootstrap",
    attrs: dict[str, Any] | None = None,
    name: str = "renormalized_matrix_element",
) -> EnsembleData:
    """Build a complex EnsembleData matrix element with dimension z."""
    coord_arr = np.asarray(coord, dtype=float)
    re_axis0 = _samples_axis_zero(np.asarray(re_samples, dtype=float), coord_arr)
    im_axis0 = _samples_axis_zero(np.asarray(im_samples, dtype=float), coord_arr)
    if re_axis0.shape != im_axis0.shape:
        raise ValueError("real and imaginary matrix-element samples must have matching shapes")
    values = [re_axis0[idx] + 1j * im_axis0[idx] for idx in range(re_axis0.shape[0])]
    return EnsembleData(
        ensemble=None,
        resample=_normalise_resample_mode(resample),
        values=values,
        dims=("z",),
        coords={"z": coord_arr.tolist()},
        attrs={key: str(value) for key, value in (attrs or {}).items() if value is not None},
        name=name,
    )


def ensemble_data_to_legacy_arrays(data: EnsembleData) -> dict[str, np.ndarray]:
    """Convert EnsembleData(z) back to legacy coord/re_samples/im_samples arrays."""
    if not isinstance(data, EnsembleData):
        raise TypeError("matrix_element_data must be an EnsembleData")
    if data.dims != ["z"]:
        raise ValueError("Fourier matrix_element_data must have physical dimension ['z']")
    values = np.asarray(data.values)
    if values.ndim != 2:
        raise ValueError("Fourier matrix_element_data values must be shaped (resample,z)")
    return {
        "coord": np.asarray(data.coords["z"], dtype=float),
        "re_samples": np.asarray(np.real(values), dtype=float),
        "im_samples": np.asarray(np.imag(values), dtype=float),
        "resample_mode": data.resample,
    }


def fourier_result_to_ensemble_data(result: dict[str, Any]) -> EnsembleData:
    """Build a complex EnsembleData(x) from Fourier workflow samples."""
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    weights = np.asarray(result.get("scheme_weights", []), dtype=float)
    if weights.shape == (ft_re.shape[0],):
        re_samples = np.sum(weights[:, None, None] * ft_re, axis=0)
        im_samples = np.sum(weights[:, None, None] * ft_im, axis=0)
    else:
        re_samples = np.mean(ft_re, axis=0)
        im_samples = np.mean(ft_im, axis=0)
    values = [re_samples[idx] + 1j * im_samples[idx] for idx in range(re_samples.shape[0])]
    attrs = {
        "method": str(result.get("method", "")),
        "order": str(result.get("order", "")),
        "observable": str(result.get("observable", "")),
        "coord_unit": str(result.get("coord_unit", "")),
        "fit_coord_unit": str(result.get("fit_coord_unit", "")),
        "part": str(result.get("part", "both")),
        "resample_mode": str(result.get("resample_mode", "")),
    }
    for key in ("pz_gev", "pz_prime_gev", "a_fm"):
        value = result.get(key)
        if value is not None:
            attrs[key] = str(value)
    for key in (
        "ft_re_mean",
        "ft_im_mean",
        "ft_re_stat_sdev",
        "ft_im_stat_sdev",
        "ft_re_sys_sdev",
        "ft_im_sys_sdev",
        "scheme_labels",
        "fit_failures",
        "scheme_weights",
        "scheme_fit_chi2_dof",
        "scheme_roughness",
        "scheme_scores",
        "best_scheme_index",
        "output_scale",
    ):
        if key in result:
            attrs[key] = json.dumps(np.asarray(result[key]).tolist())
    return EnsembleData(
        ensemble=result.get("ensemble"),
        resample=_normalise_resample_mode(str(result.get("resample_mode", "bootstrap"))),
        values=values,
        dims=("x",),
        coords={"x": np.asarray(result["k_grid"], dtype=float).tolist()},
        attrs=attrs,
        name="fourier_transform",
    )


def load_renormalized_matrix_element_samples(
    store: dict[str, Any],
    *,
    path: str,
    input_format: str | None = None,
    h5_group: str | None = None,
    coord_key: str = "coord",
    re_key: str = "re_samples",
    im_key: str = "im_samples",
    resample_mode: str = "bootstrap",
) -> dict[str, Any]:
    """Load renormalized coordinate-space matrix-element samples from NPZ or HDF5."""
    matrix_element_data, fmt, group_name = _load_matrix_element_data(
        path=path,
        input_format=input_format,
        h5_group=h5_group,
        coord_key=coord_key,
        re_key=re_key,
        im_key=im_key,
        resample_mode=resample_mode,
    )
    legacy = ensemble_data_to_legacy_arrays(matrix_element_data)
    out = "matrix_element"
    store["matrix_element_data"] = matrix_element_data
    store[out] = {
        **legacy,
        "path": str(path),
        "input_format": fmt,
        "resample_mode": matrix_element_data.resample,
    }
    if group_name is not None:
        store[out]["h5_group"] = group_name
    return {
        "out": out,
        "data": "matrix_element_data",
        "input_format": fmt,
        "h5_group": group_name,
        "resample_mode": matrix_element_data.resample,
        "n_coord": int(len(legacy["coord"])),
        "n_sample": int(legacy["re_samples"].shape[0]),
        "re_shape": list(legacy["re_samples"].shape),
        "im_shape": list(legacy["im_samples"].shape),
    }


def _load_matrix_element_data(
    *,
    path: str,
    input_format: str | None,
    h5_group: str | None,
    coord_key: str,
    re_key: str,
    im_key: str,
    resample_mode: str,
) -> tuple[EnsembleData, str, str | None]:
    """Load NPZ/HDF5 matrix-element samples and normalize them to EnsembleData."""
    if input_format is not None:
        fmt = input_format.lower()
    else:
        suffix = Path(path).suffix.lower()
        fmt = "h5" if suffix in {".h5", ".hdf5"} else suffix.lstrip(".")
    if fmt == "hdf5":
        fmt = "h5"
    if fmt == "netcdf":
        fmt = "nc"
    resample = _normalise_resample_mode(resample_mode)

    if fmt == "nc":
        data = EnsembleData.from_netcdf(path)
        return data, fmt, None

    if fmt == "npz":
        try:
            data, _extras = EnsembleData.load_npz(path)
        except ValueError:
            with np.load(path, allow_pickle=False) as npz:
                data = matrix_element_to_ensemble_data(
                    coord=np.asarray(npz[coord_key], dtype=float),
                    re_samples=np.asarray(npz[re_key], dtype=float),
                    im_samples=np.asarray(npz[im_key], dtype=float),
                    resample=resample,
                    attrs={"input_format": fmt, "path": str(path)},
                )
        if data.dims != ["z"]:
            raise ValueError("Fourier NPZ EnsembleData input must have physical dimension ['z']")
        values = np.asarray(data.values)
        if values.ndim != 2:
            raise ValueError("Fourier NPZ EnsembleData input must be shaped (resample,z)")
        return data, fmt, None

    if fmt == "h5":
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Reading HDF5 Fourier inputs requires installing lamet-agent with the analysis extra"
            ) from exc

        use_coord_key = "z_ary" if coord_key == "coord" else coord_key
        use_re_key = "Re" if re_key == "re_samples" else re_key
        use_im_key = "Im" if im_key == "im_samples" else im_key
        with h5py.File(path, "r") as h5f:
            group_names = [name for name, item in h5f.items() if isinstance(item, h5py.Group)]
            group_name = h5_group or _infer_h5_group(path, group_names)
            if group_name not in h5f:
                raise ValueError(f"HDF5 group {group_name!r} not found; available groups: {group_names}")
            group = h5f[group_name]
            data = matrix_element_to_ensemble_data(
                coord=np.asarray(group[use_coord_key], dtype=float),
                re_samples=np.asarray(group[use_re_key], dtype=float),
                im_samples=np.asarray(group[use_im_key], dtype=float),
                resample=resample,
                attrs={"input_format": fmt, "h5_group": group_name, "path": str(path)},
            )
        return data, fmt, group_name

    raise ValueError("input_format must be 'nc', 'netcdf', 'npz', 'h5', or 'hdf5'")


def _infer_h5_group(path: str, group_names: list[str]) -> str:
    match = re.search(r"(?:^|_)pz([+-]?\d+)(?:\.|_|$)", Path(path).name, flags=re.IGNORECASE)
    if match:
        group = f"Pz={match.group(1)}"
        if group in group_names:
            return group

    if len(group_names) == 1:
        return group_names[0]
    raise ValueError("h5_group is required when the HDF5 file has multiple groups and no pz can be inferred")


def _artifact_path(raw: str | None, *, default_name: str, artifacts_dir: str | Path | None = None) -> Path:
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        path = Path(raw).expanduser()
        if path.is_absolute():
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        if path.parent != Path("."):
            if artifacts_dir is not None:
                return out_dir / path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return out_dir / path
    return out_dir / default_name


def _png_companion_path(path: Path) -> Path:
    """Return a PNG companion path for Markdown image embedding."""
    return path.with_suffix(".png")


def _save_fourier_fit_info_netcdf(path: Path, result: dict[str, Any]) -> None:
    schemes = result["scheme_results"]
    fit_params = np.asarray([item["fit_params"] for item in schemes], dtype=float)
    fit_chi2 = np.asarray([item["fit_chi2"] for item in schemes], dtype=float)
    fit_dof = np.asarray([item["fit_dof"] for item in schemes], dtype=int)
    fit_q = np.asarray([item["fit_q"] for item in schemes], dtype=float)
    mean_fit_params = np.asarray([item["mean_fit_params"] for item in schemes], dtype=float)
    mean_fit_chi2 = np.asarray([item["mean_fit_chi2"] for item in schemes], dtype=float)
    mean_fit_dof = np.asarray([item["mean_fit_dof"] for item in schemes], dtype=int)
    mean_fit_q = np.asarray([item["mean_fit_q"] for item in schemes], dtype=float)

    resample_mode = _normalise_resample_mode(str(result.get("resample_mode", "bootstrap")))
    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    if fit_params.shape[1] < 2:
        fit_param_sdev = np.zeros((fit_params.shape[0], fit_params.shape[2]), dtype=float)
    else:
        fit_param_sdev = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in fit_params])

    scheme_labels = np.asarray(result["scheme_labels"])
    fit_param_labels = np.asarray(schemes[0]["fit_param_labels"])
    param_samples = np.moveaxis(fit_params, 1, 0)
    fit_info_data = EnsembleData(
        ensemble=result.get("ensemble"),
        resample=resample_mode,
        values=[param_samples[idx] for idx in range(param_samples.shape[0])],
        dims=("scheme", "parameter"),
        coords={"scheme": scheme_labels.tolist(), "parameter": fit_param_labels.tolist()},
        attrs={
            "method": str(result.get("method", "")),
            "order": str(result.get("order", "")),
            "observable": str(result.get("observable", "")),
            "part": str(result.get("part", "both")),
            "scheme_labels": json.dumps(scheme_labels.tolist()),
            "fit_param_labels": json.dumps(fit_param_labels.tolist()),
            "fit_params": json.dumps(fit_params.tolist()),
            "fit_param_center": json.dumps(np.mean(fit_params, axis=1).tolist()),
            "fit_param_sdev": json.dumps(fit_param_sdev.tolist()),
            "fit_chi2": json.dumps(fit_chi2.tolist()),
            "fit_dof": json.dumps(fit_dof.tolist()),
            "fit_q": json.dumps(fit_q.tolist()),
            "fit_chi2_dof": json.dumps(fit_chi2_dof.tolist()),
            "fit_chi2_center": json.dumps(np.mean(fit_chi2, axis=1).tolist()),
            "fit_chi2_dof_center": json.dumps(np.mean(fit_chi2_dof, axis=1).tolist()),
            "fit_q_center": json.dumps(np.mean(fit_q, axis=1).tolist()),
            "mean_fit_params": json.dumps(mean_fit_params.tolist()),
            "mean_fit_chi2": json.dumps(mean_fit_chi2.tolist()),
            "mean_fit_dof": json.dumps(mean_fit_dof.tolist()),
            "mean_fit_q": json.dumps(mean_fit_q.tolist()),
            "scheme_weights": json.dumps(np.asarray(result.get("scheme_weights", []), dtype=float).tolist()),
            "scheme_fit_chi2_dof": json.dumps(np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float).tolist()),
            "scheme_roughness": json.dumps(np.asarray(result.get("scheme_roughness", []), dtype=float).tolist()),
            "scheme_scores": json.dumps(np.asarray(result.get("scheme_scores", []), dtype=float).tolist()),
            "best_scheme_index": json.dumps(np.asarray(result.get("best_scheme_index", -1), dtype=int).tolist()),
        },
        name="fourier_fit_parameters",
    )
    fit_info_data.to_netcdf(path)


def _scan_values(spec: dict[str, Any], key: str) -> list[float]:
    values_key = f"{key}_values"
    if values_key in spec:
        return [float(item) for item in spec[values_key]]
    start = float(spec[f"{key}_start"])
    stop = float(spec[f"{key}_stop"])
    step = float(spec.get(f"{key}_step", spec.get("step", 1.0)))
    if step <= 0:
        raise ValueError(f"{key}_step must be positive")
    values = []
    current = start
    while current <= stop + 0.5 * step:
        values.append(round(current, 12))
        current += step
    return values


def _positive_grid(coord: np.ndarray) -> np.ndarray:
    positive = np.asarray(coord, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) < 4:
        raise ValueError("automatic scheme_scan needs at least four positive coordinate points")
    return positive


def _last_stable_z_index(
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    *,
    resample_mode: str,
) -> int:
    """Return the last positive-grid index before large-z data become unreliable."""
    positive_mask = np.asarray(coord, dtype=float) > 0
    re = np.asarray(re_samples, dtype=float)[:, positive_mask]
    im = np.asarray(im_samples, dtype=float)[:, positive_mask]
    re_mean = np.mean(re, axis=0)
    im_mean = np.mean(im, axis=0)
    re_sdev = _sample_sdev(re, resample_mode=resample_mode)
    im_sdev = _sample_sdev(im, resample_mode=resample_mode)

    magnitude = np.hypot(re_mean, im_mean)
    uncertainty = np.hypot(re_sdev, im_sdev)
    scale = max(float(np.max(magnitude)), 1e-12)
    rel_uncertainty = uncertainty / np.maximum(magnitude, 0.05 * scale)
    baseline = float(np.median(rel_uncertainty[: min(4, len(rel_uncertainty))]))
    uncertainty_limit = min(1.0, max(0.35, 3.0 * baseline + 0.05))

    signal = re_mean + 1j * im_mean
    signal_scale = max(float(np.max(np.abs(signal))), 1e-12)
    jitter_unstable = np.zeros_like(rel_uncertainty, dtype=bool)
    if len(signal) >= 5:
        curvature = np.abs(signal[2:] - 2.0 * signal[1:-1] + signal[:-2])
        curvature = curvature / np.maximum(np.abs(signal[1:-1]), 0.05 * signal_scale)
        curvature_baseline = float(np.median(curvature[: min(4, len(curvature))]))
        curvature_limit = min(2.0, max(0.5, 4.0 * curvature_baseline + 0.05))
        jitter_unstable[1:-1] = curvature > curvature_limit

    rel_growth = np.ones_like(rel_uncertainty, dtype=float)
    if len(rel_uncertainty) > 1:
        previous = np.maximum(rel_uncertainty[:-1], max(baseline, 1e-12))
        rel_growth[1:] = rel_uncertainty[1:] / previous
    sharp_uncertainty = (
        (rel_uncertainty > max(1.0, uncertainty_limit))
        | ((rel_uncertainty > max(0.7, 2.0 * uncertainty_limit)) & (rel_growth > 1.5))
    )
    unstable = sharp_uncertainty | jitter_unstable
    min_points = min(5, len(rel_uncertainty))
    for idx in range(min_points, len(rel_uncertainty) - 1):
        if unstable[idx] and unstable[idx + 1]:
            return max(min_points - 1, idx - 1)
    return len(rel_uncertainty) - 1


def _pick_four_tail_values(grid: np.ndarray, *, end_index: int) -> list[float]:
    end_index = int(np.clip(end_index, 0, len(grid) - 1))
    start_index = max(0, end_index - 3)
    values = grid[start_index : end_index + 1]
    if len(values) < 4:
        values = grid[: min(len(grid), 4)]
    return [float(item) for item in values[-4:]]


def _preferred_tail_start(
    *,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
) -> float | None:
    """Return the coordinate closest to z ~= 0.5 fm when unit metadata allows it."""
    unit = coord_unit.lower()
    if unit == "fm":
        return 0.5
    if unit == "lattice" and a_fm is not None and float(a_fm) > 0:
        return 0.5 / float(a_fm)
    if unit == "gev_inv":
        return 0.5 * FM_TO_GEV_INV
    if unit == "lambda" and pz_gev is not None:
        return 0.5 * FM_TO_GEV_INV * float(pz_gev)
    return None


def _tail_quality_stable_start(qualities: list[dict[str, Any]]) -> int:
    finite = [
        item
        for item in qualities
        if item["ok"] and np.isfinite(item["chi2_dof"]) and item["n_points"] >= 2
    ]
    if not finite:
        return 0

    chi = np.asarray([item["chi2_dof"] for item in finite], dtype=float)
    q_values = np.asarray([item["q_value"] for item in finite], dtype=float)
    best = float(np.min(chi))
    chi_limit = max(best * 1.25, best + 0.15, 1.0)
    for idx, item in enumerate(qualities):
        if not item["ok"] or not np.isfinite(item["chi2_dof"]):
            continue
        if item["chi2_dof"] > chi_limit:
            continue
        if item["q_value"] < 0.05 and np.nanmax(q_values) >= 0.05:
            continue
        later = [
            later_item["chi2_dof"]
            for later_item in qualities[idx : min(len(qualities), idx + 3)]
            if later_item["ok"] and np.isfinite(later_item["chi2_dof"])
        ]
        if later and max(later) <= max(chi_limit * 1.1, chi_limit + 0.1):
            return idx
    return int(np.nanargmin([item["chi2_dof"] if item["ok"] else np.inf for item in qualities]))


def _pick_four_zmin_values_by_tail_fit(
    positive: np.ndarray,
    *,
    zmax_values: list[float],
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    method: str,
    order: str,
    observable: str,
    coord_unit: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    part: str,
    preferred_zmin: float | None,
) -> list[float]:
    stable_starts = []
    required_points = _minimum_fit_points_for_parameters(len(_param_labels(method, order, observable)), part)
    for zmax in zmax_values:
        candidates = positive[positive < float(zmax)]
        candidates = np.asarray(
            [candidate for candidate in candidates if np.count_nonzero((positive >= candidate) & (positive <= zmax)) >= required_points],
            dtype=float,
        )
        if preferred_zmin is not None:
            candidates = candidates[candidates >= float(preferred_zmin)]
        if len(candidates) == 0:
            continue
        qualities = [
            fit_tail_quality_for_mean(
                coord,
                re_samples,
                im_samples,
                zmin=float(candidate),
                zmax=float(zmax),
                method=method,
                order=order,
                observable=observable,
                coord_unit=coord_unit,
                pz_gev=pz_gev,
                pz_prime_gev=pz_prime_gev,
                a_fm=a_fm,
                resample_mode=resample_mode,
                Lambda0=Lambda0,
                part=part,
            )
            for candidate in candidates
        ]
        stable_starts.append(float(candidates[_tail_quality_stable_start(qualities)]))

    stable_starts = sorted({float(item) for item in stable_starts})
    if len(stable_starts) >= 4:
        return stable_starts[:4]
    anchor = max(stable_starts) if stable_starts else (float(preferred_zmin) if preferred_zmin is not None else positive[0])
    candidates = [float(item) for item in positive if item >= anchor and any(item < zmax for zmax in zmax_values)]
    for candidate in candidates:
        if candidate not in stable_starts:
            stable_starts.append(candidate)
        if len(stable_starts) >= 4:
            break
    return stable_starts[:4]


def _default_z_ext_max(
    coord: np.ndarray,
    *,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
) -> float:
    """Return the coordinate value whose lambda is eight units past the data."""
    _fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    return float(np.max(coord) + 8.0 / ft_scale)


def _auto_fill_scheme_scan(
    spec: dict[str, Any],
    *,
    coord: np.ndarray,
    positive: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    stable_idx: int,
    coord_unit: str,
    method: str,
    order: str,
    observable: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    part: str,
) -> dict[str, Any]:
    """Fill missing scan keys with stable zmax values and tail-fit zmin diagnostics."""
    if "zmax_values" not in spec and "zmax_start" not in spec:
        spec["zmax_values"] = _pick_four_tail_values(positive, end_index=stable_idx)
    if "zmax_values" in spec:
        zmax_values = [float(item) for item in spec["zmax_values"]]
    else:
        zmax_values = _scan_values(spec, "zmax")

    if "zmin_values" not in spec and "zmin_start" not in spec:
        preferred_zmin = _preferred_tail_start(
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            a_fm=a_fm,
        )
        spec["zmin_values"] = _pick_four_zmin_values_by_tail_fit(
            positive,
            zmax_values=zmax_values,
            coord=coord,
            re_samples=re_samples,
            im_samples=im_samples,
            method=method,
            order=order,
            observable=observable,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_prime_gev=pz_prime_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            Lambda0=Lambda0,
            part=part,
            preferred_zmin=preferred_zmin,
        )
    if "z_ext_max" not in spec:
        spec["z_ext_max"] = _default_z_ext_max(
            coord,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            a_fm=a_fm,
        )
    if "smooth" not in spec:
        spec["smooth"] = "linear"
    return spec


def _scan_has_all_range_keys(spec: dict[str, Any] | None) -> bool:
    if spec is None:
        return False
    has_zmin = "zmin_values" in spec or "zmin_start" in spec
    has_zmax = "zmax_values" in spec or "zmax_start" in spec
    return has_zmin and has_zmax and "z_ext_max" in spec and "smooth" in spec


def _fill_scheme_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    spec.setdefault("y_range", [-2.0, 2.0])
    spec.setdefault("roughness_weight", 1.0)
    spec.setdefault("model_average", True)
    return spec


def _auto_scheme_scan(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    coord_unit: str,
    method: str,
    order: str,
    observable: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    part: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a conservative scan from stable zmax and tail-fit zmin diagnostics."""
    spec = dict(existing or {})
    positive = _positive_grid(coord)
    re_axis0 = np.asarray(re_samples, dtype=float)
    im_axis0 = np.asarray(im_samples, dtype=float)
    stable_idx = _last_stable_z_index(coord, re_axis0, im_axis0, resample_mode=resample_mode)
    spec = _auto_fill_scheme_scan(
        spec,
        coord=np.asarray(coord, dtype=float),
        positive=positive,
        re_samples=re_axis0,
        im_samples=im_axis0,
        stable_idx=stable_idx,
        coord_unit=coord_unit,
        method=method,
        order=order,
        observable=observable,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        a_fm=a_fm,
        resample_mode=resample_mode,
        Lambda0=Lambda0,
        part=part,
    )
    spec["auto_generated"] = True
    return spec


def _generate_scan_schemes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    zmin_values = _scan_values(spec, "zmin")
    zmax_values = _scan_values(spec, "zmax")
    z_ext_max = float(spec["z_ext_max"])
    smooth = str(spec.get("smooth", "linear"))
    max_schemes = int(spec.get("max_schemes", 200))

    schemes = []
    for zmin in zmin_values:
        for zmax in zmax_values:
            if zmax <= zmin:
                continue
            scheme = {
                "label": f"zmin_{zmin:g}_zmax_{zmax:g}".replace(".", "p"),
                "zmin": zmin,
                "zmax": zmax,
                "z_ext_max": z_ext_max,
                "smooth": smooth,
            }
            schemes.append(scheme)
            if len(schemes) >= max_schemes:
                return schemes
    if not schemes:
        raise ValueError("scheme_scan produced no valid zmin/zmax combinations")
    return schemes


def _resolve_k_grid(k_grid: list[float] | dict[str, Any]) -> list[float]:
    if isinstance(k_grid, dict):
        start = float(k_grid["start"])
        stop = float(k_grid["stop"])
        if "num" in k_grid:
            num = int(k_grid["num"])
            if num < 2:
                raise ValueError("k_grid num must be at least 2")
            return np.linspace(start, stop, num).tolist()
        step = float(k_grid["step"])
        if step <= 0:
            raise ValueError("k_grid step must be positive")
        return np.arange(start, stop + 0.5 * step, step).tolist()
    return [float(item) for item in k_grid]


def _roughness_score(k_grid: np.ndarray, ft_mean: np.ndarray, y_range: list[float] | tuple[float, float] | None) -> float:
    if y_range is None:
        mask = np.ones_like(k_grid, dtype=bool)
    else:
        low, high = float(y_range[0]), float(y_range[1])
        mask = (k_grid >= low) & (k_grid <= high)
    y = k_grid[mask]
    curve = ft_mean[mask]
    if len(y) < 3:
        return 0.0
    order = np.argsort(y)
    curve = curve[order]
    second = np.diff(curve, n=2)
    scale = max(float(np.sqrt(np.mean(curve**2))), float(np.max(np.abs(curve))), 1e-8)
    return float(np.sqrt(np.mean(second**2)) / scale)


def _apply_scheme_model_average(
    result: dict[str, Any],
    *,
    y_range: list[float] | tuple[float, float] | None,
    roughness_weight: float,
    resample_mode: str,
) -> None:
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    k_grid = np.asarray(result["k_grid"], dtype=float)
    re_mean_by_scheme = np.mean(ft_re, axis=1)
    im_mean_by_scheme = np.mean(ft_im, axis=1)
    re_stat_by_scheme = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in ft_re])
    im_stat_by_scheme = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in ft_im])

    fit_chi2 = []
    roughness = []
    for idx, scheme_result in enumerate(result["scheme_results"]):
        fit_chi2.append(float(scheme_result["mean_fit_chi2"]) / max(float(scheme_result["mean_fit_dof"]), 1.0))
        roughness.append(_roughness_score(k_grid, re_mean_by_scheme[idx], y_range))

    fit_arr = np.asarray(fit_chi2, dtype=float)
    rough_arr = np.asarray(roughness, dtype=float)
    failures = np.asarray(result["fit_failures"], dtype=float)
    scores = fit_arr + float(roughness_weight) * rough_arr + 100.0 * failures
    finite = np.isfinite(scores)
    if not np.any(finite):
        weights = np.full(len(scores), 1.0 / len(scores))
    else:
        shifted = np.where(finite, scores - np.nanmin(scores[finite]), np.inf)
        weights = np.exp(-0.5 * shifted)
        weights[~np.isfinite(weights)] = 0.0
        total = float(np.sum(weights))
        weights = weights / total if total > 0 else np.full(len(scores), 1.0 / len(scores))

    best = int(np.argmax(weights))
    re_mean = np.sum(weights[:, None] * re_mean_by_scheme, axis=0)
    im_mean = np.sum(weights[:, None] * im_mean_by_scheme, axis=0)
    re_stat = np.sqrt(np.sum(weights[:, None] * re_stat_by_scheme**2, axis=0))
    im_stat = np.sqrt(np.sum(weights[:, None] * im_stat_by_scheme**2, axis=0))
    re_sys = np.sqrt(np.sum(weights[:, None] * (re_mean_by_scheme - re_mean) ** 2, axis=0))
    im_sys = np.sqrt(np.sum(weights[:, None] * (im_mean_by_scheme - im_mean) ** 2, axis=0))

    result["ft_re_mean"] = re_mean
    result["ft_im_mean"] = im_mean
    result["ft_re_stat_sdev"] = re_stat
    result["ft_im_stat_sdev"] = im_stat
    result["ft_re_sys_sdev"] = re_sys
    result["ft_im_sys_sdev"] = im_sys
    result["scheme_weights"] = weights.tolist()
    result["scheme_fit_chi2_dof"] = fit_arr.tolist()
    result["scheme_roughness"] = rough_arr.tolist()
    result["scheme_scores"] = scores.tolist()
    result["best_scheme_index"] = best
    result["best_scheme_label"] = result["scheme_labels"][best]


def _apply_fourier_output_scale(result: dict[str, Any], output_scale: float) -> None:
    """Scale Fourier-space outputs without changing coordinate-space fits."""
    scale = float(output_scale)
    if scale == 1.0:
        result["output_scale"] = scale
        return
    for key in (
        "ft_re_samples",
        "ft_im_samples",
        "ft_re_mean",
        "ft_im_mean",
    ):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float) * scale
    error_scale = abs(scale)
    for key in (
        "ft_re_stat_sdev",
        "ft_im_stat_sdev",
        "ft_re_sys_sdev",
        "ft_im_sys_sdev",
    ):
        if key in result:
            result[key] = np.asarray(result[key], dtype=float) * error_scale
    result["output_scale"] = scale


def run_fourier_transform(
    store: dict[str, Any],
    *,
    k_grid: list[float] | dict[str, Any],
    scheme_scan: dict[str, Any] | None = None,
    method: str = "GI",
    order: str = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_prime_gev: float | None = None,
    a_fm: float | None = None,
    im_flip_for_ft: bool = False,
    Lambda0: float = 0.1,
    posterior_prior_error_scale: float = 3.0,
    fit_error_mode: str = "diagonal",
    part: str = "both",
    output_scale: float = 1.0,
    save_path: str | None = None,
    plot_fourier: dict[str, Any] | None = None,
    plot_extension: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Run local extrapolation and Fourier transform for loaded samples."""
    out = "fourier_result"
    matrix_element_data = store["matrix_element_data"]
    resample_mode = _normalise_resample_mode(getattr(matrix_element_data, "resample", "bootstrap"))
    matrix_element = ensemble_data_to_legacy_arrays(matrix_element_data)
    auto_scheme_scan = None
    coord_arr = np.asarray(matrix_element["coord"], dtype=float)
    scan_spec = _fill_scheme_defaults(dict(scheme_scan or {}))
    if not _scan_has_all_range_keys(scan_spec):
        scan_spec = _auto_scheme_scan(
            coord=coord_arr,
            re_samples=np.asarray(matrix_element["re_samples"], dtype=float),
            im_samples=np.asarray(matrix_element["im_samples"], dtype=float),
            coord_unit=coord_unit,
            method=method,
            order=order,
            observable=observable,
            pz_gev=pz_gev,
            pz_prime_gev=pz_prime_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            Lambda0=float(Lambda0),
            part=part,
            existing=scan_spec,
        )
        auto_scheme_scan = scan_spec
    scheme_scan = scan_spec
    schemes = _generate_scan_schemes(scheme_scan)
    required_points = _minimum_fit_points_for_parameters(len(_param_labels(method, order, observable)), part)
    schemes = [
        scheme
        for scheme in schemes
        if np.count_nonzero((coord_arr >= float(scheme["zmin"])) & (coord_arr <= float(scheme["zmax"])) & (coord_arr > 0))
        >= required_points
    ]
    if not schemes:
        raise ValueError("scheme_scan produced no valid zmin/zmax combinations")
    k_values = _resolve_k_grid(k_grid)
    result = run_fourier_workflow(
        matrix_element["coord"],
        matrix_element["re_samples"],
        matrix_element["im_samples"],
        k_values,
        schemes=schemes,
        method=method,
        order=order,
        observable=observable,
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        a_fm=a_fm,
        im_flip_for_ft=im_flip_for_ft,
        resample_mode=resample_mode,
        Lambda0=float(Lambda0),
        posterior_prior_error_scale=float(posterior_prior_error_scale),
        fit_error_mode=fit_error_mode,
        part=part,
    )
    result["resample_mode"] = resample_mode
    result["pz_gev"] = pz_gev
    result["pz_prime_gev"] = pz_prime_gev
    result["a_fm"] = a_fm
    result["im_flip_for_ft"] = bool(im_flip_for_ft)
    result["Lambda0"] = float(Lambda0)
    result["posterior_prior_error_scale"] = float(posterior_prior_error_scale)
    result["fit_error_mode"] = str(fit_error_mode)
    result["part"] = str(part)
    result["ensemble"] = matrix_element_data.ensemble
    if auto_scheme_scan is not None:
        result["auto_scheme_scan"] = auto_scheme_scan
    model_average = bool(scheme_scan.get("model_average", True))
    if model_average:
        _apply_scheme_model_average(
            result,
            y_range=scheme_scan.get("y_range"),
            roughness_weight=float(scheme_scan["roughness_weight"]),
            resample_mode=resample_mode,
        )
    else:
        fit_arr = [
            float(item["mean_fit_chi2"]) / max(float(item["mean_fit_dof"]), 1.0)
            for item in result["scheme_results"]
        ]
        result["scheme_weights"] = [1.0] if len(fit_arr) == 1 else [1.0 / len(fit_arr)] * len(fit_arr)
        result["scheme_fit_chi2_dof"] = fit_arr
        result["scheme_roughness"] = [0.0] * len(fit_arr)
        result["scheme_scores"] = fit_arr
        result["best_scheme_index"] = int(np.argmin(fit_arr)) if fit_arr else 0
        result["best_scheme_label"] = result["scheme_labels"][result["best_scheme_index"]]
    _apply_fourier_output_scale(result, float(output_scale))
    store["fourier_result_data"] = fourier_result_to_ensemble_data(result)
    store[out] = result
    artifact = _artifact_path(save_path, default_name=f"{out}.nc", artifacts_dir=artifacts_dir).with_suffix(".nc")
    fit_info_artifact = _artifact_path(None, default_name="fourier_fit_info.nc", artifacts_dir=artifacts_dir)
    store["fourier_result_data"].to_netcdf(artifact)
    _save_fourier_fit_info_netcdf(fit_info_artifact, result)
    result["artifact"] = str(artifact)
    result["fit_info_artifact"] = str(fit_info_artifact)
    summary = summarize_fourier_result(store)
    plot_kwargs = dict(plot_fourier or {})
    extension_kwargs = dict(plot_extension or {})
    report_kwargs = dict(report or {})
    plot = plot_fourier_result(store, artifacts_dir=artifacts_dir, **plot_kwargs)
    extension_plot = plot_fourier_extension_quality_result(store, artifacts_dir=artifacts_dir, **extension_kwargs)
    report_result = report_fourier_result(store, artifacts_dir=artifacts_dir, **report_kwargs)
    return {
        "out": out,
        "artifact": str(artifact),
        "fit_info_artifact": str(fit_info_artifact),
        "summary": summary["out"],
        "plot": plot["plot"],
        "plot_image": plot.get("plot_image"),
        "plot_re": extension_plot["plot_re"],
        "plot_re_image": extension_plot.get("plot_re_image"),
        "plot_im": extension_plot["plot_im"],
        "plot_im_image": extension_plot.get("plot_im_image"),
        "report": report_result["report"],
        "report_cn": report_result["report_cn"],
        "n_schemes": int(result["ft_re_samples"].shape[0]),
        "n_samples": int(result["ft_re_samples"].shape[1]),
        "n_k": int(result["ft_re_samples"].shape[2]),
        "scheme_labels": result["scheme_labels"],
        "fit_failures": result["fit_failures"],
        "best_scheme_index": result.get("best_scheme_index"),
        "best_scheme_label": result.get("best_scheme_label"),
        "output_scale": result.get("output_scale", 1.0),
        "auto_scheme_scan": auto_scheme_scan,
    }


def summarize_fourier_result(
    store: dict[str, Any],
) -> dict[str, Any]:
    """Store and return a compact numerical summary of the Fourier result."""
    out = "fourier_summary"
    data = store["fourier_result"]
    summary = {
        "k_grid": np.asarray(data["k_grid"]).tolist(),
        "ft_re_mean": np.asarray(data["ft_re_mean"]).tolist(),
        "ft_im_mean": np.asarray(data["ft_im_mean"]).tolist(),
        "ft_re_stat_sdev": np.asarray(data["ft_re_stat_sdev"]).tolist(),
        "ft_im_stat_sdev": np.asarray(data["ft_im_stat_sdev"]).tolist(),
        "ft_re_sys_sdev": np.asarray(data["ft_re_sys_sdev"]).tolist(),
        "ft_im_sys_sdev": np.asarray(data["ft_im_sys_sdev"]).tolist(),
        "scheme_labels": list(data["scheme_labels"]),
        "fit_failures": list(data["fit_failures"]),
        "scheme_weights": list(data.get("scheme_weights", [])),
        "scheme_fit_chi2_dof": list(data.get("scheme_fit_chi2_dof", [])),
        "scheme_roughness": list(data.get("scheme_roughness", [])),
        "scheme_scores": list(data.get("scheme_scores", [])),
        "best_scheme_index": data.get("best_scheme_index"),
        "best_scheme_label": data.get("best_scheme_label"),
        "fit_info_artifact": data.get("fit_info_artifact"),
        "output_scale": data.get("output_scale", 1.0),
    }
    store[out] = summary
    return {"out": out, **summary}


def plot_fourier_result(
    store: dict[str, Any],
    *,
    artifact_path: str | None = None,
    save_path: str | None = None,
    title: str | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Plot the Fourier-stage artifact and store the figure path."""
    source = artifact_path
    if source is None:
        source = str(_artifact_path(None, default_name="fourier_result.nc", artifacts_dir=artifacts_dir))
    output = _artifact_path(save_path, default_name="fourier_result.pdf", artifacts_dir=artifacts_dir)
    if title is not None and title.strip().lower() in {"fourier result", "fourier transform"}:
        title = None
    fig, _ = plot_fourier_artifact(source, save_path=output, title=title)
    png_output = _png_companion_path(output)
    fig.savefig(png_output, bbox_inches="tight")
    plt.close(fig)
    result = {"plot": str(output), "plot_image": str(png_output), "source": str(source)}
    store["fourier_plot"] = result
    return result


def plot_fourier_extension_quality_result(
    store: dict[str, Any],
    *,
    scheme_index: int | None = None,
    save_path: str | None = None,
    title: str | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Plot data and smoothed real-part extension for one Fourier scheme."""
    matrix_element = ensemble_data_to_legacy_arrays(store["matrix_element_data"])
    data = store["fourier_result"]
    if scheme_index is None:
        scheme_index = int(data.get("best_scheme_index", 0) or 0)
    if title is not None and title.strip().lower() in {"fourier extension quality", "lambda extrapolation"}:
        title = None
    re_output = _artifact_path(save_path, default_name="fourier_extension_re.pdf", artifacts_dir=artifacts_dir)
    im_output = _artifact_path(None, default_name="fourier_extension_im.pdf", artifacts_dir=artifacts_dir)
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        matrix_element["re_samples"],
        data,
        scheme_index=scheme_index,
        component="re",
        pz_gev=data.get("pz_gev"),
        a_fm=data.get("a_fm"),
        save_path=re_output,
        title=title,
    )
    re_png_output = _png_companion_path(re_output)
    fig.savefig(re_png_output, bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        matrix_element["im_samples"],
        data,
        scheme_index=scheme_index,
        component="im",
        pz_gev=data.get("pz_gev"),
        a_fm=data.get("a_fm"),
        save_path=im_output,
        title=title,
    )
    im_png_output = _png_companion_path(im_output)
    fig.savefig(im_png_output, bbox_inches="tight")
    plt.close(fig)
    scheme_label = data["scheme_labels"][scheme_index]
    result = {
        "plot_re": str(re_output),
        "plot_im": str(im_output),
        "plot_re_image": str(re_png_output),
        "plot_im_image": str(im_png_output),
        "scheme_label": scheme_label,
    }
    store["fourier_extension_plot"] = result
    return result


def report_fourier_result(
    store: dict[str, Any],
    *,
    save_path: str | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Write a Markdown report explaining the Fourier-stage computation."""
    data = store["fourier_result"]
    output = _artifact_path(save_path, default_name="report_fourier.md", artifacts_dir=artifacts_dir)
    artifacts = {
        "fourier_artifact": data.get("artifact")
        or str(_artifact_path(None, default_name="fourier_result.nc", artifacts_dir=artifacts_dir)),
        "fit_info_artifact": data.get("fit_info_artifact"),
    }
    if isinstance(store.get("fourier_plot"), dict):
        artifacts["fourier_plot"] = store["fourier_plot"].get("plot")
        artifacts["fourier_plot_image"] = store["fourier_plot"].get("plot_image")
        artifacts["fourier_artifact"] = store["fourier_plot"].get("source", artifacts["fourier_artifact"])
    if isinstance(store.get("fourier_extension_plot"), dict):
        artifacts["extension_plot_re"] = store["fourier_extension_plot"].get("plot_re")
        artifacts["extension_plot_im"] = store["fourier_extension_plot"].get("plot_im")
        artifacts["extension_plot_re_image"] = store["fourier_extension_plot"].get("plot_re_image")
        artifacts["extension_plot_im_image"] = store["fourier_extension_plot"].get("plot_im_image")
    paths = write_fourier_report(
        result=data,
        summary=store.get("fourier_summary") or summarize_fourier_result(store),
        artifacts=artifacts,
        path=output,
    )
    report = {
        "report": str(paths["en"]),
        "report_cn": str(paths["zh"]),
        "source": artifacts.get("fourier_artifact"),
        "fit_info_artifact": artifacts.get("fit_info_artifact"),
    }
    store["fourier_report"] = report
    return report


STAGE_TOOLS = {
    "load_renormalized_matrix_element_samples": load_renormalized_matrix_element_samples,
    "run_fourier_transform": run_fourier_transform,
    "summarize_fourier_result": summarize_fourier_result,
    "plot_fourier_result": plot_fourier_result,
    "plot_fourier_extension_quality_result": plot_fourier_extension_quality_result,
    "report_fourier_result": report_fourier_result,
}
