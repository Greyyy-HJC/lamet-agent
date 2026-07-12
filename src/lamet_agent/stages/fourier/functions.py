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
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.plotting import plot_fourier_artifact, plot_fourier_extension_quality
from lamet_agent.core.resampling import (
    normalize_resample_mode,
    normalize_sample_error_mode,
    recenter_sample_values,
    sample_mean_and_sdev,
    sample_sdev,
    sample_value_with_error,
    samples_to_gvar,
)
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
    fit_label: str | None = None
    fit_sign: float = 1.0
    fixed: float | None = None


def _normalise_resample_mode(value: str | None) -> str:
    mode = normalize_resample_mode(value, allow_raw=True)
    return {"bs": "bootstrap", "jk": "jackknife", "raw": "raw"}[mode]


def _sample_gvar(samples, *, resample_mode: str = "bootstrap", sample_error_mode: str = "covariance") -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        return gv.gvar(arr, np.zeros_like(arr, dtype=float))
    mode = _normalise_resample_mode(resample_mode)
    if mode in {"bootstrap", "jackknife"}:
        return samples_to_gvar(arr, mode=mode, sample_error_mode=sample_error_mode)
    mean = np.mean(arr, axis=0)
    sdev = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean, dtype=float)
    return gv.gvar(mean, sdev)


def _sample_sdev(samples, *, resample_mode: str = "bootstrap", sample_error_mode: str = "covariance") -> np.ndarray:
    mode = _normalise_resample_mode(resample_mode)
    if mode in {"bootstrap", "jackknife"}:
        return sample_sdev(samples, mode=mode, sample_error_mode=sample_error_mode)
    return np.asarray(gv.sdev(_sample_gvar(samples, resample_mode=resample_mode)), dtype=float)


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
    sample_error_mode: str,
    resample_mode: str,
    part: str = "both",
    re_fit_samples: np.ndarray | None = None,
    im_fit_samples: np.ndarray | None = None,
    sigma_re: np.ndarray | None = None,
    sigma_im: np.ndarray | None = None,
) -> np.ndarray:
    error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    part = _normalise_part(part)
    blocks = []
    centers = []
    floor_errors = []
    if _uses_re(part):
        if re_fit_samples is None:
            raise ValueError("sample error construction requires re_fit_samples for part='re' or 'both'")
        blocks.append(np.asarray(re_fit_samples, dtype=float))
        centers.append(np.asarray(re_fit, dtype=float))
        if sigma_re is not None:
            floor_errors.append(np.asarray(sigma_re, dtype=float))
    if _uses_im(part):
        if im_fit_samples is None:
            raise ValueError("sample error construction requires im_fit_samples for part='im' or 'both'")
        blocks.append(np.asarray(im_fit_samples, dtype=float))
        centers.append(np.asarray(im_fit, dtype=float))
        if sigma_im is not None:
            floor_errors.append(np.asarray(sigma_im, dtype=float))
    sample_matrix = np.concatenate(blocks, axis=1)
    center = np.concatenate(centers)
    if error_mode == "covariance":
        return sample_value_with_error(center, sample_matrix, mode=resample_mode, sample_error_mode=error_mode)
    template = samples_to_gvar(sample_matrix, mode=resample_mode, sample_error_mode=error_mode)
    if floor_errors:
        sigma = np.maximum(np.asarray(gv.sdev(template), dtype=float), np.concatenate(floor_errors))
        return gv.gvar(center, sigma)
    return recenter_sample_values(center, template)


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

    if np.isclose(lam[0], 0.0):
        lam_full = np.concatenate([-lam[::-1][:-1], lam])
        re_full = np.concatenate([re[::-1][:-1], re])
        im_full = np.concatenate([-im[::-1][:-1], im])
    else:
        lam_full = np.concatenate([-lam[::-1], lam])
        re_full = np.concatenate([re[::-1], re])
        im_full = np.concatenate([-im[::-1], im])
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


def _ft_scale_momentum(pz_gev: float | None, pz_out_gev: float | None = None) -> float:
    """Return the momentum used only for coord->lambda scaling."""
    return max(abs(float(pz_gev or 0.0)), abs(float(pz_out_gev or 0.0)))


def _coord_scale(
    coord_unit: str,
    *,
    pz_gev: float | None,
    a_fm: float | None,
    pz_out_gev: float | None = None,
) -> tuple[float, float]:
    """Return ``(fit_scale, ft_scale)`` from input coordinates."""
    unit = coord_unit.lower()
    ft_momentum = _ft_scale_momentum(pz_gev, pz_out_gev)
    if unit == "lambda":
        return 1.0, 1.0
    if unit == "gev_inv":
        if ft_momentum == 0.0:
            raise ValueError("pz_gev or pz_out_gev is required when coord_unit='gev_inv'")
        return 1.0, ft_momentum
    if unit == "fm":
        if ft_momentum == 0.0:
            raise ValueError("pz_gev or pz_out_gev is required when coord_unit='fm'")
        return FM_TO_GEV_INV, FM_TO_GEV_INV * ft_momentum
    if unit == "lattice":
        if a_fm is None or ft_momentum == 0.0:
            raise ValueError("a_fm and pz_gev or pz_out_gev are required when coord_unit='lattice'")
        return float(a_fm) * FM_TO_GEV_INV, float(a_fm) * FM_TO_GEV_INV * ft_momentum
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
QUARK_LIKE_AMPLITUDE_BOUND = 20.0


def _quark_like_term_names(observable: str) -> tuple[str, ...]:
    observable = _canonical_observable(observable)
    if observable not in QUARK_LIKE_TERMS:
        raise ValueError(f"unsupported quark-like observable {observable!r}")
    return QUARK_LIKE_TERMS[observable]


def _phase_scales(
    *,
    coord_unit: str,
    pz_gev: float | None,
    pz_out_gev: float | None,
    ft_scale_over_fit_scale: float,
) -> tuple[float, float | None]:
    if coord_unit.lower() == "lambda":
        phase_scale = ft_scale_over_fit_scale
        if pz_out_gev is None:
            return phase_scale, None
        if pz_gev is None:
            raise ValueError("pz_gev is required with pz_out_gev when coord_unit='lambda'")
        return phase_scale, float(pz_out_gev) / float(pz_gev)
    return float(pz_gev or 0.0), None if pz_out_gev is None else float(pz_out_gev)


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
    Lambda0: float,
) -> list[_TailParameter]:
    parameters = [
        *parameters,
        _TailParameter("m", max(0.5 - float(Lambda0), 0.05), 0.0),
    ]
    if method.upper() == "CG":
        parameters.append(_TailParameter("n", 0.5, -2.0, 4.0))
    return parameters


def _quark_like_parameters(
    order: str,
    observable: str,
    *,
    sector: str | None = None,
    hadron: str | None = None,
) -> list[_TailParameter]:
    observable = _canonical_observable(observable)
    term_names = _quark_like_term_names(observable)
    parameters = []
    for idx, name in enumerate(term_names):
        parameters.extend(
            [
                _TailParameter(f"A{name}", 1.0 if idx == 0 else 0.1, -QUARK_LIKE_AMPLITUDE_BOUND, QUARK_LIKE_AMPLITUDE_BOUND),
                _TailParameter(f"phi{name}", 0.0, -np.pi, np.pi),
            ]
        )
    if order.upper() == "NLA":
        for name in term_names:
            parameters.extend(
                [
                    _TailParameter(f"A{name}p", 0.1, -QUARK_LIKE_AMPLITUDE_BOUND, QUARK_LIKE_AMPLITUDE_BOUND),
                    _TailParameter(f"phi{name}p", 0.0, -np.pi, np.pi),
                ]
            )
    sector = str(sector or "").lower()
    hadron = str(hadron or "").lower()
    fixed = {}
    aliases = {}
    if observable == "pion_quark_quasi_pdf" and sector == "valence":
        fixed.update({"phi2": 0.0, "phi2p": 0.0})
        aliases.update({"A3": ("A1", 1.0), "phi3": ("phi1", -1.0), "A3p": ("A1p", 1.0), "phi3p": ("phi1p", -1.0)})
    if observable == "pion_quark_quasi_pdf" and sector == "sea":
        fixed.update({"A1": 0.0, "A3": 0.0, "A1p": 0.0, "A3p": 0.0})
    if observable == "meson_quasi_da" and hadron == "pion":
        aliases.update({"A2": ("A1", 1.0), "phi2": ("phi1", -1.0), "A2p": ("A1p", 1.0), "phi2p": ("phi1p", -1.0)})
    if observable == "pion_quark_quasi_gpd" and sector == "sea":
        fixed.update({"A1": 0.0, "A3": 0.0, "A1p": 0.0, "A3p": 0.0})
    return [
        _TailParameter(item.label, fixed[item.label], item.lower, item.upper, fixed=fixed[item.label])
        if item.label in fixed
        else _TailParameter(item.label, item.p0, item.lower, item.upper, aliases[item.label][0], aliases[item.label][1])
        if item.label in aliases
        else item
        for item in parameters
    ]


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


def _observable_parameters(
    order: str,
    observable: str,
    *,
    sector: str | None = None,
    hadron: str | None = None,
) -> list[_TailParameter]:
    observable = _canonical_observable(observable)
    if observable == "nucleon_gluon_quasi_pdf":
        return _nucleon_gluon_parameters(order)
    if observable == "pion_gluon_quasi_pdf":
        return _pion_gluon_parameters(order)
    return _quark_like_parameters(order, observable, sector=sector, hadron=hadron)


def _param_template(
    method: str,
    order: str,
    observable: str,
    *,
    Lambda0: float = 0.1,
    sector: str | None = None,
    hadron: str | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    method = method.upper()
    order = order.upper()
    if method not in {"GI", "CG"}:
        raise ValueError("method must be 'GI' or 'CG'")
    if order not in {"LA", "NLA"}:
        raise ValueError("order must be 'LA' or 'NLA'")

    observable = _canonical_observable(observable)
    parameters = _with_method_tail_parameters(
        _observable_parameters(order, observable, sector=sector, hadron=hadron),
        method=method,
        Lambda0=float(Lambda0),
    )
    fit_labels: set[str] = set()
    selected = []
    for item in parameters:
        if not fit:
            selected.append(item)
        elif item.fixed is None and item.fit_label is None and item.label not in fit_labels:
            selected.append(item)
            fit_labels.add(item.label)
    p0_by_label = {item.label: item.p0 for item in parameters}
    p0 = [
        item.fixed
        if item.fixed is not None
        else item.fit_sign * p0_by_label[item.fit_label]
        if item.fit_label is not None
        else item.p0
        for item in selected
    ]
    lower = [item.lower for item in selected]
    upper = [item.upper for item in selected]
    return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _param_labels(
    method: str,
    order: str,
    observable: str,
    *,
    sector: str | None = None,
    hadron: str | None = None,
    fit: bool = False,
) -> list[str]:
    method = method.upper()
    if method not in {"GI", "CG"}:
        raise ValueError("method must be 'GI' or 'CG'")
    order = order.upper()
    if order not in {"LA", "NLA"}:
        raise ValueError("order must be 'LA' or 'NLA'")
    observable = _canonical_observable(observable)
    parameters = _with_method_tail_parameters(
        _observable_parameters(order, observable, sector=sector, hadron=hadron),
        method=method,
        Lambda0=0.1,
    )
    if not fit:
        return [item.label for item in parameters]
    labels = []
    for item in parameters:
        if item.fixed is None and item.fit_label is None and item.label not in labels:
            labels.append(item.label)
    return labels


def _decay_tail(z: np.ndarray, params: Sequence[Any], *, lambda_index: int, method: str, Lambda0: float) -> Any:
    tail = gv.exp(-(params[lambda_index] + float(Lambda0)) * z)
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
    Lambda0: float,
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

    tail = _decay_tail(z, params, lambda_index=cursor, method=method, Lambda0=Lambda0)
    return re * tail, im * tail


def _nucleon_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    Lambda0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F nucleon gluon form: (A z [+ A']) exp(-Lambda z) with zero Im."""
    # LA:  A z exp(-Lambda z)
    # NLA: (A z + A') exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1]
        lambda_index = 2
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method, Lambda0=Lambda0)
    im = np.zeros_like(z, dtype=object)
    return re * tail, im * tail


def _pion_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    phase_scale: float,
    Lambda0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F pion gluon form: A2 z plus optional constant and cosine term."""
    # LA:  A2 z exp(-Lambda z)
    # NLA: (A2 z + A2' + 2 A1 cos(phi - Pz z)) exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1] + 2.0 * params[2] * gv.cos(params[3] - phase_scale * z)
        lambda_index = 4
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method, Lambda0=Lambda0)
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
    Lambda0: float = 0.1,
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
            Lambda0=Lambda0,
        )

    if observable == "pion_gluon_quasi_pdf":
        return _pion_gluon_asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            phase_scale=phase_scale,
            Lambda0=Lambda0,
        )

    return _quark_like_asymptotic_values(
        z,
        params,
        method=method,
        order=order,
        observable=observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        Lambda0=Lambda0,
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
    sector: str | None = None,
    hadron: str | None = None,
    p0: np.ndarray | None = None,
    prior: gv.BufferDict | None = None,
    Lambda0: float = 0.1,
) -> tuple[np.ndarray, gv.BufferDict | None, gv.BufferDict | None, bool, float, int, float, float]:
    default_p0, _ = _param_template(method, order, observable, Lambda0=Lambda0, sector=sector, hadron=hadron)
    fit_p0, bounds = _param_template(method, order, observable, Lambda0=Lambda0, sector=sector, hadron=hadron, fit=True)
    start = default_p0 if p0 is None else np.asarray(p0, dtype=float)
    fit_labels = _param_labels(method, order, observable, sector=sector, hadron=hadron, fit=True)
    full_items = _with_method_tail_parameters(
        _observable_parameters(order.upper(), _canonical_observable(observable), sector=sector, hadron=hadron),
        method=method.upper(),
        Lambda0=float(Lambda0),
    )
    full_labels = [item.label for item in full_items]
    if p0 is not None and len(start) != len(fit_p0):
        start = np.asarray([start[full_labels.index(label)] for label in fit_labels], dtype=float)
    else:
        start = fit_p0 if p0 is None else start

    def fcn(z: np.ndarray, p: gv.BufferDict) -> np.ndarray:
        free_values = dict(zip(fit_labels, _physical_params(p, bounds)))
        params = [
            item.fixed
            if item.fixed is not None
            else item.fit_sign * free_values[item.fit_label]
            if item.fit_label is not None
            else free_values[item.label]
            for item in full_items
        ]
        pred_re, pred_im = _asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            observable=observable,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            Lambda0=Lambda0,
        )
        return _select_fit_prediction(pred_re, pred_im, part)

    dof = max(1, _n_fit_channels(part) * len(z_fit) - len(fit_p0))
    try:
        fit_prior = prior
        if fit_prior is None:
            fit_prior = gv.BufferDict()
            for key, value in _internal_p0(start, bounds).items():
                fit_prior[key] = gv.gvar(float(value), 3.0)
        fit_args = {
            "data": (z_fit, y_data),
            "fcn": fcn,
            "p0": _internal_p0(start, bounds),
            "prior": fit_prior,
            "maxit": 2000,
            "svdcut": 1e-12,
            "fitter": "scipy_least_squares",
        }
        fit = lsqfit.nonlinear_fit(**fit_args)
        free_values = dict(zip(fit_labels, _physical_params(fit.pmean, bounds)))
        physical = [
            item.fixed
            if item.fixed is not None
            else item.fit_sign * free_values[item.fit_label]
            if item.fit_label is not None
            else free_values[item.label]
            for item in full_items
        ]
        params = np.asarray([float(item) for item in physical], dtype=float)
    except (FloatingPointError, RuntimeError, ValueError, OverflowError, AssertionError):
        return default_p0, None, None, False, float("inf"), dof, 0.0, float("-inf")

    return (
        params,
        fit.pmean,
        fit.psdev,
        bool(np.isfinite(fit.chi2)),
        float(fit.chi2),
        int(fit.dof),
        float(fit.Q),
        float(fit.logGBF),
    )


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
    pz_out_gev: float | None = None,
    a_fm: float | None = None,
    resample_mode: str = "bootstrap",
    Lambda0: float = 0.1,
    posterior_prior_error_scale: float = 3.0,
    sample_error_mode: str = "covariance",
    part: str = "both",
    sector: str | None = None,
    hadron: str | None = None,
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
    fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, pz_out_gev=pz_out_gev, a_fm=a_fm)
    fit_coord = coord_arr * fit_scale
    ft_scale_over_fit_scale = ft_scale / fit_scale
    phase_scale, phase_prime_scale = _phase_scales(
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_out_gev=pz_out_gev,
        ft_scale_over_fit_scale=ft_scale_over_fit_scale,
    )
    zmin_fit = _convert_scheme_value(zmin, fit_scale)
    zmax_fit = _convert_scheme_value(zmax, fit_scale)
    fit_mask = (fit_coord >= zmin_fit) & (fit_coord <= zmax_fit) & (fit_coord > 0)
    n_points = int(np.count_nonzero(fit_mask))
    n_params = len(_param_labels(method, order, observable, sector=sector, hadron=hadron, fit=True))
    required_points = _minimum_fit_points_for_parameters(n_params, part)
    if n_points < required_points:
        dof = max(1, _n_fit_channels(part) * n_points - n_params)
        return {
            "tail_fit_success": False,
            "chi2": float("inf"),
            "dof": int(dof),
            "chi2_dof": float("inf"),
            "q_value": 0.0,
            "n_points": n_points,
        }

    z_fit = fit_coord[fit_mask]
    mean_re, _ = sample_mean_and_sdev(re_mat[:, fit_mask], mode=resample_mode, sample_error_mode=sample_error_mode)
    mean_im, _ = sample_mean_and_sdev(im_mat[:, fit_mask], mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_re = _sample_sdev(re_mat[:, fit_mask], resample_mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_im = _sample_sdev(im_mat[:, fit_mask], resample_mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)
    y_data = _fit_y_data(
        mean_re,
        mean_im,
        sample_error_mode=sample_error_mode,
        resample_mode=resample_mode,
        part=part,
        re_fit_samples=re_mat[:, fit_mask],
        im_fit_samples=im_mat[:, fit_mask],
        sigma_re=sigma_re,
        sigma_im=sigma_im,
    )

    mean_params, mean_pmean, mean_psdev, tail_fit_success, chi2, dof, q_value, log_gbf = _fit_one_sample(
        z_fit,
        y_data=y_data,
        method=method,
        order=order,
        observable=observable,
        part=part,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        sector=sector,
        hadron=hadron,
        Lambda0=Lambda0,
    )
    if tail_fit_success and mean_pmean is not None and mean_psdev is not None:
        mean_params, _mean_pmean, _mean_psdev, tail_fit_success, chi2, dof, q_value, log_gbf = _fit_one_sample(
            z_fit,
            y_data=y_data,
            method=method,
            order=order,
            observable=observable,
            part=part,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            sector=sector,
            hadron=hadron,
            p0=mean_params,
            prior=_scaled_internal_prior(mean_pmean, mean_psdev, posterior_prior_error_scale),
            Lambda0=Lambda0,
        )
    return {
        "tail_fit_success": bool(tail_fit_success),
        "chi2": float(chi2),
        "dof": int(dof),
        "chi2_dof": float(chi2 / max(dof, 1)),
        "q_value": float(q_value),
        "logGBF": float(log_gbf),
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


def _progress(iterable, *, desc: str, leave: bool = True):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


def _scheme_ranges(scheme: dict[str, Any], coord: np.ndarray) -> tuple[float, float, float]:
    positive = coord[coord > 0]
    zmin = float(scheme.get("zmin", positive[0]))
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
    y_grid: np.ndarray,
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
    sample_error_mode: str,
    part: str,
    sector: str | None,
    hadron: str | None,
) -> dict[str, Any]:
    zmin, zmax, z_ext_max = _scheme_ranges(scheme, coord)
    label = str(scheme.get("label", f"{method}_{order}_{zmin}_{zmax}"))
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
    n_fit_params = len(_param_labels(method, order, observable, sector=sector, hadron=hadron, fit=True))
    required_points = _minimum_fit_points_for_parameters(n_fit_params, part)
    if np.count_nonzero(fit_mask) < required_points:
        raise ValueError("fit range has too few points for the selected asymptotic form")

    z_fit = fit_coord[fit_mask]
    mean_re, _ = sample_mean_and_sdev(re_samples[:, fit_mask], mode=resample_mode, sample_error_mode=sample_error_mode)
    mean_im, _ = sample_mean_and_sdev(im_samples[:, fit_mask], mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_re = _sample_sdev(re_samples[:, fit_mask], resample_mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_im = _sample_sdev(im_samples[:, fit_mask], resample_mode=resample_mode, sample_error_mode=sample_error_mode)
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)
    mean_y_data = _fit_y_data(
        mean_re,
        mean_im,
        sample_error_mode=sample_error_mode,
        resample_mode=resample_mode,
        part=part,
        re_fit_samples=re_samples[:, fit_mask],
        im_fit_samples=im_samples[:, fit_mask],
        sigma_re=sigma_re,
        sigma_im=sigma_im,
    )
    mean_params, mean_pmean, mean_psdev, mean_ok, mean_chi2, mean_dof, mean_q, mean_log_gbf = _fit_one_sample(
        z_fit,
        y_data=mean_y_data,
        method=method,
        order=order,
        observable=observable,
        part=part,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        sector=sector,
        hadron=hadron,
        Lambda0=Lambda0,
    )
    sample_prior = None
    if mean_ok and mean_pmean is not None and mean_psdev is not None:
        sample_prior = _scaled_internal_prior(mean_pmean, mean_psdev, posterior_prior_error_scale)
        mean_params, _mean_pmean, _mean_psdev, mean_ok, mean_chi2, mean_dof, mean_q, mean_log_gbf = _fit_one_sample(
            z_fit,
            y_data=mean_y_data,
            method=method,
            order=order,
            observable=observable,
            part=part,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            sector=sector,
            hadron=hadron,
            p0=mean_params,
            prior=sample_prior,
            Lambda0=Lambda0,
        )

    dz = _uniform_step(fit_coord)
    z_ext = np.arange(fit_coord[0], z_ext_fit_max + 0.5 * dz, dz)
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
    ft_re = np.empty((n_samples, len(y_grid)), dtype=float)
    ft_im = np.empty_like(ft_re)
    fit_params = np.empty((n_samples, n_params), dtype=float)
    fit_chi2 = np.empty(n_samples, dtype=float)
    fit_dof = np.empty(n_samples, dtype=int)
    fit_q = np.empty(n_samples, dtype=float)
    fit_log_gbf = np.empty(n_samples, dtype=float)
    tail_fit_success_samples = np.empty(n_samples, dtype=bool)
    failures = 0

    positive = z_ext > 0
    for sample in _progress(range(n_samples), desc=f"fourier {label}", leave=False):
        sample_y_data = _fit_y_data(
            re_samples[sample, fit_mask],
            im_samples[sample, fit_mask],
            sample_error_mode=sample_error_mode,
            resample_mode=resample_mode,
            part=part,
            re_fit_samples=re_samples[:, fit_mask],
            im_fit_samples=im_samples[:, fit_mask],
            sigma_re=sigma_re,
            sigma_im=sigma_im,
        )
        params, _sample_pmean, _sample_psdev, tail_fit_success, chi2, dof, q_value, log_gbf = _fit_one_sample(
            z_fit,
            y_data=sample_y_data,
            method=method,
            order=order,
            observable=observable,
            part=part,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            sector=sector,
            hadron=hadron,
            p0=mean_params,
            prior=sample_prior,
            Lambda0=Lambda0,
        )
        tail_fit_success_samples[sample] = bool(tail_fit_success)
        if not tail_fit_success:
            failures += 1
            params = mean_params
            chi2 = mean_chi2
            dof = mean_dof
            q_value = mean_q
            log_gbf = mean_log_gbf
        fit_params[sample] = params
        fit_chi2[sample] = chi2
        fit_dof[sample] = dof
        fit_q[sample] = q_value
        fit_log_gbf[sample] = log_gbf

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
            Lambda0=Lambda0,
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
        ft_re[sample], ft_im[sample] = sum_ft_re_im(lam_full, re_full, im_full, y_grid)

    return {
        "label": label,
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
        "fit_log_gbf": fit_log_gbf,
        "tail_fit_success_samples": tail_fit_success_samples,
        "mean_fit_params": mean_params,
        "mean_fit_chi2": mean_chi2,
        "mean_fit_dof": mean_dof,
        "mean_fit_q": mean_q,
        "mean_fit_logGBF": mean_log_gbf,
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
    y_grid: Sequence[float],
    *,
    schemes: list[dict[str, Any]] | None = None,
    method: str = "GI",
    order: str = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_out_gev: float | None = None,
    a_fm: float | None = None,
    im_flip_for_ft: bool = False,
    resample_mode: str = "bootstrap",
    Lambda0: float = 0.1,
    posterior_prior_error_scale: float = 3.0,
    sample_error_mode: str = "covariance",
    part: str = "both",
    sector: str | None = None,
    hadron: str | None = None,
) -> dict[str, Any]:
    """Run asymptotic extension and Fourier transform for resampled data.

    ``schemes`` values are in the same unit as ``coord``. The output keeps
    sample information as arrays shaped ``(scheme, sample, y)``.
    """
    coord_arr = np.asarray(coord, dtype=float)
    resample_mode = _normalise_resample_mode(resample_mode)
    sample_error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    part = _normalise_part(part)
    coord_step = _uniform_step(coord_arr)
    missing_short_distance_coord = np.arange(0.0, coord_arr[0], coord_step).tolist()

    re_mat = _as_sample_matrix("re_samples", re_samples)
    im_mat = _as_sample_matrix("im_samples", im_samples)
    if re_mat.shape != im_mat.shape:
        raise ValueError("re_samples and im_samples must have the same shape")
    if re_mat.shape[1] != len(coord_arr):
        raise ValueError("sample arrays must have one value per coordinate point")

    observable = _canonical_observable(observable)
    fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, pz_out_gev=pz_out_gev, a_fm=a_fm)
    fit_coord = coord_arr * fit_scale
    ft_scale_over_fit_scale = ft_scale / fit_scale
    phase_scale, phase_prime_scale = _phase_scales(
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_out_gev=pz_out_gev,
        ft_scale_over_fit_scale=ft_scale_over_fit_scale,
    )
    y_arr = np.asarray(y_grid, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y_grid must be one-dimensional")

    if schemes is None:
        schemes = [
            {
                "label": "default",
                "zmin": coord_arr[1],
                "zmax": coord_arr[-1],
                "z_ext_max": coord_arr[-1] + 8.0 / ft_scale,
            }
        ]

    scheme_results = []
    for scheme in _progress(schemes, desc="fourier schemes"):
        scheme_order = str(scheme.get("order", order)).upper()
        scheme_prior_width = float(scheme.get("posterior_prior_error_scale", posterior_prior_error_scale))
        scheme_results.append(
            _run_one_scheme(
                coord=coord_arr,
                fit_coord=fit_coord,
                ft_scale_over_fit_scale=ft_scale_over_fit_scale,
                re_samples=re_mat,
                im_samples=im_mat,
                y_grid=y_arr,
                scheme=scheme,
                method=method,
                order=scheme_order,
                observable=observable,
                fit_scale=fit_scale,
                im_flip_for_ft=im_flip_for_ft,
                phase_scale=phase_scale,
                phase_prime_scale=phase_prime_scale,
                resample_mode=resample_mode,
                Lambda0=Lambda0,
                posterior_prior_error_scale=scheme_prior_width,
                sample_error_mode=sample_error_mode,
                part=part,
                sector=sector,
                hadron=hadron,
            )
        )
        scheme_results[-1]["order"] = scheme_order
        scheme_results[-1]["posterior_prior_error_scale"] = scheme_prior_width

    ft_re = np.asarray([item["ft_re_samples"] for item in scheme_results])
    ft_im = np.asarray([item["ft_im_samples"] for item in scheme_results])
    re_stats = [sample_mean_and_sdev(item["ft_re_samples"], mode=resample_mode, sample_error_mode=sample_error_mode) for item in scheme_results]
    im_stats = [sample_mean_and_sdev(item["ft_im_samples"], mode=resample_mode, sample_error_mode=sample_error_mode) for item in scheme_results]
    re_mean_by_scheme = np.asarray([item[0] for item in re_stats], dtype=float)
    im_mean_by_scheme = np.asarray([item[0] for item in im_stats], dtype=float)
    re_stat_by_scheme = np.asarray([item[1] for item in re_stats], dtype=float)
    im_stat_by_scheme = np.asarray([item[1] for item in im_stats], dtype=float)

    re_mean = np.mean(re_mean_by_scheme, axis=0)
    im_mean = np.mean(im_mean_by_scheme, axis=0)
    re_stat = np.sqrt(np.mean(re_stat_by_scheme**2, axis=0))
    im_stat = np.sqrt(np.mean(im_stat_by_scheme**2, axis=0))
    re_sys = np.std(re_mean_by_scheme, axis=0, ddof=0)
    im_sys = np.std(im_mean_by_scheme, axis=0, ddof=0)

    return {
        "y_grid": y_arr,
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
        "fit_model_q": [float(item.get("mean_fit_q", 0.0)) for item in scheme_results],
        "fit_model_chi2_dof": [
            float(item["mean_fit_chi2"]) / max(float(item["mean_fit_dof"]), 1.0) for item in scheme_results
        ],
        "fit_model_logGBF": [float(item.get("mean_fit_logGBF", float("-inf"))) for item in scheme_results],
        "method": method.upper(),
        "order": order.upper() if isinstance(order, str) else ",".join(str(item).upper() for item in order),
        "observable": observable,
        "coord_unit": coord_unit,
        "fit_coord_unit": "lambda" if coord_unit.lower() == "lambda" else "gev_inv",
        "resample_mode": resample_mode,
        "sample_error_mode": sample_error_mode,
        "Lambda0": float(Lambda0),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "part": part,
        "short_distance_policy": "full_from_zero" if not missing_short_distance_coord else "truncate_missing",
        "input_coord_start": float(coord_arr[0]),
        "input_coord_step": float(coord_step),
        "missing_short_distance_coord": [float(item) for item in missing_short_distance_coord],
        "missing_short_distance_count": int(len(missing_short_distance_coord)),
        "fourier_positive_coord_start": float(coord_arr[0]),
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
    if "final_ft_re_samples" in result and "final_ft_im_samples" in result:
        re_samples = np.asarray(result["final_ft_re_samples"], dtype=float)
        im_samples = np.asarray(result["final_ft_im_samples"], dtype=float)
    else:
        ft_re = np.asarray(result["ft_re_samples"], dtype=float)
        ft_im = np.asarray(result["ft_im_samples"], dtype=float)
        re_samples = np.mean(ft_re, axis=0)
        im_samples = np.mean(ft_im, axis=0)
    values = [re_samples[idx] + 1j * im_samples[idx] for idx in range(re_samples.shape[0])]
    attrs = {
        "method": str(result.get("method", "")),
        "order": str(result.get("order", "")),
        "observable": str(result.get("observable", "")),
        "sector": str(result.get("sector", "")),
        "target_observable": str(result.get("target_observable", "")),
        "coord_unit": str(result.get("coord_unit", "")),
        "fit_coord_unit": str(result.get("fit_coord_unit", "")),
        "part": str(result.get("part", "both")),
        "im_flip_for_ft": str(result.get("im_flip_for_ft", "")),
        "resample_mode": str(result.get("resample_mode", "")),
        "sample_error_mode": str(result.get("sample_error_mode", "")),
        "average_method": str(result.get("sample_error_mode", "")),
    }
    for key in ("pz_gev", "pz_out_gev", "a_fm"):
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
        "output_scale",
        "fit_model_labels",
        "fit_model_orders",
        "fit_model_prior_widths",
        "fit_model_weights",
        "fit_model_mean_weights",
        "fit_model_q",
        "fit_model_chi2_dof",
        "fit_model_logGBF",
        "best_fit_model_index_by_sample",
        "selected_range_label",
        "selected_fit_range",
        "candidate_scheme_labels",
        "candidate_scheme_fit_chi2_dof",
        "candidate_scheme_q",
        "candidate_scheme_logGBF",
        "selected_candidate_index",
        "selected_candidate_label",
        "selection_mode",
        "short_distance_policy",
        "input_coord_start",
        "input_coord_step",
        "missing_short_distance_coord",
        "missing_short_distance_count",
        "fourier_positive_coord_start",
    ):
        if key in result:
            attrs[key] = json.dumps(np.asarray(result[key]).tolist())
    return EnsembleData(
        ensemble=result.get("ensemble"),
        resample=_normalise_resample_mode(str(result.get("resample_mode", "bootstrap"))),
        values=values,
        dims=("x",),
        coords={"x": np.asarray(result["y_grid"], dtype=float).tolist()},
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
    existing = store.get("matrix_element_data")
    if isinstance(existing, EnsembleData):
        legacy = ensemble_data_to_legacy_arrays(existing)
        out = "matrix_element"
        stored = store.get(out, {})
        fmt = stored.get("input_format", "nc") if isinstance(stored, dict) else "nc"
        group_name = stored.get("h5_group") if isinstance(stored, dict) else None
        store[out] = {
            **legacy,
            "path": str(path),
            "input_format": fmt,
            "resample_mode": existing.resample,
        }
        if group_name is not None:
            store[out]["h5_group"] = group_name
        return {
            "out": out,
            "data": "matrix_element_data",
            "input_format": fmt,
            "h5_group": group_name,
            "resample_mode": existing.resample,
            "n_coord": int(len(legacy["coord"])),
            "n_sample": int(legacy["re_samples"].shape[0]),
            "re_shape": list(legacy["re_samples"].shape),
            "im_shape": list(legacy["im_samples"].shape),
        }

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


def _svg_companion_path(path: Path) -> Path:
    """Return an SVG companion path for Markdown image embedding."""
    return path.with_suffix(".svg")


def _save_fourier_fit_info_netcdf(path: Path, result: dict[str, Any]) -> None:
    schemes = result["scheme_results"]
    fit_param_labels = []
    for item in schemes:
        for label in item["fit_param_labels"]:
            if label not in fit_param_labels:
                fit_param_labels.append(label)
    fit_params = np.full((len(schemes), schemes[0]["fit_params"].shape[0], len(fit_param_labels)), np.nan, dtype=float)
    mean_fit_params = np.full((len(schemes), len(fit_param_labels)), np.nan, dtype=float)
    for idx, item in enumerate(schemes):
        item_params = np.asarray(item["fit_params"], dtype=float)
        item_mean_params = np.asarray(item["mean_fit_params"], dtype=float)
        for local_idx, label in enumerate(item["fit_param_labels"]):
            global_idx = fit_param_labels.index(label)
            fit_params[idx, :, global_idx] = item_params[:, local_idx]
            mean_fit_params[idx, global_idx] = item_mean_params[local_idx]
    fit_chi2 = np.asarray([item["fit_chi2"] for item in schemes], dtype=float)
    fit_dof = np.asarray([item["fit_dof"] for item in schemes], dtype=int)
    fit_q = np.asarray([item["fit_q"] for item in schemes], dtype=float)
    fit_log_gbf = np.asarray([item["fit_log_gbf"] for item in schemes], dtype=float)
    mean_fit_chi2 = np.asarray([item["mean_fit_chi2"] for item in schemes], dtype=float)
    mean_fit_dof = np.asarray([item["mean_fit_dof"] for item in schemes], dtype=int)
    mean_fit_q = np.asarray([item["mean_fit_q"] for item in schemes], dtype=float)

    resample_mode = _normalise_resample_mode(str(result.get("resample_mode", "bootstrap")))
    sample_error_mode = normalize_sample_error_mode(str(result.get("sample_error_mode", "covariance")), resample_mode=resample_mode)
    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    if fit_params.shape[1] < 2:
        fit_param_sdev = np.zeros((fit_params.shape[0], fit_params.shape[2]), dtype=float)
    else:
        fit_param_sdev = np.asarray(
            [_sample_sdev(item, resample_mode=resample_mode, sample_error_mode=sample_error_mode) for item in fit_params]
        )

    scheme_labels = np.asarray(result["scheme_labels"])
    param_samples = np.moveaxis(fit_params, 1, 0)
    fit_info_data = EnsembleData(
        ensemble=result.get("ensemble"),
        resample=resample_mode,
        values=[param_samples[idx] for idx in range(param_samples.shape[0])],
        dims=("scheme", "parameter"),
        coords={"scheme": scheme_labels.tolist(), "parameter": fit_param_labels},
        attrs={
            "method": str(result.get("method", "")),
            "order": str(result.get("order", "")),
            "observable": str(result.get("observable", "")),
            "part": str(result.get("part", "both")),
            "sample_error_mode": str(result.get("sample_error_mode", "")),
            "average_method": str(result.get("sample_error_mode", "")),
            "scheme_labels": json.dumps(scheme_labels.tolist()),
            "fit_param_labels": json.dumps(fit_param_labels),
            "fit_param_labels_by_model": json.dumps([item["fit_param_labels"] for item in schemes]),
            "fit_params": json.dumps(fit_params.tolist()),
            "fit_param_center": json.dumps(np.mean(fit_params, axis=1).tolist()),
            "fit_param_sdev": json.dumps(fit_param_sdev.tolist()),
            "fit_chi2": json.dumps(fit_chi2.tolist()),
            "fit_dof": json.dumps(fit_dof.tolist()),
            "fit_q": json.dumps(fit_q.tolist()),
            "fit_log_gbf": json.dumps(fit_log_gbf.tolist()),
            "fit_chi2_dof": json.dumps(fit_chi2_dof.tolist()),
            "fit_chi2_center": json.dumps(np.mean(fit_chi2, axis=1).tolist()),
            "fit_chi2_dof_center": json.dumps(np.mean(fit_chi2_dof, axis=1).tolist()),
            "fit_q_center": json.dumps(np.mean(fit_q, axis=1).tolist()),
            "mean_fit_params": json.dumps(mean_fit_params.tolist()),
            "mean_fit_chi2": json.dumps(mean_fit_chi2.tolist()),
            "mean_fit_dof": json.dumps(mean_fit_dof.tolist()),
            "mean_fit_q": json.dumps(mean_fit_q.tolist()),
            "mean_fit_log_gbf": json.dumps(np.asarray(result.get("fit_model_logGBF", []), dtype=float).tolist()),
            "fit_model_labels": json.dumps(np.asarray(result.get("fit_model_labels", [])).tolist()),
            "fit_model_orders": json.dumps(np.asarray(result.get("fit_model_orders", [])).tolist()),
            "fit_model_prior_widths": json.dumps(
                np.asarray(result.get("fit_model_prior_widths", []), dtype=float).tolist()
            ),
            "fit_model_weights": json.dumps(np.asarray(result.get("fit_model_weights", []), dtype=float).tolist()),
            "fit_model_mean_weights": json.dumps(
                np.asarray(result.get("fit_model_mean_weights", []), dtype=float).tolist()
            ),
            "fit_model_q": json.dumps(np.asarray(result.get("fit_model_q", []), dtype=float).tolist()),
            "fit_model_chi2_dof": json.dumps(
                np.asarray(result.get("fit_model_chi2_dof", []), dtype=float).tolist()
            ),
            "fit_model_log_gbf": json.dumps(np.asarray(result.get("fit_model_logGBF", []), dtype=float).tolist()),
            "best_fit_model_index_by_sample": json.dumps(
                np.asarray(result.get("best_fit_model_index_by_sample", []), dtype=int).tolist()
            ),
            "candidate_scheme_labels": json.dumps(np.asarray(result.get("candidate_scheme_labels", [])).tolist()),
            "candidate_scheme_fit_chi2_dof": json.dumps(
                np.asarray(result.get("candidate_scheme_fit_chi2_dof", []), dtype=float).tolist()
            ),
            "candidate_scheme_q": json.dumps(
                np.asarray(result.get("candidate_scheme_q", []), dtype=float).tolist()
            ),
            "candidate_scheme_log_gbf": json.dumps(
                np.asarray(result.get("candidate_scheme_logGBF", []), dtype=float).tolist()
            ),
            "selected_candidate_index": json.dumps(
                np.asarray(result.get("selected_candidate_index", -1), dtype=int).tolist()
            ),
            "selected_candidate_label": str(result.get("selected_candidate_label", "")),
            "selection_mode": str(result.get("selection_mode", "")),
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
    sample_error_mode: str,
) -> int:
    """Return the last nonzero-signal point plus nearby zero-compatible tail points."""
    positive_mask = np.asarray(coord, dtype=float) > 0
    re = np.asarray(re_samples, dtype=float)[:, positive_mask]
    im = np.asarray(im_samples, dtype=float)[:, positive_mask]
    re_mean, re_sdev = sample_mean_and_sdev(re, mode=resample_mode, sample_error_mode=sample_error_mode)
    im_mean, im_sdev = sample_mean_and_sdev(im, mode=resample_mode, sample_error_mode=sample_error_mode)

    magnitude = np.hypot(re_mean, im_mean)
    uncertainty = np.hypot(re_sdev, im_sdev)
    nonzero = magnitude > uncertainty
    if np.any(nonzero):
        return min(len(magnitude) - 1, int(np.flatnonzero(nonzero)[-1]) + 5)
    return min(4, len(magnitude) - 1)


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
        if item["tail_fit_success"] and np.isfinite(item["chi2_dof"]) and item["n_points"] >= 2
    ]
    if not finite:
        return 0

    chi = np.asarray([item["chi2_dof"] for item in finite], dtype=float)
    q_values = np.asarray([item["q_value"] for item in finite], dtype=float)
    best = float(np.min(chi))
    chi_limit = max(best * 1.25, best + 0.15, 1.0)
    for idx, item in enumerate(qualities):
        if not item["tail_fit_success"] or not np.isfinite(item["chi2_dof"]):
            continue
        if item["chi2_dof"] > chi_limit:
            continue
        if item["q_value"] < 0.05 and np.nanmax(q_values) >= 0.05:
            continue
        later = [
            later_item["chi2_dof"]
            for later_item in qualities[idx : min(len(qualities), idx + 3)]
            if later_item["tail_fit_success"] and np.isfinite(later_item["chi2_dof"])
        ]
        if later and max(later) <= max(chi_limit * 1.1, chi_limit + 0.1):
            return idx
    return int(np.nanargmin([item["chi2_dof"] if item["tail_fit_success"] else np.inf for item in qualities]))


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
    pz_out_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0: float,
    part: str,
    sector: str | None,
    hadron: str | None,
    preferred_zmin: float | None,
) -> list[float]:
    stable_starts = []
    required_points = _minimum_fit_points_for_parameters(
        len(_param_labels(method, order, observable, sector=sector, hadron=hadron, fit=True)),
        part,
    )
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
                pz_out_gev=pz_out_gev,
                a_fm=a_fm,
                resample_mode=resample_mode,
                sample_error_mode=sample_error_mode,
                Lambda0=Lambda0,
                part=part,
                sector=sector,
                hadron=hadron,
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
    pz_out_gev: float | None,
    a_fm: float | None,
) -> float:
    """Return the coordinate value whose lambda is eight units past the data."""
    _fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, pz_out_gev=pz_out_gev, a_fm=a_fm)
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
    pz_out_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0: float,
    part: str,
    sector: str | None,
    hadron: str | None,
) -> dict[str, Any]:
    """Fill missing scan keys with stable zmax values and tail-fit zmin diagnostics."""
    if "zmax_values" not in spec and "zmax_start" not in spec:
        end_index = int(np.clip(stable_idx, 0, len(positive) - 1))
        start_index = max(0, end_index - 4)
        spec["zmax_values"] = [float(item) for item in positive[start_index : end_index + 1]]
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
            pz_out_gev=pz_out_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            Lambda0=Lambda0,
            part=part,
            sector=sector,
            hadron=hadron,
            preferred_zmin=preferred_zmin,
        )
    if "z_ext_max" not in spec:
        spec["z_ext_max"] = _default_z_ext_max(
            coord,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_out_gev=pz_out_gev,
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
    pz_out_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0: float,
    part: str,
    sector: str | None,
    hadron: str | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a conservative scan from stable zmax and tail-fit zmin diagnostics."""
    spec = dict(existing or {})
    positive = _positive_grid(coord)
    re_axis0 = np.asarray(re_samples, dtype=float)
    im_axis0 = np.asarray(im_samples, dtype=float)
    stable_idx = _last_stable_z_index(
        coord,
        re_axis0,
        im_axis0,
        resample_mode=resample_mode,
        sample_error_mode=sample_error_mode,
    )
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
        pz_out_gev=pz_out_gev,
        a_fm=a_fm,
        resample_mode=resample_mode,
        sample_error_mode=sample_error_mode,
        Lambda0=Lambda0,
        part=part,
        sector=sector,
        hadron=hadron,
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


def _resolve_y_grid(y_grid: list[float] | dict[str, Any]) -> list[float]:
    if isinstance(y_grid, dict):
        start = float(y_grid["start"])
        stop = float(y_grid["stop"])
        if "num" in y_grid:
            num = int(y_grid["num"])
            if num < 2:
                raise ValueError("y_grid num must be at least 2")
            return np.linspace(start, stop, num).tolist()
        step = float(y_grid["step"])
        if step <= 0:
            raise ValueError("y_grid step must be positive")
        return np.arange(start, stop + 0.5 * step, step).tolist()
    return [float(item) for item in y_grid]


def _fit_model_specs(order: str | Sequence[str], prior_width: float | Sequence[float]) -> list[dict[str, Any]]:
    orders = [str(item).upper() for item in (order if isinstance(order, (list, tuple)) else [order])]
    widths = [float(item) for item in (prior_width if isinstance(prior_width, (list, tuple)) else [prior_width])]
    return [
        {
            "order": order_item,
            "prior_width": width,
            "label": f"{order_item}_prior_{width:g}".replace(".", "p"),
        }
        for order_item in orders
        for width in widths
    ]


def _apply_sample_fit_model_average(
    result: dict[str, Any],
    *,
    resample_mode: str,
    sample_error_mode: str,
    model_average: bool,
) -> None:
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    log_gbf = np.asarray([item["fit_log_gbf"] for item in result["scheme_results"]], dtype=float)
    q_values = np.asarray([item["fit_q"] for item in result["scheme_results"]], dtype=float)
    tail_fit_success_mask = np.asarray(
        [item.get("tail_fit_success_samples", np.isfinite(item["fit_log_gbf"])) for item in result["scheme_results"]],
        dtype=bool,
    )
    tail_fit_success_mask &= np.isfinite(log_gbf)
    weights = np.zeros_like(log_gbf, dtype=float)
    if model_average:
        for sample in range(log_gbf.shape[1]):
            valid = tail_fit_success_mask[:, sample]
            if not np.any(valid):
                candidates = np.flatnonzero(np.isfinite(q_values[:, sample]))
                if candidates.size:
                    weights[candidates[int(np.argmax(q_values[candidates, sample]))], sample] = 1.0
                continue
            values = log_gbf[:, sample]
            shifted = np.exp(values[valid] - np.max(values[valid]))
            weights[valid, sample] = shifted / np.sum(shifted)
    else:
        for sample in range(log_gbf.shape[1]):
            passing = np.flatnonzero(tail_fit_success_mask[:, sample] & (q_values[:, sample] >= 0.05))
            if passing.size:
                best = passing[int(np.argmax(log_gbf[passing, sample]))]
            else:
                candidates = np.flatnonzero(np.isfinite(q_values[:, sample]))
                best = candidates[int(np.argmax(q_values[candidates, sample]))] if candidates.size else 0
            weights[best, sample] = 1.0

    final_re = np.sum(weights[:, :, None] * ft_re, axis=0)
    final_im = np.sum(weights[:, :, None] * ft_im, axis=0)
    re_sys_by_sample = np.sqrt(np.sum(weights[:, :, None] * (ft_re - final_re[None, :, :]) ** 2, axis=0))
    im_sys_by_sample = np.sqrt(np.sum(weights[:, :, None] * (ft_im - final_im[None, :, :]) ** 2, axis=0))

    result["final_ft_re_samples"] = final_re
    result["final_ft_im_samples"] = final_im
    result["ft_re_mean"], result["ft_re_stat_sdev"] = sample_mean_and_sdev(
        final_re,
        mode=resample_mode,
        sample_error_mode=sample_error_mode,
    )
    result["ft_im_mean"], result["ft_im_stat_sdev"] = sample_mean_and_sdev(
        final_im,
        mode=resample_mode,
        sample_error_mode=sample_error_mode,
    )
    result["ft_re_sys_sdev"] = np.mean(re_sys_by_sample, axis=0)
    result["ft_im_sys_sdev"] = np.mean(im_sys_by_sample, axis=0)
    result["fit_model_weights"] = weights.tolist()
    result["fit_model_mean_weights"] = np.mean(weights, axis=1).tolist()
    result["best_fit_model_index_by_sample"] = np.argmax(weights, axis=0).astype(int).tolist()


def _apply_fourier_output_scale(result: dict[str, Any], output_scale: float) -> None:
    """Scale Fourier-space outputs without changing coordinate-space fits."""
    scale = float(output_scale)
    if scale == 1.0:
        result["output_scale"] = scale
        return
    for key in (
        "ft_re_samples",
        "ft_im_samples",
        "final_ft_re_samples",
        "final_ft_im_samples",
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
    y_grid: list[float] | dict[str, Any],
    scheme_scan: dict[str, Any] | None = None,
    method: str = "GI",
    order: str | list[str] = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_out_gev: float | None = None,
    a_fm: float | None = None,
    im_flip_for_ft: bool = False,
    Lambda0: float = 0.1,
    posterior_prior_error_scale: float | list[float] = 3.0,
    sample_error_mode: str = "covariance",
    part: str = "both",
    output_scale: float = 1.0,
    sector: str | None = None,
    target_observable: str | None = None,
    hadron: str | None = None,
    save_path: str | None = None,
    plot_fourier: dict[str, Any] | None = None,
    plot_extension: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Run local extrapolation and Fourier transform for loaded samples."""
    out = "fourier_result"
    sector = None if sector is None else str(sector).strip().lower()
    target = str(target_observable or "").strip().lower()
    if sector is not None:
        if not target:
            observable_name = str(observable).strip().lower()
            target = "da" if observable_name == "meson_quasi_da" else "gpd" if "gpd" in observable_name else "pdf"
        if target == "da" or (target == "gpd" and sector != "sea"):
            sector = "full"
        if target == "pdf":
            if sector == "valence":
                part, output_scale, im_flip_for_ft = "re", 2.0, False
            elif sector == "total":
                part, output_scale, im_flip_for_ft = "im", 2.0, True
            elif sector == "full":
                part, output_scale, im_flip_for_ft = "both", 1.0, False
            elif sector == "sea":
                part, output_scale, im_flip_for_ft = "re", 1.0, False
        else:
            part, output_scale, im_flip_for_ft = "both", 1.0, False
    else:
        sector = {"re": "valence", "im": "total", "both": "full"}.get(str(part).lower(), str(part).lower())
        if not target:
            target = str(target_observable or "")
    matrix_element_data = store.get("matrix_element_data")
    if matrix_element_data is None:
        matrix_element_data = store["input"]
        store["matrix_element_data"] = matrix_element_data
    resample_mode = _normalise_resample_mode(getattr(matrix_element_data, "resample", "bootstrap"))
    sample_error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    matrix_element = ensemble_data_to_legacy_arrays(matrix_element_data)
    auto_scheme_scan = None
    coord_arr = np.asarray(matrix_element["coord"], dtype=float)
    range_order = str(order[0] if isinstance(order, list) else order).upper()
    range_prior_width = float(
        posterior_prior_error_scale[0] if isinstance(posterior_prior_error_scale, list) else posterior_prior_error_scale
    )
    scan_spec = _fill_scheme_defaults(dict(scheme_scan or {}))
    if not _scan_has_all_range_keys(scan_spec):
        scan_spec = _auto_scheme_scan(
            coord=coord_arr,
            re_samples=np.asarray(matrix_element["re_samples"], dtype=float),
            im_samples=np.asarray(matrix_element["im_samples"], dtype=float),
            coord_unit=coord_unit,
            method=method,
            order=range_order,
            observable=observable,
            pz_gev=pz_gev,
            pz_out_gev=pz_out_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            Lambda0=float(Lambda0),
            part=part,
            sector=sector,
            hadron=hadron,
            existing=scan_spec,
        )
        auto_scheme_scan = scan_spec
    scheme_scan = scan_spec
    schemes = _generate_scan_schemes(scheme_scan)
    required_points = _minimum_fit_points_for_parameters(
        len(_param_labels(method, range_order, observable, sector=sector, hadron=hadron, fit=True)),
        part,
    )
    schemes = [
        scheme
        for scheme in schemes
        if np.count_nonzero((coord_arr >= float(scheme["zmin"])) & (coord_arr <= float(scheme["zmax"])) & (coord_arr > 0))
        >= required_points
    ]
    if not schemes:
        raise ValueError("scheme_scan produced no valid zmin/zmax combinations")
    y_values = _resolve_y_grid(y_grid)
    model_average = bool(scheme_scan.get("model_average", True))
    candidate_labels = [str(scheme.get("label", f"scheme_{idx}")) for idx, scheme in enumerate(schemes)]
    candidate_qualities = []
    for scheme in schemes:
        candidate_qualities.append(
            fit_tail_quality_for_mean(
                matrix_element["coord"],
                matrix_element["re_samples"],
                matrix_element["im_samples"],
                zmin=float(scheme["zmin"]),
                zmax=float(scheme["zmax"]),
                method=method,
                order=range_order,
                observable=observable,
                coord_unit=coord_unit,
                pz_gev=pz_gev,
                pz_out_gev=pz_out_gev,
                a_fm=a_fm,
                resample_mode=resample_mode,
                Lambda0=float(Lambda0),
                posterior_prior_error_scale=range_prior_width,
                sample_error_mode=sample_error_mode,
                part=part,
                sector=sector,
                hadron=hadron,
            )
        )
    candidate_chi2 = [float(item["chi2_dof"]) for item in candidate_qualities]
    candidate_q = [float(item["q_value"]) for item in candidate_qualities]
    candidate_log_gbf = [float(item["logGBF"]) for item in candidate_qualities]
    candidate_diagnostics = {
        "selection_mode": "sample_range_then_sample_fit_model_average" if model_average else "sample_range_then_sample_best_fit_model",
    }
    passing_idx = [
        idx
        for idx, item in enumerate(candidate_qualities)
        if item["tail_fit_success"] and np.isfinite(item["logGBF"]) and float(item["q_value"]) >= 0.05
    ]
    if passing_idx:
        best_candidate = max(passing_idx, key=lambda idx: candidate_qualities[idx]["logGBF"])
    else:
        best_candidate = max(
            range(len(candidate_qualities)),
            key=lambda idx: candidate_qualities[idx]["q_value"] if candidate_qualities[idx]["tail_fit_success"] else float("-inf"),
        )
    selected_range = dict(schemes[best_candidate])
    fit_model_specs = []
    for spec in _fit_model_specs(order, posterior_prior_error_scale):
        n_model_params = len(_param_labels(method, spec["order"], observable, sector=sector, hadron=hadron, fit=True))
        n_model_points = np.count_nonzero(
            (coord_arr >= float(selected_range["zmin"])) & (coord_arr <= float(selected_range["zmax"])) & (coord_arr > 0)
        )
        if n_model_points >= _minimum_fit_points_for_parameters(n_model_params, part):
            fit_model_specs.append(spec)
    if not fit_model_specs:
        fit_model_specs = [_fit_model_specs(range_order, range_prior_width)[0]]
    candidate_diagnostics.update(
        {
            "candidate_scheme_labels": candidate_labels,
            "candidate_scheme_fit_chi2_dof": candidate_chi2,
            "candidate_scheme_q": candidate_q,
            "candidate_scheme_logGBF": candidate_log_gbf,
            "selected_candidate_index": best_candidate,
            "selected_candidate_label": candidate_labels[best_candidate],
            "selected_range_label": candidate_labels[best_candidate],
            "selected_fit_range": [float(selected_range["zmin"]), float(selected_range["zmax"])],
            "fit_model_labels": [spec["label"] for spec in fit_model_specs],
            "fit_model_orders": [spec["order"] for spec in fit_model_specs],
            "fit_model_prior_widths": [spec["prior_width"] for spec in fit_model_specs],
        }
    )
    schemes = []
    for spec in fit_model_specs:
        model_scheme = dict(selected_range)
        model_scheme["label"] = spec["label"]
        model_scheme["order"] = spec["order"]
        model_scheme["posterior_prior_error_scale"] = spec["prior_width"]
        schemes.append(model_scheme)
    if sector == "sea" and target == "pdf":
        total_result = run_fourier_workflow(
            matrix_element["coord"],
            matrix_element["re_samples"],
            matrix_element["im_samples"],
            y_values,
            schemes=schemes,
            method=method,
            order=order,
            observable=observable,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_out_gev=pz_out_gev,
            a_fm=a_fm,
            im_flip_for_ft=True,
            resample_mode=resample_mode,
            Lambda0=float(Lambda0),
            posterior_prior_error_scale=range_prior_width,
            sample_error_mode=sample_error_mode,
            part="im",
            sector=sector,
            hadron=hadron,
        )
        valence_result = run_fourier_workflow(
            matrix_element["coord"],
            matrix_element["re_samples"],
            matrix_element["im_samples"],
            y_values,
            schemes=schemes,
            method=method,
            order=order,
            observable=observable,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_out_gev=pz_out_gev,
            a_fm=a_fm,
            im_flip_for_ft=False,
            resample_mode=resample_mode,
            Lambda0=float(Lambda0),
            posterior_prior_error_scale=range_prior_width,
            sample_error_mode=sample_error_mode,
            part="re",
            sector=sector,
            hadron=hadron,
        )
        for sector_result in (total_result, valence_result):
            sector_result.update(candidate_diagnostics)
            _apply_sample_fit_model_average(
                sector_result,
                resample_mode=resample_mode,
                sample_error_mode=sample_error_mode,
                model_average=model_average,
            )
            _apply_fourier_output_scale(sector_result, 2.0)
        result = total_result
        for key in ("ft_re_samples", "ft_im_samples", "final_ft_re_samples", "final_ft_im_samples", "ft_re_mean", "ft_im_mean"):
            result[key] = 0.5 * (np.asarray(total_result[key], dtype=float) - np.asarray(valence_result[key], dtype=float))
        result["ft_re_mean"], result["ft_re_stat_sdev"] = sample_mean_and_sdev(
            np.asarray(result["final_ft_re_samples"], dtype=float),
            mode=resample_mode,
            sample_error_mode=sample_error_mode,
        )
        result["ft_im_mean"], result["ft_im_stat_sdev"] = sample_mean_and_sdev(
            np.asarray(result["final_ft_im_samples"], dtype=float),
            mode=resample_mode,
            sample_error_mode=sample_error_mode,
        )
        result["ft_re_sys_sdev"] = 0.5 * np.hypot(
            np.asarray(total_result["ft_re_sys_sdev"], dtype=float),
            np.asarray(valence_result["ft_re_sys_sdev"], dtype=float),
        )
        result["ft_im_sys_sdev"] = 0.5 * np.hypot(
            np.asarray(total_result["ft_im_sys_sdev"], dtype=float),
            np.asarray(valence_result["ft_im_sys_sdev"], dtype=float),
        )
        for total_scheme, valence_scheme in zip(result["scheme_results"], valence_result["scheme_results"]):
            for key in ("extended_re_samples", "extended_im_samples"):
                total_scheme[key] = np.asarray(total_scheme[key], dtype=float) - np.asarray(valence_scheme[key], dtype=float)
            for key in ("ft_re_samples", "ft_im_samples"):
                total_scheme[key] = 0.5 * (np.asarray(total_scheme[key], dtype=float) - np.asarray(valence_scheme[key], dtype=float))
        part, output_scale, im_flip_for_ft = "sea", 1.0, False
    else:
        result = run_fourier_workflow(
            matrix_element["coord"],
            matrix_element["re_samples"],
            matrix_element["im_samples"],
            y_values,
            schemes=schemes,
            method=method,
            order=order,
            observable=observable,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_out_gev=pz_out_gev,
            a_fm=a_fm,
            im_flip_for_ft=im_flip_for_ft,
            resample_mode=resample_mode,
            Lambda0=float(Lambda0),
            posterior_prior_error_scale=range_prior_width,
            sample_error_mode=sample_error_mode,
            part=part,
            sector=sector,
            hadron=hadron,
        )
    result["resample_mode"] = resample_mode
    result["sample_error_mode"] = sample_error_mode
    result["pz_gev"] = pz_gev
    result["pz_out_gev"] = pz_out_gev
    result["a_fm"] = a_fm
    result["im_flip_for_ft"] = bool(im_flip_for_ft)
    result["sector"] = sector
    result["target_observable"] = target
    result["Lambda0"] = float(Lambda0)
    result["posterior_prior_error_scale"] = (
        range_prior_width
        if len(candidate_diagnostics["fit_model_prior_widths"]) == 1
        else candidate_diagnostics["fit_model_prior_widths"]
    )
    result["part"] = str(part)
    result["ensemble"] = matrix_element_data.ensemble
    result.update(candidate_diagnostics)
    if auto_scheme_scan is not None:
        result["auto_scheme_scan"] = auto_scheme_scan
    if not (sector == "sea" and target == "pdf"):
        _apply_sample_fit_model_average(
            result,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            model_average=model_average,
        )
        _apply_fourier_output_scale(result, float(output_scale))
    store["fourier_result_data"] = fourier_result_to_ensemble_data(result)
    store[out] = result
    artifact = _artifact_path(save_path, default_name=f"{out}.nc", artifacts_dir=artifacts_dir).with_suffix(".nc")
    fit_info_artifact = _artifact_path(None, default_name=f"{artifact.stem}_fit_info.nc", artifacts_dir=artifacts_dir)
    store["fourier_result_data"].to_netcdf(artifact)
    _save_fourier_fit_info_netcdf(fit_info_artifact, result)
    result["artifact"] = str(artifact)
    result["fit_info_artifact"] = str(fit_info_artifact)
    summary = summarize_fourier_result(store)
    plot_kwargs = dict(plot_fourier or {})
    plot_kwargs.setdefault("artifact_path", str(artifact))
    extension_kwargs = dict(plot_extension or {})
    plot = plot_fourier_result(store, artifacts_dir=artifacts_dir, **plot_kwargs)
    extension_plot = plot_fourier_extension_quality_result(store, artifacts_dir=artifacts_dir, **extension_kwargs)
    report_result = {}
    if isinstance(report, dict) and report.get("enabled"):
        report_kwargs = dict(report)
        report_kwargs.pop("enabled", None)
        report_result = report_fourier_result(store, artifacts_dir=artifacts_dir, **report_kwargs)
    store["output"] = store["fourier_result_data"]
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
        "report": report_result.get("report"),
        "n_schemes": int(result["ft_re_samples"].shape[0]),
        "n_samples": int(np.asarray(result["final_ft_re_samples"]).shape[0]),
        "n_y": int(np.asarray(result["final_ft_re_samples"]).shape[1]),
        "scheme_labels": result["scheme_labels"],
        "fit_model_labels": result.get("fit_model_labels", []),
        "fit_model_mean_weights": result.get("fit_model_mean_weights", []),
        "fit_failures": result["fit_failures"],
        "selected_range_label": result.get("selected_range_label"),
        "output_scale": result.get("output_scale", 1.0),
        "sector": result.get("sector", sector),
        "auto_scheme_scan": auto_scheme_scan,
    }


def summarize_fourier_result(
    store: dict[str, Any],
) -> dict[str, Any]:
    """Store and return a compact numerical summary of the Fourier result."""
    out = "fourier_summary"
    data = store["fourier_result"]
    summary = {
        "y_grid": np.asarray(data["y_grid"]).tolist(),
        "ft_re_mean": np.asarray(data["ft_re_mean"]).tolist(),
        "ft_im_mean": np.asarray(data["ft_im_mean"]).tolist(),
        "ft_re_stat_sdev": np.asarray(data["ft_re_stat_sdev"]).tolist(),
        "ft_im_stat_sdev": np.asarray(data["ft_im_stat_sdev"]).tolist(),
        "ft_re_sys_sdev": np.asarray(data["ft_re_sys_sdev"]).tolist(),
        "ft_im_sys_sdev": np.asarray(data["ft_im_sys_sdev"]).tolist(),
        "scheme_labels": list(data["scheme_labels"]),
        "fit_failures": list(data["fit_failures"]),
        "fit_model_labels": list(data.get("fit_model_labels", [])),
        "fit_model_mean_weights": list(data.get("fit_model_mean_weights", [])),
        "fit_model_orders": list(data.get("fit_model_orders", [])),
        "fit_model_prior_widths": list(data.get("fit_model_prior_widths", [])),
        "fit_model_chi2_dof": list(data.get("fit_model_chi2_dof", [])),
        "fit_model_logGBF": list(data.get("fit_model_logGBF", [])),
        "selected_range_label": data.get("selected_range_label"),
        "selected_fit_range": data.get("selected_fit_range"),
        "fit_info_artifact": data.get("fit_info_artifact"),
        "output_scale": data.get("output_scale", 1.0),
        "sector": data.get("sector", data.get("part", "full")),
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
    svg_output = _svg_companion_path(output)
    fig.savefig(svg_output, bbox_inches="tight")
    plt.close(fig)
    result = {"plot": str(output), "plot_image": str(svg_output), "source": str(source)}
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
        scheme_index = 0
    if title is not None and title.strip().lower() in {"fourier extension quality", "lambda extrapolation"}:
        title = None
    re_output = _artifact_path(save_path, default_name="fourier_extension_re.pdf", artifacts_dir=artifacts_dir)
    im_output = re_output.with_name(f"{re_output.stem}_im.pdf")
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
    re_svg_output = _svg_companion_path(re_output)
    fig.savefig(re_svg_output, bbox_inches="tight")
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
    im_svg_output = _svg_companion_path(im_output)
    fig.savefig(im_svg_output, bbox_inches="tight")
    plt.close(fig)
    scheme_label = data["scheme_labels"][scheme_index]
    result = {
        "plot_re": str(re_output),
        "plot_im": str(im_output),
        "plot_re_image": str(re_svg_output),
        "plot_im_image": str(im_svg_output),
        "scheme_label": scheme_label,
    }
    store["fourier_extension_plot"] = result
    return result


def report_fourier_result(
    store: dict[str, Any],
    *,
    save_path: str | None = None,
    artifacts_dir: str | None = None,
    report_language: str = "en",
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
        report_language=report_language,
    )
    report = {
        "report": str(paths["report"]),
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
