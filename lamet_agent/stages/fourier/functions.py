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
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
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

from lamet_agent.core.data import EnsembleData, EnsembleInfo, HBAR_C_GEV_FM
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
from lamet_agent.stages.fourier.validation import (
    INFERRED_OBSERVABLES,
    PUBLIC_OBSERVABLES,
    quasi_y_ls_error,
    resolve_grid_spec,
)

FM_TO_GEV_INV = 1.0 / HBAR_C_GEV_FM
_FOURIER_SAMPLE_EXECUTOR: ContextVar[ProcessPoolExecutor | None] = ContextVar(
    "fourier_sample_executor",
    default=None,
)

POLARIZED_OBSERVABLES = {
    (f"{hadron}_quark_quasi_{target}", polarization): f"{hadron}_quark_{polarization}_quasi_{target}"
    for hadron in ("pion", "nucleon")
    for target in ("pdf", "gpd")
    for polarization in ("unpolarized", "helicity", "transversity")
}
POLARIZED_OBSERVABLES.update(
    {
        ("pion_gluon_quasi_pdf", "unpolarized"): "pion_gluon_unpolarized_quasi_pdf",
        ("nucleon_gluon_quasi_pdf", "unpolarized"): "nucleon_gluon_unpolarized_quasi_pdf",
    }
)

OBSERVABLE_BACKENDS = {
    "pion_quark_unpolarized_quasi_pdf": "pion_quark_quasi_pdf",
    "pion_quark_helicity_quasi_pdf": "pion_quark_quasi_pdf",
    "pion_quark_transversity_quasi_pdf": "pion_quark_quasi_pdf",
    "nucleon_quark_unpolarized_quasi_pdf": "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_quark_helicity_quasi_pdf": "nucleon_quark_unpolarized_quasi_pdf",
    "nucleon_quark_transversity_quasi_pdf": "nucleon_quark_transversity_quasi_pdf",
    "pion_gluon_unpolarized_quasi_pdf": "pion_gluon_quasi_pdf",
    "nucleon_gluon_unpolarized_quasi_pdf": "nucleon_gluon_quasi_pdf",
    "meson_quasi_da": "meson_quasi_da",
    "pion_quark_unpolarized_quasi_gpd": "pion_quark_quasi_gpd",
    "pion_quark_helicity_quasi_gpd": "pion_quark_quasi_gpd",
    "pion_quark_transversity_quasi_gpd": "pion_quark_quasi_gpd",
    "nucleon_quark_unpolarized_quasi_gpd": "nucleon_quark_quasi_gpd",
    "nucleon_quark_helicity_quasi_gpd": "nucleon_quark_quasi_gpd",
    "nucleon_quark_transversity_quasi_gpd": "nucleon_quark_quasi_gpd",
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


def _nearest_positive_coord(coord: np.ndarray, target: float) -> float:
    """Return the nearest strictly positive coordinate to ``target``."""
    positive = np.asarray(coord, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        raise ValueError("z grid has no positive coordinates to snap onto")
    return float(positive[int(np.argmin(np.abs(positive - float(target))))])


def _snap_fit_interval(coord: np.ndarray, zmin: float, zmax: float) -> tuple[float, float]:
    """Snap a fit window onto existing positive coordinates so inclusive masks cannot drop a site."""
    zmin_snap = _nearest_positive_coord(coord, zmin)
    zmax_snap = _nearest_positive_coord(coord, zmax)
    if zmax_snap < zmin_snap:
        zmin_snap, zmax_snap = zmax_snap, zmin_snap
    return zmin_snap, zmax_snap


def _fit_coord_mask(coord: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Inclusive mask on ``coord`` after snapping ``zmin``/``zmax`` onto the grid."""
    coord_arr = np.asarray(coord, dtype=float)
    zmin_snap, zmax_snap = _snap_fit_interval(coord_arr, zmin, zmax)
    return (coord_arr >= zmin_snap) & (coord_arr <= zmax_snap) & (coord_arr > 0)


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
    k = np.asarray(output_k)
    diffs = np.diff(x)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        pref = abs(float(diffs[0])) / (2 * np.pi)
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

    weights = np.empty_like(x, dtype=float)
    weights[0] = abs(float(x[1] - x[0])) / 2.0
    weights[-1] = abs(float(x[-1] - x[-2])) / 2.0
    weights[1:-1] = np.abs(x[2:] - x[:-2]) / 2.0
    pref = weights / (2 * np.pi)

    if k.ndim == 0:
        phase = x * k
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        val_re = np.sum(pref * cos_phase * fx_re) - np.sum(pref * sin_phase * fx_im)
        val_im = np.sum(pref * sin_phase * fx_re) + np.sum(pref * cos_phase * fx_im)
        return val_re, val_im

    phase = np.multiply.outer(x, k)
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    val_re = np.sum(pref[:, None] * cos_phase * fx_re[:, None], axis=0) - np.sum(
        pref[:, None] * sin_phase * fx_im[:, None], axis=0
    )
    val_im = np.sum(pref[:, None] * sin_phase * fx_re[:, None], axis=0) + np.sum(
        pref[:, None] * cos_phase * fx_im[:, None], axis=0
    )
    return val_re, val_im


def _project_da_symmetry(
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    *,
    phase_scale: float,
) -> np.ndarray:
    """Project ``exp(+i lambda/2) h`` to real values and rotate back."""
    phase = np.exp(0.5j * np.asarray(coord, dtype=float) * float(phase_scale))[None, :]
    rotated = (
        np.asarray(re_samples, dtype=float) + 1j * np.asarray(im_samples, dtype=float)
    ) * phase
    return np.real(rotated) * np.conjugate(phase)


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


def _ft_scale_momentum(momentum_gev: float | None, final_momentum_gev: float | None = None) -> float:
    """Return the boost used for coord->lambda scaling.

    NonBreit quasi-GPD inputs provide both momenta and use the average hadron
    momentum. Forward/PDF/DA inputs provide one momentum and are unchanged.
    """
    if final_momentum_gev is None:
        return abs(float(momentum_gev or 0.0))
    return abs((float(momentum_gev or 0.0) + float(final_momentum_gev)) / 2.0)


def _canonical_observable(observable: str) -> str:
    key = observable.lower().replace("-", "_").replace(" ", "_")
    if key not in PUBLIC_OBSERVABLES and key not in OBSERVABLE_BACKENDS:
        allowed = ", ".join(sorted(PUBLIC_OBSERVABLES))
        raise ValueError(f"observable must be one of: {allowed}")
    return key


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
    backend = OBSERVABLE_BACKENDS.get(observable, observable)
    if backend not in QUARK_LIKE_TERMS:
        raise ValueError(f"unsupported quark-like observable {observable!r}")
    return QUARK_LIKE_TERMS[backend]


def _quark_like_phase_scales(
    observable: str,
    *,
    phase_scale: float,
    phase_prime_scale: float | None,
) -> tuple[float, ...]:
    observable = _canonical_observable(observable)
    observable = OBSERVABLE_BACKENDS.get(observable, observable)
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
    Lambda0_gev: float,
) -> list[_TailParameter]:
    parameters = [
        *parameters,
        _TailParameter("m", max(0.5 - float(Lambda0_gev), 0.05), 0.0),
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
    psi1_flavor_class: str | None = None,
    psi2_flavor_class: str | None = None,
) -> list[_TailParameter]:
    observable = _canonical_observable(observable)
    canonical_observable = observable
    observable = OBSERVABLE_BACKENDS.get(observable, observable)
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
    psi1_class = str(psi1_flavor_class or "heavy").lower()
    psi2_class = str(psi2_flavor_class or "heavy").lower()
    fixed = {}
    aliases = {}
    if canonical_observable == "pion_quark_unpolarized_quasi_pdf" and sector == "valence":
        fixed.update({"phi2": 0.0, "phi2p": 0.0})
        aliases.update({"A3": ("A1", 1.0), "phi3": ("phi1", -1.0), "A3p": ("A1p", 1.0), "phi3p": ("phi1p", -1.0)})
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "light":
        aliases.update({"A2": ("A1", 1.0), "phi2": ("phi1", -1.0), "A2p": ("A1p", 1.0), "phi2p": ("phi1p", -1.0)})
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "heavy":
        fixed.update({"A1": 0.0, "phi1": 0.0, "A1p": 0.0, "phi1p": 0.0})
    if observable == "meson_quasi_da" and psi1_class == "heavy" and psi2_class == "light":
        fixed.update({"A2": 0.0, "phi2": 0.0, "A2p": 0.0, "phi2p": 0.0})
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
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
) -> list[_TailParameter]:
    observable = _canonical_observable(observable)
    backend = OBSERVABLE_BACKENDS.get(observable, observable)
    if backend == "nucleon_gluon_quasi_pdf":
        return _nucleon_gluon_parameters(order)
    if backend == "pion_gluon_quasi_pdf":
        return _pion_gluon_parameters(order)
    return _quark_like_parameters(
        order,
        observable,
        sector=sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
    )


def _param_template(
    method: str,
    order: str,
    observable: str,
    *,
    Lambda0_gev: float = 0.0,
    sector: str | None = None,
    hadron: str | None = None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
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
        _observable_parameters(
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        ),
        method=method,
        Lambda0_gev=float(Lambda0_gev),
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
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
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
        _observable_parameters(
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        ),
        method=method,
        Lambda0_gev=0.0,
    )
    if not fit:
        return [item.label for item in parameters]
    labels = []
    for item in parameters:
        if item.fixed is None and item.fit_label is None and item.label not in labels:
            labels.append(item.label)
    return labels


def _decay_tail(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    lambda_index: int,
    method: str,
    Lambda0_gev: float,
) -> Any:
    tail = gv.exp(-(params[lambda_index] + float(Lambda0_gev)) * z)
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
    Lambda0_gev: float,
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

    tail = _decay_tail(z, params, lambda_index=cursor, method=method, Lambda0_gev=Lambda0_gev)
    return re * tail, im * tail


def _nucleon_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    Lambda0_gev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F nucleon gluon form: (A z [+ A']) exp(-Lambda z) with zero Im."""
    # LA:  A z exp(-Lambda z)
    # NLA: (A z + A') exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1]
        lambda_index = 2
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method, Lambda0_gev=Lambda0_gev)
    im = np.zeros_like(z, dtype=object)
    return re * tail, im * tail


def _pion_gluon_asymptotic_values(
    z: np.ndarray,
    params: Sequence[Any],
    *,
    method: str,
    order: str,
    phase_scale: float,
    Lambda0_gev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix-F pion gluon form: A2 z plus optional constant and cosine term."""
    # LA:  A2 z exp(-Lambda z)
    # NLA: (A2 z + A2' + 2 A1 cos(phi - Pz z)) exp(-Lambda z)
    re = params[0] * z
    lambda_index = 1
    if order.upper() == "NLA":
        re = re + params[1] + 2.0 * params[2] * gv.cos(params[3] - phase_scale * z)
        lambda_index = 4
    tail = _decay_tail(z, params, lambda_index=lambda_index, method=method, Lambda0_gev=Lambda0_gev)
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
    Lambda0_gev: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("asymptotic form requires positive coordinates")

    observable = _canonical_observable(observable)
    backend = OBSERVABLE_BACKENDS.get(observable, observable)
    if backend == "nucleon_gluon_quasi_pdf":
        return _nucleon_gluon_asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            Lambda0_gev=Lambda0_gev,
        )

    if backend == "pion_gluon_quasi_pdf":
        return _pion_gluon_asymptotic_values(
            z,
            params,
            method=method,
            order=order,
            phase_scale=phase_scale,
            Lambda0_gev=Lambda0_gev,
        )

    return _quark_like_asymptotic_values(
        z,
        params,
        method=method,
        order=order,
        observable=observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        Lambda0_gev=Lambda0_gev,
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
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    p0: np.ndarray | None = None,
    prior: gv.BufferDict | None = None,
    Lambda0_gev: float = 0.0,
) -> tuple[np.ndarray, gv.BufferDict | None, gv.BufferDict | None, bool, float, int, float, float]:
    default_p0, _ = _param_template(
        method,
        order,
        observable,
        Lambda0_gev=Lambda0_gev,
        sector=sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
    )
    fit_p0, bounds = _param_template(
        method,
        order,
        observable,
        Lambda0_gev=Lambda0_gev,
        sector=sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        fit=True,
    )
    start = default_p0 if p0 is None else np.asarray(p0, dtype=float)
    fit_labels = _param_labels(
        method,
        order,
        observable,
        sector=sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        fit=True,
    )
    full_items = _with_method_tail_parameters(
        _observable_parameters(
            order.upper(),
            _canonical_observable(observable),
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        ),
        method=method.upper(),
        Lambda0_gev=float(Lambda0_gev),
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
            Lambda0_gev=Lambda0_gev,
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
    momentum_gev: float | None = None,
    final_momentum_gev: float | None = None,
    resample_mode: str = "bootstrap",
    Lambda0_gev: float = 0.0,
    posterior_prior_error_scale: float = 3.0,
    sample_error_mode: str = "covariance",
    part: str = "both",
    sector: str | None = None,
    hadron: str | None = None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
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
    fit_coord = coord_arr * FM_TO_GEV_INV
    phase_scale = float(momentum_gev or 0.0)
    phase_prime_scale = None if final_momentum_gev is None else float(final_momentum_gev)
    fit_mask = _fit_coord_mask(coord_arr, zmin, zmax)
    n_points = int(np.count_nonzero(fit_mask))
    n_params = len(
        _param_labels(
            method,
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            fit=True,
        )
    )
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
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        Lambda0_gev=Lambda0_gev,
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
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            p0=mean_params,
            prior=_scaled_internal_prior(mean_pmean, mean_psdev, posterior_prior_error_scale),
            Lambda0_gev=Lambda0_gev,
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


def _sample_batches(n_samples: int, workers: int) -> list[list[int]]:
    """Split sample indices into at most ``workers`` non-empty batches."""
    n_batches = min(int(workers), int(n_samples))
    return [batch.tolist() for batch in np.array_split(np.arange(n_samples), n_batches) if batch.size]


def _fit_fourier_sample_batch(payload: bytes, sample_indices: list[int]) -> list[dict[str, Any]]:
    """Fit one batch of Fourier samples in a worker process."""
    context = gv.loads(payload)
    results: list[dict[str, Any]] = []
    for sample in sample_indices:
        sample_y_data = _fit_y_data(
            context["re_fit_samples"][sample],
            context["im_fit_samples"][sample],
            sample_error_mode=context["sample_error_mode"],
            resample_mode=context["resample_mode"],
            part=context["part"],
            re_fit_samples=context["re_fit_samples"],
            im_fit_samples=context["im_fit_samples"],
            sigma_re=context["sigma_re"],
            sigma_im=context["sigma_im"],
        )
        params, _pmean, _psdev, success, chi2, dof, q_value, log_gbf = _fit_one_sample(
            context["z_fit"],
            y_data=sample_y_data,
            method=context["method"],
            order=context["order"],
            observable=context["observable"],
            part=context["part"],
            phase_scale=context["phase_scale"],
            phase_prime_scale=context["phase_prime_scale"],
            sector=context["sector"],
            hadron=context["hadron"],
            psi1_flavor_class=context["psi1_flavor_class"],
            psi2_flavor_class=context["psi2_flavor_class"],
            p0=context["mean_params"],
            prior=context["sample_prior"],
            Lambda0_gev=context["Lambda0_gev"],
        )
        results.append(
            {
                "sample": sample,
                "params": params,
                "success": bool(success),
                "chi2": float(chi2),
                "dof": int(dof),
                "q_value": float(q_value),
                "log_gbf": float(log_gbf),
            }
        )
    return results


def _with_fourier_sample_executor(func):
    """Own one process pool for the full Fourier tool invocation."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        value = kwargs.get("workers", 1)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError("workers must be a positive integer")
        workers = int(value)
        kwargs["workers"] = workers
        if workers == 1:
            return func(*args, **kwargs)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            token = _FOURIER_SAMPLE_EXECUTOR.set(executor)
            try:
                return func(*args, **kwargs)
            finally:
                _FOURIER_SAMPLE_EXECUTOR.reset(token)

    return wrapped


def _scheme_ranges(scheme: dict[str, Any], coord: np.ndarray) -> tuple[float, float, float]:
    positive = coord[coord > 0]
    zmin = float(scheme.get("zmin", positive[0]))
    zmax = float(scheme.get("zmax", coord[-1]))
    zmin, zmax = _snap_fit_interval(coord, zmin, zmax)
    z_ext_max = float(scheme["z_ext_max"])
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
    im_flip_for_ft: bool,
    phase_scale: float,
    phase_prime_scale: float | None,
    resample_mode: str,
    Lambda0_gev: float,
    posterior_prior_error_scale: float,
    sample_error_mode: str,
    part: str,
    sector: str | None,
    hadron: str | None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    executor: ProcessPoolExecutor | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    zmin, zmax, z_ext_max = _scheme_ranges(scheme, coord)
    label = str(scheme.get("label", f"{method}_{order}_{zmin}_{zmax}"))
    zmin_fit = zmin * FM_TO_GEV_INV
    zmax_fit = zmax * FM_TO_GEV_INV
    z_ext_fit_max = z_ext_max * FM_TO_GEV_INV

    if zmin_fit <= 0:
        raise ValueError("zmin must be positive; asymptotic forms are singular at zero")
    if zmax_fit <= zmin_fit:
        raise ValueError("zmax must be larger than zmin")
    if z_ext_fit_max < zmax_fit:
        raise ValueError("z_ext_max must be >= zmax")

    fit_mask = _fit_coord_mask(coord, zmin, zmax)
    n_params = len(
        _param_labels(
            method,
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        )
    )
    n_fit_params = len(
        _param_labels(
            method,
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            fit=True,
        )
    )
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
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        Lambda0_gev=Lambda0_gev,
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
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            p0=mean_params,
            prior=sample_prior,
            Lambda0_gev=Lambda0_gev,
        )

    diffs = np.diff(fit_coord)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        dz = _uniform_step(fit_coord)
        z_ext = np.arange(fit_coord[0], z_ext_fit_max + 0.5 * dz, dz)
    else:
        dz = float(np.min(diffs))
        z_ext = fit_coord[fit_coord <= z_ext_fit_max + 0.5 * dz]
        z_tail = np.arange(z_ext[-1] + dz, z_ext_fit_max + 0.5 * dz, dz)
        z_ext = np.concatenate([z_ext, z_tail])
    lambda_ext = z_ext * ft_scale_over_fit_scale

    data_re = _interp_samples(fit_coord, re_samples, z_ext)
    data_im = _interp_samples(fit_coord, im_samples, z_ext)

    trusted_stop = min(zmax_fit, fit_coord[-1])
    smooth = str(scheme["smooth"]).lower()
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

    parallel_fit_results: dict[int, dict[str, Any]] | None = None
    if executor is not None:
        payload = gv.dumps(
            {
                "z_fit": z_fit,
                "re_fit_samples": re_samples[:, fit_mask],
                "im_fit_samples": im_samples[:, fit_mask],
                "sigma_re": sigma_re,
                "sigma_im": sigma_im,
                "sample_error_mode": sample_error_mode,
                "resample_mode": resample_mode,
                "part": part,
                "method": method,
                "order": order,
                "observable": observable,
                "phase_scale": phase_scale,
                "phase_prime_scale": phase_prime_scale,
                "sector": sector,
                "hadron": hadron,
                "psi1_flavor_class": psi1_flavor_class,
                "psi2_flavor_class": psi2_flavor_class,
                "mean_params": mean_params,
                "sample_prior": sample_prior,
                "Lambda0_gev": Lambda0_gev,
            }
        )
        futures = [
            executor.submit(_fit_fourier_sample_batch, payload, batch)
            for batch in _sample_batches(n_samples, workers)
        ]
        parallel_fit_results = {
            item["sample"]: item
            for future in futures
            for item in future.result()
        }

    positive = z_ext > 0
    for sample in range(n_samples):
        if parallel_fit_results is None:
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
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                p0=mean_params,
                prior=sample_prior,
                Lambda0_gev=Lambda0_gev,
            )
        else:
            fit_result = parallel_fit_results[sample]
            params = np.asarray(fit_result["params"], dtype=float)
            tail_fit_success = bool(fit_result["success"])
            chi2 = float(fit_result["chi2"])
            dof = int(fit_result["dof"])
            q_value = float(fit_result["q_value"])
            log_gbf = float(fit_result["log_gbf"])
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
            Lambda0_gev=Lambda0_gev,
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
        "fit_param_labels": _param_labels(
            method,
            order,
            observable,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        ),
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
    observable: str,
    momentum_gev: float | None = None,
    final_momentum_gev: float | None = None,
    im_flip_for_ft: bool = False,
    resample_mode: str = "bootstrap",
    Lambda0_gev: float = 0.0,
    posterior_prior_error_scale: float = 3.0,
    sample_error_mode: str = "covariance",
    part: str = "both",
    sector: str | None = None,
    hadron: str | None = None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    workers: int = 1,
) -> dict[str, Any]:
    """Run asymptotic extension and Fourier transform for resampled data.

    ``coord`` and ``schemes`` values are physical distances in fm. The output keeps
    sample information as arrays shaped ``(scheme, sample, y)``.
    """
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    coord_arr = np.asarray(coord, dtype=float)
    resample_mode = _normalise_resample_mode(resample_mode)
    sample_error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    part = _normalise_part(part)
    coord_diffs = np.diff(coord_arr)
    coord_step = (
        _uniform_step(coord_arr)
        if np.allclose(coord_diffs, coord_diffs[0], rtol=1e-7, atol=1e-12)
        else float(np.min(coord_diffs))
    )
    missing_short_distance_coord = np.arange(0.0, coord_arr[0] - 0.5 * coord_step, coord_step).tolist()

    re_mat = _as_sample_matrix("re_samples", re_samples)
    im_mat = _as_sample_matrix("im_samples", im_samples)
    if re_mat.shape != im_mat.shape:
        raise ValueError("re_samples and im_samples must have the same shape")
    if re_mat.shape[1] != len(coord_arr):
        raise ValueError("sample arrays must have one value per coordinate point")

    observable = _canonical_observable(observable)
    fit_coord = coord_arr * FM_TO_GEV_INV
    ft_scale_over_fit_scale = _ft_scale_momentum(momentum_gev, final_momentum_gev)
    ft_scale = FM_TO_GEV_INV * ft_scale_over_fit_scale
    phase_scale = float(momentum_gev or 0.0)
    phase_prime_scale = None if final_momentum_gev is None else float(final_momentum_gev)
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

    shared_executor = _FOURIER_SAMPLE_EXECUTOR.get()
    owned_executor = (
        ProcessPoolExecutor(max_workers=min(workers, re_mat.shape[0]))
        if workers > 1 and shared_executor is None
        else None
    )
    sample_executor = shared_executor or owned_executor
    scheme_results = []
    try:
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
                    im_flip_for_ft=im_flip_for_ft,
                    phase_scale=phase_scale,
                    phase_prime_scale=phase_prime_scale,
                    resample_mode=resample_mode,
                    Lambda0_gev=Lambda0_gev,
                    posterior_prior_error_scale=scheme_prior_width,
                    sample_error_mode=sample_error_mode,
                    part=part,
                    sector=sector,
                    hadron=hadron,
                    psi1_flavor_class=psi1_flavor_class,
                    psi2_flavor_class=psi2_flavor_class,
                    executor=sample_executor,
                    workers=workers,
                )
            )
            scheme_results[-1]["order"] = scheme_order
            scheme_results[-1]["posterior_prior_error_scale"] = scheme_prior_width
    finally:
        if owned_executor is not None:
            owned_executor.shutdown()

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
        "coord_unit": "fm",
        "fit_coord_unit": "gev_inv",
        "resample_mode": resample_mode,
        "sample_error_mode": sample_error_mode,
        "Lambda0_gev": float(Lambda0_gev),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "part": part,
        "hadron": hadron,
        "psi1_flavor_class": str(psi1_flavor_class or "heavy").lower(),
        "psi2_flavor_class": str(psi2_flavor_class or "heavy").lower(),
        "workers": int(workers),
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


def fourier_result_to_ensemble_data(result: dict[str, Any], source_ensemble: EnsembleInfo | None = None) -> EnsembleData:
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
        "hadron": str(result.get("hadron", "")),
        "psi1_flavor_class": str(result.get("psi1_flavor_class", "heavy")),
        "psi2_flavor_class": str(result.get("psi2_flavor_class", "heavy")),
        "coord_unit": str(result.get("coord_unit", "")),
        "fit_coord_unit": str(result.get("fit_coord_unit", "")),
        "part": str(result.get("part", "both")),
        "im_flip_for_ft": str(result.get("im_flip_for_ft", "")),
        "symmetry_guarantee": str(result.get("symmetry_guarantee", False)),
        "Lambda0_gev": str(result.get("Lambda0_gev", 0.0)),
        "resample_mode": str(result.get("resample_mode", "")),
        "sample_error_mode": str(result.get("sample_error_mode", "")),
        "average_method": str(result.get("sample_error_mode", "")),
        "workers": str(result.get("workers", 1)),
    }
    if str(result.get("target_observable", "")).lower() in {"pdf", "gpd"}:
        attrs.update(
            {
                "observable_backend": str(result.get("observable_backend", "")),
                "parton": str(result.get("parton", "")),
                "current_operator": str(result.get("current_operator", "")),
                "polarization": str(result.get("polarization", "unpolarized")),
            }
        )
    for key in (
        "momentum",
        "volume",
        "bz_direction",
        "ensemble",
        "momentum_gev",
        "final_momentum_gev",
        "lattice_spacing_fm",
        "zs_fm",
    ):
        value = result.get(key)
        if value is not None:
            attrs[key] = str(value)
    for key in ("selected_range_label", "selected_candidate_label"):
        if key in result:
            attrs[key] = str(result[key])
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
        "selected_fit_range",
        "candidate_scheme_labels",
        "candidate_scheme_fit_chi2_dof",
        "candidate_scheme_q",
        "candidate_scheme_logGBF",
        "selected_candidate_index",
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
    ensemble_label = str(result.get("ensemble") or (source_ensemble.id if source_ensemble is not None else ""))
    return EnsembleData(
        ensemble=EnsembleInfo(
            source_ensemble.series if source_ensemble is not None else "",
            ensemble_label,
            source_ensemble.a_s if source_ensemble is not None else float(result.get("lattice_spacing_fm") or 1.0),
            source_ensemble.a_t if source_ensemble is not None else float(result.get("lattice_spacing_fm") or 1.0),
            source_ensemble.L_s if source_ensemble is not None else 1,
            source_ensemble.L_t if source_ensemble is not None else 1,
            source_ensemble.m_pi if source_ensemble is not None else 0.0,
        ),
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


def _save_fourier_fit_info_netcdf(path: Path, result: dict[str, Any], source_ensemble: EnsembleInfo | None = None) -> None:
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
    ensemble_label = str(result.get("ensemble", ""))
    fit_info_data = EnsembleData(
        ensemble=EnsembleInfo(
            source_ensemble.series if source_ensemble is not None else "",
            ensemble_label,
            source_ensemble.a_s if source_ensemble is not None else float(result.get("lattice_spacing_fm") or 1.0),
            source_ensemble.a_t if source_ensemble is not None else float(result.get("lattice_spacing_fm") or 1.0),
            source_ensemble.L_s if source_ensemble is not None else 1,
            source_ensemble.L_t if source_ensemble is not None else 1,
            source_ensemble.m_pi if source_ensemble is not None else 0.0,
        ),
        resample=resample_mode,
        values=[param_samples[idx] for idx in range(param_samples.shape[0])],
        dims=("scheme", "parameter"),
        coords={"scheme": scheme_labels.tolist(), "parameter": fit_param_labels},
        attrs={
            "method": str(result.get("method", "")),
            "order": str(result.get("order", "")),
            "observable": str(result.get("observable", "")),
            "part": str(result.get("part", "both")),
            "hadron": str(result.get("hadron", "")),
            **(
                {
                    "observable_backend": str(result.get("observable_backend", "")),
                    "parton": str(result.get("parton", "")),
                    "current_operator": str(result.get("current_operator", "")),
                    "polarization": str(result.get("polarization", "unpolarized")),
                }
                if str(result.get("target_observable", "")).lower() in {"pdf", "gpd"}
                else {}
            ),
            "psi1_flavor_class": str(result.get("psi1_flavor_class", "heavy")),
            "psi2_flavor_class": str(result.get("psi2_flavor_class", "heavy")),
            "symmetry_guarantee": str(result.get("symmetry_guarantee", False)),
            "Lambda0_gev": str(result.get("Lambda0_gev", 0.0)),
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


def _pick_four_zmin_fm_by_tail_fit(
    positive: np.ndarray,
    *,
    zmax_candidates: list[float],
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    method: str,
    order: str,
    observable: str,
    momentum_gev: float | None,
    final_momentum_gev: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0_gev: float,
    part: str,
    sector: str | None,
    hadron: str | None,
    preferred_zmin: float | None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
) -> list[float]:
    stable_starts = []
    required_points = _minimum_fit_points_for_parameters(
        len(
            _param_labels(
                method,
                order,
                observable,
                sector=sector,
                hadron=hadron,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                fit=True,
            )
        ),
        part,
    )
    for zmax in zmax_candidates:
        zmax_snap = _nearest_positive_coord(positive, zmax)
        candidates = positive[positive < zmax_snap]
        candidates = np.asarray(
            [candidate for candidate in candidates if int(np.count_nonzero(_fit_coord_mask(positive, candidate, zmax))) >= required_points],
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
                momentum_gev=momentum_gev,
                final_momentum_gev=final_momentum_gev,
                resample_mode=resample_mode,
                sample_error_mode=sample_error_mode,
                Lambda0_gev=Lambda0_gev,
                part=part,
                sector=sector,
                hadron=hadron,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
            )
            for candidate in candidates
        ]
        stable_starts.append(float(candidates[_tail_quality_stable_start(qualities)]))

    stable_starts = sorted({float(item) for item in stable_starts})
    if len(stable_starts) >= 4:
        return stable_starts[:4]
    anchor = max(stable_starts) if stable_starts else (float(preferred_zmin) if preferred_zmin is not None else positive[0])
    candidates = [float(item) for item in positive if item >= anchor and any(item < zmax for zmax in zmax_candidates)]
    for candidate in candidates:
        if candidate not in stable_starts:
            stable_starts.append(candidate)
        if len(stable_starts) >= 4:
            break
    return stable_starts[:4]


def _auto_fill_scheme_scan(
    spec: dict[str, Any],
    *,
    coord: np.ndarray,
    positive: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    stable_idx: int,
    method: str,
    order: str,
    observable: str,
    momentum_gev: float | None,
    final_momentum_gev: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0_gev: float,
    part: str,
    sector: str | None,
    hadron: str | None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
) -> dict[str, Any]:
    """Fill missing scan keys with stable zmax values and tail-fit zmin diagnostics."""
    if "zmax_fm" not in spec:
        end_index = int(np.clip(stable_idx, 0, len(positive) - 1))
        start_index = max(0, end_index - 4)
        spec["zmax_fm"] = [float(item) for item in positive[start_index : end_index + 1]]
    zmax_candidates = [float(item) for item in spec["zmax_fm"]]

    if "zmin_fm" not in spec:
        spec["zmin_fm"] = _pick_four_zmin_fm_by_tail_fit(
            positive,
            zmax_candidates=zmax_candidates,
            coord=coord,
            re_samples=re_samples,
            im_samples=im_samples,
            method=method,
            order=order,
            observable=observable,
            momentum_gev=momentum_gev,
            final_momentum_gev=final_momentum_gev,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            Lambda0_gev=Lambda0_gev,
            part=part,
            sector=sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            preferred_zmin=0.5,
        )
    if "zmax_ext_fm" not in spec:
        spec["zmax_ext_fm"] = float(
            np.max(coord) + 8.0 / (FM_TO_GEV_INV * _ft_scale_momentum(momentum_gev, final_momentum_gev))
        )
    if "smooth" not in spec:
        spec["smooth"] = "linear"
    return spec


def _scan_has_all_range_keys(spec: dict[str, Any] | None) -> bool:
    if spec is None:
        return False
    return all(key in spec for key in ("zmin_fm", "zmax_fm", "zmax_ext_fm", "smooth"))


def _auto_scheme_scan(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    method: str,
    order: str,
    observable: str,
    momentum_gev: float | None,
    final_momentum_gev: float | None,
    resample_mode: str,
    sample_error_mode: str,
    Lambda0_gev: float,
    part: str,
    sector: str | None,
    hadron: str | None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
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
        method=method,
        order=order,
        observable=observable,
        momentum_gev=momentum_gev,
        final_momentum_gev=final_momentum_gev,
        resample_mode=resample_mode,
        sample_error_mode=sample_error_mode,
        Lambda0_gev=Lambda0_gev,
        part=part,
        sector=sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
    )
    spec["auto_generated"] = True
    return spec


def _generate_scan_schemes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    zmin_candidates = [float(item) for item in spec["zmin_fm"]]
    zmax_candidates = [float(item) for item in spec["zmax_fm"]]
    z_ext_max = float(spec["zmax_ext_fm"])
    smooth = str(spec.get("smooth", "linear"))
    max_schemes = int(spec.get("max_schemes", 200))

    schemes = []
    for zmin in zmin_candidates:
        for zmax in zmax_candidates:
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


def _resolve_quasi_y_ls(quasi_y_ls: list[float] | dict[str, Any]) -> list[float]:
    grid = np.asarray(resolve_grid_spec(quasi_y_ls, name="quasi_y_ls"), dtype=float)
    message = quasi_y_ls_error(grid)
    if message:
        raise ValueError(message)
    return grid.tolist()


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


@_with_fourier_sample_executor
def run_fourier_transform(
    store: dict[str, Any],
    *,
    quasi_y_ls: list[float] | dict[str, Any],
    scheme_scan: dict[str, Any],
    zmin_shift: int,
    method: str,
    order: str | list[str],
    target_observable: str,
    parton: str,
    hadron: str,
    momentum: str | None = None,
    volume: str | None = None,
    bz_direction: str | None = None,
    momentum_gev: float | None = None,
    final_momentum_gev: float | None = None,
    ensemble: str | None = None,
    lattice_spacing_fm: float | None = None,
    zs_fm: float | None = None,
    im_flip_for_ft: bool,
    symmetry_guarantee: bool | None = None,
    Lambda0_gev: float,
    posterior_prior_error_scale: float | list[float],
    sample_error_mode: str,
    part: str,
    output_scale: float,
    sector: str,
    current_operator: str | None = None,
    polarization: str | None = None,
    psi1_flavor_class: str | None = None,
    psi2_flavor_class: str | None = None,
    save_path: str | None = None,
    plot_fourier: dict[str, Any] | None = None,
    plot_extension: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    artifacts_dir: str | None = None,
    workers: int,
) -> dict[str, Any]:
    """Run local extrapolation and Fourier transform for loaded samples."""
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if symmetry_guarantee is not None and not isinstance(symmetry_guarantee, bool):
        raise ValueError("symmetry_guarantee must be a boolean")
    out = "fourier_result"
    sector = None if sector is None else str(sector).strip().lower()
    parton = str(parton).strip().lower()
    hadron = str(hadron).strip().lower()
    polarization = str(polarization or "").strip().lower()
    psi1_flavor_class = str(psi1_flavor_class or "").strip().lower()
    psi2_flavor_class = str(psi2_flavor_class or "").strip().lower()
    target = str(target_observable).strip().lower()
    if target in {"pdf", "gpd"}:
        hadron = "nucleon" if hadron == "proton" else hadron
        public_observable = INFERRED_OBSERVABLES[(target, parton, hadron)]
        if not polarization:
            raise ValueError("polarization is required for PDF/GPD Fourier transforms")
        if (public_observable, polarization) not in POLARIZED_OBSERVABLES:
            raise ValueError(f"polarization={polarization!r} is not supported for {public_observable!r}")
        observable = POLARIZED_OBSERVABLES[(public_observable, polarization)]
    else:
        public_observable = observable = "meson_quasi_da"
    if sector is not None:
        if target == "da":
            sector = "full"
        if parton == "gluon":
            sector, part, output_scale, im_flip_for_ft = "full", "both", 1.0, False
        elif target in {"pdf", "gpd"}:
            part, output_scale, im_flip_for_ft = {
                "valence": (("im" if polarization == "helicity" else "re"), 2.0, False),
                "singlet": (("re" if polarization == "helicity" else "im"), 2.0, False),
                "sea": ("both", 1.0, False),
                "full": ("both", 1.0, False),
            }[sector]
        else:
            part, output_scale, im_flip_for_ft = "both", 1.0, False
    else:
        if parton == "gluon":
            sector, part, output_scale, im_flip_for_ft = "full", "both", 1.0, False
        elif polarization == "helicity" and target in {"pdf", "gpd"}:
            sector = {"re": "singlet", "im": "valence", "both": "full"}.get(str(part).lower(), str(part).lower())
        else:
            sector = {"re": "valence", "im": "singlet", "both": "full"}.get(str(part).lower(), str(part).lower())
    fit_sector = "full" if sector == "sea" else sector
    matrix_element_data = store.get("matrix_element_data")
    if matrix_element_data is None:
        matrix_element_data = store["input"]
        store["matrix_element_data"] = matrix_element_data
    if zs_fm is None:
        upstream_zs = getattr(matrix_element_data, "attrs", {}).get("zs_fm")
        if upstream_zs not in {None, ""}:
            zs_fm = float(upstream_zs)
    resample_mode = _normalise_resample_mode(matrix_element_data.resample)
    sample_error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    matrix_element = ensemble_data_to_legacy_arrays(matrix_element_data)
    coord_arr = np.asarray(matrix_element["coord"], dtype=float)
    if target == "da" and not isinstance(symmetry_guarantee, bool):
        raise ValueError("DA Fourier transforms require symmetry_guarantee")
    if target == "da" and (not psi1_flavor_class or not psi2_flavor_class):
        raise ValueError("DA Fourier transforms require both psi flavor classes")
    if target == "da" and symmetry_guarantee:
        ft_scale = FM_TO_GEV_INV * _ft_scale_momentum(momentum_gev, final_momentum_gev)
        projected = _project_da_symmetry(
            coord_arr,
            matrix_element["re_samples"],
            matrix_element["im_samples"],
            phase_scale=ft_scale,
        )
        matrix_element["re_samples"] = np.real(projected)
        matrix_element["im_samples"] = np.imag(projected)
        matrix_element_data.array.values = projected
    auto_scheme_scan = None
    range_order = str(order[0] if isinstance(order, list) else order).upper()
    range_prior_width = float(
        posterior_prior_error_scale[0] if isinstance(posterior_prior_error_scale, list) else posterior_prior_error_scale
    )
    scan_spec = {"model_average": True, **dict(scheme_scan)}
    if not _scan_has_all_range_keys(scan_spec):
        scan_spec = _auto_scheme_scan(
            coord=coord_arr,
            re_samples=np.asarray(matrix_element["re_samples"], dtype=float),
            im_samples=np.asarray(matrix_element["im_samples"], dtype=float),
            method=method,
            order=range_order,
            observable=observable,
            momentum_gev=momentum_gev,
            final_momentum_gev=final_momentum_gev,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            Lambda0_gev=float(Lambda0_gev),
            part=part,
            sector=fit_sector,
            hadron=hadron,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            existing=scan_spec,
        )
        auto_scheme_scan = scan_spec
    scheme_scan = scan_spec
    schemes = _generate_scan_schemes(scheme_scan)
    required_points = _minimum_fit_points_for_parameters(
        len(
            _param_labels(
                method,
                range_order,
                observable,
                sector=fit_sector,
                hadron=hadron,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                fit=True,
            )
        ),
        part,
    )
    schemes = [
        scheme
        for scheme in schemes
        if np.count_nonzero(_fit_coord_mask(coord_arr, float(scheme["zmin"]), float(scheme["zmax"])))
        >= required_points
    ]
    if not schemes:
        raise ValueError("scheme_scan produced no valid zmin/zmax combinations")
    y_values = _resolve_quasi_y_ls(quasi_y_ls)
    model_average = bool(scheme_scan["model_average"])
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
                momentum_gev=momentum_gev,
                final_momentum_gev=final_momentum_gev,
                resample_mode=resample_mode,
                Lambda0_gev=float(Lambda0_gev),
                posterior_prior_error_scale=range_prior_width,
                sample_error_mode=sample_error_mode,
                part=part,
                sector=fit_sector,
                hadron=hadron,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
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
    if int(zmin_shift):
        positive_coord = coord_arr[coord_arr > 0]
        zmin_candidates = positive_coord[positive_coord < float(selected_range["zmax"])]
        current_index = int(np.argmin(np.abs(zmin_candidates - float(selected_range["zmin"]))))
        shifted_index = min(max(current_index + int(zmin_shift), 0), len(zmin_candidates) - 1)
        selected_range["zmin"] = float(zmin_candidates[shifted_index])
    fit_model_specs = []
    for spec in _fit_model_specs(order, posterior_prior_error_scale):
        n_model_params = len(
            _param_labels(
                method,
                spec["order"],
                observable,
                sector=fit_sector,
                hadron=hadron,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                fit=True,
            )
        )
        n_model_points = np.count_nonzero(
            _fit_coord_mask(coord_arr, float(selected_range["zmin"]), float(selected_range["zmax"]))
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
            "selected_range_label": f"zmin_{float(selected_range['zmin']):g}_zmax_{float(selected_range['zmax']):g}".replace(".", "p"),
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
    sea_projection = sector == "sea"
    result = run_fourier_workflow(
        matrix_element["coord"],
        matrix_element["re_samples"],
        matrix_element["im_samples"],
        -np.asarray(y_values, dtype=float) if sea_projection else y_values,
        schemes=schemes,
        method=method,
        order=order,
        observable=observable,
        momentum_gev=momentum_gev,
        final_momentum_gev=final_momentum_gev,
        im_flip_for_ft=im_flip_for_ft,
        resample_mode=resample_mode,
        Lambda0_gev=float(Lambda0_gev),
        posterior_prior_error_scale=range_prior_width,
        sample_error_mode=sample_error_mode,
        part=part,
        sector=fit_sector,
        hadron=hadron,
        psi1_flavor_class=psi1_flavor_class,
        psi2_flavor_class=psi2_flavor_class,
        workers=workers,
    )
    result["resample_mode"] = resample_mode
    result["sample_error_mode"] = sample_error_mode
    result["momentum"] = momentum
    result["volume"] = volume
    result["bz_direction"] = bz_direction
    result["momentum_gev"] = momentum_gev
    result["final_momentum_gev"] = final_momentum_gev
    result["lattice_spacing_fm"] = lattice_spacing_fm
    result["zs_fm"] = zs_fm
    result["im_flip_for_ft"] = bool(im_flip_for_ft)
    result["symmetry_guarantee"] = bool(target == "da" and symmetry_guarantee)
    result["sector"] = sector
    result["target_observable"] = target
    if target in {"pdf", "gpd"}:
        result["observable_backend"] = OBSERVABLE_BACKENDS.get(str(result.get("observable", "")), "")
        result["observable"] = public_observable
        result["parton"] = parton
        result["current_operator"] = current_operator
        result["polarization"] = polarization
    result["Lambda0_gev"] = float(Lambda0_gev)
    result["posterior_prior_error_scale"] = (
        range_prior_width
        if len(candidate_diagnostics["fit_model_prior_widths"]) == 1
        else candidate_diagnostics["fit_model_prior_widths"]
    )
    result["part"] = str(part)
    result["hadron"] = hadron
    result["psi1_flavor_class"] = psi1_flavor_class
    result["psi2_flavor_class"] = psi2_flavor_class
    result["workers"] = int(workers)
    result["ensemble"] = str(ensemble or "")
    result.update(candidate_diagnostics)
    if auto_scheme_scan is not None:
        result["auto_scheme_scan"] = auto_scheme_scan
    _apply_sample_fit_model_average(
        result,
        resample_mode=resample_mode,
        sample_error_mode=sample_error_mode,
        model_average=model_average,
    )
    sea_sign = 1.0 if polarization == "helicity" else -1.0
    _apply_fourier_output_scale(result, sea_sign if sea_projection else float(output_scale))
    if sea_projection:
        result["y_grid"] = np.asarray(y_values, dtype=float)
        result["output_scale"] = 1.0
        if sea_sign < 0:
            for scheme_result in result["scheme_results"]:
                for key in ("ft_re_samples", "ft_im_samples"):
                    scheme_result[key] = -np.asarray(scheme_result[key], dtype=float)
    store["fourier_result_data"] = fourier_result_to_ensemble_data(result, source_ensemble=matrix_element_data.ensemble)
    store[out] = result
    artifact = _artifact_path(save_path, default_name=f"{out}.nc", artifacts_dir=artifacts_dir).with_suffix(".nc")
    fit_info_artifact = _artifact_path(None, default_name=f"{artifact.stem}_fit_info.nc", artifacts_dir=artifacts_dir)
    store["fourier_result_data"].to_netcdf(artifact)
    _save_fourier_fit_info_netcdf(fit_info_artifact, result, source_ensemble=matrix_element_data.ensemble)
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
        **(
            {
                "observable": result.get("observable"),
                "parton": result.get("parton"),
                "polarization": result.get("polarization"),
            }
            if target in {"pdf", "gpd"}
            else {}
        ),
        "symmetry_guarantee": result.get("symmetry_guarantee", False),
        "auto_scheme_scan": auto_scheme_scan,
        "Lambda0_gev": result.get("Lambda0_gev", 0.0),
        "workers": int(workers),
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
        "symmetry_guarantee": data.get("symmetry_guarantee", False),
        "Lambda0_gev": data.get("Lambda0_gev", 0.0),
        "sector": data.get("sector", data.get("part", "full")),
    }
    if str(data.get("target_observable", "")).lower() in {"pdf", "gpd"}:
        summary.update(
            {
                "observable": data.get("observable"),
                "parton": data.get("parton"),
                "polarization": data.get("polarization"),
            }
        )
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
    output = _artifact_path(save_path, default_name="fourier_xdep.pdf", artifacts_dir=artifacts_dir)
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
    re_output = _artifact_path(save_path, default_name="fourier_re.pdf", artifacts_dir=artifacts_dir)
    im_stem = f"{re_output.stem[:-3]}_im" if re_output.stem.endswith("_re") else f"{re_output.stem}_im"
    im_output = re_output.with_name(f"{im_stem}.pdf")
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        matrix_element["re_samples"],
        data,
        scheme_index=scheme_index,
        component="re",
        momentum_gev=data.get("momentum_gev"),
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
        momentum_gev=data.get("momentum_gev"),
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
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
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
        backend=backend,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
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
