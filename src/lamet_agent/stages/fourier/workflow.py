"""Sample-preserving extrapolation and Fourier-transform workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

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
    "meson_quasi_da": "meson_quasi_da",
    "quasi_da": "meson_quasi_da",
    "pion_quark_quasi_gpd": "pion_quark_quasi_gpd",
    "pion_gpd": "pion_quark_quasi_gpd",
    "nucleon_quark_quasi_gpd": "nucleon_quark_quasi_gpd",
    "nucleon_gpd": "nucleon_quark_quasi_gpd",
}


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


def _param_template(method: str, order: str, observable: str) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    method = method.upper()
    order = order.upper()
    if method not in {"GI", "CG"}:
        raise ValueError("method must be 'GI' or 'CG'")
    if order not in {"LA", "NLA", "EMPIRICAL"}:
        raise ValueError("order must be 'LA', 'NLA', or 'Empirical'")

    if order == "EMPIRICAL":
        p0 = [1.0, 0.1, 1.0, 1.0, 5.0]
        lower = [-np.inf, -np.inf, -5.0, -5.0, 0.1]
        upper = [np.inf, np.inf, 5.0, 5.0, 100.0]
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

    p0.append(0.5)
    lower.append(0.1)
    upper.append(3.0)

    if method == "CG":
        p0.append(0.5)
        lower.append(-2.0)
        upper.append(4.0)

    return np.asarray(p0, dtype=float), (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _param_labels(method: str, order: str, observable: str) -> list[str]:
    if order.upper() == "EMPIRICAL":
        return ["c1", "c2", "a", "b", "lambda0"]
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

    if order.upper() == "EMPIRICAL":
        lambda_pos = phase_scale * z
        c1, c2, a, b, lambda0 = params[:5]
        values = (c1 / (1j * lambda_pos) ** a + np.exp(-1j * lambda_pos) * c2 / (-1j * lambda_pos) ** b)
        values = values * np.exp(-lambda_pos / lambda0)
        return np.real(values), np.imag(values)

    term_names = _observable_term_names(observable)
    phase_scales = _term_phase_scales(
        observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
    )
    amp = np.zeros_like(z, dtype=complex)
    cursor = 0
    for phase in phase_scales:
        amp = amp + params[cursor] * np.exp(1j * (params[cursor + 1] + phase * z))
        cursor += 2

    if order.upper() == "NLA":
        subleading = np.zeros_like(z, dtype=complex)
        for phase in phase_scales:
            subleading = subleading + params[cursor] * np.exp(1j * (params[cursor + 1] + phase * z))
            cursor += 2
        amp = amp + subleading / z

    lam = params[cursor]
    tail = np.exp(-lam * z)
    if method.upper() == "CG":
        n = params[-1]
        tail = tail / z**n

    values = amp * tail
    return np.real(values), np.imag(values)


def _fit_one_sample(
    z_fit: np.ndarray,
    re_fit: np.ndarray,
    im_fit: np.ndarray,
    *,
    method: str,
    order: str,
    observable: str,
    phase_scale: float,
    phase_prime_scale: float | None = None,
    p0: np.ndarray | None = None,
    sigma_re: np.ndarray | None = None,
    sigma_im: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, float, int, float]:
    default_p0, bounds = _param_template(method, order, observable)
    start = default_p0 if p0 is None else np.asarray(p0, dtype=float)
    re_scale = np.ones_like(re_fit, dtype=float) if sigma_re is None else np.asarray(sigma_re, dtype=float)
    im_scale = np.ones_like(im_fit, dtype=float) if sigma_im is None else np.asarray(sigma_im, dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        pred_re, pred_im = _asymptotic_values(
            z_fit,
            params,
            method=method,
            order=order,
            observable=observable,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
        )
        return np.concatenate([(pred_re - re_fit) / re_scale, (pred_im - im_fit) / im_scale])

    dof = max(1, 2 * len(z_fit) - len(default_p0))
    try:
        fit = least_squares(residual, start, bounds=bounds, max_nfev=2000)
    except ValueError:
        return default_p0, False, float("inf"), dof, 0.0

    chi2 = float(np.sum(residual(fit.x) ** 2))
    q_value = float(chi2_distribution.sf(chi2, dof))
    return fit.x, bool(fit.success), chi2, dof, q_value


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
    n_params = len(_param_template(method, order, observable)[0])
    if 2 * n_points < n_params or n_points < 2:
        dof = max(1, 2 * n_points - n_params)
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
    sigma_re = _sample_sdev(re_mat[:, fit_mask])
    sigma_im = _sample_sdev(im_mat[:, fit_mask])
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)

    _params, ok, chi2, dof, q_value = _fit_one_sample(
        z_fit,
        mean_re,
        mean_im,
        method=method,
        order=order,
        observable=observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        sigma_re=sigma_re,
        sigma_im=sigma_im,
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
    if 2 * np.count_nonzero(fit_mask) < len(_param_template(method, order, observable)[0]):
        raise ValueError("fit range has too few points for the selected asymptotic form")

    z_fit = fit_coord[fit_mask]
    mean_re = np.mean(re_samples, axis=0)[fit_mask]
    mean_im = np.mean(im_samples, axis=0)[fit_mask]
    sigma_re = _sample_sdev(re_samples[:, fit_mask])
    sigma_im = _sample_sdev(im_samples[:, fit_mask])
    sigma_floor = max(1e-8, 0.02 * max(float(np.max(np.abs(mean_re))), float(np.max(np.abs(mean_im))), 1.0))
    sigma_re = np.maximum(sigma_re, sigma_floor)
    sigma_im = np.maximum(sigma_im, sigma_floor)
    mean_params, _, mean_chi2, mean_dof, mean_q = _fit_one_sample(
        z_fit,
        mean_re,
        mean_im,
        method=method,
        order=order,
        observable=observable,
        phase_scale=phase_scale,
        phase_prime_scale=phase_prime_scale,
        sigma_re=sigma_re,
        sigma_im=sigma_im,
    )

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
    n_params = len(_param_template(method, order, observable)[0])
    fit_params = np.empty((n_samples, n_params), dtype=float)
    fit_chi2 = np.empty(n_samples, dtype=float)
    fit_dof = np.empty(n_samples, dtype=int)
    fit_q = np.empty(n_samples, dtype=float)
    failures = 0

    positive = z_ext > 0
    for sample in range(n_samples):
        params, ok, chi2, dof, q_value = _fit_one_sample(
            z_fit,
            re_samples[sample, fit_mask],
            im_samples[sample, fit_mask],
            method=method,
            order=order,
            observable=observable,
            phase_scale=phase_scale,
            phase_prime_scale=phase_prime_scale,
            p0=mean_params,
            sigma_re=sigma_re,
            sigma_im=sigma_im,
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

        fit_re_samples[sample] = fit_re
        fit_im_samples[sample] = fit_im
        ext_re[sample] = fit_weight * fit_re + (1.0 - fit_weight) * data_re[sample]
        ext_im[sample] = fit_weight * fit_im + (1.0 - fit_weight) * data_im[sample]

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


def _sample_sdev(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 2:
        return np.zeros(samples.shape[1], dtype=float)
    return np.std(samples, axis=0, ddof=1)


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
) -> dict[str, Any]:
    """Run asymptotic extension and Fourier transform for resampled data.

    ``schemes`` values are in the same unit as ``coord``. The output keeps
    sample information as arrays shaped ``(scheme, sample, k)``.
    """
    coord_arr = np.asarray(coord, dtype=float)
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
        schemes = [{"label": "default", "zmin": coord_arr[1], "zmax": coord_arr[-1], "z_ext_max": coord_arr[-1]}]

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
        )
        for scheme in schemes
    ]

    ft_re = np.asarray([item["ft_re_samples"] for item in scheme_results])
    ft_im = np.asarray([item["ft_im_samples"] for item in scheme_results])
    re_mean_by_scheme = np.mean(ft_re, axis=1)
    im_mean_by_scheme = np.mean(ft_im, axis=1)
    re_stat_by_scheme = np.asarray([_sample_sdev(item["ft_re_samples"]) for item in scheme_results])
    im_stat_by_scheme = np.asarray([_sample_sdev(item["ft_im_samples"]) for item in scheme_results])

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
    }
