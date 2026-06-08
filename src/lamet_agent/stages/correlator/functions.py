"""Correlator-analysis stage tools.

Purpose:
- provide fixed Python tools for 2pt ground-state and 3pt/2pt ratio analysis
- read correlators, resample, fit windows, model-average, and plot fit-on-data

Expected inputs:
- 2pt HDF5: ``source_sink/gamma/momentum`` with shape (Lt, n_cfg)
- 3pt HDF5: ``source_sink/gamma/momentum/b_dir/eta/bT*/bz*`` with shape (tsep+1, n_cfg)
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- gvar arrays and fit scans in a per-stage artifact store
- model-averaged parameters and PDFs under artifacts/

Example usage:
- from lamet_agent.stages.correlator.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["read_pt2"](store, path="examples/fake_data/data/fake_2pt.h5")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gvar as gv
import h5py
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np

#! ignore numpy overflow
np.seterr(over='ignore')

from lamet_agent.core.plotting import (
    COLOR_CYCLE,
    ERRORBAR_STYLE,
    FONT_SIZE,
    LEGEND_SETS,
    default_plot,
    plot_pt2_fit_on_data,
    plot_pt3_ratio_fit_on_data,
)
from lamet_agent.core.resampling import (
    bootstrap,
    bootstrap_by_indices as _bootstrap_by_indices,
    bootstrap_indices as _bootstrap_indices,
    bs_ls_avg,
    jackknife,
    jk_ls_avg,
    resample_config_samples as _resample_config_samples,
    sample_mean_err as _sample_mean_err,
    samples_to_gvar as _samples_to_gvar,
)
from lamet_agent.core.tools import log_nonlinear_fit_quality, resolve_plot_save_path, setup_logger


# --- data reading (trimmed from LaMETLat correlators/pt2.py) ----------------


def read_pt2(
    store: dict[str, Any],
    *,
    path: str,
    source_sink: str = "SS",
    gamma: str = "5",
    momentum: str = "PX0PY0PZ0",
    out: str = "pt2_samples",
    imag_out: str = "pt2_imag_samples",
) -> dict[str, Any]:
    """Read one 2pt dataset; store real and imag as (n_cfg, Lt) samples."""
    with h5py.File(path, "r") as h5f:
        data = np.swapaxes(np.asarray(h5f[source_sink][gamma][momentum]), 0, 1)
    store[out] = np.real(data)
    store[imag_out] = np.imag(data)
    store["Lt"] = int(store[out].shape[1])
    return {
        "out": out,
        "imag_out": imag_out,
        "n_cfg": int(store[out].shape[0]),
        "Lt": int(store[out].shape[1]),
    }


def _pt3_dataset_path(
    *,
    source_sink: str,
    gamma: str,
    momentum: str,
    b_dir: str,
    eta: str,
    bt: str,
    bz: str,
) -> str:
    return f"{source_sink}/{gamma}/{momentum}/{b_dir}/{eta}/{bt}/{bz}"


def read_pt3(
    store: dict[str, Any],
    *,
    path: str,
    tsep: int | None = None,
    source_sink: str = "SS",
    gamma: str = "T",
    momentum: str = "PX0PY0PZ0",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bt: str = "bT0",
    bz: str = "bz0",
    re_out: str = "pt3_samples_re",
    im_out: str = "pt3_samples_im",
    append: bool = True,
    out: str | None = None,
) -> dict[str, Any]:
    """Read one 3pt slice; merge into per-tsep dicts keyed by integer tsep.

    ``out`` is accepted for agent compatibility but ignored (use ``re_out`` / ``im_out``).
    """
    del out
    dset = _pt3_dataset_path(
        source_sink=source_sink,
        gamma=gamma,
        momentum=momentum,
        b_dir=b_dir,
        eta=eta,
        bt=bt,
        bz=bz,
    )
    with h5py.File(path, "r") as h5f:
        data = np.swapaxes(np.asarray(h5f[dset]), 0, 1)
    inferred = int(data.shape[1]) - 1
    tsep_key = int(tsep) if tsep is not None else inferred
    if tsep is not None and tsep_key != inferred:
        raise ValueError(
            f"tsep={tsep_key} does not match data length {data.shape[1]} (expected tsep={inferred})"
        )

    re_dict: dict[int, np.ndarray] = dict(store[re_out]) if append and re_out in store else {}
    im_dict: dict[int, np.ndarray] = dict(store[im_out]) if append and im_out in store else {}
    re_dict[tsep_key] = np.real(data)
    im_dict[tsep_key] = np.imag(data)
    store[re_out] = re_dict
    store[im_out] = im_dict
    return {
        "re_out": re_out,
        "im_out": im_out,
        "tsep": tsep_key,
        "n_cfg": int(data.shape[0]),
        "ntau": int(data.shape[1]),
        "tsep_keys": sorted(re_dict.keys()),
    }


# --- resampling --------------------------------------------------------------


def resample_to_gvar(
    store: dict[str, Any],
    *,
    samples: str = "pt2_samples",
    mode: str = "bs",
    n_samples: int = 200,
    seed: int | None = 1984,
    out: str = "pt2_gv",
) -> dict[str, Any]:
    """Resample stored samples and reduce them to a gvar correlator array."""
    data = store[samples]
    if mode == "bs":
        resampled = bootstrap(data, n_samples=n_samples, seed=seed)
        gv_arr = bs_ls_avg(resampled)
    elif mode == "jk":
        resampled = jackknife(data)
        gv_arr = jk_ls_avg(resampled)
    else:
        raise ValueError(f"unsupported resampling mode: {mode!r}")
    store[out] = gv_arr
    store["Lt"] = int(len(gv_arr))
    return {"out": out, "mode": mode, "Lt": int(len(gv_arr))}


def _bs_dict_to_gvar(
    data: dict[int, np.ndarray],
    *,
    n_samples: int,
    seed: int | None,
) -> dict[int, np.ndarray]:
    return {
        key: bs_ls_avg(bootstrap(arr, n_samples=n_samples, seed=seed))
        for key, arr in data.items()
    }


def _jk_dict_to_gvar(data: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {key: jk_ls_avg(jackknife(arr)) for key, arr in data.items()}


def compute_pt3_ratio(
    store: dict[str, Any],
    *,
    pt2_samples: str = "pt2_samples",
    pt2_imag_samples: str = "pt2_imag_samples",
    pt3_samples_re: str = "pt3_samples_re",
    pt3_samples_im: str = "pt3_samples_im",
    re_out: str = "ratio_samples_re",
    im_out: str = "ratio_samples_im",
    out: str | None = None,
) -> dict[str, Any]:
    """Build per-tsep 3pt/2pt ratio sample dicts from stored correlators."""
    del out
    pt2_re = np.asarray(store[pt2_samples])
    pt2_im = np.asarray(store[pt2_imag_samples])
    if pt2_re.shape != pt2_im.shape:
        raise ValueError("pt2 real and imag samples must have the same shape")
    pt2_complex = pt2_re + 1j * pt2_im

    pt3_re = store[pt3_samples_re]
    pt3_im = store[pt3_samples_im]
    if set(pt3_re) != set(pt3_im):
        raise ValueError("pt3 real and imag dicts must share the same tsep keys")

    ratio_re: dict[int, np.ndarray] = {}
    ratio_im: dict[int, np.ndarray] = {}
    for tsep in sorted(pt3_re):
        tsep_i = int(tsep)
        if not (0 <= tsep_i < pt2_complex.shape[1]):
            raise ValueError(
                f"tsep {tsep_i} out of range for pt2 with Lt={pt2_complex.shape[1]}"
            )
        pt3_complex = np.asarray(pt3_re[tsep_i]) + 1j * np.asarray(pt3_im[tsep_i])
        if pt3_complex.shape[0] != pt2_complex.shape[0]:
            raise ValueError(f"pt3[{tsep_i}] sample count mismatch with pt2")
        ratio = pt3_complex / pt2_complex[:, tsep_i][:, None]
        ratio_re[tsep_i] = np.real(ratio)
        ratio_im[tsep_i] = np.imag(ratio)

    store[re_out] = ratio_re
    store[im_out] = ratio_im
    return {"re_out": re_out, "im_out": im_out, "tsep_keys": sorted(ratio_re.keys())}


def resample_ratio_to_gvar(
    store: dict[str, Any],
    *,
    ratio_samples_re: str = "ratio_samples_re",
    ratio_samples_im: str = "ratio_samples_im",
    mode: str = "bs",
    n_samples: int = 200,
    seed: int | None = 1984,
    re_out: str = "ratio_real_gv",
    im_out: str = "ratio_imag_gv",
    out: str | None = None,
) -> dict[str, Any]:
    """Resample per-tsep ratio samples into gvar arrays."""
    del out
    re_data: dict[int, np.ndarray] = store[ratio_samples_re]
    im_data: dict[int, np.ndarray] = store[ratio_samples_im]
    if mode == "bs":
        store[re_out] = _bs_dict_to_gvar(re_data, n_samples=n_samples, seed=seed)
        store[im_out] = _bs_dict_to_gvar(im_data, n_samples=n_samples, seed=seed)
    elif mode == "jk":
        store[re_out] = _jk_dict_to_gvar(re_data)
        store[im_out] = _jk_dict_to_gvar(im_data)
    else:
        raise ValueError(f"unsupported resampling mode: {mode!r}")
    return {"re_out": re_out, "im_out": im_out, "mode": mode, "tsep_keys": sorted(re_data.keys())}


# --- ground-state fit (copied from LaMETLat ground_state) -------------------


def _validate_correlator_rescale(correlator_rescale: float) -> float:
    scale = float(correlator_rescale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"correlator_rescale must be positive and finite, got {correlator_rescale!r}")
    return scale


def _overlap_rescale(correlator_rescale: float) -> float:
    return float(np.sqrt(_validate_correlator_rescale(correlator_rescale)))


def _physical_overlap(value: gv.GVar, correlator_rescale: float) -> gv.GVar:
    return value / _overlap_rescale(correlator_rescale)


def _physical_overlap_diagnostics(p: dict, nstate: int, correlator_rescale: float) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "correlator_rescale": float(correlator_rescale),
        "overlap_rescale": _overlap_rescale(correlator_rescale),
    }
    physical: list[gv.GVar] = []
    for state in range(int(nstate)):
        key = f"z{state}"
        if key not in p:
            continue
        z_phys = _physical_overlap(p[key], correlator_rescale)
        physical.append(z_phys)
        diagnostics[f"{key}_physical"] = z_phys
    if len(physical) >= 2 and gv.mean(physical[0]) != 0.0:
        diagnostics["z1_over_z0_physical"] = physical[1] / physical[0]
    return diagnostics


def inspect_correlator_scale(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_windows: list[dict[str, int]] | None = None,
    source_sink: str = "SS",
    gamma: str = "5",
    momentum: str = "PX0PY0PZ0",
    selectors: dict[str, Any] | None = None,
    out: str = "correlator_scale_inspection",
) -> dict[str, Any]:
    """Inspect 2pt correlator magnitudes so the agent can choose a rescale factor."""
    if selectors is not None:
        source_sink = str(selectors.get("source_sink") or source_sink)
        gamma = str(selectors.get("gamma") or selectors.get("pt2_gamma") or gamma)
        momentum = str(selectors.get("momentum") or momentum)
    pt2_complex = _read_pt2_complex(
        pt2_path,
        source_sink=source_sink,
        gamma=gamma,
        momentum=momentum,
    )
    pt2_real = np.real(pt2_complex)
    n_cfg, Lt = pt2_real.shape
    windows = _normalise_pt2_windows(pt2_windows, Lt=int(Lt))
    window_stats: list[dict[str, Any]] = []
    for window in windows:
        tmin = int(window["tmin"])
        tmax = int(window["tmax"])
        values = np.abs(pt2_real[:, tmin:tmax]).reshape(-1)
        nonzero = values[values > 0.0]
        min_abs_nonzero = float(np.min(nonzero)) if nonzero.size else None
        window_stats.append(
            {
                "tmin": tmin,
                "tmax": tmax,
                "median_abs": float(np.median(values)),
                "max_abs": float(np.max(values)),
                "min_abs_nonzero": min_abs_nonzero,
            }
        )
    result = {
        "out": out,
        "pt2_path": pt2_path,
        "source_sink": source_sink,
        "gamma": gamma,
        "momentum": momentum,
        "n_cfg": int(n_cfg),
        "Lt": int(Lt),
        "windows": window_stats,
        "target_typical_abs_range": [0.1, 1.0],
    }
    store[out] = result
    return result


def pt2_re_fcn(t: np.ndarray, p: dict, Lt: int, nstate: int = 2) -> np.ndarray:
    """Real part of the n-state two-point correlator."""
    val = 0.0
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]
        z = p[f"z{state}"]
        val = val + z**2 / (2 * energy) * (
            np.exp(-energy * t) + np.exp(-energy * (Lt - t))
        )
    return val


def pt2_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for an n-state two-point ground-state fit."""
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(1, 10)
    for state in range(1, nstate):
        prior[f"log(dE{state})"] = gv.gvar(0, 10)
    for state in range(nstate):
        prior[f"z{state}"] = gv.gvar(1, 2) / 10**state 
    return prior


def pt2_fit(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    nstate: int = 2,
    svdcut: float = 1e-2,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with an n-state spectral decomposition.

    ``svdcut`` regularizes the strongly correlated 2pt covariance; without it
    the correlated chi-square is dominated by near-singular noise modes.
    """
    scale = _validate_correlator_rescale(correlator_rescale)
    fit_t = np.arange(tmin, tmax, dtype=int)
    fit_pt2 = np.asarray(pt2_gv)[fit_t] * scale

    def fcn(t: np.ndarray, p: dict) -> np.ndarray:
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

    kwargs: dict[str, Any] = {}
    if p0 is not None:
        kwargs["p0"] = p0
    return lsf.nonlinear_fit(
        data=(fit_t, fit_pt2),
        prior=pt2_prior(nstate),
        fcn=fcn,
        svdcut=svdcut,
        maxit=10000,
        **kwargs,
    )


# --- 3pt ratio fit (copied from LaMETLat ground_state) ----------------------

PT2_PRIOR_ERROR_SCALE = 5.0


def _validate_nstate(nstate: int) -> None:
    if isinstance(nstate, bool) or not isinstance(nstate, int) or nstate < 1:
        raise ValueError("nstate must be a positive integer")


def _fit_parts(part: str) -> tuple[str, ...]:
    if part == "both":
        return ("re", "im")
    if part in {"re", "im"}:
        return (part,)
    raise ValueError("part must be 're', 'im', or 'both'")


def _asymptotic_ratio_gvar(
    O00: gv.GVar,
    E0: gv.GVar,
    *,
    tsep: int,
    Lt: int,
) -> gv.GVar:
    """Ratio at symmetric tau when only the ground state contributes (wrap-aware)."""
    tsep_f = float(int(tsep))
    forward = gv.exp(-E0 * tsep_f)
    backward = gv.exp(-E0 * (float(int(Lt)) - tsep_f))
    return O00 * forward / (2 * E0 * (forward + backward))


def asymptotic_ratio_real_gvar(
    O00_re: gv.GVar,
    E0: gv.GVar,
    *,
    tsep: int,
    Lt: int,
) -> gv.GVar:
    """Real ratio plateau; equals ``O00_re/(2*E0)`` when backward wrap is negligible."""
    return _asymptotic_ratio_gvar(O00_re, E0, tsep=tsep, Lt=Lt)


def asymptotic_ratio_imag_gvar(
    O00_im: gv.GVar,
    E0: gv.GVar,
    *,
    tsep: int,
    Lt: int,
) -> gv.GVar:
    """Imaginary ratio plateau (same kinematic factor as the real part)."""
    return _asymptotic_ratio_gvar(O00_im, E0, tsep=tsep, Lt=Lt)


def pt3_ratio_re_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Real part of the n-state 3pt/2pt ratio."""
    return _pt3_ratio_fcn(ra_t, ra_tau, p, Lt, nstate=nstate, part="re")


def pt3_ratio_im_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Imaginary part of the n-state 3pt/2pt ratio."""
    return _pt3_ratio_fcn(ra_t, ra_tau, p, Lt, nstate=nstate, part="im")


def _pt3_ratio_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int,
    part: str,
) -> float | np.ndarray:
    _validate_nstate(nstate)
    energies = []
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]
        energies.append(energy)

    numerator = 0.0
    for source_state, source_energy in enumerate(energies):
        for sink_state, sink_energy in enumerate(energies):
            row = min(source_state, sink_state)
            col = max(source_state, sink_state)
            matrix_element = p[f"O{row}{col}_{part}"]
            numerator = numerator + (
                matrix_element
                * p[f"z{source_state}"]
                * p[f"z{sink_state}"]
                * np.exp(-source_energy * (ra_t - ra_tau))
                * np.exp(-sink_energy * ra_tau)
                / (2 * source_energy)
                / (2 * sink_energy)
            )
    return numerator / pt2_re_fcn(ra_t, p, Lt, nstate=nstate)


def pt3_ratio_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for an n-state 3pt/2pt ratio fit."""
    _validate_nstate(nstate)
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(1, 10)
    for state in range(1, nstate):
        prior[f"log(dE{state})"] = gv.gvar(0, 10)
    for state in range(nstate):
        prior[f"z{state}"] = gv.gvar(1, 2) / 10**state 
    for row in range(nstate):
        for col in range(row, nstate):
            prior[f"O{row}{col}_re"] = gv.gvar(1, 10)
            prior[f"O{row}{col}_im"] = gv.gvar(1, 10)
    return prior


def _pt2_ground_state_prior_keys(nstate: int) -> list[str]:
    """2pt ground-state parameters anchored from 2pt posteriors into 3pt ratio priors."""
    _validate_nstate(nstate)
    return ["E0", "z0"]


def _pt2_avg_store_keys(nstate: int, *, suffix: str = "_avg") -> list[str]:
    return [f"{key}{suffix}" for key in _pt2_ground_state_prior_keys(nstate)]


def _pt2_posterior_as_prior(
    posterior: gv.GVar,
    *,
    error_scale: float = PT2_PRIOR_ERROR_SCALE,
) -> gv.GVar:
    """Use a 2pt posterior mean as a 3pt prior with inflated uncertainty."""
    return gv.gvar(gv.mean(posterior), gv.sdev(posterior) * error_scale)


def _update_prior_from_pt2_fit(
    prior: gv.BufferDict,
    pt2_fit_res: lsf.nonlinear_fit,
    nstate: int,
    *,
    error_scale: float = PT2_PRIOR_ERROR_SCALE,
) -> None:
    for key in _pt2_ground_state_prior_keys(nstate):
        prior[key] = _pt2_posterior_as_prior(pt2_fit_res.p[key], error_scale=error_scale)


def _ensure_pt2_avg_priors(
    store: dict[str, Any],
    *,
    pt2_scan: str,
    nstate: int,
    suffix: str = "_avg",
    window_indices: list[int] | None = None,
) -> list[str]:
    """Run ``model_average`` for missing ``E0_avg`` / ``z0_avg`` needed by 3pt fits."""
    if pt2_scan not in store:
        raise ValueError(
            f"cannot build 2pt priors: {pt2_scan!r} missing from store; run fit_window first."
        )
    indices = (
        list(window_indices)
        if window_indices is not None
        else list(range(len(store[pt2_scan])))
    )
    filled: list[str] = []
    for key in _pt2_ground_state_prior_keys(nstate):
        out_key = f"{key}{suffix}"
        if out_key in store:
            continue
        model_average(
            store,
            scan=pt2_scan,
            param=key,
            window_indices=indices,
            out=out_key,
        )
        filled.append(out_key)
    return filled


def _update_prior_from_pt2_avg(
    prior: gv.BufferDict,
    store: dict[str, Any],
    *,
    nstate: int,
    suffix: str = "_avg",
    error_scale: float = PT2_PRIOR_ERROR_SCALE,
) -> list[str]:
    """Pin E0 and z0 to widened 2pt BMA values; other params stay on ``pt3_ratio_prior``."""
    applied: list[str] = []
    missing: list[str] = []
    for key in _pt2_ground_state_prior_keys(nstate):
        store_key = f"{key}{suffix}"
        if store_key not in store:
            missing.append(store_key)
            continue
        prior[key] = _pt2_posterior_as_prior(store[store_key], error_scale=error_scale)
        applied.append(store_key)
    if missing:
        raise ValueError(
            "3pt ratio fit requires 2pt model-averaged ground-state posteriors: "
            f"{missing}. Run model_average on scan for "
            f"{_pt2_ground_state_prior_keys(nstate)} before fit_pt3_window."
        )
    return applied


def pt3_ratio_prior_from_pt2_avg(
    store: dict[str, Any],
    *,
    nstate: int = 2,
    suffix: str = "_avg",
) -> gv.BufferDict:
    """Return 3pt ratio priors: broad defaults except widened ``E0`` and ``z0`` from 2pt BMA."""
    prior = pt3_ratio_prior(nstate=nstate)
    _update_prior_from_pt2_avg(prior, store, nstate=nstate, suffix=suffix)
    return prior


def _resolve_pt3_prior(
    store: dict[str, Any],
    *,
    nstate: int,
    use_pt2_avg_prior: bool,
    pt2_scan: str,
    pt2_window_index: int | None,
    avg_suffix: str = "_avg",
) -> tuple[gv.BufferDict, list[str]]:
    """Build 3pt priors from 2pt BMA (default) or a single 2pt fit window."""
    prior = pt3_ratio_prior(nstate=nstate)
    if use_pt2_avg_prior:
        applied = _update_prior_from_pt2_avg(
            prior, store, nstate=nstate, suffix=avg_suffix
        )
        return prior, applied
    if pt2_window_index is not None:
        pt2_fit_res = store[pt2_scan][int(pt2_window_index)]["fit"]
        _update_prior_from_pt2_fit(prior, pt2_fit_res, nstate)
        return prior, [f"{pt2_scan}[{pt2_window_index}]"]
    return prior, []


def pt3_ratio_fit(
    tsep_ls: list[int],
    tau_cut: int,
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray],
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    part: str = "both",
    svdcut: float = 1e-2,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Fit real and imaginary 3pt/2pt ratio data with an n-state ansatz."""
    _validate_correlator_rescale(correlator_rescale)
    parts = _fit_parts(part)
    if prior is None:
        priors = pt3_ratio_prior(nstate=nstate)
        if pt2_fit_res is not None:
            _update_prior_from_pt2_fit(priors, pt2_fit_res, nstate)
    else:
        priors = prior

    ts: list[int] = []
    taus: list[int] = []
    fit_real: list = []
    fit_imag: list = []
    for tsep in tsep_ls:
        if tsep not in ratio_real or tsep not in ratio_imag:
            raise KeyError(f"ratio data missing tsep {tsep}")
        tau_range = range(int(tau_cut), int(tsep) + 1 - int(tau_cut))
        if len(tau_range) == 0:
            raise ValueError(f"empty tau fit window for tsep {tsep} and tau_cut {tau_cut}")
        real_row = np.asarray(ratio_real[tsep], dtype=object)
        imag_row = np.asarray(ratio_imag[tsep], dtype=object)
        for tau in tau_range:
            ts.append(int(tsep))
            taus.append(int(tau))
            fit_real.append(real_row[tau])
            fit_imag.append(imag_row[tau])

    x_vecs = [np.array(ts, dtype=float), np.array(taus, dtype=float)]
    all_y = {"re": fit_real, "im": fit_imag}
    y_data = {key: all_y[key] for key in parts}

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values = {
            "re": pt3_ratio_re_fcn(x[0], x[1], p, Lt, nstate=nstate),
            "im": pt3_ratio_im_fcn(x[0], x[1], p, Lt, nstate=nstate),
        }
        return {key: values[key] for key in parts}

    kwargs: dict[str, Any] = {}
    if p0 is not None:
        kwargs["p0"] = p0
    return lsf.nonlinear_fit(
        data=(x_vecs, y_data),
        prior=priors,
        fcn=fcn,
        svdcut=svdcut,
        maxit=10000,
        **kwargs,
    )


def _n_pt3_fit_params(nstate: int) -> int:
    """Nonlinear parameters for an n-state 3pt ratio fit (re+im)."""
    return 2 * int(nstate) + int(nstate) * (int(nstate) + 1)


def _count_pt3_data(tsep_ls: list[int], tau_cut: int, *, n_parts: int = 2) -> int:
    return int(n_parts) * sum(int(tsep) + 1 - 2 * int(tau_cut) for tsep in tsep_ls)


def _validate_pt3_fit_window(
    *,
    tsep_ls: list[int],
    tau_cut: int,
    nstate: int,
    part: str,
    append: bool,
    n_existing: int,
    ratio_real: dict[int, np.ndarray],
) -> None:
    if int(tau_cut) < 1:
        raise ValueError(f"tau_cut must be >= 1 for 3pt ratio fits, got {tau_cut}")
    if not tsep_ls:
        raise ValueError("tsep_ls must be a non-empty list of integers")
    for tsep in tsep_ls:
        if int(tsep) not in ratio_real:
            raise ValueError(f"tsep {tsep} not found in ratio data")
        if len(range(int(tau_cut), int(tsep) + 1 - int(tau_cut))) == 0:
            raise ValueError(f"empty tau window for tsep {tsep} and tau_cut {tau_cut}")
    n_parts = len(_fit_parts(part))
    n_data = _count_pt3_data([int(t) for t in tsep_ls], int(tau_cut), n_parts=n_parts)
    n_params = _n_pt3_fit_params(int(nstate))
    if n_data < n_params:
        raise ValueError(
            f"3pt fit has {n_data} data points but needs at least {n_params} "
            f"for nstate={nstate} and part={part!r}"
        )
    if append and n_existing >= MAX_PT3_FIT_WINDOWS:
        raise ValueError(
            f"at most {MAX_PT3_FIT_WINDOWS} 3pt fit windows when append=True "
            f"(already have {n_existing})"
        )


def _pt3_fit_record(
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray],
    *,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    prior: gv.BufferDict | None = None,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> dict[str, Any]:
    scale = _validate_correlator_rescale(correlator_rescale)
    fit = pt3_ratio_fit(
        [int(t) for t in tsep_ls],
        int(tau_cut),
        ratio_real,
        ratio_imag,
        int(Lt),
        nstate=int(nstate),
        part=part,
        svdcut=float(svdcut),
        prior=prior,
        p0=p0,
        correlator_rescale=scale,
    )
    return {
        "tsep_ls": [int(t) for t in tsep_ls],
        "tau_cut": int(tau_cut),
        "nstate": int(nstate),
        "part": part,
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "logGBF": float(fit.logGBF),
        "correlator_rescale": scale,
        "fit": fit,
    }


def pt2_ratio_joint_fit(
    pt2_gv: np.ndarray,
    *,
    tmin: int,
    tmax: int,
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    nstate: int = 2,
    prior: gv.BufferDict | None = None,
    part: str = "both",
    svdcut: float = 1e-2,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Jointly fit 2pt data and real/imag 3pt/2pt ratios."""
    scale = _validate_correlator_rescale(correlator_rescale)
    parts = _fit_parts(part)
    priors = prior if prior is not None else pt3_ratio_prior(nstate=nstate)
    fit_t = np.arange(int(tmin), int(tmax), dtype=int)
    fit_pt2 = np.asarray(pt2_gv)[fit_t] * scale

    ts: list[int] = []
    taus: list[int] = []
    fit_real: list = []
    fit_imag: list = []
    for tsep in tsep_ls:
        if int(tsep) not in ratio_real or int(tsep) not in ratio_imag:
            raise KeyError(f"ratio data missing tsep {tsep}")
        tau_range = range(int(tau_cut), int(tsep) + 1 - int(tau_cut))
        if len(tau_range) == 0:
            raise ValueError(f"empty tau fit window for tsep {tsep} and tau_cut {tau_cut}")
        real_row = np.asarray(ratio_real[int(tsep)], dtype=object)
        imag_row = np.asarray(ratio_imag[int(tsep)], dtype=object)
        for tau in tau_range:
            ts.append(int(tsep))
            taus.append(int(tau))
            fit_real.append(real_row[tau])
            fit_imag.append(imag_row[tau])

    x_data = {
        "pt2_t": fit_t,
        "ratio_t": np.array(ts, dtype=float),
        "ratio_tau": np.array(taus, dtype=float),
    }
    y_data: dict[str, Any] = {"pt2": fit_pt2}
    if "re" in parts:
        y_data["ratio_re"] = fit_real
    if "im" in parts:
        y_data["ratio_im"] = fit_imag

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {
            "pt2": pt2_re_fcn(x["pt2_t"], p, int(Lt), nstate=int(nstate)),
        }
        if "re" in parts:
            values["ratio_re"] = pt3_ratio_re_fcn(
                x["ratio_t"], x["ratio_tau"], p, int(Lt), nstate=int(nstate)
            )
        if "im" in parts:
            values["ratio_im"] = pt3_ratio_im_fcn(
                x["ratio_t"], x["ratio_tau"], p, int(Lt), nstate=int(nstate)
            )
        return values

    kwargs: dict[str, Any] = {}
    if p0 is not None:
        kwargs["p0"] = p0
    return lsf.nonlinear_fit(
        data=(x_data, y_data),
        prior=priors,
        fcn=fcn,
        svdcut=float(svdcut),
        maxit=10000,
        **kwargs,
    )


def _joint_fit_record(
    pt2_gv: np.ndarray,
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray],
    *,
    tmin: int,
    tmax: int,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    prior: gv.BufferDict | None = None,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> dict[str, Any]:
    scale = _validate_correlator_rescale(correlator_rescale)
    fit = pt2_ratio_joint_fit(
        pt2_gv,
        tmin=int(tmin),
        tmax=int(tmax),
        ratio_real=ratio_real,
        ratio_imag=ratio_imag,
        tsep_ls=[int(t) for t in tsep_ls],
        tau_cut=int(tau_cut),
        Lt=int(Lt),
        nstate=int(nstate),
        prior=prior,
        part=part,
        svdcut=float(svdcut),
        p0=p0,
        correlator_rescale=scale,
    )
    return {
        "fit_mode": "joint_2pt_ratio",
        "tmin": int(tmin),
        "tmax": int(tmax),
        "tsep_ls": [int(t) for t in tsep_ls],
        "tau_cut": int(tau_cut),
        "nstate": int(nstate),
        "part": part,
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "logGBF": float(fit.logGBF),
        "correlator_rescale": scale,
        "fit": fit,
    }


def fit_pt3_window(
    store: dict[str, Any],
    *,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int | None = None,
    ratio_real_gv: str = "ratio_real_gv",
    ratio_imag_gv: str = "ratio_imag_gv",
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    pt2_scan: str = "scan",
    use_pt2_avg_prior: bool = True,
    pt2_window_index: int | None = None,
    pt2_ma_window_indices: list[int] | None = None,
    correlator_rescale: float = 1.0,
    out: str = "pt3_scan",
    append: bool = True,
) -> dict[str, Any]:
    """Fit one (tsep_ls, tau_cut) 3pt ratio window; append to ``out``.

    By default, ``E0`` and ``z0`` use model-averaged 2pt posteriors with errors
    inflated by ``PT2_PRIOR_ERROR_SCALE`` (5); ``log(dE*)``, other ``z*``, and
    ``O_ij`` keep the broad ``pt3_ratio_prior`` defaults.
    """
    if Lt is None:
        Lt = _infer_Lt(store)
    n_existing = len(store[out]) if append and out in store else 0
    ratio_real = store[ratio_real_gv]
    ratio_imag = store[ratio_imag_gv]
    autofill: list[str] = []
    if use_pt2_avg_prior:
        autofill = _ensure_pt2_avg_priors(
            store,
            pt2_scan=pt2_scan,
            nstate=int(nstate),
            window_indices=pt2_ma_window_indices,
        )
    _validate_pt3_fit_window(
        tsep_ls=tsep_ls,
        tau_cut=int(tau_cut),
        nstate=int(nstate),
        part=part,
        append=bool(append),
        n_existing=n_existing,
        ratio_real=ratio_real,
    )
    prior, prior_sources = _resolve_pt3_prior(
        store,
        nstate=int(nstate),
        use_pt2_avg_prior=bool(use_pt2_avg_prior),
        pt2_scan=pt2_scan,
        pt2_window_index=pt2_window_index,
    )
    record = _pt3_fit_record(
        ratio_real,
        ratio_imag,
        tsep_ls=tsep_ls,
        tau_cut=int(tau_cut),
        Lt=int(Lt),
        nstate=int(nstate),
        part=part,
        svdcut=float(svdcut),
        prior=prior,
        correlator_rescale=correlator_rescale,
    )
    if append and out in store:
        records = list(store[out])
        records.append(record)
    else:
        records = [record]
    store[out] = records
    index = len(records) - 1
    fit = record["fit"]
    result: dict[str, Any] = {
        "out": out,
        "index": index,
        "tsep_ls": record["tsep_ls"],
        "tau_cut": record["tau_cut"],
        "chi2_dof": record["chi2_dof"],
        "Q": record["Q"],
        "logGBF": record["logGBF"],
        "O00_re": str(fit.p["O00_re"]),
        "E0": str(fit.p["E0"]),
        "correlator_rescale": float(record["correlator_rescale"]),
        "n_windows": len(records),
    }
    if prior_sources:
        result["pt2_prior_from"] = prior_sources
    if autofill:
        result["pt2_prior_autofill"] = autofill
    return result


# --- fit windows and model averaging ----------------------------------------

MAX_FIT_WINDOWS = 6
MAX_PT3_FIT_WINDOWS = 2


def _n_fit_params(nstate: int) -> int:
    """Number of nonlinear fit parameters for an n-state 2pt spectral fit."""
    return 2 * int(nstate)


def _validate_fit_window(
    *,
    tmin: int,
    tmax: int,
    Lt: int,
    nstate: int,
    append: bool,
    n_existing: int,
) -> None:
    if int(tmin) < 1:
        raise ValueError(f"tmin must be >= 1, got {tmin}")
    n_data = int(tmax) - int(tmin)
    n_params = _n_fit_params(nstate)
    if n_data < n_params:
        raise ValueError(
            f"fit window has {n_data} data points but needs at least {n_params} "
            f"for nstate={nstate} (tmax - tmin >= {n_params})"
        )
    if append and n_existing >= MAX_FIT_WINDOWS:
        raise ValueError(
            f"at most {MAX_FIT_WINDOWS} fit windows when append=True "
            f"(already have {n_existing}); use append=False to reset or pick "
            "window_indices from existing scans"
        )
    half = int(Lt) // 2
    if int(tmax) > half or int(tmin) >= half:
        raise ValueError(
            f"fit window must stay in the first half (tmax <= {half}, tmin < {half})"
        )


def _fit_window_warning(tmin: int, tmax: int, Lt: int) -> str | None:
    """Return a soft warning when the window extends past Lt/2."""
    half = Lt // 2
    if tmax > half or tmin >= half:
        return (
            "2pt is symmetric about t = Lt/2; fit windows are usually kept "
            f"in the first half (tmax <= {half}, tmin < Lt/2)."
        )
    return None


def _fit_record(
    data: np.ndarray,
    *,
    tmin: int,
    tmax: int,
    Lt: int,
    nstate: int,
    svdcut: float,
    p0: dict[str, float] | None = None,
    correlator_rescale: float = 1.0,
) -> dict[str, Any]:
    scale = _validate_correlator_rescale(correlator_rescale)
    fit = pt2_fit(
        data,
        int(tmin),
        int(tmax),
        int(Lt),
        nstate=int(nstate),
        svdcut=float(svdcut),
        p0=p0,
        correlator_rescale=scale,
    )
    return {
        "tmin": int(tmin),
        "tmax": int(tmax),
        "nstate": int(nstate),
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "logGBF": float(fit.logGBF),
        "correlator_rescale": scale,
        "fit": fit,
    }


def fit_window(
    store: dict[str, Any],
    *,
    pt2_gv: str = "pt2_gv",
    tmin: int,
    tmax: int,
    Lt: int,
    nstate: int = 2,
    svdcut: float = 1e-2,
    correlator_rescale: float = 1.0,
    out: str = "scan",
    append: bool = True,
) -> dict[str, Any]:
    """Fit one [tmin, tmax) window and append or replace records in ``out``."""
    n_existing = len(store[out]) if append and out in store else 0
    _validate_fit_window(
        tmin=int(tmin),
        tmax=int(tmax),
        Lt=int(Lt),
        nstate=int(nstate),
        append=bool(append),
        n_existing=n_existing,
    )
    data = store[pt2_gv]
    record = _fit_record(
        data,
        tmin=int(tmin),
        tmax=int(tmax),
        Lt=int(Lt),
        nstate=int(nstate),
        svdcut=float(svdcut),
        correlator_rescale=correlator_rescale,
    )
    if append and out in store:
        records = list(store[out])
        records.append(record)
    else:
        records = [record]
    store[out] = records
    index = len(records) - 1
    warning = _fit_window_warning(int(tmin), int(tmax), int(Lt))
    result: dict[str, Any] = {
        "out": out,
        "index": index,
        "tmin": record["tmin"],
        "tmax": record["tmax"],
        "chi2_dof": record["chi2_dof"],
        "Q": record["Q"],
        "logGBF": record["logGBF"],
        "E0": str(record["fit"].p["E0"]),
        "correlator_rescale": float(record["correlator_rescale"]),
        "n_windows": len(records),
    }
    if warning is not None:
        result["warning"] = warning
    return result


def _select_records(
    records: list[dict[str, Any]],
    window_indices: list[int] | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    if window_indices is None:
        return records, list(range(len(records)))
    selected = [records[i] for i in window_indices]
    return selected, list(window_indices)


def _logGBF_weights(records: list[dict[str, Any]]) -> np.ndarray:
    log_gbf = np.array([rec["logGBF"] for rec in records], dtype=float)
    weights = np.exp(log_gbf - np.max(log_gbf))
    return weights / np.sum(weights)


def _bayesian_average(values: np.ndarray, weights: np.ndarray) -> gv.GVar:
    """Combine fit values with statistical and systematic spread (BMA)."""
    mean = np.sum(weights * gv.mean(values))
    var = np.sum(weights * (gv.sdev(values) ** 2 + gv.mean(values) ** 2)) - mean**2
    return gv.gvar(mean, np.sqrt(var))


def model_average(
    store: dict[str, Any],
    *,
    scan: str = "scan",
    param: str = "E0",
    window_indices: list[int] | None = None,
    out: str | None = None,
) -> dict[str, Any]:
    """logGBF-weighted model average of one fit parameter across windows."""
    records, indices = _select_records(store[scan], window_indices)
    if not records:
        raise ValueError("no fit windows selected for model_average")
    weights = _logGBF_weights(records)
    values = np.array([rec["fit"].p[param] for rec in records], dtype=object)
    averaged = _bayesian_average(values, weights)

    stat = float(np.sum(weights * gv.sdev(values)))
    sys = float(np.sqrt(max(averaged.sdev**2 - stat**2, 0.0)))
    key = out if out is not None else f"{param}_avg"
    store[key] = averaged
    return {
        "out": key,
        "param": param,
        "value": str(averaged),
        "mean": float(averaged.mean),
        "sdev": float(averaged.sdev),
        "stat": stat,
        "sys": sys,
        "n_windows": len(records),
        "window_indices": indices,
    }


def _window_fit_band(rec: dict[str, Any], Lt: int) -> tuple[np.ndarray, np.ndarray]:
    fit_t = np.arange(rec["tmin"], rec["tmax"], dtype=int)
    fit_gv = pt2_re_fcn(fit_t, rec["fit"].p, int(Lt), nstate=rec["nstate"])
    fit_gv = fit_gv / float(rec.get("correlator_rescale", 1.0))
    return fit_t, fit_gv


def plot_fit_on_data(
    store: dict[str, Any],
    *,
    pt2_gv: str = "pt2_gv",
    scan: str = "scan",
    window_indices: list[int] | None = None,
    E0_avg: str = "E0_avg",
    Lt: int | None = None,
    boundary: str = "periodic",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str | None = None,
) -> dict[str, Any]:
    """Plot per-window fit bands on C2pt and meff, plus model-averaged E0 on meff."""
    del out
    if Lt is None:
        Lt = _infer_Lt(store)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_save = resolve_plot_save_path(save_path, artifacts_dir=out_dir)
    data = store[pt2_gv]
    records, indices = _select_records(store[scan], window_indices)
    if not records:
        raise ValueError("no fit windows selected for plot_fit_on_data")

    fit_bands = []
    for i, rec in enumerate(records):
        fit_t, fit_gv = _window_fit_band(rec, int(Lt))
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        fit_bands.append(
            {
                "fit_t": fit_t,
                "fit_gv": fit_gv,
                "label": f"t=[{rec['tmin']},{rec['tmax']})",
                "color": color,
            }
        )

    e0_band = store.get(E0_avg)
    plot_pt2_fit_on_data(
        data,
        boundary=boundary,
        fit_bands=fit_bands,
        E0_band=e0_band,
        save_path=resolved_save,
    )
    result: dict[str, Any] = {
        "c2pt_pdf": f"{resolved_save}_c2pt.pdf",
        "meff_pdf": f"{resolved_save}_meff.pdf",
        "n_bands": len(fit_bands),
        "window_indices": indices,
    }
    if e0_band is not None:
        result["E0_band"] = str(e0_band)
    return result


TSEP = r"$t_{\mathrm{sep}}$"


def _pt3_window_fit_bands(
    rec: dict[str, Any],
    Lt: int,
) -> list[dict[str, Any]]:
    """Build smooth fit bands per tsep for one 3pt scan record."""
    bands = []
    tau_cut = int(rec["tau_cut"])
    nstate = int(rec["nstate"])
    p = rec["fit"].p
    for i, tsep in enumerate(rec["tsep_ls"]):
        fit_tau = np.linspace(tau_cut - 0.5, tsep - tau_cut + 0.5, 200)
        fit_t = np.full_like(fit_tau, float(tsep))
        fit_re = pt3_ratio_re_fcn(fit_t, fit_tau, p, int(Lt), nstate=nstate)
        fit_im = pt3_ratio_im_fcn(fit_t, fit_tau, p, int(Lt), nstate=nstate)
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        bands.append(
            {
                "tsep": int(tsep),
                "tau_cut": tau_cut,
                "fit_tau": fit_tau,
                "fit_re": fit_re,
                "fit_im": fit_im,
                "label": f"{TSEP}={tsep}",
                "color": color,
            }
        )
    return bands


def plot_pt3_fit_on_data(
    store: dict[str, Any],
    *,
    ratio_real_gv: str = "ratio_real_gv",
    ratio_imag_gv: str = "ratio_imag_gv",
    scan: str = "pt3_scan",
    window_indices: list[int] | None = None,
    O00_re_avg: str = "O00_re_avg",
    O00_im_avg: str = "O00_im_avg",
    Lt: int | None = None,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Plot per-window 3pt ratio fit bands and optional model-averaged matrix element."""
    if Lt is None:
        Lt = _infer_Lt(store)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_save = resolve_plot_save_path(save_path, artifacts_dir=out_dir)
    ratio_real = store[ratio_real_gv]
    ratio_imag = store[ratio_imag_gv]
    records, indices = _select_records(store[scan], window_indices)
    if not records:
        raise ValueError("no fit windows selected for plot_pt3_fit_on_data")

    window_bands = []
    for rec in records:
        window_bands.append(
            {
                "record_label": (
                    f"tsep={rec['tsep_ls']}, tau_cut={rec['tau_cut']}"
                ),
                "bands": _pt3_window_fit_bands(rec, int(Lt)),
                "fit": rec["fit"],
            }
        )

    e0_avg = store.get("E0_avg")
    if e0_avg is not None and O00_im_avg not in store:
        model_average(
            store,
            scan=scan,
            param="O00_im",
            window_indices=indices,
            out=O00_im_avg,
        )

    correction_energy = e0_avg if e0_avg is not None else records[0]["fit"].p["E0"]
    plateau_re = None
    plateau_im = None
    if e0_avg is not None:
        o00_re_avg = store.get(O00_re_avg)
        if o00_re_avg is not None:
            plateau_re = o00_re_avg / (2 * e0_avg)
        o00_im_avg = store.get(O00_im_avg)
        if o00_im_avg is not None:
            plateau_im = o00_im_avg / (2 * e0_avg)

    plot_pt3_ratio_fit_on_data(
        ratio_real,
        ratio_imag,
        denominator_correction_energy=correction_energy,
        denominator_correction_Lt=int(Lt),
        window_bands=window_bands,
        plateau_ref_re=plateau_re,
        plateau_ref_im=plateau_im,
        save_path=resolved_save,
    )
    return {
        "ratio_re_pdf": f"{resolved_save}_pt3_ratio_re.pdf",
        "ratio_im_pdf": f"{resolved_save}_pt3_ratio_im.pdf",
        "n_windows": len(window_bands),
        "window_indices": indices,
    }


# --- batch bare-matrix export ------------------------------------------------


def _progress(iterable, *, desc: str):
    """Use tqdm when available, otherwise return the iterable unchanged."""
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc)



def _recenter_gvar(mean: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Use ``template`` covariance with a replacement mean vector."""
    return gv.gvar(np.asarray(mean, dtype=float), gv.evalcov(template))


def _fit_p0_from_prior(fit: lsf.nonlinear_fit, prior: gv.BufferDict) -> dict[str, float]:
    """Build an lsqfit p0 dict using only keys present in the prior."""
    p0: dict[str, float] = {}
    for key in prior:
        try:
            p0[key] = float(gv.mean(fit.p[key]))
        except Exception:
            p0[key] = float(gv.mean(prior[key]))
    return p0


def _scaled_posterior_as_prior(
    fit: lsf.nonlinear_fit,
    template: gv.BufferDict,
    *,
    error_scale: float = 3.0,
) -> gv.BufferDict:
    """Use a fit posterior as a prior with inflated uncertainties."""
    prior = gv.BufferDict()
    for key in template:
        value = fit.p[key] if key in fit.p else template[key]
        prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * float(error_scale))
    return prior


def _fit_posterior_is_usable(
    fit: lsf.nonlinear_fit,
    template: gv.BufferDict,
    *,
    sdev_floor: float = 1e-12,
    e0_floor: float = 1e-4,
) -> tuple[bool, str | None]:
    """Reject non-finite or numerically degenerate posteriors before sample fits."""
    for key in template:
        if key not in fit.p:
            return False, f"missing posterior key {key}"
        mean = float(gv.mean(fit.p[key]))
        sdev = float(gv.sdev(fit.p[key]))
        if not np.isfinite(mean) or not np.isfinite(sdev):
            return False, f"non-finite posterior for {key}: mean={mean}, sdev={sdev}"
        if sdev <= float(sdev_floor):
            return False, f"degenerate posterior for {key}: sdev={sdev}"
    e0 = float(gv.mean(fit.p["E0"]))
    if e0 <= float(e0_floor):
        return False, f"non-physical E0 posterior: E0={e0}"
    return True, None


def _scan_joint_average_windows(
    pt2_avg_gv: np.ndarray,
    ratio_real_avg: dict[int, np.ndarray],
    ratio_imag_avg: dict[int, np.ndarray],
    *,
    pt2_windows: list[dict[str, int]],
    pt3_windows: list[dict[str, Any]],
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    prior: gv.BufferDict,
    correlator_rescale: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan joint 2pt+ratio windows and return usable records plus rejections."""
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    template = pt3_ratio_prior(nstate=int(nstate))
    for pt2_window in pt2_windows:
        for pt3_window in pt3_windows:
            meta = {
                "tmin": int(pt2_window["tmin"]),
                "tmax": int(pt2_window["tmax"]),
                "tsep_ls": [int(t) for t in pt3_window["tsep_ls"]],
                "tau_cut": int(pt3_window["tau_cut"]),
            }
            try:
                _validate_fit_window(
                    tmin=meta["tmin"],
                    tmax=meta["tmax"],
                    Lt=int(Lt),
                    nstate=int(nstate),
                    append=False,
                    n_existing=0,
                )
                _validate_pt3_fit_window(
                    tsep_ls=meta["tsep_ls"],
                    tau_cut=meta["tau_cut"],
                    nstate=int(nstate),
                    part=part,
                    append=False,
                    n_existing=0,
                    ratio_real=ratio_real_avg,
                )
                record = _joint_fit_record(
                    pt2_avg_gv,
                    ratio_real_avg,
                    ratio_imag_avg,
                    tmin=meta["tmin"],
                    tmax=meta["tmax"],
                    tsep_ls=meta["tsep_ls"],
                    tau_cut=meta["tau_cut"],
                    Lt=int(Lt),
                    nstate=int(nstate),
                    part=part,
                    svdcut=float(svdcut),
                    prior=prior,
                    correlator_rescale=correlator_rescale,
                )
                usable, reason = _fit_posterior_is_usable(record["fit"], template)
                if not usable:
                    rejected.append({**meta, "reason": reason})
                    continue
                records.append(record)
            except Exception as exc:
                rejected.append({**meta, "reason": str(exc)})
    if not records:
        reasons = "; ".join(str(item) for item in rejected[:5])
        raise ValueError("all sample-average joint fit windows failed or were rejected: " + reasons)
    return records, rejected


def _write_sample0_ratio_plot(
    *,
    ratio_real_sample: dict[int, np.ndarray],
    ratio_imag_sample: dict[int, np.ndarray],
    fit_record: dict[str, Any],
    Lt: int,
    log_dir: Path,
    momentum: str,
    z: int,
    fit_label: str = "joint_fit",
) -> dict[str, str]:
    """Save sample-0 ratio fit-on-data plots only."""
    stem = log_dir / f"{fit_label}_{momentum}_z{int(z)}_sample0"
    fit_params = fit_record["fit"].p
    plateau_re = fit_params["O00_re"] / (2 * fit_params["E0"])
    plateau_im = fit_params["O00_im"] / (2 * fit_params["E0"])
    figures = plot_pt3_ratio_fit_on_data(
        ratio_real_sample,
        ratio_imag_sample,
        window_bands=[
            {
                "record_label": f"joint t=[{fit_record['tmin']},{fit_record['tmax']}), tau_cut={fit_record['tau_cut']}",
                "bands": _pt3_window_fit_bands(fit_record, int(Lt)),
                "fit": fit_record["fit"],
            }
        ],
        plateau_ref_re=plateau_re,
        plateau_ref_im=plateau_im,
        plateau_label=r"Sample-0 fit $\mathcal{O}_{00}/(2E_0)$",
        denominator_correction_energy=fit_params["E0"],
        denominator_correction_Lt=int(Lt),
        save_path=stem,
    )
    for fig, _ax in figures:
        plt.close(fig)
    return {
        "ratio_re_pdf": str(stem.with_name(f"{stem.name}_pt3_ratio_re.pdf")),
        "ratio_im_pdf": str(stem.with_name(f"{stem.name}_pt3_ratio_im.pdf")),
    }


def _normalise_pt2_windows(
    windows: list[dict[str, int]] | None,
    *,
    Lt: int,
) -> list[dict[str, int]]:
    if windows is not None:
        return [{"tmin": int(w["tmin"]), "tmax": int(w["tmax"])} for w in windows]
    quarter = max(int(Lt) // 4, 1)
    # Default windows target the common two-state fit, which needs at least 4 points.
    max_tmin = quarter - 4
    tmins = list(range(2, max_tmin + 1))
    return [{"tmin": tmin, "tmax": quarter} for tmin in tmins[:MAX_FIT_WINDOWS]]


def _normalise_pt3_windows(
    windows: list[dict[str, Any]] | None,
    *,
    tsep_ls: list[int],
    tau_cuts: list[int] | None,
) -> list[dict[str, Any]]:
    if windows is not None:
        return [
            {
                "tsep_ls": [int(t) for t in w.get("tsep_ls", tsep_ls)],
                "tau_cut": int(w["tau_cut"]),
            }
            for w in windows
        ]
    cuts = [int(cut) for cut in (tau_cuts if tau_cuts is not None else [1, 2, 3, 4])]
    return [{"tsep_ls": [int(t) for t in tsep_ls], "tau_cut": cut} for cut in cuts]


def _select_best_fit_index(
    records: list[dict[str, Any]],
    *,
    q_min: float = 0.05,
) -> tuple[int, bool]:
    """Select best window: max logGBF among Q-passing fits, otherwise max Q."""
    if not records:
        raise ValueError("cannot select a fit window from an empty scan")
    passing = [i for i, rec in enumerate(records) if float(rec.get("Q", 0.0)) >= float(q_min)]
    if passing:
        return max(passing, key=lambda i: float(records[i].get("logGBF", -np.inf))), False
    return max(range(len(records)), key=lambda i: float(records[i].get("Q", -np.inf))), True


def _fit_summary(rec: dict[str, Any], *, fallback: bool, index: int) -> dict[str, Any]:
    summary = {
        "index": int(index),
        "chi2_dof": float(rec["chi2_dof"]),
        "Q": float(rec["Q"]),
        "logGBF": float(rec["logGBF"]),
        "fallback_no_q_passing": bool(fallback),
    }
    for key in ("tmin", "tmax", "tsep_ls", "tau_cut", "nstate", "part", "correlator_rescale"):
        if key in rec:
            summary[key] = rec[key]
    fit = rec.get("fit")
    if fit is not None and "nstate" in rec:
        overlap = _physical_overlap_diagnostics(
            fit.p,
            int(rec["nstate"]),
            float(rec.get("correlator_rescale", 1.0)),
        )
        for key, value in overlap.items():
            summary[key] = str(value) if isinstance(value, gv.GVar) else value
    return summary


def _read_pt2_complex(
    path: str,
    *,
    source_sink: str,
    gamma: str,
    momentum: str,
) -> np.ndarray:
    with h5py.File(path, "r") as h5f:
        return np.swapaxes(np.asarray(h5f[source_sink][gamma][momentum]), 0, 1)


def _read_pt3_complex(
    path: str,
    *,
    source_sink: str,
    gamma: str,
    momentum: str,
    b_dir: str,
    eta: str,
    bt: str,
    bz: str,
    tsep: int,
) -> np.ndarray:
    dset = _pt3_dataset_path(
        source_sink=source_sink,
        gamma=gamma,
        momentum=momentum,
        b_dir=b_dir,
        eta=eta,
        bt=bt,
        bz=bz,
    )
    with h5py.File(path, "r") as h5f:
        data = np.swapaxes(np.asarray(h5f[dset]), 0, 1)
    expected_ntau = int(tsep) + 1
    if data.shape[1] != expected_ntau:
        raise ValueError(
            f"{path}:{dset} has ntau={data.shape[1]}, expected {expected_ntau} for tsep={tsep}"
        )
    return data


def _scan_pt2_average_windows(
    pt2_avg_gv: np.ndarray,
    *,
    windows: list[dict[str, int]],
    Lt: int,
    nstate: int,
    svdcut: float,
    correlator_rescale: float = 1.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for window in windows:
        try:
            _validate_fit_window(
                tmin=int(window["tmin"]),
                tmax=int(window["tmax"]),
                Lt=int(Lt),
                nstate=int(nstate),
                append=False,
                n_existing=0,
            )
            records.append(
                _fit_record(
                    pt2_avg_gv,
                    tmin=int(window["tmin"]),
                    tmax=int(window["tmax"]),
                    Lt=int(Lt),
                    nstate=int(nstate),
                    svdcut=float(svdcut),
                    correlator_rescale=correlator_rescale,
                )
            )
        except Exception as exc:
            errors.append(f"pt2 window {window}: {exc}")
    if not records:
        raise ValueError("all sample-average 2pt fit windows failed: " + "; ".join(errors))
    return records


def _scan_pt3_average_windows(
    ratio_real_avg: dict[int, np.ndarray],
    ratio_imag_avg: dict[int, np.ndarray],
    *,
    windows: list[dict[str, Any]],
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    prior: gv.BufferDict,
    correlator_rescale: float = 1.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for window in windows:
        try:
            tseps = [int(t) for t in window["tsep_ls"]]
            tau_cut = int(window["tau_cut"])
            _validate_pt3_fit_window(
                tsep_ls=tseps,
                tau_cut=tau_cut,
                nstate=int(nstate),
                part=part,
                append=False,
                n_existing=0,
                ratio_real=ratio_real_avg,
            )
            records.append(
                _pt3_fit_record(
                    ratio_real_avg,
                    ratio_imag_avg,
                    tsep_ls=tseps,
                    tau_cut=tau_cut,
                    Lt=int(Lt),
                    nstate=int(nstate),
                    part=part,
                    svdcut=float(svdcut),
                    prior=prior,
                    correlator_rescale=correlator_rescale,
                )
            )
        except Exception as exc:
            errors.append(f"pt3 window {window}: {exc}")
    if not records:
        raise ValueError("all sample-average 3pt fit windows failed: " + "; ".join(errors))
    return records


def _bare_matrix_samples_from_records(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    real: list[float] = []
    imag: list[float] = []
    for rec in records:
        fit = rec.get("fit")
        if fit is None:
            real.append(np.nan)
            imag.append(np.nan)
            continue
        real.append(float(gv.mean(fit.p["O00_re"] / (2 * fit.p["E0"]))))
        imag.append(float(gv.mean(fit.p["O00_im"] / (2 * fit.p["E0"]))))
    return np.asarray(real, dtype=float), np.asarray(imag, dtype=float)


def _bare_filename(
    *,
    ensemble: str,
    tag: str,
    variant: str,
    direction: str,
    momentum: str,
    b_label: str,
    z: int,
) -> str:
    return f"{ensemble}_{tag}_{variant}_{direction}_{momentum}_{b_label}_z{int(z)}.txt"


def _write_bare_matrix_grid_outputs(
    records: list[dict[str, Any]],
    *,
    artifacts_dir: str | Path,
    save_path: str | None,
    ensemble: str,
    tag: str,
    variant: str,
    direction: str,
    momentum: str,
    b_label: str,
    resample_mode: str,
    output_subdir: str = "bare_qpdf",
    ylim: tuple[float, float] = (-0.2, 1.2),
) -> dict[str, Any]:
    """Write per-z sample text files, summary plot, and JSON report."""
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = out_dir / output_subdir
    txt_dir.mkdir(parents=True, exist_ok=True)
    resolved_save = resolve_plot_save_path(
        save_path,
        artifacts_dir=out_dir,
        default_stem="bare_matrix_elements",
    )

    z_values: list[int] = []
    real_mean: list[float] = []
    real_err: list[float] = []
    imag_mean: list[float] = []
    imag_err: list[float] = []
    outputs: list[dict[str, Any]] = []

    for rec in sorted(records, key=lambda item: int(item["z"])):
        z = int(rec["z"])
        real = np.asarray(rec["real_samples"], dtype=float)
        imag = np.asarray(rec["imag_samples"], dtype=float)
        path = txt_dir / _bare_filename(
            ensemble=ensemble,
            tag=tag,
            variant=variant,
            direction=direction,
            momentum=momentum,
            b_label=b_label,
            z=z,
        )
        np.savetxt(path, np.column_stack([real, imag]), fmt="%.10e")
        r_mean, r_err = _sample_mean_err(real, mode=resample_mode)
        i_mean, i_err = _sample_mean_err(imag, mode=resample_mode)
        z_values.append(z)
        real_mean.append(r_mean)
        real_err.append(r_err)
        imag_mean.append(i_mean)
        imag_err.append(i_err)
        outputs.append(
            {
                "z": z,
                "path": str(path),
                "n_samples": int(real.shape[0]),
                "n_failed_samples": int(np.count_nonzero(~np.isfinite(real) | ~np.isfinite(imag))),
                "real_mean": r_mean,
                "real_sdev": r_err,
                "imag_mean": i_mean,
                "imag_sdev": i_err,
                "pt3_window": rec["pt3_window"],
                "joint_window": rec.get("joint_window", rec["pt3_window"]),
                "sample0_plot_paths": rec.get("sample0_plot_paths", {}),
            }
        )

    fig, ax = default_plot()
    ax.errorbar(
        z_values,
        real_mean,
        real_err,
        label="Re",
        color=COLOR_CYCLE[0],
        **ERRORBAR_STYLE,
    )
    ax.errorbar(
        z_values,
        imag_mean,
        imag_err,
        label="Im",
        color=COLOR_CYCLE[1],
        marker="s",
        **ERRORBAR_STYLE,
    )
    ax.set_xlabel(r"$z/a$", **FONT_SIZE)
    ax.set_ylabel(r"Bare matrix element $O_{00}/(2E_0)$", **FONT_SIZE)
    ax.set_title(f"{ensemble} {momentum} {direction} bare matrix elements", **FONT_SIZE)
    ax.set_ylim(*ylim)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    pdf_path = f"{resolved_save}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    report = {
        "ensemble": ensemble,
        "tag": tag,
        "variant": variant,
        "direction": direction,
        "momentum": momentum,
        "b_label": b_label,
        "resample_mode": resample_mode,
        "plot_ylim": [float(ylim[0]), float(ylim[1])],
        "output_subdir": str(txt_dir),
        "plot_pdf": pdf_path,
        "outputs": outputs,
    }
    report_path = f"{resolved_save}_report.json"
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "txt_dir": str(txt_dir),
        "plot_pdf": pdf_path,
        "report_json": report_path,
        "n_z": len(records),
        "n_txt": len(outputs),
        "plot_ylim": [float(ylim[0]), float(ylim[1])],
        "outputs": outputs,
    }



def _normalise_fit_strategy(value: str | None) -> tuple[str, str]:
    raw = "joint" if value is None else str(value).strip().lower()
    aliases = {
        "joint": ("joint", "joint_2pt_ratio"),
        "joint_2pt_ratio": ("joint", "joint_2pt_ratio"),
        "joint-fit": ("joint", "joint_2pt_ratio"),
        "chained": ("chained", "chained_2pt_ratio"),
        "chained_2pt_ratio": ("chained", "chained_2pt_ratio"),
        "chain": ("chained", "chained_2pt_ratio"),
    }
    if raw not in aliases:
        raise ValueError("fit_strategy must be 'joint' or 'chained', got %r" % value)
    return aliases[raw]


def _split_fit_log_paths(
    *,
    log_dir: Path,
    log_path: str | Path | None,
    ensemble: str,
    tag: str,
    variant: str,
    direction: str,
    momentum: str,
    b_label: str,
    fit_mode: str,
) -> tuple[Path, Path]:
    if log_path is not None:
        base = Path(log_path)
        suffix = base.suffix or ".log"
        return (
            base.with_name(f"{base.stem}_tuning{suffix}"),
            base.with_name(f"{base.stem}_samples{suffix}"),
        )
    stem = f"{ensemble}_{tag}_{variant}_{direction}_{momentum}_{b_label}_{fit_mode}"
    return log_dir / f"{stem}_tuning.log", log_dir / f"{stem}_samples.log"


def _normalise_pt3_paths(
    pt3_paths: dict[str, str] | list[str],
    *,
    tsep_ls: list[int],
) -> dict[int, str]:
    if isinstance(pt3_paths, dict):
        return {int(k): str(v) for k, v in pt3_paths.items()}
    if len(pt3_paths) != len(tsep_ls):
        raise ValueError("pt3_paths list length must match tsep_ls")
    return {int(tsep): str(path) for tsep, path in zip(tsep_ls, pt3_paths)}


def fit_bare_matrix_grid(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt3_paths: dict[str, str] | list[str],
    tsep_ls: list[int],
    z_values: list[int],
    ensemble: str,
    tag: str,
    momentum: str,
    direction: str = "X",
    variant: str = "free",
    source_sink: str = "SS",
    pt2_gamma: str = "5",
    pt3_gamma: str = "T",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bt: str = "bT0",
    b_label: str = "b0",
    pt2_windows: list[dict[str, int]] | None = None,
    pt3_windows: list[dict[str, Any]] | None = None,
    pt3_tau_cuts: list[int] | None = None,
    fit_strategy: str = "joint",
    nstate: int = 2,
    resample_mode: str = "bs",
    n_boot: int = 200,
    seed: int | None = 1984,
    svdcut: float = 1e-2,
    part: str = "both",
    q_min: float = 0.05,
    posterior_prior_error_scale: float = 3.0,
    correlator_rescale: float = 1.0,
    output_subdir: str = "bare_qpdf",
    save_path: str | None = None,
    log_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "bare_matrix_grid",
) -> dict[str, Any]:
    """Batch-fit bare matrix elements over z and export resampled samples."""
    del out
    strategy, fit_mode = _normalise_fit_strategy(fit_strategy)
    scale = _validate_correlator_rescale(correlator_rescale)
    if strategy == "chained":
        return _fit_bare_matrix_grid_chained(
            store,
            pt2_path=pt2_path,
            pt3_paths=pt3_paths,
            tsep_ls=tsep_ls,
            z_values=z_values,
            ensemble=ensemble,
            tag=tag,
            momentum=momentum,
            direction=direction,
            variant=variant,
            source_sink=source_sink,
            pt2_gamma=pt2_gamma,
            pt3_gamma=pt3_gamma,
            b_dir=b_dir,
            eta=eta,
            bt=bt,
            b_label=b_label,
            pt2_windows=pt2_windows,
            pt3_windows=pt3_windows,
            pt3_tau_cuts=pt3_tau_cuts,
            nstate=nstate,
            resample_mode=resample_mode,
            n_boot=n_boot,
            seed=seed,
            svdcut=svdcut,
            part=part,
            q_min=q_min,
            posterior_prior_error_scale=posterior_prior_error_scale,
            correlator_rescale=scale,
            output_subdir=output_subdir,
            save_path=save_path,
            log_dir=log_dir,
            log_path=log_path,
            artifacts_dir=artifacts_dir,
        )

    mode = str(resample_mode)
    if mode not in {"bs", "jk"}:
        raise ValueError(f"resample_mode must be 'bs' or 'jk', got {resample_mode!r}")
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    tuning_log_path, sample_log_path = _split_fit_log_paths(
        log_dir=fit_log_dir,
        log_path=log_path,
        ensemble=ensemble,
        tag=tag,
        variant=variant,
        direction=direction,
        momentum=momentum,
        b_label=b_label,
        fit_mode=fit_mode,
    )
    tuning_logger = setup_logger(tuning_log_path, console_output=False, logger_name="correlator_tuning_logger")
    sample_logger = setup_logger(sample_log_path, console_output=False, logger_name="correlator_sample_logger")
    tuning_logger.info("Starting joint 2pt+ratio bare matrix grid fit")
    tuning_logger.info("correlator_rescale=%s overlap_rescale=%s", scale, _overlap_rescale(scale))
    tuning_logger.info("ensemble=%s tag=%s momentum=%s direction=%s z_values=%s", ensemble, tag, momentum, direction, z_values)
    sample_logger.info("Starting joint 2pt+ratio per-sample fit log")

    tseps = [int(t) for t in tsep_ls]
    z_list = [int(z) for z in z_values]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    missing = [tsep for tsep in tseps if tsep not in paths_by_tsep]
    if missing:
        raise ValueError(f"pt3_paths missing tsep entries: {missing}")

    pt2_complex = _read_pt2_complex(
        pt2_path,
        source_sink=source_sink,
        gamma=pt2_gamma,
        momentum=momentum,
    )
    pt2_real = np.real(pt2_complex)
    n_cfg, Lt = pt2_real.shape
    pt2_samples, indices = _resample_config_samples(
        pt2_real,
        mode=mode,
        n_boot=int(n_boot),
        seed=seed,
    )
    pt2_gv = _samples_to_gvar(pt2_samples, mode=mode)
    pt2_window_specs = _normalise_pt2_windows(pt2_windows, Lt=int(Lt))
    pt3_window_specs = _normalise_pt3_windows(
        pt3_windows,
        tsep_ls=tseps,
        tau_cuts=pt3_tau_cuts,
    )
    tuning_logger.info("Lt=%s n_cfg=%s resample_mode=%s n_samples=%s", Lt, n_cfg, mode, pt2_samples.shape[0])
    tuning_logger.info("pt2_windows=%s pt3_windows=%s svdcut=%s", pt2_window_specs, pt3_window_specs, svdcut)

    z_records: list[dict[str, Any]] = []
    z_report: list[dict[str, Any]] = []
    joint_template = pt3_ratio_prior(nstate=int(nstate))

    for z in _progress(z_list, desc=f"fit bare matrix {ensemble} {momentum} {direction}"):
        tuning_logger.info("=== z=%s ===", z)
        bz = f"bz{z}"
        ratio_samples_re: dict[int, np.ndarray] = {}
        ratio_samples_im: dict[int, np.ndarray] = {}
        ratio_real_gv: dict[int, np.ndarray] = {}
        ratio_imag_gv: dict[int, np.ndarray] = {}

        for tsep in tseps:
            pt3 = _read_pt3_complex(
                paths_by_tsep[tsep],
                source_sink=source_sink,
                gamma=pt3_gamma,
                momentum=momentum,
                b_dir=b_dir,
                eta=eta,
                bt=bt,
                bz=bz,
                tsep=tsep,
            )
            if pt3.shape[0] != n_cfg:
                raise ValueError(
                    f"3pt n_cfg mismatch for z={z}, tsep={tsep}: {pt3.shape[0]} != {n_cfg}"
                )
            ratio = pt3 / pt2_complex[:, int(tsep)][:, None]
            ratio_samples, _ = _resample_config_samples(
                ratio,
                mode=mode,
                n_boot=int(n_boot),
                seed=seed,
                indices=indices,
            )
            ratio_samples_re[tsep] = np.real(ratio_samples)
            ratio_samples_im[tsep] = np.imag(ratio_samples)
            ratio_real_gv[tsep] = _samples_to_gvar(ratio_samples_re[tsep], mode=mode)
            ratio_imag_gv[tsep] = _samples_to_gvar(ratio_samples_im[tsep], mode=mode)

        avg_records, rejected_windows = _scan_joint_average_windows(
            pt2_gv,
            ratio_real_gv,
            ratio_imag_gv,
            pt2_windows=pt2_window_specs,
            pt3_windows=pt3_window_specs,
            Lt=int(Lt),
            nstate=int(nstate),
            part=part,
            svdcut=float(svdcut),
            prior=joint_template,
            correlator_rescale=scale,
        )
        for idx, rec in enumerate(avg_records):
            tuning_logger.info(
                "candidate z=%s idx=%s t=[%s,%s) tau_cut=%s Q=%.6g chi2/dof=%.6g logGBF=%.6g E0=%s O00_re=%s",
                z,
                idx,
                rec["tmin"],
                rec["tmax"],
                rec["tau_cut"],
                rec["Q"],
                rec["chi2_dof"],
                rec["logGBF"],
                rec["fit"].p["E0"],
                rec["fit"].p["O00_re"],
            )
            overlap_diag = _physical_overlap_diagnostics(rec["fit"].p, int(nstate), scale)
            tuning_logger.info(
                "candidate z=%s idx=%s physical overlaps z0=%s z1=%s z1/z0=%s",
                z,
                idx,
                overlap_diag.get("z0_physical"),
                overlap_diag.get("z1_physical"),
                overlap_diag.get("z1_over_z0_physical"),
            )
            log_nonlinear_fit_quality(
                rec["fit"],
                kind="sample-average joint 2pt+ratio",
                label=f"z={z} idx={idx} t=[{rec['tmin']},{rec['tmax']}) tau_cut={rec['tau_cut']}",
                logger=tuning_logger,
                q_min=float(q_min),
            )
        for rejected in rejected_windows:
            tuning_logger.info("rejected z=%s window=%s", z, rejected)

        best_index, fallback = _select_best_fit_index(avg_records, q_min=float(q_min))
        avg_best = avg_records[best_index]
        tuning_logger.info(
            "selected z=%s index=%s fallback=%s t=[%s,%s) tau_cut=%s Q=%.6g chi2/dof=%.6g logGBF=%.6g",
            z,
            best_index,
            fallback,
            avg_best["tmin"],
            avg_best["tmax"],
            avg_best["tau_cut"],
            avg_best["Q"],
            avg_best["chi2_dof"],
            avg_best["logGBF"],
        )
        tuning_logger.info("selected fit format for z=%s:\n%s", z, avg_best["fit"].format(100))

        sample_prior = _scaled_posterior_as_prior(
            avg_best["fit"],
            joint_template,
            error_scale=float(posterior_prior_error_scale),
        )
        sample_p0 = _fit_p0_from_prior(avg_best["fit"], sample_prior)

        n_samples = int(ratio_samples_re[tseps[0]].shape[0])
        sample_records: list[dict[str, Any]] = []
        sample_failures: list[dict[str, Any]] = []
        sample0_plot_paths: dict[str, str] = {}
        for sample_index in range(n_samples):
            try:
                pt2_sample = _recenter_gvar(pt2_samples[sample_index], pt2_gv)
                ratio_real_sample = {
                    tsep: _recenter_gvar(ratio_samples_re[tsep][sample_index], ratio_real_gv[tsep])
                    for tsep in tseps
                }
                ratio_imag_sample = {
                    tsep: _recenter_gvar(ratio_samples_im[tsep][sample_index], ratio_imag_gv[tsep])
                    for tsep in tseps
                }
                sample_rec = _joint_fit_record(
                    pt2_sample,
                    ratio_real_sample,
                    ratio_imag_sample,
                    tmin=avg_best["tmin"],
                    tmax=avg_best["tmax"],
                    tsep_ls=avg_best["tsep_ls"],
                    tau_cut=avg_best["tau_cut"],
                    Lt=int(Lt),
                    nstate=int(nstate),
                    part=part,
                    svdcut=float(svdcut),
                    prior=sample_prior,
                    p0=sample_p0,
                    correlator_rescale=scale,
                )
                sample_records.append(sample_rec)
                log_nonlinear_fit_quality(
                    sample_rec["fit"],
                    kind="joint 2pt+ratio",
                    label=f"z={z} sample={sample_index}",
                    logger=sample_logger,
                    q_min=float(q_min),
                )
                if sample_index == 0:
                    sample_logger.info(
                        "sample0 z=%s Q=%.6g chi2/dof=%.6g logGBF=%.6g O00/(2E0)=(%s,%s)",
                        z,
                        sample_rec["Q"],
                        sample_rec["chi2_dof"],
                        sample_rec["logGBF"],
                        sample_rec["fit"].p["O00_re"] / (2 * sample_rec["fit"].p["E0"]),
                        sample_rec["fit"].p["O00_im"] / (2 * sample_rec["fit"].p["E0"]),
                    )
                    overlap_diag = _physical_overlap_diagnostics(sample_rec["fit"].p, int(nstate), scale)
                    sample_logger.info(
                        "sample0 z=%s physical overlaps z0=%s z1=%s z1/z0=%s correlator_rescale=%s",
                        z,
                        overlap_diag.get("z0_physical"),
                        overlap_diag.get("z1_physical"),
                        overlap_diag.get("z1_over_z0_physical"),
                        scale,
                    )
                    sample_logger.info("sample0 fit format for z=%s:\n%s", z, sample_rec["fit"].format(100))
                    sample0_plot_paths = _write_sample0_ratio_plot(
                        ratio_real_sample=ratio_real_sample,
                        ratio_imag_sample=ratio_imag_sample,
                        fit_record=sample_rec,
                        Lt=int(Lt),
                        log_dir=fit_log_dir,
                        momentum=momentum,
                        z=z,
                    )
            except Exception as exc:
                sample_records.append({"fit": None})
                sample_failures.append({"sample": sample_index, "stage": "joint_2pt_ratio", "error": str(exc)})
                sample_logger.info("Bad joint 2pt+ratio z=%s sample=%s: %s", z, sample_index, exc)

        real_samples, imag_samples = _bare_matrix_samples_from_records(sample_records)
        if not np.any(np.isfinite(real_samples)):
            raise ValueError(f"all resampled joint fits failed for z={z}")
        real_mean, real_sdev = _sample_mean_err(real_samples, mode=mode)
        imag_mean, imag_sdev = _sample_mean_err(imag_samples, mode=mode)
        sample_logger.info(
            "summary z=%s real=%s +/- %s imag=%s +/- %s failed_samples=%s",
            z,
            real_mean,
            real_sdev,
            imag_mean,
            imag_sdev,
            len(sample_failures),
        )
        window_summary = _fit_summary(avg_best, fallback=fallback, index=best_index)
        z_records.append(
            {
                "z": z,
                "real_samples": real_samples,
                "imag_samples": imag_samples,
                "pt3_window": window_summary,
                "joint_window": window_summary,
                "sample0_plot_paths": sample0_plot_paths,
            }
        )
        z_report.append(
            {
                "z": z,
                "joint_window": window_summary,
                "rejected_windows": rejected_windows,
                "sample0_plot_paths": sample0_plot_paths,
                "n_failed_samples": len(sample_failures),
                "sample_failures": sample_failures[:10],
            }
        )

    output = _write_bare_matrix_grid_outputs(
        z_records,
        artifacts_dir=out_dir,
        save_path=save_path,
        ensemble=ensemble,
        tag=tag,
        variant=variant,
        direction=direction,
        momentum=momentum,
        b_label=b_label,
        resample_mode=mode,
        output_subdir=output_subdir,
    )
    report_path = Path(output["report_json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    n_samples = int(pt2_samples.shape[0])
    report.update(
        {
            "fit_mode": "joint_2pt_ratio",
            "selection_rule": "sample average joint fit: choose max logGBF among Q >= q_min; otherwise choose max Q",
            "sample_fit_prior": "sample-average joint posterior used as prior and p0 after error inflation",
            "posterior_prior_error_scale": float(posterior_prior_error_scale),
            "correlator_rescale": scale,
            "overlap_rescale": _overlap_rescale(scale),
            "fit_strategy": strategy,
            "fit_log_path": str(tuning_log_path),
            "tuning_log_path": str(tuning_log_path),
            "sample_log_path": str(sample_log_path),
            "q_min": float(q_min),
            "resample_mode": mode,
            "n_samples": n_samples,
            "n_boot": int(n_boot) if mode == "bs" else None,
            "seed": seed if mode == "bs" else None,
            "svdcut": float(svdcut),
            "tsep_ls": tseps,
            "z_values": z_list,
            "z_fits": z_report,
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store["bare_matrix_grid_report"] = report
    return {
        **output,
        "fit_mode": "joint_2pt_ratio",
        "fit_strategy": strategy,
        "fit_log_path": str(tuning_log_path),
        "tuning_log_path": str(tuning_log_path),
        "sample_log_path": str(sample_log_path),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "correlator_rescale": scale,
        "overlap_rescale": _overlap_rescale(scale),
        "resample_mode": mode,
        "n_samples": n_samples,
        "n_boot": report["n_boot"],
        "z_values": z_list,
        "selection_rule": report["selection_rule"],
    }



def _fit_bare_matrix_grid_chained(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt3_paths: dict[str, str] | list[str],
    tsep_ls: list[int],
    z_values: list[int],
    ensemble: str,
    tag: str,
    momentum: str,
    direction: str,
    variant: str,
    source_sink: str,
    pt2_gamma: str,
    pt3_gamma: str,
    b_dir: str,
    eta: str,
    bt: str,
    b_label: str,
    pt2_windows: list[dict[str, int]] | None,
    pt3_windows: list[dict[str, Any]] | None,
    pt3_tau_cuts: list[int] | None,
    nstate: int,
    resample_mode: str,
    n_boot: int,
    seed: int | None,
    svdcut: float,
    part: str,
    q_min: float,
    posterior_prior_error_scale: float,
    correlator_rescale: float,
    output_subdir: str,
    save_path: str | None,
    log_dir: str | Path | None,
    log_path: str | Path | None,
    artifacts_dir: str | Path | None,
) -> dict[str, Any]:
    """Batch chained 2pt -> ratio bare matrix export."""
    strategy, fit_mode = _normalise_fit_strategy("chained")
    scale = _validate_correlator_rescale(correlator_rescale)
    mode = str(resample_mode)
    if mode not in {"bs", "jk"}:
        raise ValueError(f"resample_mode must be 'bs' or 'jk', got {resample_mode!r}")
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    tuning_log_path, sample_log_path = _split_fit_log_paths(
        log_dir=fit_log_dir,
        log_path=log_path,
        ensemble=ensemble,
        tag=tag,
        variant=variant,
        direction=direction,
        momentum=momentum,
        b_label=b_label,
        fit_mode=fit_mode,
    )
    tuning_logger = setup_logger(tuning_log_path, console_output=False, logger_name="correlator_chained_tuning_logger")
    sample_logger = setup_logger(sample_log_path, console_output=False, logger_name="correlator_chained_sample_logger")
    tuning_logger.info("Starting chained 2pt -> ratio bare matrix grid fit")
    tuning_logger.info("correlator_rescale=%s overlap_rescale=%s", scale, _overlap_rescale(scale))
    sample_logger.info("Starting chained 2pt -> ratio per-sample fit log")

    tseps = [int(t) for t in tsep_ls]
    z_list = [int(z) for z in z_values]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    missing = [tsep for tsep in tseps if tsep not in paths_by_tsep]
    if missing:
        raise ValueError(f"pt3_paths missing tsep entries: {missing}")

    pt2_complex = _read_pt2_complex(
        pt2_path,
        source_sink=source_sink,
        gamma=pt2_gamma,
        momentum=momentum,
    )
    pt2_real = np.real(pt2_complex)
    n_cfg, Lt = pt2_real.shape
    pt2_samples, indices = _resample_config_samples(
        pt2_real,
        mode=mode,
        n_boot=int(n_boot),
        seed=seed,
    )
    pt2_gv = _samples_to_gvar(pt2_samples, mode=mode)
    pt2_window_specs = _normalise_pt2_windows(pt2_windows, Lt=int(Lt))
    pt3_window_specs = _normalise_pt3_windows(pt3_windows, tsep_ls=tseps, tau_cuts=pt3_tau_cuts)
    tuning_logger.info("Lt=%s n_cfg=%s resample_mode=%s n_samples=%s", Lt, n_cfg, mode, pt2_samples.shape[0])
    tuning_logger.info("pt2_windows=%s pt3_windows=%s svdcut=%s", pt2_window_specs, pt3_window_specs, svdcut)

    pt2_records = _scan_pt2_average_windows(
        pt2_gv,
        windows=pt2_window_specs,
        Lt=int(Lt),
        nstate=int(nstate),
        svdcut=float(svdcut),
        correlator_rescale=scale,
    )
    for idx, rec in enumerate(pt2_records):
        log_nonlinear_fit_quality(
            rec["fit"],
            kind="sample-average 2pt",
            label=f"idx={idx} t=[{rec['tmin']},{rec['tmax']})",
            logger=tuning_logger,
            q_min=float(q_min),
        )
    pt2_best_index, pt2_fallback = _select_best_fit_index(pt2_records, q_min=float(q_min))
    pt2_best = pt2_records[pt2_best_index]
    tuning_logger.info(
        "selected 2pt index=%s fallback=%s t=[%s,%s) Q=%.6g chi2/dof=%.6g logGBF=%.6g",
        pt2_best_index,
        pt2_fallback,
        pt2_best["tmin"],
        pt2_best["tmax"],
        pt2_best["Q"],
        pt2_best["chi2_dof"],
        pt2_best["logGBF"],
    )

    z_records: list[dict[str, Any]] = []
    z_report: list[dict[str, Any]] = []
    template = pt3_ratio_prior(nstate=int(nstate))

    for z in _progress(z_list, desc=f"fit bare matrix {ensemble} {momentum} {direction}"):
        tuning_logger.info("=== z=%s ===", z)
        bz = f"bz{z}"
        ratio_samples_re: dict[int, np.ndarray] = {}
        ratio_samples_im: dict[int, np.ndarray] = {}
        ratio_real_gv: dict[int, np.ndarray] = {}
        ratio_imag_gv: dict[int, np.ndarray] = {}
        for tsep in tseps:
            pt3 = _read_pt3_complex(
                paths_by_tsep[tsep],
                source_sink=source_sink,
                gamma=pt3_gamma,
                momentum=momentum,
                b_dir=b_dir,
                eta=eta,
                bt=bt,
                bz=bz,
                tsep=tsep,
            )
            if pt3.shape[0] != n_cfg:
                raise ValueError(f"3pt n_cfg mismatch for z={z}, tsep={tsep}: {pt3.shape[0]} != {n_cfg}")
            ratio = pt3 / pt2_complex[:, int(tsep)][:, None]
            ratio_samples, _ = _resample_config_samples(
                ratio,
                mode=mode,
                n_boot=int(n_boot),
                seed=seed,
                indices=indices,
            )
            ratio_samples_re[tsep] = np.real(ratio_samples)
            ratio_samples_im[tsep] = np.imag(ratio_samples)
            ratio_real_gv[tsep] = _samples_to_gvar(ratio_samples_re[tsep], mode=mode)
            ratio_imag_gv[tsep] = _samples_to_gvar(ratio_samples_im[tsep], mode=mode)

        avg_prior = pt3_ratio_prior(nstate=int(nstate))
        _update_prior_from_pt2_fit(avg_prior, pt2_best["fit"], int(nstate))
        avg_records = _scan_pt3_average_windows(
            ratio_real_gv,
            ratio_imag_gv,
            windows=pt3_window_specs,
            Lt=int(Lt),
            nstate=int(nstate),
            part=part,
            svdcut=float(svdcut),
            prior=avg_prior,
            correlator_rescale=scale,
        )
        for idx, rec in enumerate(avg_records):
            log_nonlinear_fit_quality(
                rec["fit"],
                kind="sample-average chained ratio",
                label=f"z={z} idx={idx} tau_cut={rec['tau_cut']}",
                logger=tuning_logger,
                q_min=float(q_min),
            )
            overlap_diag = _physical_overlap_diagnostics(rec["fit"].p, int(nstate), scale)
            tuning_logger.info(
                "candidate chained z=%s idx=%s physical overlaps z0=%s z1=%s z1/z0=%s",
                z,
                idx,
                overlap_diag.get("z0_physical"),
                overlap_diag.get("z1_physical"),
                overlap_diag.get("z1_over_z0_physical"),
            )
        best_index, fallback = _select_best_fit_index(avg_records, q_min=float(q_min))
        avg_best = avg_records[best_index]
        avg_best["tmin"] = pt2_best["tmin"]
        avg_best["tmax"] = pt2_best["tmax"]
        tuning_logger.info(
            "selected z=%s ratio index=%s fallback=%s tau_cut=%s Q=%.6g chi2/dof=%.6g logGBF=%.6g",
            z,
            best_index,
            fallback,
            avg_best["tau_cut"],
            avg_best["Q"],
            avg_best["chi2_dof"],
            avg_best["logGBF"],
        )

        sample_prior = _scaled_posterior_as_prior(
            avg_best["fit"],
            template,
            error_scale=float(posterior_prior_error_scale),
        )
        sample_p0 = _fit_p0_from_prior(avg_best["fit"], sample_prior)
        n_samples = int(ratio_samples_re[tseps[0]].shape[0])
        sample_records: list[dict[str, Any]] = []
        sample_failures: list[dict[str, Any]] = []
        sample0_plot_paths: dict[str, str] = {}
        for sample_index in range(n_samples):
            try:
                ratio_real_sample = {
                    tsep: _recenter_gvar(ratio_samples_re[tsep][sample_index], ratio_real_gv[tsep])
                    for tsep in tseps
                }
                ratio_imag_sample = {
                    tsep: _recenter_gvar(ratio_samples_im[tsep][sample_index], ratio_imag_gv[tsep])
                    for tsep in tseps
                }
                sample_rec = _pt3_fit_record(
                    ratio_real_sample,
                    ratio_imag_sample,
                    tsep_ls=avg_best["tsep_ls"],
                    tau_cut=avg_best["tau_cut"],
                    Lt=int(Lt),
                    nstate=int(nstate),
                    part=part,
                    svdcut=float(svdcut),
                    prior=sample_prior,
                    p0=sample_p0,
                    correlator_rescale=scale,
                )
                sample_rec["tmin"] = pt2_best["tmin"]
                sample_rec["tmax"] = pt2_best["tmax"]
                sample_records.append(sample_rec)
                log_nonlinear_fit_quality(
                    sample_rec["fit"],
                    kind="chained ratio",
                    label=f"z={z} sample={sample_index}",
                    logger=sample_logger,
                    q_min=float(q_min),
                )
                if sample_index == 0:
                    sample_logger.info(
                        "sample0 z=%s Q=%.6g chi2/dof=%.6g logGBF=%.6g O00/(2E0)=(%s,%s)",
                        z,
                        sample_rec["Q"],
                        sample_rec["chi2_dof"],
                        sample_rec["logGBF"],
                        sample_rec["fit"].p["O00_re"] / (2 * sample_rec["fit"].p["E0"]),
                        sample_rec["fit"].p["O00_im"] / (2 * sample_rec["fit"].p["E0"]),
                    )
                    overlap_diag = _physical_overlap_diagnostics(sample_rec["fit"].p, int(nstate), scale)
                    sample_logger.info(
                        "sample0 z=%s physical overlaps z0=%s z1=%s z1/z0=%s correlator_rescale=%s",
                        z,
                        overlap_diag.get("z0_physical"),
                        overlap_diag.get("z1_physical"),
                        overlap_diag.get("z1_over_z0_physical"),
                        scale,
                    )
                    sample_logger.info("sample0 fit format for z=%s:\n%s", z, sample_rec["fit"].format(100))
                    sample0_plot_paths = _write_sample0_ratio_plot(
                        ratio_real_sample=ratio_real_sample,
                        ratio_imag_sample=ratio_imag_sample,
                        fit_record=sample_rec,
                        Lt=int(Lt),
                        log_dir=fit_log_dir,
                        momentum=momentum,
                        z=z,
                        fit_label="chained_fit",
                    )
            except Exception as exc:
                sample_records.append({"fit": None})
                sample_failures.append({"sample": sample_index, "stage": "chained_2pt_ratio", "error": str(exc)})
                sample_logger.info("Bad chained ratio z=%s sample=%s: %s", z, sample_index, exc)

        real_samples, imag_samples = _bare_matrix_samples_from_records(sample_records)
        if not np.any(np.isfinite(real_samples)):
            raise ValueError(f"all resampled chained fits failed for z={z}")
        real_mean, real_sdev = _sample_mean_err(real_samples, mode=mode)
        imag_mean, imag_sdev = _sample_mean_err(imag_samples, mode=mode)
        sample_logger.info(
            "summary z=%s real=%s +/- %s imag=%s +/- %s failed_samples=%s",
            z,
            real_mean,
            real_sdev,
            imag_mean,
            imag_sdev,
            len(sample_failures),
        )
        window_summary = _fit_summary(avg_best, fallback=fallback, index=best_index)
        pt2_summary = _fit_summary(pt2_best, fallback=pt2_fallback, index=pt2_best_index)
        z_records.append(
            {
                "z": z,
                "real_samples": real_samples,
                "imag_samples": imag_samples,
                "pt3_window": window_summary,
                "chained_pt2_window": pt2_summary,
                "sample0_plot_paths": sample0_plot_paths,
            }
        )
        z_report.append(
            {
                "z": z,
                "pt2_window": pt2_summary,
                "pt3_window": window_summary,
                "sample0_plot_paths": sample0_plot_paths,
                "n_failed_samples": len(sample_failures),
                "sample_failures": sample_failures[:10],
            }
        )

    output = _write_bare_matrix_grid_outputs(
        z_records,
        artifacts_dir=out_dir,
        save_path=save_path,
        ensemble=ensemble,
        tag=tag,
        variant=variant,
        direction=direction,
        momentum=momentum,
        b_label=b_label,
        resample_mode=mode,
        output_subdir=output_subdir,
    )
    report_path = Path(output["report_json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    n_samples = int(pt2_samples.shape[0])
    report.update(
        {
            "fit_strategy": strategy,
            "fit_mode": fit_mode,
            "selection_rule": "sample average chained fit: choose max logGBF among Q >= q_min; otherwise choose max Q",
            "sample_fit_prior": "sample-average chained ratio posterior used as prior and p0 after error inflation",
            "posterior_prior_error_scale": float(posterior_prior_error_scale),
            "correlator_rescale": scale,
            "overlap_rescale": _overlap_rescale(scale),
            "fit_log_path": str(tuning_log_path),
            "tuning_log_path": str(tuning_log_path),
            "sample_log_path": str(sample_log_path),
            "q_min": float(q_min),
            "resample_mode": mode,
            "n_samples": n_samples,
            "n_boot": int(n_boot) if mode == "bs" else None,
            "seed": seed if mode == "bs" else None,
            "svdcut": float(svdcut),
            "tsep_ls": tseps,
            "z_values": z_list,
            "z_fits": z_report,
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store["bare_matrix_grid_report"] = report
    return {
        **output,
        "fit_strategy": strategy,
        "fit_mode": fit_mode,
        "fit_log_path": str(tuning_log_path),
        "tuning_log_path": str(tuning_log_path),
        "sample_log_path": str(sample_log_path),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "correlator_rescale": scale,
        "overlap_rescale": _overlap_rescale(scale),
        "resample_mode": mode,
        "n_samples": n_samples,
        "n_boot": report["n_boot"],
        "z_values": z_list,
        "selection_rule": report["selection_rule"],
    }

def _infer_Lt(store: dict[str, Any]) -> int:
    if "Lt" in store:
        return int(store["Lt"])
    if "pt2_samples" in store:
        return int(np.asarray(store["pt2_samples"]).shape[1])
    if "pt2_gv" in store:
        return int(len(store["pt2_gv"]))
    raise ValueError("cannot infer Lt from correlator store")


STAGE_TOOLS = {
    "read_pt2": read_pt2,
    "read_pt3": read_pt3,
    "compute_pt3_ratio": compute_pt3_ratio,
    "resample_to_gvar": resample_to_gvar,
    "resample_ratio_to_gvar": resample_ratio_to_gvar,
    "inspect_correlator_scale": inspect_correlator_scale,
    "fit_window": fit_window,
    "fit_pt3_window": fit_pt3_window,
    "model_average": model_average,
    "plot_fit_on_data": plot_fit_on_data,
    "plot_pt3_fit_on_data": plot_pt3_fit_on_data,
    "fit_bare_matrix_grid": fit_bare_matrix_grid,
}
