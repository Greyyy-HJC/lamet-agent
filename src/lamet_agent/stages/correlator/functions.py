"""Correlator-analysis stage tools.

Purpose:
- provide fixed Python tools for 2pt ground-state and 3pt/2pt ratio analysis
- read correlators, resample, fit windows, model-average, and plot fit-on-data

Expected inputs:
- 2pt HDF5: ``source_sink/gamma/momentum`` with shape (Lt, n_cfg)
- 3pt HDF5: ``source_sink/gamma/momentum/b_dir/eta/bT*/bz*`` with shape (tsep+2, n_cfg)
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

from pathlib import Path
from typing import Any

import gvar as gv
import h5py
import lsqfit as lsf
import numpy as np

from lamet_agent.core.plotting import COLOR_CYCLE, plot_pt2_fit_on_data, plot_pt3_ratio_fit_on_data
from lamet_agent.core.tools import resolve_plot_save_path


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
    inferred = int(data.shape[1]) - 2
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


# --- resampling (copied from LaMETLat correlators/resampling.py) ------------


def bootstrap(data: np.ndarray, n_samples: int, axis: int = 0, seed: int | None = 1984) -> np.ndarray:
    """Generate bootstrap sample averages from ensemble data."""
    data = np.asarray(data)
    n_conf = data.shape[axis]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_conf, (n_samples, n_conf), replace=True)
    return np.take(data, indices, axis=axis).mean(axis=axis + 1)


def jackknife(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Generate leave-one-out jackknife sample averages from ensemble data."""
    data = np.asarray(data)
    n_conf = data.shape[axis]
    total = data.sum(axis=axis, keepdims=True)
    return (total - data) / (n_conf - 1)


def bs_ls_avg(bs_ls: np.ndarray) -> np.ndarray:
    """Average bootstrap samples (sample axis first) into a gvar array."""
    bs_arr = np.asarray(bs_ls)
    bs_flat = bs_arr.reshape(bs_arr.shape[0], -1)
    mean = np.mean(bs_flat, axis=0)
    cov = np.cov(bs_flat, rowvar=False)
    return gv.gvar(mean, cov).reshape(bs_arr.shape[1:])


def jk_ls_avg(jk_ls: np.ndarray) -> np.ndarray:
    """Average jackknife samples (sample axis first) into a gvar array."""
    jk_arr = np.asarray(jk_ls)
    jk_flat = jk_arr.reshape(jk_arr.shape[0], -1)
    n_sample = jk_flat.shape[0]
    mean = np.mean(jk_flat, axis=0)
    cov = np.cov(jk_flat, rowvar=False) * (n_sample - 1)
    return gv.gvar(mean, cov).reshape(jk_arr.shape[1:])


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
        prior[f"z{state}"] = gv.gvar(1, 10)
    return prior


def pt2_fit(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    nstate: int = 2,
    svdcut: float = 1e-2,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with an n-state spectral decomposition.

    ``svdcut`` regularizes the strongly correlated 2pt covariance; without it
    the correlated chi-square is dominated by near-singular noise modes.
    """
    fit_t = np.arange(tmin, tmax, dtype=int)
    fit_pt2 = np.asarray(pt2_gv)[fit_t]

    def fcn(t: np.ndarray, p: dict) -> np.ndarray:
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

    return lsf.nonlinear_fit(
        data=(fit_t, fit_pt2), prior=pt2_prior(nstate), fcn=fcn, svdcut=svdcut, maxit=10000
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
        prior[f"z{state}"] = gv.gvar(1, 10)
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
) -> lsf.nonlinear_fit:
    """Fit real and imaginary 3pt/2pt ratio data with an n-state ansatz."""
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

    return lsf.nonlinear_fit(
        data=(x_vecs, y_data),
        prior=priors,
        fcn=fcn,
        svdcut=svdcut,
        maxit=10000,
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
) -> dict[str, Any]:
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
    )
    return {
        "tsep_ls": [int(t) for t in tsep_ls],
        "tau_cut": int(tau_cut),
        "nstate": int(nstate),
        "part": part,
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "logGBF": float(fit.logGBF),
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
) -> dict[str, Any]:
    fit = pt2_fit(data, int(tmin), int(tmax), int(Lt), nstate=int(nstate), svdcut=float(svdcut))
    return {
        "tmin": int(tmin),
        "tmax": int(tmax),
        "nstate": int(nstate),
        "chi2_dof": float(fit.chi2 / fit.dof),
        "Q": float(fit.Q),
        "logGBF": float(fit.logGBF),
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

    plateau_re = None
    plateau_im = None
    if e0_avg is not None:
        tsep_ref = max(int(t) for t in ratio_real.keys())
        o00_re_avg = store.get(O00_re_avg)
        if o00_re_avg is not None:
            plateau_re = asymptotic_ratio_real_gvar(
                o00_re_avg, e0_avg, tsep=tsep_ref, Lt=int(Lt)
            )
        o00_im_avg = store.get(O00_im_avg)
        if o00_im_avg is not None:
            plateau_im = asymptotic_ratio_imag_gvar(
                o00_im_avg, e0_avg, tsep=tsep_ref, Lt=int(Lt)
            )

    plot_pt3_ratio_fit_on_data(
        ratio_real,
        ratio_imag,
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
    "fit_window": fit_window,
    "fit_pt3_window": fit_pt3_window,
    "model_average": model_average,
    "plot_fit_on_data": plot_fit_on_data,
    "plot_pt3_fit_on_data": plot_pt3_fit_on_data,
}
