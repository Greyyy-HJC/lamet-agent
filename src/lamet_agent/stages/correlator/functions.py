"""Correlator-analysis stage tools.

Purpose:
- provide the fixed Python tools the agent calls for 2pt ground-state analysis
- read 2pt data, resample, fit ground-state windows, model-average, and plot

Expected inputs:
- a 2pt HDF5 file laid out as ``source_sink/gamma/momentum`` with shape (Lt, n_cfg)
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- numbers and gvar artifacts stored in a per-stage artifact store
- a model-averaged ground-state energy / matrix element and a fit-on-data PDF

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

from lamet_agent.core.plotting import COLOR_CYCLE, plot_pt2_fit_on_data
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
) -> dict[str, Any]:
    """Read one 2pt dataset and store its real part as (n_cfg, Lt) samples."""
    with h5py.File(path, "r") as h5f:
        data = np.swapaxes(np.asarray(h5f[source_sink][gamma][momentum]), 0, 1)
    samples = np.real(data)
    store[out] = samples
    return {"out": out, "n_cfg": int(samples.shape[0]), "Lt": int(samples.shape[1])}


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
    return {"out": out, "mode": mode, "Lt": int(len(gv_arr))}


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


# --- fit windows and model averaging ----------------------------------------

MAX_FIT_WINDOWS = 6


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
    Lt: int,
    boundary: str = "periodic",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Plot per-window fit bands on C2pt and meff, plus model-averaged E0 on meff."""
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


STAGE_TOOLS = {
    "read_pt2": read_pt2,
    "resample_to_gvar": resample_to_gvar,
    "fit_window": fit_window,
    "model_average": model_average,
    "plot_fit_on_data": plot_fit_on_data,
}
