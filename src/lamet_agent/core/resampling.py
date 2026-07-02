"""Reusable resampling helpers for bootstrap and jackknife analyses."""

from __future__ import annotations

import gvar as gv
import numpy as np


def bin_data(data: np.ndarray, bin_size: int, axis: int = 0) -> np.ndarray:
    """Average adjacent configurations into bins along ``axis``."""
    if bin_size < 1:
        raise ValueError("bin_size must be a positive integer")
    data = np.moveaxis(np.asarray(data), axis, 0)
    n_bins = data.shape[0] // bin_size
    data = data[: n_bins * bin_size]
    data = data.reshape(n_bins, bin_size, *data.shape[1:]).mean(axis=1)
    return np.moveaxis(data, 0, axis)


def bootstrap(data: np.ndarray, n_samples: int, axis: int = 0, seed: int | None = 1984, bin_size: int = 1) -> np.ndarray:
    """Generate bootstrap sample averages from ensemble data."""
    data = np.asarray(data)
    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)
    n_conf = data.shape[axis]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_conf, (n_samples, n_conf), replace=True)
    return np.take(data, indices, axis=axis).mean(axis=axis + 1)


def jackknife(data: np.ndarray, axis: int = 0, bin_size: int = 1) -> np.ndarray:
    """Generate leave-one-bin-out jackknife sample averages from ensemble data."""
    data = np.asarray(data)
    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)
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


def bootstrap_indices(n_cfg: int, n_samples: int, seed: int | None) -> np.ndarray:
    """Return shared bootstrap configuration indices with shape (n_samples, n_cfg)."""
    rng = np.random.default_rng(seed)
    return rng.choice(int(n_cfg), (int(n_samples), int(n_cfg)), replace=True)


def bootstrap_by_indices(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Bootstrap-average configuration-axis samples using precomputed indices."""
    return np.asarray(data)[np.asarray(indices, dtype=int)].mean(axis=1)


def resample_config_samples(
    data: np.ndarray,
    *,
    mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int = 1,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return resampled configuration averages and optional bootstrap indices.

    ``bin_size`` bins the configuration axis before resampling; ``indices``
    (when provided) are assumed to already be in the binned index space.
    """
    data_arr = np.asarray(data)
    if bin_size > 1:
        data_arr = bin_data(data_arr, bin_size, axis=0)
    if mode == "jk":
        return jackknife(data_arr), None
    if mode == "bs":
        use_indices = indices
        if use_indices is None:
            use_indices = bootstrap_indices(data_arr.shape[0], int(n_boot), seed)
        return bootstrap_by_indices(data_arr, use_indices), use_indices
    raise ValueError(f"unsupported resample_mode: {mode!r}")


def samples_to_gvar(samples: np.ndarray, *, mode: str) -> np.ndarray:
    """Convert bootstrap or jackknife samples into a gvar array."""
    if mode == "jk":
        return jk_ls_avg(samples)
    if mode == "bs":
        return bs_ls_avg(samples)
    raise ValueError(f"unsupported resample_mode: {mode!r}")


def sample_mean_err(samples: np.ndarray, *, mode: str) -> tuple[float, float]:
    """Return a sample mean and bootstrap/jackknife-scaled error."""
    finite = np.asarray(samples, dtype=float)[np.isfinite(samples)]
    if finite.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(finite))
    if finite.size == 1:
        return mean, 0.0
    if mode == "jk":
        err = float(np.sqrt((finite.size - 1) * np.mean((finite - mean) ** 2)))
    elif mode == "bs":
        err = float(np.std(finite, ddof=1))
    else:
        raise ValueError(f"unsupported resample_mode: {mode!r}")
    return mean, err
