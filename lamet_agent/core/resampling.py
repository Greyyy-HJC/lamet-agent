"""Reusable resampling helpers for bootstrap and jackknife analyses."""

from __future__ import annotations

import gvar as gv
import numpy as np

RESAMPLE_MODE_ALIASES = {
    "bs": "bs",
    "boot": "bs",
    "bootstrap": "bs",
    "jk": "jk",
    "jackknife": "jk",
}
SAMPLE_ERROR_MODES = frozenset({"mean", "median", "covariance"})


def normalize_resample_mode(value: str | None, *, allow_raw: bool = False) -> str:
    """Return the canonical short resample mode, ``bs`` or ``jk``."""
    mode = "bs" if value is None else str(value).strip().lower()
    aliases = dict(RESAMPLE_MODE_ALIASES)
    if allow_raw:
        aliases["raw"] = "raw"
    if mode not in aliases:
        allowed = "'bs'/'bootstrap', 'jk'/'jackknife'" + (", or 'raw'" if allow_raw else "")
        raise ValueError(f"resample_mode must be {allowed}")
    return aliases[mode]


def normalize_sample_error_mode(value: str | None, *, resample_mode: str | None = None) -> str:
    """Return the canonical sample-error mode and validate mode combinations."""
    mode = "covariance" if value is None else str(value).strip().lower()
    if mode not in SAMPLE_ERROR_MODES:
        allowed = "', '".join(sorted(SAMPLE_ERROR_MODES))
        raise ValueError(f"sample_error_mode must be one of: '{allowed}'")
    if resample_mode is not None and normalize_resample_mode(resample_mode, allow_raw=True) == "jk" and mode == "median":
        raise ValueError("sample_error_mode='median' is not supported with resample_mode='jk'")
    return mode


def _move_sample_axis(samples: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim == 0:
        raise ValueError("samples must have a sample axis")
    return np.moveaxis(arr, axis, 0) if axis != 0 else arr


def _reshape_gvar(values: object, shape: tuple[int, ...]) -> object:
    return values if shape == () else np.asarray(values, dtype=object).reshape(shape)


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


def bs_ls_avg(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples (sample axis first) into a gvar array."""
    bs_arr = _move_sample_axis(np.asarray(bs_ls), axis)
    if bs_arr.shape[0] < 2:
        mean = np.mean(bs_arr, axis=0)
        return gv.gvar(mean, np.zeros_like(mean, dtype=float))
    bs_flat = bs_arr.reshape(bs_arr.shape[0], -1)
    mean = np.mean(bs_flat, axis=0)
    if bs_flat.shape[1] == 1:
        out = gv.gvar(mean[0], np.std(bs_flat[:, 0], ddof=1))
        return _reshape_gvar(out, bs_arr.shape[1:])
    cov = np.cov(bs_flat, rowvar=False)
    return gv.gvar(mean, cov).reshape(bs_arr.shape[1:])


def jk_ls_avg(jk_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average jackknife samples (sample axis first) into a gvar array."""
    jk_arr = _move_sample_axis(np.asarray(jk_ls), axis)
    if jk_arr.shape[0] < 2:
        mean = np.mean(jk_arr, axis=0)
        return gv.gvar(mean, np.zeros_like(mean, dtype=float))
    jk_flat = jk_arr.reshape(jk_arr.shape[0], -1)
    n_sample = jk_flat.shape[0]
    mean = np.mean(jk_flat, axis=0)
    if jk_flat.shape[1] == 1:
        out = gv.gvar(mean[0], np.std(jk_flat[:, 0], ddof=1) * np.sqrt(n_sample - 1))
        return _reshape_gvar(out, jk_arr.shape[1:])
    cov = np.cov(jk_flat, rowvar=False) * (n_sample - 1)
    return gv.gvar(mean, cov).reshape(jk_arr.shape[1:])


def bs_ls_avg_percentile(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples with median centers and percentile widths."""
    bs_arr = _move_sample_axis(np.asarray(bs_ls), axis)
    if bs_arr.shape[0] < 2:
        mid = np.median(bs_arr, axis=0)
        return gv.gvar(mid, np.zeros_like(mid, dtype=float))
    shape = bs_arr.shape
    bs_flat = bs_arr.reshape(shape[0], -1)
    mid = np.median(bs_flat, axis=0)
    p16, p84 = np.percentile(bs_flat, [16, 84], axis=0)
    sdev = 0.5 * (p84 - p16)
    return gv.gvar(mid, sdev).reshape(shape[1:])


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
    mode = normalize_resample_mode(mode)
    if mode == "jk":
        return jackknife(data_arr), None
    if mode == "bs":
        use_indices = indices
        if use_indices is None:
            use_indices = bootstrap_indices(data_arr.shape[0], int(n_boot), seed)
        return bootstrap_by_indices(data_arr, use_indices), use_indices
    raise ValueError(f"unsupported resample_mode: {mode!r}")


def samples_to_gvar(
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str = "covariance",
    axis: int = 0,
) -> np.ndarray:
    """Convert bootstrap or jackknife samples into a gvar array."""
    resample_mode = normalize_resample_mode(mode)
    error_mode = normalize_sample_error_mode(sample_error_mode, resample_mode=resample_mode)
    if error_mode == "median":
        return bs_ls_avg_percentile(samples, axis=axis)
    covariance_avg = jk_ls_avg(samples, axis=axis) if resample_mode == "jk" else bs_ls_avg(samples, axis=axis)
    if error_mode == "covariance":
        return covariance_avg
    return gv.gvar(gv.mean(covariance_avg), gv.sdev(covariance_avg))


def sample_mean_and_sdev(
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str = "covariance",
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sample-average centers and standard deviations."""
    avg = samples_to_gvar(samples, mode=mode, sample_error_mode=sample_error_mode, axis=axis)
    return np.asarray(gv.mean(avg)), np.asarray(gv.sdev(avg), dtype=float)


def sample_sdev(
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str = "covariance",
    axis: int = 0,
) -> np.ndarray:
    """Return standard deviations from the configured sample average."""
    return sample_mean_and_sdev(samples, mode=mode, sample_error_mode=sample_error_mode, axis=axis)[1]


def recenter_sample_values(sample_values: np.ndarray, template: object) -> object:
    """Copy a gvar template's errors/covariance onto new central values."""
    center = np.asarray(sample_values, dtype=float)
    flat_center = center.reshape(-1)
    template_arr = np.asarray(template, dtype=object)
    flat_template = template_arr.reshape(-1)
    if flat_center.shape[0] != flat_template.shape[0]:
        raise ValueError("sample values and gvar template must have matching shape")
    if flat_center.shape[0] == 1:
        out = gv.gvar(float(flat_center[0]), float(gv.sdev(flat_template[0])))
    else:
        out = gv.gvar(flat_center, gv.evalcov(flat_template))
    return _reshape_gvar(out, center.shape)


def sample_value_with_error(
    sample_values: np.ndarray,
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str = "covariance",
    axis: int = 0,
) -> object:
    """Attach the ensemble sample error to one sample or center vector."""
    template = samples_to_gvar(samples, mode=mode, sample_error_mode=sample_error_mode, axis=axis)
    return recenter_sample_values(sample_values, template)


def add_error_to_sample(
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str = "covariance",
    axis: int = 0,
) -> np.ndarray:
    """Return each sample with the configured ensemble errors attached."""
    arr = _move_sample_axis(np.asarray(samples, dtype=float), axis)
    template = samples_to_gvar(arr, mode=mode, sample_error_mode=sample_error_mode, axis=0)
    with_errors = [recenter_sample_values(arr[idx], template) for idx in range(arr.shape[0])]
    return np.asarray(with_errors, dtype=object)
