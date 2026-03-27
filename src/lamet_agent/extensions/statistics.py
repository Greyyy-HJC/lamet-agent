"""Reusable statistical helpers migrated from incoming analysis drafts.

This module centralizes resampling and uncertainty utilities so stage code can
reuse a consistent implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from lamet_agent.errors import OptionalDependencyError

try:  # pragma: no cover - optional dependency checks are environment dependent.
    import gvar as gv
except ModuleNotFoundError:  # pragma: no cover - optional dependency checks are environment dependent.
    gv = None

try:  # pragma: no cover - optional dependency checks are environment dependent.
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency checks are environment dependent.
    h5py = None

try:  # pragma: no cover - optional dependency checks are environment dependent.
    import lsqfit
except ModuleNotFoundError:  # pragma: no cover - optional dependency checks are environment dependent.
    lsqfit = None


def _require_gvar() -> None:
    if gv is None:
        raise OptionalDependencyError("gvar is required for this helper. Install with `pip install gvar`.")


def _require_h5py() -> None:
    if h5py is None:
        raise OptionalDependencyError("h5py is required for this helper. Install with `pip install h5py`.")


def _require_lsqfit() -> None:
    if lsqfit is None:
        raise OptionalDependencyError("lsqfit is required for this helper. Install with `pip install lsqfit`.")


def bin_data(data: np.ndarray, bin_size: int, axis: int = 0) -> np.ndarray:
    """Average contiguous blocks to reduce autocorrelation along ``axis``."""
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    array = np.asarray(data)
    if bin_size == 1:
        return array
    axis_length = array.shape[axis]
    binned_length = axis_length // bin_size
    if binned_length == 0:
        raise ValueError("bin_size is larger than the selected axis length")
    trimmed = np.take(array, np.arange(binned_length * bin_size), axis=axis)
    moved = np.moveaxis(trimmed, axis, 0)
    reshaped = moved.reshape(binned_length, bin_size, *moved.shape[1:])
    averaged = reshaped.mean(axis=1)
    return np.moveaxis(averaged, 0, axis)


def bootstrap(
    data: Sequence[float] | np.ndarray,
    sample_count: int,
    sample_size: int | None = None,
    axis: int = 0,
    bin_size: int = 1,
    seed: int = 1984,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap means and sampled indices."""
    array = np.asarray(data)
    if bin_size > 1:
        array = bin_data(array, bin_size=bin_size, axis=axis)

    n_cfg = array.shape[axis]
    size = n_cfg if sample_size is None else sample_size
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(n_cfg, size=(sample_count, size), replace=True)
    sampled = np.take(array, sampled_indices, axis=axis)
    sample_means = sampled.mean(axis=axis + 1)
    return sample_means, sampled_indices


def jackknife(data: Sequence[float] | np.ndarray, axis: int = 0) -> np.ndarray:
    """Return leave-one-out jackknife sample means."""
    array = np.asarray(data)
    n_cfg = array.shape[axis]
    if n_cfg < 2:
        raise ValueError("jackknife requires at least two configurations")
    total = np.sum(array, axis=axis, keepdims=True)
    return (total - array) / (n_cfg - 1)


def _average_samples_to_gvar(samples: np.ndarray, axis: int, is_jackknife: bool):
    _require_gvar()
    arranged = np.asarray(samples)
    if axis != 0:
        arranged = np.swapaxes(arranged, 0, axis)
    shape = arranged.shape
    flat = arranged.reshape(shape[0], -1)

    mean = np.mean(flat, axis=0)
    if flat.shape[1] == 1:
        stdev = np.std(flat, axis=0, ddof=0)
        if is_jackknife:
            stdev *= np.sqrt(flat.shape[0] - 1)
        return gv.gvar(mean, stdev)[0]

    covariance = np.cov(flat, rowvar=False)
    if is_jackknife:
        covariance *= flat.shape[0] - 1
    averaged = gv.gvar(mean, covariance)
    return np.reshape(averaged, shape[1:])


def bootstrap_average(samples: np.ndarray, axis: int = 0):
    """Convert bootstrap samples to correlated ``gvar`` values."""
    return _average_samples_to_gvar(samples=samples, axis=axis, is_jackknife=False)


def jackknife_average(samples: np.ndarray, axis: int = 0):
    """Convert jackknife samples to correlated ``gvar`` values."""
    return _average_samples_to_gvar(samples=samples, axis=axis, is_jackknife=True)


def bootstrap_average_no_correlation(samples: np.ndarray, axis: int = 0):
    """Convert bootstrap samples to ``gvar`` values using percentile errors only."""
    _require_gvar()
    arranged = np.asarray(samples)
    if axis != 0:
        arranged = np.swapaxes(arranged, 0, axis)
    shape = arranged.shape
    flat = arranged.reshape(shape[0], -1)
    median = np.median(flat, axis=0)
    p16 = np.percentile(flat, 16, axis=0)
    p84 = np.percentile(flat, 84, axis=0)
    err = 0.5 * (p84 - p16)
    averaged = gv.gvar(median, err)
    return np.reshape(averaged, shape[1:])


def gvar_list_to_correlated_samples(gvar_values: Sequence, sample_count: int) -> np.ndarray:
    """Draw correlated Gaussian samples from a 1D sequence of ``gvar`` values."""
    _require_gvar()
    mean = np.array([value.mean for value in gvar_values], dtype=float)
    cov = gv.evalcov(gvar_values)
    rng = np.random.default_rng()
    return rng.multivariate_normal(mean, cov, size=sample_count)


def gvar_dict_to_correlated_samples(gvar_mapping: Mapping[str, Sequence], sample_count: int) -> dict[str, np.ndarray]:
    """Draw correlated Gaussian samples for each key in a ``gvar`` mapping."""
    key_lengths = {key: len(values) for key, values in gvar_mapping.items()}
    flattened = [entry for values in gvar_mapping.values() for entry in values]
    all_samples = gvar_list_to_correlated_samples(flattened, sample_count)
    by_component = list(np.swapaxes(all_samples, 0, 1))

    output: dict[str, np.ndarray] = {}
    for key, length in key_lengths.items():
        key_components = [by_component.pop(0) for _ in range(length)]
        output[key] = np.swapaxes(np.asarray(key_components), 0, 1)
    return output


def save_gvar_dict_samples_to_h5(gvar_mapping: Mapping[str, Sequence], sample_count: int, file_path: str | Path) -> None:
    """Persist correlated samples from each dictionary entry to an HDF5 file."""
    _require_h5py()
    samples = gvar_dict_to_correlated_samples(gvar_mapping, sample_count)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in samples.items():
            handle.create_dataset(key, data=value)


def constant_fit(values: Sequence, prior_mean: float = 0.0, prior_sigma: float = 100.0):
    """Fit data to a constant model and return the posterior ``gvar`` coefficient."""
    _require_gvar()
    _require_lsqfit()

    def model(x_axis: np.ndarray, params):
        return np.zeros_like(x_axis, dtype=float) + params["const"]

    x_axis = np.arange(len(values), dtype=float)
    priors = gv.BufferDict({"const": gv.gvar(prior_mean, prior_sigma)})
    fit = lsqfit.nonlinear_fit(
        data=(x_axis, values),
        prior=priors,
        fcn=model,
        maxit=10_000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )
    return fit.p["const"]
