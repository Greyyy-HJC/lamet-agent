"""Correlated nonlinear fits with deterministic resampling and sample parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence
import warnings

import gvar as gv
import numpy as np
from ..data import EnsembleData
from ._pool import _ParallelPool


@dataclass(frozen=True)
class _FitResult:
    """The sample-average fit and ordered per-sample parameter estimates."""

    fit: Any
    samples: tuple[gv.BufferDict | None, ...]
    resample: str
    sample_errors: tuple[str | None, ...] = ()
    sample_diagnostics: tuple[dict[str, float] | None, ...] = ()
    sample_posteriors: tuple[gv.BufferDict | None, ...] = ()

    @property
    def n_failed_samples(self) -> int:
        return sum(error is not None for error in self.sample_errors)

    @property
    def p(self) -> gv.BufferDict:
        return self.fit.p

    @property
    def pmean(self) -> gv.BufferDict:
        return self.fit.pmean

    @property
    def chi2(self) -> float:
        return float(self.fit.chi2)

    @property
    def dof(self) -> int:
        return int(self.fit.dof)

    @property
    def Q(self) -> float:
        return float(self.fit.Q)

    @property
    def logGBF(self) -> float:
        return float(self.fit.logGBF)


class FitNumericalError(RuntimeError):
    """A numerically unusable fit point or candidate, rather than a contract error."""


_NUMERICAL_FIT_ERRORS = (FloatingPointError, OverflowError, ZeroDivisionError, RuntimeError, ValueError)


@contextmanager
def _fit_warning_scope():
    """Suppress only expected optimizer trial-point warnings inside one fit call."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"overflow encountered in (exp|dot)", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=r"invalid value encountered in dot", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=r"det\(fit\.cov\) < 0.*", category=UserWarning)
        yield


def _sample_fit(
    task: tuple[Any, np.ndarray, np.ndarray, Callable[..., Any], Mapping[str, Any], Mapping[str, Any], bool],
) -> tuple[gv.BufferDict | None, str | None, dict[str, float] | None, gv.BufferDict | None]:
    x, mean, covariance, fcn, prior, options, capture_posterior = task
    import lsqfit

    sample_data = gv.gvar(mean, covariance)
    fit_data = sample_data if x is None else (x, sample_data)
    try:
        with _fit_warning_scope():
            fit = lsqfit.nonlinear_fit(data=fit_data, fcn=fcn, prior=prior, **dict(options))
    except _NUMERICAL_FIT_ERRORS as exc:
        return None, f"{type(exc).__name__}: {exc}", None, None
    return (
        fit.pmean,
        None,
        {
            "chi2": float(fit.chi2),
            "dof": float(fit.dof),
            "Q": float(fit.Q),
            "logGBF": float(fit.logGBF),
        },
        fit.p if capture_posterior else None,
    )


def _posterior_prior(fit: Any, template: Mapping[str, Any], scale: float) -> gv.BufferDict:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("sample_prior_scale must be finite and positive")
    prior = gv.BufferDict()
    for key in template:
        value = fit.p[key]
        width = np.asarray(gv.sdev(value)) * scale
        if np.any(~np.isfinite(width)) or np.any(width <= 0):
            raise ValueError(f"center fit produced an invalid posterior width for '{key}'")
        prior[key] = gv.gvar(gv.mean(value), width)
    return prior


def nonlinear_fit(
    data: EnsembleData | tuple[Any, EnsembleData],
    fcn: Callable[..., Any],
    prior: Mapping[str, Any],
    *,
    resampling: Literal["jackknife", "bootstrap"] | None = None,
    n_resample: int | None = None,
    seed: int | None = None,
    workers: int = 1,
    sample_prior_scale: float | None = None,
    covariance: np.ndarray | None = None,
    sample_error_mode: Literal["covariance", "variance", "one_sigma"] = "covariance",
    mode: Literal["center", "resamples"] = "resamples",
    tolerate_sample_failures: bool = False,
    capture_sample_posteriors: Sequence[int] = (),
    _parallel: _ParallelPool | None = None,
    **options: Any,
) -> _FitResult:
    """Fit either the ensemble center or every supplied resample.

    ``data`` is either an ``EnsembleData`` or ``(x, EnsembleData)``. Center
    mode averages raw, jackknife, or bootstrap source data with its
    corresponding covariance and performs no sample scheduling. Resamples mode
    accepts existing jackknife/bootstrap samples, or creates them from raw data
    when ``resampling`` is supplied, then fits every stored sample in order.
    ``capture_sample_posteriors`` retains full gvar posteriors only for the
    requested sample indices; ordinary callers continue to receive means only.
    """
    if isinstance(data, tuple):
        if len(data) != 2 or not isinstance(data[1], EnsembleData):
            raise TypeError("data must be EnsembleData or (x, EnsembleData)")
        x, samples = data
    elif isinstance(data, EnsembleData):
        x, samples = None, data
    else:
        raise TypeError("data must be EnsembleData or (x, EnsembleData)")
    if not callable(fcn) or not isinstance(prior, Mapping) or not prior:
        raise TypeError("fcn must be callable and prior must be a nonempty mapping")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if mode not in {"center", "resamples"}:
        raise ValueError("mode must be 'center' or 'resamples'")
    capture_indices = tuple(capture_sample_posteriors)
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in capture_indices):
        raise ValueError("capture_sample_posteriors must contain nonnegative integer indices")
    if len(set(capture_indices)) != len(capture_indices):
        raise ValueError("capture_sample_posteriors must not contain duplicates")
    if samples.resample == "gvar" and mode != "center":
        raise ValueError("fitting requires raw, jackknife, or bootstrap data")
    if resampling is not None:
        if samples.resample != "raw":
            raise ValueError("resampling can only be requested for raw data")
        if resampling == "bootstrap":
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("bootstrap fitting requires a nonnegative integer seed")
            if isinstance(n_resample, bool) or not isinstance(n_resample, int) or n_resample < 2:
                raise ValueError("bootstrap fitting requires at least two resamples")
            samples = samples.bootstrap(n_resample, seed=seed)
        else:
            if seed is not None or n_resample is not None:
                raise ValueError("seed and n_resample are only valid for bootstrap fitting")
            samples = samples.jackknife()
    elif seed is not None or n_resample is not None:
        raise ValueError("seed and n_resample require resampling='bootstrap'")
    if mode == "resamples" and samples.resample not in {"jackknife", "bootstrap"}:
        raise ValueError("resamples mode requires jackknife or bootstrap samples")
    if samples.resample != "gvar" and samples.n_sample < 2:
        raise ValueError("fitting requires at least two source samples")

    import lsqfit

    average = samples.average(sample_error_mode)
    if covariance is None:
        center_data = average
    else:
        sample_values = np.asarray(samples.values)
        if np.iscomplexobj(sample_values):
            raise TypeError("covariance override requires real sample data")
        flat_size = int(np.prod(sample_values.shape[1:]))
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (flat_size, flat_size) or np.any(~np.isfinite(covariance)):
            raise ValueError("covariance must be a finite square matrix matching one flattened sample")
        center_data = gv.gvar(np.asarray(gv.mean(average)).reshape(-1), covariance).reshape(sample_values.shape[1:])
    fit_data = center_data if x is None else (x, center_data)
    try:
        with _fit_warning_scope():
            fitted_center = lsqfit.nonlinear_fit(data=fit_data, fcn=fcn, prior=prior, **options)
    except _NUMERICAL_FIT_ERRORS as exc:
        raise FitNumericalError(f"sample-average fit failed: {type(exc).__name__}: {exc}") from exc
    try:
        sample_prior = (
            prior if sample_prior_scale is None else _posterior_prior(fitted_center, prior, sample_prior_scale)
        )
    except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError) as exc:
        raise FitNumericalError(f"sample-average posterior is unusable: {type(exc).__name__}: {exc}") from exc
    if mode == "center":
        if capture_indices:
            raise ValueError("capture_sample_posteriors requires mode='resamples'")
        return _FitResult(fitted_center, (), samples.resample, (), (), ())
    sample_options = dict(options)
    sample_options["p0"] = {
        key: np.asarray(gv.mean(fitted_center.p[key])).item()
        if np.asarray(gv.mean(fitted_center.p[key])).ndim == 0
        else np.asarray(gv.mean(fitted_center.p[key]))
        for key in prior
    }
    covariance = np.asarray(gv.evalcov(center_data))
    if any(index >= samples.n_sample for index in capture_indices):
        raise ValueError("capture_sample_posteriors contains an out-of-range sample index")
    capture_set = set(capture_indices)
    tasks = [
        (x, np.asarray(sample), covariance, fcn, sample_prior, sample_options, index in capture_set)
        for index, sample in enumerate(samples.values)
    ]
    if _parallel is None:
        with _ParallelPool(min(workers, len(tasks))) as parallel:
            outcomes = parallel.map(
                _sample_fit,
                tasks,
            )
    else:
        outcomes = _parallel.map(
            _sample_fit,
            tasks,
        )
    fitted_samples = tuple(parameters for parameters, _error, _diagnostics, _posterior in outcomes)
    sample_errors = tuple(error for _parameters, error, _diagnostics, _posterior in outcomes)
    sample_diagnostics = tuple(diagnostics for _parameters, _error, diagnostics, _posterior in outcomes)
    sample_posteriors = tuple(posterior for _parameters, _error, _diagnostics, posterior in outcomes)
    if not tolerate_sample_failures and any(error is not None for error in sample_errors):
        failed_index = next(index for index, error in enumerate(sample_errors) if error is not None)
        raise FitNumericalError(f"sample fit {failed_index} failed: {sample_errors[failed_index]}")
    return _FitResult(
        fitted_center,
        fitted_samples,
        samples.resample,
        sample_errors,
        sample_diagnostics,
        sample_posteriors,
    )


__all__ = ["FitNumericalError", "nonlinear_fit"]
