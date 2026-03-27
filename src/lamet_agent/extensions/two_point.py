"""Reusable helpers for two-point correlator analysis.

This module centralizes the standard two-point workflow used by the
``correlator_analysis`` stage:

- resample raw configuration measurements
- compute effective masses
- build default n-state priors
- fit the correlator with a fixed lattice-inspired model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from lamet_agent.errors import OptionalDependencyError
from lamet_agent.extensions.statistics import (
    bootstrap,
    bootstrap_average,
    gv,
    jackknife,
    jackknife_average,
    lsqfit,
)

DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_PRIOR_WIDTH = 10.0


def _require_gvar() -> None:
    if gv is None:
        raise OptionalDependencyError("gvar is required for two-point uncertainty propagation.")


def _require_lsqfit() -> None:
    if lsqfit is None:
        raise OptionalDependencyError("lsqfit is required for two-point correlator fitting.")


@dataclass(slots=True)
class ResampledCorrelator:
    """Container for resampled two-point correlator averages."""

    method: str
    sample_means: np.ndarray | None
    mean: np.ndarray
    error: np.ndarray
    average: Sequence[Any] | None
    configuration_count: int
    resample_count: int
    bin_size: int


@dataclass(slots=True)
class EffectiveMassResult:
    """Container for effective-mass estimates."""

    method: str
    times: np.ndarray
    mean: np.ndarray
    error: np.ndarray
    average: Sequence[Any] | None


def resample_two_point_correlator(
    raw_samples: np.ndarray,
    *,
    method: str = "jackknife",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_sample_size: int | None = None,
    bin_size: int = 1,
    seed: int = 1984,
) -> ResampledCorrelator:
    """Resample raw two-point measurements and return ensemble averages.

    Parameters
    ----------
    raw_samples
        Array with shape ``(n_t, n_cfg)``.
    method
        One of ``"jackknife"``, ``"bootstrap"``, or ``"none"``.
    """

    samples = np.asarray(raw_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Two-point raw samples must have shape (n_t, n_cfg).")

    configuration_count = samples.shape[1]
    method_name = method.lower()
    if method_name == "jackknife":
        sample_means = jackknife(samples, axis=1).T
        average = jackknife_average(sample_means, axis=0) if gv is not None else None
    elif method_name == "bootstrap":
        bootstrap_means, _ = bootstrap(
            samples,
            sample_count=bootstrap_samples,
            sample_size=bootstrap_sample_size,
            axis=1,
            bin_size=bin_size,
            seed=seed,
        )
        sample_means = bootstrap_means.T
        average = bootstrap_average(sample_means, axis=0) if gv is not None else None
    elif method_name == "none":
        sample_means = None
        average = gv.gvar(samples.mean(axis=1), samples.std(axis=1, ddof=1)) if gv is not None else None
    else:
        raise ValueError("Unsupported resampling method. Expected 'jackknife', 'bootstrap', or 'none'.")

    if average is not None:
        mean = np.asarray(gv.mean(average), dtype=float)
        error = np.asarray(gv.sdev(average), dtype=float)
    else:
        mean = samples.mean(axis=1)
        error = samples.std(axis=1, ddof=1) if configuration_count > 1 else np.zeros(samples.shape[0], dtype=float)

    resample_count = 0 if sample_means is None else int(sample_means.shape[0])
    return ResampledCorrelator(
        method=method_name,
        sample_means=sample_means,
        mean=mean,
        error=error,
        average=average,
        configuration_count=configuration_count,
        resample_count=resample_count,
        bin_size=bin_size,
    )


def effective_mass_from_correlator(
    correlator: ResampledCorrelator | Sequence[Any] | np.ndarray,
    *,
    method: str | None = None,
    boundary: str = "periodic",
) -> EffectiveMassResult:
    """Compute the effective mass from a correlator average or resampled samples."""

    boundary_name = boundary.lower()
    method_name = _resolve_effective_mass_method(method=method, boundary=boundary_name)

    if isinstance(correlator, ResampledCorrelator):
        base_values = correlator.mean
        if correlator.sample_means is None:
            averaged = None
        else:
            effective_samples = np.asarray(
                [_effective_mass_array(sample, method=method_name, boundary=boundary_name) for sample in correlator.sample_means],
                dtype=float,
            )
            if correlator.method == "jackknife":
                averaged = jackknife_average(effective_samples, axis=0) if gv is not None else None
            else:
                averaged = bootstrap_average(effective_samples, axis=0) if gv is not None else None
    else:
        base_values = np.asarray(correlator, dtype=float)
        averaged = None

    mean_values = _effective_mass_array(base_values, method=method_name, boundary=boundary_name)
    if averaged is not None:
        mean = np.asarray(gv.mean(averaged), dtype=float)
        error = np.asarray(gv.sdev(averaged), dtype=float)
    else:
        mean = mean_values
        error = np.zeros_like(mean_values)

    if method_name in {"cosh", "sinh"}:
        times = np.arange(1, 1 + len(mean), dtype=float)
    else:
        times = np.arange(len(mean), dtype=float)

    return EffectiveMassResult(
        method=method_name,
        times=times,
        mean=mean,
        error=error,
        average=averaged,
    )


def build_two_point_priors(
    state_count: int,
    prior_overrides: dict[str, Any] | None = None,
):
    """Return the default prior set for an ``n``-state two-point fit."""

    _require_gvar()
    if state_count < 1:
        raise ValueError("state_count must be >= 1")

    priors = gv.BufferDict()
    priors["E0"] = gv.gvar(1.0, DEFAULT_PRIOR_WIDTH)
    for state_index in range(1, state_count):
        priors[f"log(dE{state_index})"] = gv.gvar(0.0, DEFAULT_PRIOR_WIDTH)
    for state_index in range(state_count):
        priors[f"z{state_index}"] = gv.gvar(1.0, DEFAULT_PRIOR_WIDTH)

    for key, value in (prior_overrides or {}).items():
        target_key, mean, sigma = _coerce_prior_override(key, value, priors)
        priors[target_key] = gv.gvar(mean, sigma)
    return priors


def extract_state_energies(parameters, state_count: int) -> list[Any]:
    """Return the ordered energy levels implied by the fit parameters."""

    energies = [parameters["E0"]]
    current_energy = parameters["E0"]
    for state_index in range(1, state_count):
        current_energy = current_energy + np.exp(parameters[f"log(dE{state_index})"])
        energies.append(current_energy)
    return energies


def two_point_fit_function(
    times: Sequence[float] | np.ndarray,
    parameters,
    temporal_extent: int,
    *,
    state_count: int,
    boundary: str = "periodic",
):
    """Evaluate the standard ``n``-state two-point fit function."""

    times_array = np.asarray(times, dtype=float)
    boundary_name = boundary.lower()
    values = 0.0
    for state_index, energy in enumerate(extract_state_energies(parameters, state_count)):
        overlap = parameters[f"z{state_index}"]
        forward = overlap**2 / (2 * energy) * np.exp(-energy * times_array)
        if boundary_name == "periodic":
            backward = overlap**2 / (2 * energy) * np.exp(-energy * (temporal_extent - times_array))
            values = values + forward + backward
        elif boundary_name == "anti-periodic":
            backward = overlap**2 / (2 * energy) * np.exp(-energy * (temporal_extent - times_array))
            values = values + forward - backward
        elif boundary_name == "none":
            values = values + forward
        else:
            raise ValueError(f"Unsupported boundary condition: {boundary}.")
    return values


def fit_two_point_correlator(
    averaged_correlator,
    *,
    temporal_extent: int,
    tmin: int,
    tmax: int,
    state_count: int = 2,
    boundary: str = "periodic",
    normalize: bool = True,
    prior_overrides: dict[str, Any] | None = None,
):
    """Fit a resampled two-point correlator with the default lattice model."""

    _require_gvar()
    _require_lsqfit()
    if tmax <= tmin:
        raise ValueError("tmax must be larger than tmin for a two-point fit.")

    priors = build_two_point_priors(state_count=state_count, prior_overrides=prior_overrides)
    times = np.arange(tmin, tmax, dtype=float)
    fit_data = averaged_correlator[tmin:tmax]
    normalization = 1.0
    if normalize:
        normalization = abs(float(gv.mean(averaged_correlator[0]))) or 1.0
        fit_data = fit_data / normalization

    def model(t_sep: np.ndarray, params):
        return (
            two_point_fit_function(
                t_sep,
                params,
                temporal_extent=temporal_extent,
                state_count=state_count,
                boundary=boundary,
            )
            / normalization
        )

    return lsqfit.nonlinear_fit(
        data=(times, fit_data),
        prior=priors,
        fcn=model,
        maxit=10_000,
    )


def summarize_two_point_fit(fit_result, state_count: int) -> dict[str, Any]:
    """Convert an ``lsqfit`` result into JSON-safe fit diagnostics."""

    _require_gvar()
    summary: dict[str, Any] = {
        "state_count": state_count,
        "chi2": float(fit_result.chi2),
        "dof": int(fit_result.dof),
        "chi2_per_dof": float(fit_result.chi2 / fit_result.dof) if fit_result.dof else float("nan"),
        "Q": float(fit_result.Q),
        "loggbf": float(fit_result.logGBF),
        "parameters": {},
        "energies": {},
    }
    for key, value in fit_result.p.items():
        summary["parameters"][key] = {
            "mean": float(gv.mean(value)),
            "sdev": float(gv.sdev(value)),
        }
    for state_index, energy in enumerate(extract_state_energies(fit_result.p, state_count=state_count)):
        summary["energies"][f"E{state_index}"] = {
            "mean": float(gv.mean(energy)),
            "sdev": float(gv.sdev(energy)),
        }
    return summary


def _resolve_effective_mass_method(method: str | None, boundary: str) -> str:
    if method is not None:
        method_name = method.lower()
    elif boundary in {"periodic", "anti-periodic"}:
        method_name = "cosh" if boundary == "periodic" else "sinh"
    else:
        method_name = "log"
    if method_name not in {"cosh", "sinh", "log"}:
        raise ValueError("Effective-mass method must be one of 'cosh', 'sinh', or 'log'.")
    return method_name


def _effective_mass_array(values: np.ndarray, *, method: str, boundary: str) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    if len(samples) < 3 and method in {"cosh", "sinh"}:
        raise ValueError("At least three time slices are required for cosh/sinh effective masses.")
    if len(samples) < 2 and method == "log":
        raise ValueError("At least two time slices are required for log effective masses.")

    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "cosh":
            ratio = (samples[2:] + samples[:-2]) / (2.0 * samples[1:-1])
            result = np.arccosh(ratio)
        elif method == "sinh":
            ratio = (samples[2:] + samples[:-2]) / (2.0 * samples[1:-1])
            result = np.arcsinh(ratio)
        else:
            result = np.log(samples[:-1] / samples[1:])
    result = np.asarray(result, dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result


def _coerce_prior_override(key: str, value: Any, priors) -> tuple[str, float, float]:
    target_key = key
    mean, sigma = _parse_prior_override_value(value)
    if key not in priors:
        log_key = f"log({key})"
        if log_key not in priors:
            raise KeyError(f"Unknown two-point prior override: {key}.")
        if mean <= 0:
            raise ValueError(f"Prior override {key!r} must have positive mean to map into log space.")
        target_key = log_key
        sigma = abs(float(sigma) / float(mean)) if sigma else 1e-12
        mean = float(np.log(mean))
    return target_key, float(mean), float(abs(sigma))


def _parse_prior_override_value(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        if "mean" not in value or "sdev" not in value:
            raise ValueError("Prior override dictionaries must define 'mean' and 'sdev'.")
        return float(value["mean"]), float(value["sdev"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("Prior overrides must be {'mean': x, 'sdev': y} or [x, y].")
