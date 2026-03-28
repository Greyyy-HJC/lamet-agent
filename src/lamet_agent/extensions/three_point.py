"""Reusable helpers for three-point correlator analysis and fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

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

MAD_SCALE = 1.4826
DEFAULT_BAD_POINT_ZCUT = 8.0
DEFAULT_BAD_POINT_RATIO_CUT = 50.0
DEFAULT_ABSOLUTE_THRESHOLD = 1e12
DEFAULT_POSTERIOR_SAMPLES = 200


def _require_gvar() -> None:
    if gv is None:
        raise OptionalDependencyError("gvar is required for three-point uncertainty propagation.")


def _require_lsqfit() -> None:
    if lsqfit is None:
        raise OptionalDependencyError("lsqfit is required for three-point correlator fitting.")


@dataclass(slots=True)
class BadPointFilterDiagnostics:
    """Summary of a bad-point filtering pass."""

    mode: str
    replacement: str
    flagged_count: int
    total_count: int
    mad_zcut: float
    ratio_cut: float
    absolute_threshold: float


@dataclass(slots=True)
class ResampledObservable:
    """Generic resampled observable container."""

    method: str
    sample_means: np.ndarray | None
    mean: np.ndarray
    error: np.ndarray
    average: Any | None
    configuration_count: int
    resample_count: int
    bin_size: int


def filter_bad_points(
    samples: np.ndarray,
    *,
    axis: int = -1,
    mode: str = "mad",
    mad_zcut: float = DEFAULT_BAD_POINT_ZCUT,
    ratio_cut: float = DEFAULT_BAD_POINT_RATIO_CUT,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
    replacement: str = "median",
) -> tuple[np.ndarray, BadPointFilterDiagnostics]:
    """Filter isolated bad points along the configuration axis."""

    array = np.asarray(samples, dtype=float)
    moved = np.moveaxis(array.copy(), axis, -1)
    flagged_total = 0

    if mode not in {"mad", "absolute"}:
        raise ValueError("bad-point filter mode must be 'mad' or 'absolute'.")
    if replacement not in {"median", "zero", "sign"}:
        raise ValueError("bad-point filter replacement must be 'median', 'zero', or 'sign'.")

    for index in np.ndindex(moved.shape[:-1]):
        slice_values = moved[index]
        if mode == "absolute":
            mask = np.abs(slice_values) > absolute_threshold
            replacement_values = np.full(slice_values.shape, _replacement_value(slice_values, replacement))
        else:
            median = float(np.median(slice_values))
            abs_deviation = np.abs(slice_values - median)
            mad = float(np.median(abs_deviation))
            if mad > 0.0:
                scale = MAD_SCALE * mad
                mask = abs_deviation > mad_zcut * scale
            else:
                abs_values = np.abs(slice_values)
                sorted_abs = np.sort(abs_values)
                reference_source = sorted_abs[:-1] if len(sorted_abs) > 1 else sorted_abs
                reference = float(np.median(reference_source))
                if reference <= 0.0:
                    positive = abs_values[abs_values > 0.0]
                    reference = float(np.median(positive)) if positive.size else 0.0
                mask = abs_values > ratio_cut * reference if reference > 0.0 else np.zeros_like(slice_values, dtype=bool)
            replacement_values = np.full(slice_values.shape, _replacement_value(slice_values, replacement))

        flagged_total += int(np.count_nonzero(mask))
        if np.any(mask):
            slice_values[mask] = replacement_values[mask]

    filtered = np.moveaxis(moved, -1, axis)
    return filtered, BadPointFilterDiagnostics(
        mode=mode,
        replacement=replacement,
        flagged_count=flagged_total,
        total_count=int(array.size),
        mad_zcut=float(mad_zcut),
        ratio_cut=float(ratio_cut),
        absolute_threshold=float(absolute_threshold),
    )


def resample_observable(
    raw_samples: np.ndarray,
    *,
    method: str = "jackknife",
    axis: int = -1,
    bootstrap_samples: int = 500,
    bootstrap_sample_size: int | None = None,
    bin_size: int = 1,
    seed: int = 1984,
) -> ResampledObservable:
    """Resample a real-valued observable along the configuration axis."""

    array = np.asarray(raw_samples, dtype=float)
    normalized_axis = axis if axis >= 0 else array.ndim + axis
    configuration_count = int(array.shape[normalized_axis])
    method_name = method.lower()
    if method_name == "jackknife":
        sample_means = np.moveaxis(jackknife(array, axis=normalized_axis), normalized_axis, 0)
        average = jackknife_average(sample_means, axis=0) if gv is not None else None
    elif method_name == "bootstrap":
        bootstrapped, _ = bootstrap(
            array,
            sample_count=bootstrap_samples,
            sample_size=bootstrap_sample_size,
            axis=normalized_axis,
            bin_size=bin_size,
            seed=seed,
        )
        sample_means = np.moveaxis(bootstrapped, normalized_axis, 0)
        average = bootstrap_average(sample_means, axis=0) if gv is not None else None
    elif method_name == "none":
        sample_means = None
        average = gv.gvar(array.mean(axis=normalized_axis), array.std(axis=normalized_axis, ddof=1)) if gv is not None else None
    else:
        raise ValueError("Unsupported resampling method. Expected 'jackknife', 'bootstrap', or 'none'.")

    if average is not None:
        mean = np.asarray(gv.mean(average), dtype=float)
        error = np.asarray(gv.sdev(average), dtype=float)
    else:
        mean = np.mean(array, axis=normalized_axis)
        error = np.std(array, axis=normalized_axis, ddof=1) if configuration_count > 1 else np.zeros_like(mean, dtype=float)

    return ResampledObservable(
        method=method_name,
        sample_means=sample_means,
        mean=mean,
        error=error,
        average=average,
        configuration_count=configuration_count,
        resample_count=0 if sample_means is None else int(sample_means.shape[0]),
        bin_size=bin_size,
    )


def build_three_point_priors(prior_overrides: dict[str, Any] | None = None):
    """Return default priors for ratio/FH/joint fits."""

    _require_gvar()
    priors = gv.BufferDict()
    priors["E0"] = gv.gvar(1.0, 10.0)
    priors["log(dE1)"] = gv.gvar(0.0, 10.0)
    priors["O00_re"] = gv.gvar(1.0, 10.0)
    priors["O00_im"] = gv.gvar(1.0, 10.0)
    priors["O01_re"] = gv.gvar(1.0, 10.0)
    priors["O01_im"] = gv.gvar(1.0, 10.0)
    priors["O11_re"] = gv.gvar(1.0, 10.0)
    priors["O11_im"] = gv.gvar(1.0, 10.0)
    priors["z0"] = gv.gvar(1.0, 10.0)
    priors["z1"] = gv.gvar(1.0, 10.0)
    priors["re_b1"] = gv.gvar(0.0, 10.0)
    priors["re_b2"] = gv.gvar(0.0, 10.0)
    priors["re_b3"] = gv.gvar(0.0, 10.0)
    priors["re_c1"] = gv.gvar(0.0, 10.0)
    priors["re_c2"] = gv.gvar(0.0, 10.0)
    priors["im_b1"] = gv.gvar(0.0, 10.0)
    priors["im_b2"] = gv.gvar(0.0, 10.0)
    priors["im_b3"] = gv.gvar(0.0, 10.0)
    priors["im_c1"] = gv.gvar(0.0, 10.0)
    priors["im_c2"] = gv.gvar(0.0, 10.0)
    for key, value in (prior_overrides or {}).items():
        if key not in priors:
            raise KeyError(f"Unknown prior key for three-point fit: {key}")
        if isinstance(value, (tuple, list)) and len(value) == 2:
            priors[key] = gv.gvar(float(value[0]), float(value[1]))
        else:
            priors[key] = value
    return priors


def ratio_real_function(ra_t, ra_tau, parameters, temporal_extent: int, *, nstate: int = 2):
    """Evaluate the real part of the ratio fit model."""

    e0 = parameters["E0"]
    e1 = parameters["E0"] + np.exp(parameters["log(dE1)"])
    z0 = parameters["z0"]
    z1 = parameters["z1"]
    if nstate == 1:
        z1 = 0.0
    numerator = (
        parameters["O00_re"] * z0**2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + parameters["O01_re"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau) / (2 * e0) / (2 * e1)
        + parameters["O01_re"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau) / (2 * e1) / (2 * e0)
        + parameters["O11_re"] * z1**2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = (
        z0**2 / (2 * e0) * (np.exp(-e0 * ra_t) + np.exp(-e0 * (temporal_extent - ra_t)))
        + z1**2 / (2 * e1) * (np.exp(-e1 * ra_t) + np.exp(-e1 * (temporal_extent - ra_t)))
    )
    return numerator / denominator


def ratio_imag_function(ra_t, ra_tau, parameters, temporal_extent: int, *, nstate: int = 2):
    """Evaluate the imaginary part of the ratio fit model."""

    e0 = parameters["E0"]
    e1 = parameters["E0"] + np.exp(parameters["log(dE1)"])
    z0 = parameters["z0"]
    z1 = parameters["z1"]
    if nstate == 1:
        z1 = 0.0
    numerator = (
        parameters["O00_im"] * z0**2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + parameters["O01_im"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau) / (2 * e0) / (2 * e1)
        + parameters["O01_im"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau) / (2 * e1) / (2 * e0)
        + parameters["O11_im"] * z1**2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = (
        z0**2 / (2 * e0) * (np.exp(-e0 * ra_t) + np.exp(-e0 * (temporal_extent - ra_t)))
        + z1**2 / (2 * e1) * (np.exp(-e1 * ra_t) + np.exp(-e1 * (temporal_extent - ra_t)))
    )
    return numerator / denominator


def fh_real_function(tsep, parameters):
    """Evaluate the one-state FH model."""

    e0 = parameters["E0"]
    return parameters["O00_re"] / (2 * e0) + 0.0 * tsep


def fh_imag_function(tsep, parameters):
    """Evaluate the one-state FH model."""

    e0 = parameters["E0"]
    return parameters["O00_im"] / (2 * e0) + 0.0 * tsep


def build_ratio_samples(
    two_point_real_samples: np.ndarray,
    two_point_imag_samples: np.ndarray,
    three_point_real_samples: np.ndarray,
    three_point_imag_samples: np.ndarray,
    tsep_axis: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ratio samples from aligned two-point and three-point samples."""

    two_point_complex = np.asarray(two_point_real_samples, dtype=float) + 1j * np.asarray(two_point_imag_samples, dtype=float)
    three_point_complex = np.asarray(three_point_real_samples, dtype=float) + 1j * np.asarray(three_point_imag_samples, dtype=float)
    if two_point_complex.shape[0] != three_point_complex.shape[0]:
        raise ValueError("Two-point and three-point resamples must share the same sample axis.")

    ratio = np.empty_like(three_point_complex, dtype=complex)
    tsep_indices = np.asarray(tsep_axis, dtype=int)
    for index, tsep in enumerate(tsep_indices):
        denominator = two_point_complex[:, tsep]
        ratio[:, index, :] = three_point_complex[:, index, :] / denominator[:, None]
    return np.real(ratio), np.imag(ratio)


def average_resampled_samples(sample_means: np.ndarray, method: str):
    """Convert a sample-mean array to correlated gvars along the sample axis."""

    _require_gvar()
    if method == "jackknife":
        return jackknife_average(sample_means, axis=0)
    if method == "bootstrap":
        return bootstrap_average(sample_means, axis=0)
    raise ValueError("average_resampled_samples expects method 'jackknife' or 'bootstrap'.")


def add_error_to_resampled_samples(sample_means: np.ndarray, method: str) -> np.ndarray:
    """Attach the ensemble covariance to each resampled sample.

    This mirrors the ``add_error_to_sample`` pattern used in the legacy
    proton-analysis scripts: each resample mean is kept as the central value,
    while the covariance matrix is taken from the full resampled ensemble.
    """

    _require_gvar()
    array = np.asarray(sample_means, dtype=float)
    average = average_resampled_samples(array, method)
    covariance = gv.evalcov(average)
    return np.asarray([gv.gvar(sample, covariance) for sample in array], dtype=object)


def build_summed_ratio_samples(
    ratio_samples: np.ndarray,
    tsep_axis: Sequence[float],
    tau_axis: Sequence[float],
    *,
    tau_cut: int,
    fit_tsep: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build summed ratio samples over the allowed tau window."""

    fit_indices = _axis_indices(np.asarray(tsep_axis, dtype=int), fit_tsep)
    tau_values = np.asarray(tau_axis, dtype=int)
    summed_samples = []
    selected_tsep = []
    for index in fit_indices:
        tsep = int(tsep_axis[index])
        valid_mask = (tau_values >= tau_cut) & (tau_values <= tsep - tau_cut)
        summed_samples.append(np.sum(ratio_samples[:, index, :][:, valid_mask], axis=1))
        selected_tsep.append(tsep)
    return np.asarray(selected_tsep, dtype=float), np.stack(summed_samples, axis=1)


def build_fh_samples(
    summed_ratio_samples: np.ndarray,
    summed_tsep_axis: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build FH samples from summed ratio samples."""

    tsep = np.asarray(summed_tsep_axis, dtype=float)
    delta_t = np.diff(tsep)
    return tsep[:-1], (summed_ratio_samples[:, 1:] - summed_ratio_samples[:, :-1]) / delta_t[None, :]


def _collect_ratio_fit_data(
    ratio_average: dict[int, Sequence[Any]],
    *,
    fit_tsep: Sequence[int],
    tau_axis: Sequence[float],
    tau_cut: int,
) -> tuple[list[float], list[float], list[Any]]:
    tau_values = np.asarray(tau_axis, dtype=int)
    tsep_values: list[float] = []
    shifted_tau_values: list[float] = []
    data_values: list[Any] = []
    for tsep in fit_tsep:
        valid_mask = (tau_values >= tau_cut) & (tau_values <= tsep - tau_cut)
        for tau in tau_values[valid_mask]:
            tsep_values.append(float(tsep))
            shifted_tau_values.append(float(tau))
            data_values.append(ratio_average[int(tsep)][int(tau)])
    return tsep_values, shifted_tau_values, data_values


def fit_ratio_correlator(
    ratio_real_average: dict[int, Sequence[Any]],
    ratio_imag_average: dict[int, Sequence[Any]],
    *,
    fit_window: dict[str, dict[str, Any]],
    tau_axis: Sequence[float],
    temporal_extent: int,
    two_point_fit_result,
    prior_overrides: dict[str, Any] | None = None,
):
    """Fit the ratio observable with shared 2pt energies and overlaps."""

    _require_gvar()
    _require_lsqfit()
    priors = build_three_point_priors(prior_overrides=prior_overrides)
    priors.update({key: two_point_fit_result.p[key] for key in ("E0", "log(dE1)", "z0", "z1")})
    real_t, real_tau, ratio_re = _collect_ratio_fit_data(
        ratio_real_average,
        fit_tsep=fit_window["real"]["fit_tsep"],
        tau_axis=tau_axis,
        tau_cut=int(fit_window["real"]["tau_cut"]),
    )
    imag_t, imag_tau, ratio_im = _collect_ratio_fit_data(
        ratio_imag_average,
        fit_tsep=fit_window["imag"]["fit_tsep"],
        tau_axis=tau_axis,
        tau_cut=int(fit_window["imag"]["tau_cut"]),
    )
    x_values = {
        "re": [np.asarray(real_t, dtype=float), np.asarray(real_tau, dtype=float)],
        "im": [np.asarray(imag_t, dtype=float), np.asarray(imag_tau, dtype=float)],
    }
    y_values = {"re": ratio_re, "im": ratio_im}

    def model(x, parameters):
        return {
            "re": ratio_real_function(x["re"][0], x["re"][1], parameters, temporal_extent),
            "im": ratio_imag_function(x["im"][0], x["im"][1], parameters, temporal_extent),
        }

    return lsqfit.nonlinear_fit(data=(x_values, y_values), prior=priors, fcn=model, maxit=10_000)


def fit_fh_correlator(
    fh_real_average,
    fh_imag_average,
    *,
    fh_tsep: dict[str, Sequence[float]],
    two_point_fit_result,
    prior_overrides: dict[str, Any] | None = None,
):
    """Fit the FH observable with a one-state model."""

    _require_gvar()
    _require_lsqfit()
    priors = build_three_point_priors(prior_overrides=prior_overrides)
    priors["E0"] = two_point_fit_result.p["E0"]
    x_values = {
        "re": np.asarray(fh_tsep["real"], dtype=float),
        "im": np.asarray(fh_tsep["imag"], dtype=float),
    }
    y_values = {"re": fh_real_average, "im": fh_imag_average}

    def model(x, parameters):
        return {
            "re": fh_real_function(x["re"], parameters),
            "im": fh_imag_function(x["im"], parameters),
        }

    return lsqfit.nonlinear_fit(data=(x_values, y_values), prior=priors, fcn=model, maxit=10_000)


def fit_joint_ratio_fh_correlator(
    ratio_real_average: dict[int, Sequence[Any]],
    ratio_imag_average: dict[int, Sequence[Any]],
    fh_real_average,
    fh_imag_average,
    *,
    ratio_fit_window: dict[str, dict[str, Any]],
    tau_axis: Sequence[float],
    temporal_extent: int,
    fh_tsep: dict[str, Sequence[float]],
    two_point_fit_result,
    prior_overrides: dict[str, Any] | None = None,
):
    """Jointly fit the ratio and FH observables."""

    _require_gvar()
    _require_lsqfit()
    priors = build_three_point_priors(prior_overrides=prior_overrides)
    priors.update({key: two_point_fit_result.p[key] for key in ("E0", "log(dE1)", "z0", "z1")})
    real_t, real_tau, ratio_re = _collect_ratio_fit_data(
        ratio_real_average,
        fit_tsep=ratio_fit_window["real"]["fit_tsep"],
        tau_axis=tau_axis,
        tau_cut=int(ratio_fit_window["real"]["tau_cut"]),
    )
    imag_t, imag_tau, ratio_im = _collect_ratio_fit_data(
        ratio_imag_average,
        fit_tsep=ratio_fit_window["imag"]["fit_tsep"],
        tau_axis=tau_axis,
        tau_cut=int(ratio_fit_window["imag"]["tau_cut"]),
    )

    x_values = {
        "ratio_re": [np.asarray(real_t, dtype=float), np.asarray(real_tau, dtype=float)],
        "ratio_im": [np.asarray(imag_t, dtype=float), np.asarray(imag_tau, dtype=float)],
        "fh_re": np.asarray(fh_tsep["real"], dtype=float),
        "fh_im": np.asarray(fh_tsep["imag"], dtype=float),
    }
    y_values = {
        "ratio_re": ratio_re,
        "ratio_im": ratio_im,
        "fh_re": fh_real_average,
        "fh_im": fh_imag_average,
    }

    def model(x, parameters):
        return {
            "ratio_re": ratio_real_function(x["ratio_re"][0], x["ratio_re"][1], parameters, temporal_extent),
            "ratio_im": ratio_imag_function(x["ratio_im"][0], x["ratio_im"][1], parameters, temporal_extent),
            "fh_re": fh_real_function(x["fh_re"], parameters),
            "fh_im": fh_imag_function(x["fh_im"], parameters),
        }

    return lsqfit.nonlinear_fit(data=(x_values, y_values), prior=priors, fcn=model, maxit=10_000)


def posterior_samples_from_fit_result(fit_result, sample_count: int = DEFAULT_POSTERIOR_SAMPLES, seed: int = 1984) -> list[dict[str, float]]:
    """Draw posterior parameter samples from an lsqfit result."""

    _require_gvar()
    parameter_keys = list(fit_result.p.keys())
    parameter_values = [fit_result.p[key] for key in parameter_keys]
    mean = np.asarray([gv.mean(value) for value in parameter_values], dtype=float)
    covariance = np.asarray(gv.evalcov(parameter_values), dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=sample_count)
    return [{key: float(value) for key, value in zip(parameter_keys, draw, strict=True)} for draw in draws]


def summarize_three_point_fit(
    fit_result,
    *,
    mode: str,
    fit_windows: dict[str, Any] | None = None,
    fit_tsep: Sequence[int] | None = None,
    tau_cut: int | None = None,
) -> dict[str, Any]:
    """Serialize common diagnostics for a ratio/FH/joint fit."""

    _require_gvar()
    bare_real = fit_result.p["O00_re"] / (2 * fit_result.p["E0"])
    bare_imag = fit_result.p["O00_im"] / (2 * fit_result.p["E0"])
    summary: dict[str, Any] = {
        "performed": True,
        "mode": mode,
        "chi2": float(fit_result.chi2),
        "dof": int(fit_result.dof),
        "chi2_per_dof": float(fit_result.chi2 / fit_result.dof) if fit_result.dof else float("nan"),
        "Q": float(fit_result.Q),
        "loggbf": float(fit_result.logGBF),
        "quality": "good" if fit_result.Q >= 0.05 else "poor",
        "parameters": {},
        "bare_matrix_element": {
            "real": {"mean": float(gv.mean(bare_real)), "sdev": float(gv.sdev(bare_real))},
            "imag": {"mean": float(gv.mean(bare_imag)), "sdev": float(gv.sdev(bare_imag))},
        },
    }
    if fit_windows is not None:
        summary["fit_windows"] = fit_windows
    elif fit_tsep is not None and tau_cut is not None:
        summary["fit_tsep"] = [int(value) for value in fit_tsep]
        summary["tau_cut"] = int(tau_cut)
    for key, value in fit_result.p.items():
        summary["parameters"][key] = {"mean": float(gv.mean(value)), "sdev": float(gv.sdev(value))}
    return summary


def summarize_three_point_fit_samples(
    sample_records: Sequence[dict[str, Any]],
    representative_fit_result,
    *,
    method: str,
    mode: str,
    fit_windows: dict[str, Any] | None = None,
    fit_tsep: Sequence[int] | None = None,
    tau_cut: int | None = None,
) -> dict[str, Any]:
    """Serialize sample-by-sample fit diagnostics using the resampling ensemble."""

    _require_gvar()
    if not sample_records:
        raise ValueError("summarize_three_point_fit_samples requires at least one fit result.")

    representative = representative_fit_result
    bare_real_samples = np.asarray(
        [float(record["bare_real"]) for record in sample_records],
        dtype=float,
    )
    bare_imag_samples = np.asarray(
        [float(record["bare_imag"]) for record in sample_records],
        dtype=float,
    )
    bare_real = average_resampled_samples(bare_real_samples, method)
    bare_imag = average_resampled_samples(bare_imag_samples, method)

    parameters: dict[str, dict[str, float]] = {}
    for key in representative.p.keys():
        sample_values = np.asarray([float(record["parameter_means"][key]) for record in sample_records], dtype=float)
        averaged = average_resampled_samples(sample_values, method)
        parameters[key] = {"mean": float(gv.mean(averaged)), "sdev": float(gv.sdev(averaged))}

    chi2_per_dof = np.asarray(
        [float(record["chi2"] / record["dof"]) if record["dof"] else float("nan") for record in sample_records],
        dtype=float,
    )
    q_values = np.asarray([float(record["Q"]) for record in sample_records], dtype=float)

    return {
        "performed": True,
        "mode": mode,
        "sample_fit_count": int(len(sample_records)),
        "chi2": float(representative.chi2),
        "dof": int(representative.dof),
        "chi2_per_dof": float(np.nanmedian(chi2_per_dof)),
        "Q": float(np.nanmedian(q_values)),
        "loggbf": float(representative.logGBF),
        "quality": "good" if np.nanmedian(q_values) >= 0.05 else "poor",
        "representative_Q": float(representative.Q),
        "representative_chi2_per_dof": float(representative.chi2 / representative.dof)
        if representative.dof
        else float("nan"),
        "parameters": parameters,
        "bare_matrix_element": {
            "real": {"mean": float(gv.mean(bare_real)), "sdev": float(gv.sdev(bare_real))},
            "imag": {"mean": float(gv.mean(bare_imag)), "sdev": float(gv.sdev(bare_imag))},
        },
    } | (
        {"fit_windows": fit_windows}
        if fit_windows is not None
        else {
            "fit_tsep": [int(value) for value in fit_tsep or ()],
            "tau_cut": int(tau_cut or 0),
        }
    )


def evaluate_ratio_band(
    fit_result,
    *,
    tsep: int,
    tau_values: Sequence[float],
    temporal_extent: int,
    part: str,
    sample_count: int = DEFAULT_POSTERIOR_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a posterior band for one ratio time slice."""

    draws = posterior_samples_from_fit_result(fit_result, sample_count=sample_count)
    tau_array = np.asarray(tau_values, dtype=float)
    evaluator = ratio_real_function if part == "real" else ratio_imag_function
    samples = np.asarray(
        [evaluator(float(tsep), tau_array, draw, temporal_extent) for draw in draws],
        dtype=float,
    )
    return np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)


def evaluate_fh_band(
    fit_result,
    *,
    tsep_values: Sequence[float],
    part: str,
    sample_count: int = DEFAULT_POSTERIOR_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a posterior band for the FH observable."""

    draws = posterior_samples_from_fit_result(fit_result, sample_count=sample_count)
    tsep_array = np.asarray(tsep_values, dtype=float)
    evaluator = fh_real_function if part == "real" else fh_imag_function
    samples = np.asarray([evaluator(tsep_array, draw) for draw in draws], dtype=float)
    return np.mean(samples, axis=0), np.std(samples, axis=0, ddof=1)


def _axis_indices(axis_values: np.ndarray, selected_values: Sequence[int]) -> list[int]:
    indices = []
    for value in selected_values:
        matches = np.where(axis_values == int(value))[0]
        if len(matches) == 0:
            raise ValueError(f"Requested tsep {value} is not present in the three-point dataset.")
        indices.append(int(matches[0]))
    return indices


def _replacement_value(values: np.ndarray, replacement: str) -> float:
    if replacement == "zero":
        return 0.0
    if replacement == "sign":
        median = float(np.median(values))
        return 1.0 if median >= 0.0 else -1.0
    return float(np.median(values))
