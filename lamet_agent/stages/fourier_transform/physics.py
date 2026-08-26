"""Signed-coordinate completion, tail connection, and discrete Fourier physics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel import FitNumericalError, fourier_transform, nonlinear_fit
from lamet_agent.parallel._pool import _ParallelPool


def load_data(value: Any) -> EnsembleData:
    """Load one explicit coordinate-space EnsembleData source."""
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, Path):
        if value.suffix.lower() != ".nc":
            raise ValueError("Fourier input must be a .nc artifact")
        return EnsembleData.from_netcdf(value)
    raise TypeError("Fourier input is neither EnsembleData nor a NetCDF Path")


def _symmetry_mapping(value: Mapping[str, str]) -> dict[str, str]:
    """Validate the authored component-wise signed-z convention."""
    if not isinstance(value, Mapping) or set(value) != {"real", "imag"}:
        raise ValueError("symmetry must contain exactly real and imag")
    real = value["real"]
    imag = value["imag"]
    if real not in {"even", "odd", "explicit"} or imag not in {"even", "odd", "explicit"}:
        raise ValueError("symmetry components must be even, odd, or explicit")
    return {"real": real, "imag": imag}


def _stored_symmetry(attrs: Mapping[str, object]) -> dict[str, str]:
    """Read a JSON-serializable symmetry convention from data attrs."""
    value = attrs.get("symmetry")
    if value is None:
        raise ValueError("tail data is missing its authored symmetry convention")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("stored symmetry attr must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError("stored symmetry attr must be a mapping")
    return _symmetry_mapping(value)


def _signed_from_positive(
    positive_values: np.ndarray, extended_z: np.ndarray, symmetry: Mapping[str, str]
) -> np.ndarray:
    """Apply component-wise parity to values sampled on a nonnegative grid."""
    negative_mask = extended_z < 0
    negative = positive_values[negative_mask]
    real = np.real(negative)
    imag = np.imag(negative)
    if symmetry["real"] == "explicit" or symmetry["imag"] == "explicit":
        raise ValueError("explicit symmetry requires both positive and negative input coordinates")
    if symmetry["real"] == "odd":
        real = -real
    if symmetry["imag"] == "odd":
        imag = -imag
    return np.concatenate([real + 1j * imag, positive_values[~negative_mask]])


def _tail_parameter_names_base(
    model_id: str, order: str, observable: str, psi1_flavor_class: str, psi2_flavor_class: str
) -> list[str]:
    """Return the independent parameters of one PDF or meson-DA endpoint tail."""
    if observable == "DA":
        names = []
        if psi1_flavor_class != "light" or psi2_flavor_class != "heavy":
            names.extend(["A1", "phi1"])
        if psi1_flavor_class != "heavy" or psi2_flavor_class != "light":
            if not (psi1_flavor_class == "light" and psi2_flavor_class == "light"):
                names.extend(["A2", "phi2"])
        if order == "NLA":
            names.extend([name + "p" for name in names])
        names.append("Lambda")
    else:
        names = ["A2", "phi2"]
        if order == "NLA":
            names.extend(["A2p", "phi2p"])
        names.append("Lambda")
    if model_id == "cg_nla":
        names.append("n")
    return names


def _tail_model_values_base(
    z_fm: np.ndarray,
    model_id: str,
    parameters: Mapping[str, float],
    *,
    order: str = "NLA",
    observable: str = "PDF",
    momentum_gev: float | None = None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
) -> np.ndarray:
    """Evaluate the locked GI/CG LA or NLA tail family on signed coordinates.

    ``gi_nla`` is ``[A2 exp(i phi2 sign(z)) + A2p exp(i phi2p sign(z))/|z|]
    exp(-Lambda |z|/(hbar*c))``.  ``cg_nla`` divides the same family by
    ``|z|**n``.  The model is evaluated away from the origin; callers use the
    measured origin value because the displayed asymptotic family is not a
    short-distance ansatz.
    """
    if model_id not in {"gi_nla", "cg_nla"}:
        raise ValueError(f"unsupported tail model '{model_id}'")
    order = order.upper()
    if order not in {"LA", "NLA"}:
        raise ValueError("tail order must be LA or NLA")
    if (
        observable not in {"PDF", "DA"}
        or psi1_flavor_class not in {"light", "heavy"}
        or psi2_flavor_class not in {"light", "heavy"}
    ):
        raise ValueError("tail observable and DA flavor classes are invalid")
    expected = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class)
    if set(parameters) != set(expected):
        raise ValueError(f"tail parameters must contain exactly {expected}")
    z = np.asarray(z_fm, dtype=float)
    absolute = np.abs(z)
    if np.any(absolute <= 0):
        raise ValueError("tail model is undefined at z=0")
    sign = np.sign(z)
    lambda_value = float(parameters["Lambda"])
    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("tail Lambda must be finite and positive")
    if observable == "DA":
        if (
            not isinstance(momentum_gev, (int, float))
            or isinstance(momentum_gev, bool)
            or not math.isfinite(float(momentum_gev))
            or float(momentum_gev) <= 0
        ):
            raise ValueError("DA tails require finite positive momentum_gev")
        light_light = psi1_flavor_class == "light" and psi2_flavor_class == "light"
        first = (
            0.0
            if psi1_flavor_class == "light" and psi2_flavor_class == "heavy"
            else float(parameters["A1"])
            * np.exp(1j * (float(parameters["phi1"]) - float(momentum_gev) * absolute / HBAR_C_GEV_FM))
        )
        second = (
            0.0
            if psi1_flavor_class == "heavy" and psi2_flavor_class == "light"
            else (
                float(parameters["A1"]) * np.exp(-1j * float(parameters["phi1"]))
                if light_light
                else float(parameters["A2"]) * np.exp(1j * float(parameters["phi2"]))
            )
        )
        result = first + second
        if order == "NLA":
            first_prime = (
                0.0
                if psi1_flavor_class == "light" and psi2_flavor_class == "heavy"
                else float(parameters["A1p"])
                * np.exp(1j * (float(parameters["phi1p"]) - float(momentum_gev) * absolute / HBAR_C_GEV_FM))
            )
            second_prime = (
                0.0
                if psi1_flavor_class == "heavy" and psi2_flavor_class == "light"
                else (
                    float(parameters["A1p"]) * np.exp(-1j * float(parameters["phi1p"]))
                    if light_light
                    else float(parameters["A2p"]) * np.exp(1j * float(parameters["phi2p"]))
                )
            )
            result = result + (first_prime + second_prime) / absolute
        result = np.real(result) + 1j * sign * np.imag(result)
    else:
        result = float(parameters["A2"]) * np.exp(1j * float(parameters["phi2"]) * sign)
        if order == "NLA":
            result = result + float(parameters["A2p"]) * np.exp(1j * float(parameters["phi2p"]) * sign) / absolute
    result = result * np.exp(-lambda_value * absolute / HBAR_C_GEV_FM)
    if model_id == "cg_nla":
        exponent = float(parameters["n"])
        if not math.isfinite(exponent) or exponent <= 0:
            raise ValueError("CG tail power n must be finite and positive")
        result = result / absolute**exponent
    return result


def _tail_fit_fcn_base(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate real and imaginary tail channels for the shared fitter."""
    z = np.asarray(x["z"], dtype=float)
    absolute = np.abs(z)
    sign = np.sign(z)
    decay = np.exp(-(parameters["Lambda"] + float(x["lambda0_gev"])) * absolute / HBAR_C_GEV_FM)
    if x["observable"] == "DA":
        light_light = x["psi1_flavor_class"] == "light" and x["psi2_flavor_class"] == "light"
        first_phase = parameters.get("phi1", 0.0) - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
        first_amplitude = (
            0.0 if x["psi1_flavor_class"] == "light" and x["psi2_flavor_class"] == "heavy" else parameters["A1"]
        )
        second_phase = -parameters["phi1"] if light_light else parameters.get("phi2", 0.0)
        second_amplitude = (
            0.0
            if x["psi1_flavor_class"] == "heavy" and x["psi2_flavor_class"] == "light"
            else (parameters["A1"] if light_light else parameters["A2"])
        )
        real = first_amplitude * gv.cos(first_phase) + second_amplitude * gv.cos(second_phase)
        imag = first_amplitude * gv.sin(first_phase) + second_amplitude * gv.sin(second_phase)
        if x["order"] == "NLA":
            first_prime_phase = parameters.get("phi1p", 0.0) - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
            first_prime_amplitude = (
                0.0 if x["psi1_flavor_class"] == "light" and x["psi2_flavor_class"] == "heavy" else parameters["A1p"]
            )
            second_prime_phase = -parameters["phi1p"] if light_light else parameters.get("phi2p", 0.0)
            second_prime_amplitude = (
                0.0
                if x["psi1_flavor_class"] == "heavy" and x["psi2_flavor_class"] == "light"
                else (parameters["A1p"] if light_light else parameters["A2p"])
            )
            real = (
                real
                + (
                    first_prime_amplitude * gv.cos(first_prime_phase)
                    + second_prime_amplitude * gv.cos(second_prime_phase)
                )
                / absolute
            )
            imag = (
                imag
                + (
                    first_prime_amplitude * gv.sin(first_prime_phase)
                    + second_prime_amplitude * gv.sin(second_prime_phase)
                )
                / absolute
            )
        imag = sign * imag
    else:
        real = parameters["A2"] * np.cos(parameters["phi2"] * sign)
        imag = parameters["A2"] * np.sin(parameters["phi2"] * sign)
        if x["order"] == "NLA":
            real = real + parameters["A2p"] * np.cos(parameters["phi2p"] * sign) / absolute
            imag = imag + parameters["A2p"] * np.sin(parameters["phi2p"] * sign) / absolute
    real = real * decay
    imag = imag * decay
    if x["model_id"] == "cg_nla":
        real = real / absolute ** parameters["n"]
        imag = imag / absolute ** parameters["n"]
    if x["component"] == "re":
        return real
    if x["component"] == "im":
        return imag
    return np.concatenate([real, imag])


def _tail_parameter_names(
    model_id: str,
    order: str,
    observable: str,
    psi1_flavor_class: str,
    psi2_flavor_class: str,
    sector: str = "full",
    hadron: str = "",
) -> list[str]:
    """Return the reference endpoint parameter set for the selected observable."""
    if observable == "PDF" and hadron.lower() == "pion" and sector.lower() == "valence":
        names = ["A2", "A1", "phi1"]
        if order.upper() == "NLA":
            names.extend(["A2p", "A1p", "phi1p"])
        names.append("Lambda")
        if model_id == "cg_nla":
            names.append("n")
        return names
    return _tail_parameter_names_base(model_id, order, observable, psi1_flavor_class, psi2_flavor_class)


def tail_model_values(
    z_fm: np.ndarray,
    model_id: str,
    parameters: Mapping[str, float],
    *,
    order: str = "NLA",
    observable: str = "PDF",
    momentum_gev: float | None = None,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    sector: str = "full",
    hadron: str = "",
) -> np.ndarray:
    """Evaluate the reference pion-valence tail or the generic migrated family."""
    if observable != "PDF" or hadron.lower() != "pion" or sector.lower() != "valence":
        return _tail_model_values_base(
            z_fm,
            model_id,
            parameters,
            order=order,
            observable=observable,
            momentum_gev=momentum_gev,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
        )
    expected = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class, sector, hadron)
    if set(parameters) != set(expected):
        raise ValueError(f"tail parameters must contain exactly {expected}")
    if (
        not isinstance(momentum_gev, (int, float))
        or isinstance(momentum_gev, bool)
        or not math.isfinite(float(momentum_gev))
        or float(momentum_gev) <= 0
    ):
        raise ValueError("pion PDF tails require finite positive momentum_gev")
    absolute = np.abs(np.asarray(z_fm, dtype=float))
    if np.any(absolute <= 0):
        raise ValueError("tail model is undefined at z=0")
    phase = float(parameters["phi1"]) - float(momentum_gev) * absolute / HBAR_C_GEV_FM
    result = float(parameters["A2"]) + 2.0 * float(parameters["A1"]) * np.cos(phase)
    if order.upper() == "NLA":
        phase_prime = float(parameters["phi1p"]) - float(momentum_gev) * absolute / HBAR_C_GEV_FM
        result = (
            result
            + (float(parameters["A2p"]) + 2.0 * float(parameters["A1p"]) * np.cos(phase_prime))
            * HBAR_C_GEV_FM
            / absolute
        )
    result = result * np.exp(-float(parameters["Lambda"]) * absolute / HBAR_C_GEV_FM)
    if model_id == "cg_nla":
        result = result / (absolute / HBAR_C_GEV_FM) ** float(parameters["n"])
    return np.asarray(result, dtype=complex)


def tail_fit_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the reference pion-valence fit family or the generic family."""
    if (
        x["observable"] != "PDF"
        or str(x.get("hadron", "")).lower() != "pion"
        or str(x.get("sector", "")).lower() != "valence"
    ):
        return _tail_fit_fcn_base(x, parameters)
    absolute = np.abs(np.asarray(x["z"], dtype=float))
    phase = parameters["phi1"] - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
    real = parameters["A2"] + 2.0 * parameters["A1"] * gv.cos(phase)
    if x["order"] == "NLA":
        phase_prime = parameters["phi1p"] - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
        real = real + (parameters["A2p"] + 2.0 * parameters["A1p"] * gv.cos(phase_prime)) * HBAR_C_GEV_FM / absolute
    real = real * gv.exp(-(parameters["Lambda"] + float(x["lambda0_gev"])) * absolute / HBAR_C_GEV_FM)
    if x["model_id"] == "cg_nla":
        real = real / (absolute / HBAR_C_GEV_FM) ** parameters["n"]
    if x["component"] == "im":
        return np.zeros_like(absolute)
    if x["component"] == "both":
        return np.concatenate([real, np.zeros_like(absolute)])
    return real


def _tail_bounds(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for name in names:
        if name.startswith("A"):
            lower.append(-20.0)
            upper.append(20.0)
        elif name.startswith("phi"):
            lower.append(-np.pi)
            upper.append(np.pi)
        elif name == "Lambda":
            lower.append(0.0)
            upper.append(np.inf)
        elif name == "n":
            lower.append(-2.0)
            upper.append(4.0)
        else:
            raise ValueError(f"unsupported tail parameter '{name}'")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _bounded_to_internal(value: float, lower: float, upper: float) -> float:
    if np.isfinite(lower) and np.isfinite(upper):
        width = upper - lower
        clipped = min(max(float(value), lower + 1e-8 * width), upper - 1e-8 * width)
        ratio = (clipped - lower) / width
        return float(np.log(ratio / (1.0 - ratio)))
    if np.isfinite(lower):
        return float(np.log(max(float(value) - lower, 1e-8)))
    if np.isfinite(upper):
        return float(np.log(max(upper - float(value), 1e-8)))
    return float(value)


def _internal_to_bounded(value: Any, lower: float, upper: float) -> Any:
    if np.isfinite(lower) and np.isfinite(upper):
        return lower + (upper - lower) / (1.0 + gv.exp(-value))
    if np.isfinite(lower):
        return lower + gv.exp(value)
    if np.isfinite(upper):
        return upper - gv.exp(value)
    return value


def _internal_start(initial: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> gv.BufferDict:
    lower, upper = bounds
    result = gv.BufferDict()
    for index, value in enumerate(initial):
        result[f"u{index}"] = _bounded_to_internal(float(value), float(lower[index]), float(upper[index]))
    return result


def _physical_tail_parameters(
    parameters: Mapping[str, Any],
    names: list[str],
    bounds: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    lower, upper = bounds
    return {
        name: _internal_to_bounded(parameters[f"u{index}"], float(lower[index]), float(upper[index]))
        for index, name in enumerate(names)
    }


def _bounded_tail_fit_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    names = list(x["parameter_names"])
    bounds = (np.asarray(x["lower_bounds"], dtype=float), np.asarray(x["upper_bounds"], dtype=float))
    return tail_fit_fcn(x, _physical_tail_parameters(parameters, names, bounds))


def fit_tail_parameters(
    data: EnsembleData,
    *,
    model_id: str,
    z_min_fm: float,
    z_max_fm: float,
    prior_means: Mapping[str, float],
    prior_widths: Mapping[str, float],
    order: str = "NLA",
    component: str = "both",
    lambda0_gev: float = 0.0,
    observable: str = "PDF",
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    sector: str = "full",
    hadron: str = "",
    workers: int = 1,
    mode: Literal["center", "resamples"] = "resamples",
    posterior_prior_scale: float | None = None,
    _parallel: _ParallelPool | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Fit one tail model at the ensemble center or across all resamples."""
    if "z" not in data.dims or data.dims != ["z"]:
        raise ValueError("tail fitting requires one z dimension")
    z = np.asarray(data.coords["z"], dtype=float)
    mask = z >= float(z_min_fm) - 1e-12
    mask &= z <= float(z_max_fm) + 1e-12
    mask &= z > 0
    if model_id not in {"gi_nla", "cg_nla"}:
        raise ValueError(f"unsupported tail model '{model_id}'")
    order = order.upper()
    if order not in {"LA", "NLA"} or component not in {"re", "im", "both"}:
        raise ValueError("tail order and component must be LA/NLA and re/im/both")
    if not math.isfinite(lambda0_gev) or lambda0_gev < 0:
        raise ValueError("lambda0_gev must be finite and nonnegative")
    if (
        observable not in {"PDF", "DA"}
        or psi1_flavor_class not in {"light", "heavy"}
        or psi2_flavor_class not in {"light", "heavy"}
    ):
        raise ValueError("tail observable and DA flavor classes are invalid")
    momentum = data.attrs.get("momentum_gev")
    if observable == "DA" and (
        not isinstance(momentum, (int, float))
        or isinstance(momentum, bool)
        or not math.isfinite(float(momentum))
        or float(momentum) <= 0
    ):
        raise ValueError("DA tail fitting requires finite positive momentum_gev")
    names = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class, sector, hadron)
    channel_count = 2 if component == "both" else 1
    required_points = max(int(math.ceil(len(names) / channel_count)), 2)
    if int(np.count_nonzero(mask)) < required_points:
        raise ValueError("tail fit range has too few points for the selected model")
    if set(prior_means) != set(names) or set(prior_widths) != set(names):
        raise ValueError(f"tail priors must contain exactly {names}")
    initial = np.asarray([float(prior_means[name]) for name in names], dtype=float)
    widths = np.asarray([float(prior_widths[name]) for name in names], dtype=float)
    if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("tail prior means must be finite and widths must be finite and positive")
    if initial[names.index("Lambda")] <= 0 or ("n" in names and initial[names.index("n")] <= 0):
        raise ValueError("tail Lambda and CG power prior means must be positive")
    samples = np.asarray(data.values)
    if samples.shape[0] < 2:
        raise ValueError("tail fitting requires at least two resamples")
    selected = samples[:, mask]
    real_selected = np.real(selected)
    imag_selected = np.imag(selected)
    observations = (
        real_selected
        if component == "re"
        else imag_selected
        if component == "im"
        else np.concatenate([real_selected, imag_selected], axis=1)
    )
    fit_data = EnsembleData(
        data.ensemble,
        data.resample,
        list(observations),
        ["observation"],
        {"observation": list(range(observations.shape[1]))},
    )
    sample_error_mode = str(data.attrs.get("sample_error_mode", "covariance"))
    if sample_error_mode == "covariance":
        covariance = np.asarray(gv.evalcov(fit_data.average("covariance")), dtype=float)
    else:
        real_data = EnsembleData(data.ensemble, data.resample, list(real_selected), ["z"], {"z": z[mask].tolist()})
        imag_data = EnsembleData(data.ensemble, data.resample, list(imag_selected), ["z"], {"z": z[mask].tolist()})
        error_floor = max(
            1e-8,
            0.02
            * max(
                float(np.max(np.abs(np.mean(real_selected, axis=0)))),
                float(np.max(np.abs(np.mean(imag_selected, axis=0)))),
                1.0,
            ),
        )
        real_error = np.maximum(np.asarray(gv.sdev(real_data.average(sample_error_mode)), dtype=float), error_floor)
        imag_error = np.maximum(np.asarray(gv.sdev(imag_data.average(sample_error_mode)), dtype=float), error_floor)
        fit_error = (
            real_error
            if component == "re"
            else imag_error
            if component == "im"
            else np.concatenate([real_error, imag_error])
        )
        covariance = np.diag(fit_error**2)
    bounds = _tail_bounds(names)
    internal_start = _internal_start(initial, bounds)
    prior = gv.BufferDict()
    for index, width in enumerate(widths):
        prior[f"u{index}"] = gv.gvar(float(internal_start[f"u{index}"]), float(width))
    fit_x = {
        "z": z[mask],
        "model_id": model_id,
        "order": order,
        "component": component,
        "lambda0_gev": float(lambda0_gev),
        "observable": observable,
        "momentum_gev": momentum,
        "psi1_flavor_class": psi1_flavor_class,
        "psi2_flavor_class": psi2_flavor_class,
        "sector": sector,
        "hadron": hadron,
        "parameter_names": names,
        "lower_bounds": bounds[0],
        "upper_bounds": bounds[1],
    }

    fit_options = {
        "maxit": 2000,
        "svdcut": 1e-12,
    }
    if posterior_prior_scale is not None:
        if not math.isfinite(posterior_prior_scale) or posterior_prior_scale <= 0:
            raise ValueError("posterior_prior_scale must be finite and positive")
        initial_result = nonlinear_fit(
            (fit_x, fit_data),
            _bounded_tail_fit_fcn,
            prior,
            workers=workers,
            covariance=covariance,
            sample_error_mode=sample_error_mode,
            mode="center",
            **fit_options,
        )
        posterior_prior = gv.BufferDict()
        for key in prior:
            center = initial_result.fit.p[key]
            width = float(gv.sdev(center)) * posterior_prior_scale
            posterior_prior[key] = gv.gvar(float(gv.mean(center)), max(width, 1e-8))
        prior = posterior_prior
    result = nonlinear_fit(
        (fit_x, fit_data),
        _bounded_tail_fit_fcn,
        prior,
        workers=workers,
        covariance=covariance,
        sample_error_mode=sample_error_mode,
        mode=mode,
        tolerate_sample_failures=mode == "resamples",
        _parallel=_parallel,
        **fit_options,
    )
    center_parameters = result.pmean
    fitted_parameters = (
        (center_parameters,)
        if mode == "center"
        else tuple(parameters if parameters is not None else center_parameters for parameters in result.samples)
    )
    records = []
    for parameters in fitted_parameters:
        physical = _physical_tail_parameters(parameters, names, bounds)
        records.append({name: float(physical[name]) + (lambda0_gev if name == "Lambda" else 0.0) for name in names})
    center_diagnostics = {
        "chi2": result.chi2,
        "dof": float(result.dof),
        "chi2_dof": result.chi2 / result.dof,
        "Q": result.Q,
        "logGBF": result.logGBF,
        "aic": result.chi2 + 2.0 * len(names),
    }
    if mode == "center":
        return records, center_diagnostics
    sample_diagnostics = [
        diagnostics if diagnostics is not None else center_diagnostics for diagnostics in result.sample_diagnostics
    ]
    return records, {
        **center_diagnostics,
        "sample_diagnostics": sample_diagnostics,
        "sample_failures": list(result.sample_errors),
        "n_failed_samples": result.n_failed_samples,
    }


def complete_signed_z(data: EnsembleData, symmetry: Mapping[str, str]) -> EnsembleData:
    """Complete a positive ``z`` grid using explicit real/imaginary parity.

    ``even`` and ``odd`` are applied independently to the real and imaginary
    components. Existing signed grids are sorted and retained; ``explicit``
    therefore requires both positive and negative input coordinates.
    """
    if "z" not in data.dims:
        raise ValueError("Fourier input must have a z dimension")
    z = np.asarray(data.coords["z"], dtype=float)
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z coordinates must be finite and strictly increasing")
    convention = _symmetry_mapping(symmetry)
    if np.any(z < 0) and np.any(z > 0):
        attrs = data.attrs
        attrs["symmetry"] = json.dumps(convention, sort_keys=True)
        return EnsembleData(
            data.ensemble,
            data.resample,
            [np.asarray(sample) for sample in data.values],
            data.dims,
            data.coords,
            attrs=attrs,
            name=data.name,
        ).sort_dim("z")
    if convention["real"] == "explicit" or convention["imag"] == "explicit":
        raise ValueError("explicit symmetry requires both positive and negative input coordinates")
    positive_indices = np.where(z >= 0)[0]
    positive_z = z[positive_indices]
    negative_z = -positive_z[positive_z > 0][::-1]
    output_z = np.concatenate([negative_z, positive_z])
    values = []
    for sample in data.values:
        positive = np.asarray(sample)[positive_indices]
        positive_nonzero = positive[positive_z > 0][::-1]
        negative_real = np.real(positive_nonzero)
        negative_imag = np.imag(positive_nonzero)
        if convention["real"] == "odd":
            negative_real = -negative_real
        if convention["imag"] == "odd":
            negative_imag = -negative_imag
        negative = negative_real + 1j * negative_imag
        values.append(np.concatenate([negative, positive]))
    attrs = data.attrs
    attrs["signed_z_completion"] = json.dumps(convention, sort_keys=True)
    attrs["symmetry"] = json.dumps(convention, sort_keys=True)
    return EnsembleData(
        data.ensemble,
        data.resample,
        values,
        data.dims,
        {**data.coords, "z": output_z.tolist()},
        attrs=attrs,
        name=data.name,
    )


def extend_tail(
    data: EnsembleData,
    *,
    z_max_fm: float,
    z_min_fm: float,
    smoothing_method: str,
    smoothing_width_fm: float,
    model_id: str,
    tail_parameters: list[Mapping[str, float]],
    order: str = "NLA",
    observable: str = "PDF",
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    sector: str = "full",
    hadron: str = "",
) -> EnsembleData:
    """Connect an exponentially damped endpoint tail on the input spacing."""
    if (
        z_max_fm <= 0
        or z_min_fm < 0
        or z_max_fm <= z_min_fm
        or smoothing_width_fm <= 0
        or smoothing_method not in {"linear", "none"}
    ):
        raise ValueError("tail ranges and smoothing width are invalid")
    z = np.asarray(data.coords["z"], dtype=float)
    positive = z[z >= 0]
    if positive.size < 2:
        raise ValueError("tail extension requires at least two nonnegative z points")
    spacing = float(np.min(np.diff(positive)))
    steps = int(math.floor(float(z_max_fm) / spacing + 0.5))
    if steps < 1:
        raise ValueError("tail extent does not reach one input-grid step")
    extended_positive = np.arange(steps + 1, dtype=float) * spacing
    extended_z = np.concatenate([-extended_positive[extended_positive > 0][::-1], extended_positive])
    symmetry = _stored_symmetry(data.attrs)
    completed = complete_signed_z(data, symmetry) if not (np.any(z < 0) and np.any(z > 0)) else data
    source_z = np.asarray(completed.coords["z"], dtype=float)
    symmetry = _stored_symmetry(completed.attrs)
    positive_indices = np.where(source_z >= 0)[0]
    source_positive_z = source_z[positive_indices]
    if source_positive_z.size < 2 or np.any(np.diff(source_positive_z) <= 0):
        raise ValueError("tail extension requires a unique nonnegative source grid")

    explicit = symmetry["real"] == "explicit" or symmetry["imag"] == "explicit"
    source_negative_mask = source_z < 0
    source_negative_z = source_z[source_negative_mask]

    def interpolate_sample(sample: np.ndarray) -> np.ndarray:
        """Interpolate parity-completed or explicitly signed source branches."""
        positive_sample = np.asarray(sample)[positive_indices]
        if not explicit:
            right_real = np.real(positive_sample[-1])
            right_imag = np.imag(positive_sample[-1])
            real = np.interp(
                np.abs(extended_z),
                source_positive_z,
                np.real(positive_sample),
                left=np.real(positive_sample[0]),
                right=right_real,
            )
            imag = np.interp(
                np.abs(extended_z),
                source_positive_z,
                np.imag(positive_sample),
                left=np.imag(positive_sample[0]),
                right=right_imag,
            )
            return _signed_from_positive(real + 1j * imag, extended_z, symmetry)
        if source_negative_z.size == 0:
            raise ValueError("explicit symmetry requires both signed source branches")
        output = np.empty(extended_z.size, dtype=complex)
        negative_sample = np.asarray(sample)[source_negative_mask]
        positive_target = extended_z >= 0
        negative_target = ~positive_target
        positive_right = np.asarray(positive_sample)[-1]
        negative_right = np.asarray(negative_sample)[-1]
        output[positive_target] = np.interp(
            extended_z[positive_target],
            source_positive_z,
            np.real(positive_sample),
            left=np.real(positive_sample[0]),
            right=np.real(positive_right),
        ) + 1j * np.interp(
            extended_z[positive_target],
            source_positive_z,
            np.imag(positive_sample),
            left=np.imag(positive_sample[0]),
            right=np.imag(positive_right),
        )
        output[negative_target] = np.interp(
            extended_z[negative_target],
            source_negative_z,
            np.real(negative_sample),
            left=np.real(negative_sample[0]),
            right=np.real(negative_right),
        ) + 1j * np.interp(
            extended_z[negative_target],
            source_negative_z,
            np.imag(negative_sample),
            left=np.imag(negative_sample[0]),
            right=np.imag(negative_right),
        )
        return output

    values = []
    for sample_index, sample in enumerate(completed.values):
        measured = interpolate_sample(np.asarray(sample))
        tail_start = min(max(z_min_fm, 0.0), z_max_fm)
        if sample_index >= len(tail_parameters):
            raise ValueError("tail parameter records are not aligned with input samples")
        nonzero = np.where(np.abs(extended_z) <= np.finfo(float).eps, np.finfo(float).eps, extended_z)
        extension = tail_model_values(
            nonzero,
            model_id,
            tail_parameters[sample_index],
            order=order,
            observable=observable,
            momentum_gev=data.attrs.get("momentum_gev"),
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            sector=sector,
            hadron=hadron,
        )
        extension[np.abs(extended_z) <= np.finfo(float).eps] = measured[np.abs(extended_z) <= np.finfo(float).eps]
        u = (np.abs(extended_z) - tail_start) / smoothing_width_fm
        weight = (
            np.where(np.abs(extended_z) <= z_max_fm, 1.0, 0.0)
            if smoothing_method == "none"
            else np.where(u <= 0, 1.0, np.where(u >= 1, 0.0, 1.0 - u))
        )
        values.append(weight * measured + (1.0 - weight) * extension)
    attrs = completed.attrs
    attrs.update(
        {
            "tail_model": model_id,
            "tail_order": order.upper(),
            "tail_extent_fm": float(z_max_fm),
            "smoothing_method": smoothing_method,
        }
    )
    return EnsembleData(
        completed.ensemble,
        completed.resample,
        values,
        ["z"],
        {"z": extended_z.tolist()},
        attrs=attrs,
        name=completed.name,
    )


def _scan_tail_priors(
    *,
    model_id: str,
    order: str,
    lambda0_gev: float,
    observable: str = "PDF",
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    sector: str = "full",
    hadron: str = "",
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the original fixed tail starts and first-pass prior widths."""
    names = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class, sector, hadron)
    means = {}
    widths = {}
    amplitude_index = 0
    for name in names:
        if name == "Lambda":
            means[name] = max(0.5 - lambda0_gev, 0.05)
        elif name == "n":
            means[name] = 0.5
        elif name.startswith("phi"):
            means[name] = 0.0
        else:
            means[name] = 1.0 if amplitude_index == 0 else 0.1
            amplitude_index += 1
        widths[name] = 3.0
    return means, widths


def _select_fourier_range(candidates: list[dict[str, Any]], *, q_min: float) -> dict[str, Any]:
    """Apply the original center-fit range selection rule."""
    successful = [
        candidate
        for candidate in candidates
        if candidate.get("fit_success", False) and math.isfinite(float(candidate.get("Q", float("nan"))))
    ]
    if not successful:
        raise FitNumericalError("no Fourier range candidate has a usable center fit")
    passing = [
        candidate
        for candidate in successful
        if float(candidate["Q"]) >= q_min and math.isfinite(float(candidate.get("logGBF", float("nan"))))
    ]
    if passing:
        return max(passing, key=lambda candidate: float(candidate["logGBF"]))
    return max(successful, key=lambda candidate: float(candidate["Q"]))


def _sample_model_weights(
    candidates: list[dict[str, Any]],
    *,
    n_sample: int,
    q_min: float,
    model_average: bool,
) -> np.ndarray:
    """Apply the original per-sample model choice or evidence average."""
    weights = np.zeros((len(candidates), n_sample), dtype=float)
    for sample_index in range(n_sample):
        diagnostics = [candidate["sample_diagnostics"][sample_index] for candidate in candidates]
        valid = np.asarray(
            [
                candidate["sample_failures"][sample_index] is None and math.isfinite(float(item["logGBF"]))
                for candidate, item in zip(candidates, diagnostics)
            ],
            dtype=bool,
        )
        q_values = np.asarray([float(item["Q"]) for item in diagnostics], dtype=float)
        log_gbf = np.asarray([float(item["logGBF"]) for item in diagnostics], dtype=float)
        if model_average and np.any(valid):
            shifted = np.exp(log_gbf[valid] - np.max(log_gbf[valid]))
            weights[valid, sample_index] = shifted / np.sum(shifted)
            continue
        passing = np.flatnonzero(valid & (q_values >= q_min))
        if passing.size:
            selected = int(passing[np.argmax(log_gbf[passing])])
        else:
            fallback = np.flatnonzero(np.isfinite(q_values))
            selected = int(fallback[np.argmax(q_values[fallback])]) if fallback.size else 0
        weights[selected, sample_index] = 1.0
    return weights


def scan_fourier_transform(
    data: EnsembleData,
    x_grid: list[float],
    *,
    transform: Mapping[str, Any],
    tail: Mapping[str, Any],
    scan: Mapping[str, Any],
    observable: str = "PDF",
    phase_transfer_da: bool = False,
    psi1_flavor_class: str = "heavy",
    psi2_flavor_class: str = "heavy",
    workers: int = 1,
    _parallel: _ParallelPool | None = None,
) -> dict[str, Any]:
    """Fit, transform, and select one complete native Fourier candidate scan."""
    transform_keys = {"phase_sign", "x_shift", "prefactor"}
    tail_keys = {"models", "z_min_fm", "z_max_fm", "extent_fm", "smoothing_method"}
    scan_keys = {
        "orders",
        "sector",
        "lambda0_gev",
        "prior_widths",
        "model_average",
        "max_schemes",
        "component",
        "output_scale",
        "q_min",
    }
    if set(transform) != transform_keys or set(tail) != tail_keys or set(scan) != scan_keys:
        raise ValueError("Fourier transform, tail, and scan mappings do not match the native interface")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if observable not in {"PDF", "DA"} or not isinstance(phase_transfer_da, bool):
        raise ValueError("observable and phase_transfer_da are invalid")
    if psi1_flavor_class not in {"light", "heavy"} or psi2_flavor_class not in {"light", "heavy"}:
        raise ValueError("DA flavor classes must be light or heavy")
    models = list(tail["models"])
    orders = [str(value).upper() for value in scan["orders"]]
    prior_widths = [float(value) for value in scan["prior_widths"]]
    smoothing_method = str(tail["smoothing_method"])
    if not models or any(model not in {"gi_nla", "cg_nla"} for model in models):
        raise ValueError("tail scan needs at least one supported model")
    if not orders or any(order not in {"LA", "NLA"} for order in orders):
        raise ValueError("orders must be a nonempty LA/NLA list")
    if not prior_widths or any(not math.isfinite(value) or value <= 0 for value in prior_widths):
        raise ValueError("prior_widths must be finite and positive")
    if smoothing_method not in {"linear", "none"}:
        raise ValueError("smoothing_method must be linear or none")
    component = str(scan["component"])
    if component not in {"re", "im", "both"}:
        raise ValueError("component must be re, im, or both")
    lambda0_gev = float(scan["lambda0_gev"])
    output_scale = float(scan["output_scale"])
    q_min = float(scan["q_min"])
    max_schemes = int(scan["max_schemes"])
    if (
        not math.isfinite(lambda0_gev)
        or lambda0_gev < 0
        or not math.isfinite(output_scale)
        or output_scale <= 0
        or not 0 <= q_min <= 1
        or max_schemes < 1
    ):
        raise ValueError("scan scales, quality threshold, and maximum scheme count are invalid")
    momentum = data.attrs.get("momentum_gev")
    if (
        not isinstance(momentum, (int, float))
        or isinstance(momentum, bool)
        or not math.isfinite(float(momentum))
        or float(momentum) <= 0
    ):
        raise ValueError("Fourier scan requires finite positive momentum_gev")
    hadron = str(data.attrs.get("hadron", ""))
    if observable == "DA" and phase_transfer_da:
        z = np.asarray(data.coords["z"], dtype=float)
        phase = np.exp(0.5j * z * float(momentum) / HBAR_C_GEV_FM)[None, :]
        projected_values = np.real(np.asarray(data.values) * phase) * np.conjugate(phase)
        attrs = data.attrs
        attrs["phase_transfer_da"] = "true"
        data = EnsembleData(
            data.ensemble, data.resample, list(projected_values), data.dims, data.coords, attrs=attrs, name=data.name
        )
    ranges = [
        (model, float(z_min), float(z_max))
        for model in models
        for z_min in tail["z_min_fm"]
        for z_max in tail["z_max_fm"]
        if float(z_min) < float(z_max)
    ][:max_schemes]
    if not ranges:
        raise ValueError("Fourier scan contains no ordered tail range")
    range_order = orders[0]
    range_prior_width = prior_widths[0]
    range_records = []
    for model_id, z_min_fm, z_max_fm in ranges:
        names = _tail_parameter_names(
            model_id,
            range_order,
            observable,
            psi1_flavor_class,
            psi2_flavor_class,
            scan["sector"],
            hadron,
        )
        z = np.asarray(data.coords["z"], dtype=float)
        mask = (z >= z_min_fm - 1e-12) & (z <= z_max_fm + 1e-12) & (z > 0)
        channel_count = 2 if component == "both" else 1
        required_points = max(int(math.ceil(len(names) / channel_count)), 2)
        if int(np.count_nonzero(mask)) < required_points:
            continue
        means, widths = _scan_tail_priors(
            model_id=model_id,
            order=range_order,
            lambda0_gev=lambda0_gev,
            observable=observable,
            psi1_flavor_class=psi1_flavor_class,
            psi2_flavor_class=psi2_flavor_class,
            sector=str(scan["sector"]),
            hadron=hadron,
        )
        record = {
            "model_id": model_id,
            "z_min_fm": z_min_fm,
            "z_max_fm": z_max_fm,
            "order": range_order,
            "prior_width": range_prior_width,
        }
        try:
            range_parameters, diagnostics = fit_tail_parameters(
                data,
                model_id=model_id,
                z_min_fm=z_min_fm,
                z_max_fm=z_max_fm,
                prior_means=means,
                prior_widths=widths,
                order=range_order,
                component=component,
                lambda0_gev=lambda0_gev,
                observable=observable,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                sector=str(scan["sector"]),
                hadron=hadron,
                workers=workers,
                mode="center",
                posterior_prior_scale=range_prior_width,
            )
            record.update({"fit_success": True, "fit_parameters": range_parameters[0], **diagnostics})
        except FitNumericalError as exc:
            record.update({"fit_success": False, "error": str(exc)})
        range_records.append(record)
    selected_range = _select_fourier_range(range_records, q_min=q_min)
    selected_model_id = str(selected_range["model_id"])
    selected_z_min = float(selected_range["z_min_fm"])
    selected_z_max = float(selected_range["z_max_fm"])

    selected_z = np.asarray(data.coords["z"], dtype=float)
    selected_mask = (selected_z >= selected_z_min - 1e-12) & (selected_z <= selected_z_max + 1e-12) & (selected_z > 0)
    fit_model_specs = []
    for order in orders:
        names = _tail_parameter_names(
            selected_model_id,
            order,
            observable,
            psi1_flavor_class,
            psi2_flavor_class,
            scan["sector"],
            hadron,
        )
        channel_count = 2 if component == "both" else 1
        required_points = max(int(math.ceil(len(names) / channel_count)), 2)
        if int(np.count_nonzero(selected_mask)) >= required_points:
            fit_model_specs.extend((order, prior_width) for prior_width in prior_widths)
    if not fit_model_specs:
        fit_model_specs = [(range_order, range_prior_width)]

    parallel = _parallel or _ParallelPool(min(workers, data.n_sample))
    try:
        candidates = []
        for order, prior_width in fit_model_specs:
            means, widths = _scan_tail_priors(
                model_id=selected_model_id,
                order=order,
                lambda0_gev=lambda0_gev,
                observable=observable,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                sector=str(scan["sector"]),
                hadron=hadron,
            )
            parameters, diagnostics = fit_tail_parameters(
                data,
                model_id=selected_model_id,
                z_min_fm=selected_z_min,
                z_max_fm=selected_z_max,
                prior_means=means,
                prior_widths=widths,
                order=order,
                component=component,
                lambda0_gev=lambda0_gev,
                observable=observable,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                sector=str(scan["sector"]),
                hadron=hadron,
                workers=workers,
                mode="resamples",
                posterior_prior_scale=prior_width,
                _parallel=parallel,
            )
            smoothing_width = selected_z_max - selected_z_min
            extended = extend_tail(
                data,
                z_max_fm=float(tail["extent_fm"]),
                z_min_fm=selected_z_min,
                smoothing_method=smoothing_method,
                smoothing_width_fm=smoothing_width,
                model_id=selected_model_id,
                tail_parameters=parameters,
                order=order,
                observable=observable,
                psi1_flavor_class=psi1_flavor_class,
                psi2_flavor_class=psi2_flavor_class,
                sector=str(scan["sector"]),
                hadron=hadron,
            )
            projected_values = (
                np.real(extended.values)
                if component == "re"
                else 1j * np.imag(extended.values)
                if component == "im"
                else np.asarray(extended.values)
            )
            projected = EnsembleData(
                extended.ensemble,
                extended.resample,
                list(projected_values),
                extended.dims,
                extended.coords,
                attrs=extended.attrs,
                name=extended.name,
            )
            transformed = fourier_transform(
                projected,
                x_grid,
                momentum_gev=float(momentum),
                phase_sign=int(transform["phase_sign"]),
                x_shift=float(transform["x_shift"]),
                prefactor=str(transform["prefactor"]),
                workers=workers,
                _parallel=parallel,
            )
            label = (
                f"{selected_model_id}_{selected_z_min:g}_{selected_z_max:g}_{order}_w{prior_width:g}_{smoothing_method}"
            )
            parameter_names = list(parameters[0])
            parameter_values = {
                name: np.asarray([sample[name] for sample in parameters], dtype=float) for name in parameter_names
            }
            candidates.append(
                {
                    "label": label,
                    "model_id": selected_model_id,
                    "z_min_fm": selected_z_min,
                    "z_max_fm": selected_z_max,
                    "order": order,
                    "prior_width": prior_width,
                    "smoothing_method": smoothing_method,
                    "parameter_mean": {name: float(np.mean(values)) for name, values in parameter_values.items()},
                    "parameter_sdev": {
                        name: float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                        for name, values in parameter_values.items()
                    },
                    "extended": extended,
                    "data": transformed,
                    **diagnostics,
                }
            )
    finally:
        if _parallel is None:
            parallel.close()
    if not candidates:
        raise ValueError("the selected Fourier range produces no model candidates")
    center_passing = [
        candidate
        for candidate in candidates
        if float(candidate["Q"]) >= q_min and math.isfinite(float(candidate["logGBF"]))
    ]
    if center_passing:
        best = max(center_passing, key=lambda candidate: float(candidate["logGBF"]))
    else:
        best = max(candidates, key=lambda candidate: float(candidate["Q"]))
    sample_weights = _sample_model_weights(
        candidates, n_sample=data.n_sample, q_min=q_min, model_average=bool(scan["model_average"])
    )
    transformed_values = np.asarray([candidate["data"].values for candidate in candidates])
    values = np.sum(sample_weights[:, :, None] * transformed_values, axis=0) * output_scale
    mean_weights = np.mean(sample_weights, axis=1)
    selected = [candidate for candidate, weight in zip(candidates, mean_weights) if weight > 0.0]
    selected_range = {"model_id": selected_model_id, "z_min_fm": selected_z_min, "z_max_fm": selected_z_max}
    attrs = dict(best["data"].attrs)
    attrs.update(
        {
            "sector": str(scan["sector"]),
            "component": component,
            "output_scale": output_scale,
            "model_average": str(bool(scan["model_average"])).lower(),
            "selected_range": json.dumps([selected_range["z_min_fm"], selected_range["z_max_fm"]]),
            "selected_models": json.dumps([candidate["label"] for candidate in selected]),
            "model_weights": json.dumps(mean_weights.tolist()),
            "phase_transfer_da": str(observable == "DA" and phase_transfer_da).lower(),
            "psi1_flavor_class": psi1_flavor_class,
            "psi2_flavor_class": psi2_flavor_class,
        }
    )
    output = EnsembleData(
        data.ensemble, data.resample, list(values), ["x"], best["data"].coords, attrs=attrs, name="quasi_distribution"
    )
    return {
        "data": output,
        "selected_candidate": best,
        "selected_range": selected_range,
        "range_candidates": range_records,
        "model_candidates": candidates,
        "selected_labels": [candidate["label"] for candidate in selected],
        "weights": mean_weights.tolist(),
        "sample_model_weights": sample_weights.tolist(),
        "workers": workers,
    }
