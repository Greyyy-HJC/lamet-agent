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
from lamet_agent.ui import track


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


def _zero_odd_origin_imag(values: np.ndarray, z: np.ndarray, symmetry: Mapping[str, str]) -> np.ndarray:
    """Force Im(z=0)=0 so an odd imaginary part remains odd at the origin."""
    if symmetry.get("imag") != "odd":
        return values
    origin = np.isclose(np.asarray(z, dtype=float), 0.0)
    if not np.any(origin):
        return values
    output = np.array(values, copy=True)
    output[origin] = np.real(output[origin]) + 0.0j
    return output


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
    return _zero_odd_origin_imag(
        np.concatenate([real + 1j * imag, positive_values[~negative_mask]]),
        extended_z,
        symmetry,
    )


_NUCLEON_HADRONS = {"nucleon", "proton"}


def _tail_family(
    observable: str,
    hadron: str,
    sector: str,
    psi1_flavor_class: str,
    psi2_flavor_class: str,
) -> str:
    """Select the paper's large-distance family from physical provenance."""
    observable = observable.upper()
    hadron = hadron.lower()
    sector = sector.lower()
    if observable == "PDF":
        if sector not in {"valence", "singlet", "full"}:
            raise ValueError("PDF tail sector must be valence, singlet, or full")
        if hadron == "pion":
            return "pion_pdf_valence" if sector == "valence" else "pion_pdf"
        if hadron in _NUCLEON_HADRONS:
            return "nucleon_pdf"
        raise ValueError(f"PDF tails are not implemented for hadron '{hadron or '<missing>'}'")
    if observable == "DA":
        if sector != "full":
            raise ValueError("DA tail sector must be full")
        if hadron == "pion" and psi1_flavor_class == psi2_flavor_class == "light":
            return "pion_da"
        return "meson_da"
    if observable == "GPD":
        if sector not in {"sea", "valence", "singlet", "full"}:
            raise ValueError("GPD tail sector must be sea, valence, singlet, or full")
        if hadron == "pion":
            return "pion_gpd_sea" if sector == "sea" else "pion_gpd"
        if hadron in _NUCLEON_HADRONS:
            return "nucleon_gpd"
        raise ValueError(f"GPD tails are not implemented for hadron '{hadron or '<missing>'}'")
    raise ValueError(f"unsupported tail observable '{observable}'")


def _tail_parameter_names(
    model_id: str,
    order: str,
    observable: str,
    psi1_flavor_class: str,
    psi2_flavor_class: str,
    sector: str = "full",
    hadron: str = "",
) -> list[str]:
    """Return the independent parameters for the selected paper family."""
    if model_id not in {"gi_nla", "cg_nla"}:
        raise ValueError(f"unsupported tail model '{model_id}'")
    order = order.upper()
    if order not in {"LA", "NLA"}:
        raise ValueError("tail order must be LA or NLA")
    family = _tail_family(observable, hadron, sector, psi1_flavor_class, psi2_flavor_class)
    if family == "pion_pdf_valence":
        names = ["A2", "A1", "phi1"]
    elif family == "pion_pdf":
        names = ["A2", "phi2", "A1", "phi1", "A3", "phi3"]
    elif family == "nucleon_pdf":
        names = ["A2", "phi2"]
    elif family == "pion_da":
        names = ["A1", "phi1"]
    elif family == "meson_da":
        names = []
        if not (psi1_flavor_class == "light" and psi2_flavor_class == "heavy"):
            names.extend(["A1", "phi1"])
        if not (psi1_flavor_class == "heavy" and psi2_flavor_class == "light"):
            names.extend(["A2", "phi2"])
    elif family in {"pion_gpd_sea", "nucleon_gpd"}:
        names = ["A2", "phi2", "At2", "phit2"]
    else:
        names = ["A1", "phi1", "A3", "phi3", "A2", "phi2", "At2", "phit2"]
    if order == "NLA":
        names.extend([name + "p" for name in names])
    names.append("Lambda")
    if model_id == "cg_nla":
        names.append("n")
    return names


def _tail_fit_fcn_base(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one channel-specific GI/CG tail for the fitter."""
    z = np.asarray(x["z"], dtype=float)
    absolute = np.abs(z)
    sign = np.sign(z)
    z_gev_inv = z / HBAR_C_GEV_FM
    decay = gv.exp(-(parameters["Lambda"] + float(x["lambda0_gev"])) * absolute / HBAR_C_GEV_FM)
    family = _tail_family(
        str(x["observable"]),
        str(x.get("hadron", "")),
        str(x.get("sector", "full")),
        str(x["psi1_flavor_class"]),
        str(x["psi2_flavor_class"]),
    )
    if family == "pion_pdf_valence":
        phase = parameters["phi1"] - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
        real = parameters["A2"] + 2.0 * parameters["A1"] * gv.cos(phase)
        imag = np.zeros_like(absolute)
        if x["order"] == "NLA":
            phase_prime = parameters["phi1p"] - float(x["momentum_gev"]) * absolute / HBAR_C_GEV_FM
            real = real + (parameters["A2p"] + 2.0 * parameters["A1p"] * gv.cos(phase_prime)) / absolute
    elif family in {"pion_pdf", "nucleon_pdf"}:
        terms = (
            (("2", 0.0),)
            if family == "nucleon_pdf"
            else (
                ("2", 0.0),
                ("1", -float(x["momentum_gev"])),
                ("3", float(x["momentum_gev"])),
            )
        )
        real = np.zeros_like(absolute, dtype=object)
        imag = np.zeros_like(absolute, dtype=object)
        for term, frequency in terms:
            phase = sign * parameters[f"phi{term}"] + frequency * z_gev_inv
            real = real + parameters[f"A{term}"] * gv.cos(phase)
            imag = imag + parameters[f"A{term}"] * gv.sin(phase)
        if x["order"] == "NLA":
            for term, frequency in terms:
                phase = sign * parameters[f"phi{term}p"] + frequency * z_gev_inv
                real = real + parameters[f"A{term}p"] * gv.cos(phase) / absolute
                imag = imag + parameters[f"A{term}p"] * gv.sin(phase) / absolute
    elif family in {"pion_da", "meson_da"}:
        terms = []
        if "A1" in parameters:
            terms.append((parameters["A1"], sign * parameters["phi1"] - float(x["momentum_gev"]) * z_gev_inv))
        if family == "pion_da":
            terms.append((parameters["A1"], -sign * parameters["phi1"]))
        elif "A2" in parameters:
            terms.append((parameters["A2"], sign * parameters["phi2"]))
        real = sum(amplitude * gv.cos(phase) for amplitude, phase in terms)
        imag = sum(amplitude * gv.sin(phase) for amplitude, phase in terms)
        if x["order"] == "NLA":
            prime_terms = []
            if "A1p" in parameters:
                prime_terms.append(
                    (parameters["A1p"], sign * parameters["phi1p"] - float(x["momentum_gev"]) * z_gev_inv)
                )
            if family == "pion_da":
                prime_terms.append((parameters["A1p"], -sign * parameters["phi1p"]))
            elif "A2p" in parameters:
                prime_terms.append((parameters["A2p"], sign * parameters["phi2p"]))
            real = real + sum(amplitude * gv.cos(phase) for amplitude, phase in prime_terms) / absolute
            imag = imag + sum(amplitude * gv.sin(phase) for amplitude, phase in prime_terms) / absolute
    else:
        initial = float(x["initial_momentum_gev"])
        final = float(x["final_momentum_gev"])
        delta = float(x["delta_momentum_gev"])
        phase_transfer = str(x["phase_transfer_gpd"])
        shift = {"barpsi_at_0": 0.0, "mid_at_0": 0.5 * delta, "psi_at_0": delta}[phase_transfer]
        base_terms = {
            "1": -final,
            "3": initial,
            "2": 0.0,
            "t2": -delta,
        }
        terms = ("2", "t2") if family in {"pion_gpd_sea", "nucleon_gpd"} else ("1", "3", "2", "t2")
        real = np.zeros_like(absolute, dtype=object)
        imag = np.zeros_like(absolute, dtype=object)
        for term in terms:
            phase = sign * parameters[f"phi{term}"] + (base_terms[term] + shift) * z_gev_inv
            real = real + parameters[f"A{term}"] * gv.cos(phase)
            imag = imag + parameters[f"A{term}"] * gv.sin(phase)
        if x["order"] == "NLA":
            for term in terms:
                phase = sign * parameters[f"phi{term}p"] + (base_terms[term] + shift) * z_gev_inv
                real = real + parameters[f"A{term}p"] * gv.cos(phase) / absolute
                imag = imag + parameters[f"A{term}p"] * gv.sin(phase) / absolute
    real = real * decay
    imag = imag * decay
    if x["model_id"] == "cg_nla":
        real = real / absolute ** parameters["n"]
        imag = imag / absolute ** parameters["n"]
    if x["source_component"] == "re":
        return real
    if x["source_component"] == "im":
        return imag
    return np.concatenate([real, imag])


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
    initial_momentum_gev: float | None = None,
    final_momentum_gev: float | None = None,
    delta_momentum_gev: float | None = None,
    phase_transfer_gpd: str = "barpsi_at_0",
) -> np.ndarray:
    """Evaluate the same channel-specific tail used by the fitter."""
    if model_id not in {"gi_nla", "cg_nla"}:
        raise ValueError(f"unsupported tail model '{model_id}'")
    order = order.upper()
    if order not in {"LA", "NLA"}:
        raise ValueError("tail order must be LA or NLA")
    if (
        observable not in {"PDF", "DA", "GPD"}
        or psi1_flavor_class not in {"light", "heavy"}
        or psi2_flavor_class not in {"light", "heavy"}
    ):
        raise ValueError("tail observable and DA flavor classes are invalid")
    expected = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class, sector, hadron)
    if set(parameters) != set(expected):
        raise ValueError(f"tail parameters must contain exactly {expected}")
    z = np.asarray(z_fm, dtype=float)
    absolute = np.abs(z)
    if np.any(absolute <= 0):
        raise ValueError("tail model is undefined at z=0")
    family = _tail_family(observable, hadron, sector, psi1_flavor_class, psi2_flavor_class)
    if family in {"pion_pdf_valence", "pion_pdf", "pion_da", "meson_da"} and (
        not isinstance(momentum_gev, (int, float))
        or isinstance(momentum_gev, bool)
        or not math.isfinite(float(momentum_gev))
        or float(momentum_gev) <= 0
    ):
        raise ValueError(f"{family} tails require finite positive momentum_gev")
    lambda_value = float(parameters["Lambda"])
    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("tail Lambda must be finite and positive")
    if model_id == "cg_nla":
        exponent = float(parameters["n"])
        if not math.isfinite(exponent) or exponent <= 0:
            raise ValueError("CG tail power n must be finite and positive")
    if observable == "GPD":
        gpd_momenta = (initial_momentum_gev, final_momentum_gev, delta_momentum_gev)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
            for value in gpd_momenta
        ):
            raise ValueError("GPD tails require finite initial, final, and delta momenta")
        if not np.isclose(float(final_momentum_gev) - float(initial_momentum_gev), float(delta_momentum_gev)):
            raise ValueError("GPD delta momentum must equal final minus initial momentum")
        if phase_transfer_gpd not in {"mid_at_0", "barpsi_at_0", "psi_at_0"}:
            raise ValueError("phase_transfer_gpd must be mid_at_0, barpsi_at_0, or psi_at_0")
    values = tail_fit_fcn(
        {
            "z": z,
            "model_id": model_id,
            "order": order,
            "source_component": "both",
            "lambda0_gev": 0.0,
            "observable": observable,
            "momentum_gev": 0.0 if momentum_gev is None else momentum_gev,
            "psi1_flavor_class": psi1_flavor_class,
            "psi2_flavor_class": psi2_flavor_class,
            "sector": sector,
            "hadron": hadron,
            "initial_momentum_gev": initial_momentum_gev,
            "final_momentum_gev": final_momentum_gev,
            "delta_momentum_gev": delta_momentum_gev,
            "phase_transfer_gpd": phase_transfer_gpd,
        },
        parameters,
    )
    return np.asarray(values[: z.size], dtype=complex) + 1j * np.asarray(values[z.size :], dtype=complex)


def tail_fit_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the channel-specific tail for fitting or extension."""
    return _tail_fit_fcn_base(x, parameters)


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
            lower.append(0.0)
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
    source_component: str = "both",
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
    if order not in {"LA", "NLA"} or source_component not in {"re", "im", "both"}:
        raise ValueError("tail order and source_component must be LA/NLA and re/im/both")
    if not math.isfinite(lambda0_gev) or lambda0_gev < 0:
        raise ValueError("lambda0_gev must be finite and nonnegative")
    if (
        observable not in {"PDF", "DA", "GPD"}
        or psi1_flavor_class not in {"light", "heavy"}
        or psi2_flavor_class not in {"light", "heavy"}
    ):
        raise ValueError("tail observable and DA flavor classes are invalid")
    family = _tail_family(observable, hadron, sector, psi1_flavor_class, psi2_flavor_class)
    momentum = data.attrs.get("momentum_gev")
    if family in {"pion_pdf_valence", "pion_pdf", "pion_da", "meson_da"} and (
        not isinstance(momentum, (int, float))
        or isinstance(momentum, bool)
        or not math.isfinite(float(momentum))
        or float(momentum) <= 0
    ):
        raise ValueError(f"{family} tail fitting requires finite positive momentum_gev")
    initial_momentum = data.attrs.get("initial_momentum_gev")
    final_momentum = data.attrs.get("final_momentum_gev")
    delta_momentum = data.attrs.get("delta_momentum_gev")
    phase_transfer_gpd = str(data.attrs.get("phase_transfer_gpd", "barpsi_at_0"))
    if observable == "GPD":
        gpd_momenta = (initial_momentum, final_momentum, delta_momentum)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
            for value in gpd_momenta
        ):
            raise ValueError("GPD tail fitting requires finite initial, final, and delta momenta")
        if not np.isclose(float(final_momentum) - float(initial_momentum), float(delta_momentum)):
            raise ValueError("GPD delta momentum must equal final minus initial momentum")
        if phase_transfer_gpd not in {"mid_at_0", "barpsi_at_0", "psi_at_0"}:
            raise ValueError("phase_transfer_gpd must be mid_at_0, barpsi_at_0, or psi_at_0")
    names = _tail_parameter_names(model_id, order, observable, psi1_flavor_class, psi2_flavor_class, sector, hadron)
    channel_count = 2 if source_component == "both" else 1
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
        if source_component == "re"
        else imag_selected
        if source_component == "im"
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
            if source_component == "re"
            else imag_error
            if source_component == "im"
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
        "source_component": source_component,
        "lambda0_gev": float(lambda0_gev),
        "observable": observable,
        "momentum_gev": momentum,
        "psi1_flavor_class": psi1_flavor_class,
        "psi2_flavor_class": psi2_flavor_class,
        "sector": sector,
        "hadron": hadron,
        "initial_momentum_gev": initial_momentum,
        "final_momentum_gev": final_momentum,
        "delta_momentum_gev": delta_momentum,
        "phase_transfer_gpd": phase_transfer_gpd,
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
        values.append(_zero_odd_origin_imag(np.concatenate([negative, positive]), output_z, convention))
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
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("tail extension requires a positive lattice spacing")
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
            initial_momentum_gev=data.attrs.get("initial_momentum_gev"),
            final_momentum_gev=data.attrs.get("final_momentum_gev"),
            delta_momentum_gev=data.attrs.get("delta_momentum_gev"),
            phase_transfer_gpd=str(data.attrs.get("phase_transfer_gpd", "barpsi_at_0")),
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
            "tail_family": _tail_family(
                observable,
                hadron,
                sector,
                psi1_flavor_class,
                psi2_flavor_class,
            ),
            "power_coordinate_unit": "fm",
            "cg_power_applied": str(model_id == "cg_nla").lower(),
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


def _select_fourier_model(candidates: list[dict[str, Any]], *, q_min: float) -> dict[str, Any]:
    """Select a usable center model, preferring evidence above the Q threshold."""
    usable = [
        candidate
        for candidate in candidates
        if candidate.get("error") is None and candidate.get("Q") is not None and math.isfinite(float(candidate["Q"]))
    ]
    if not usable:
        raise FitNumericalError("no Fourier model candidate has a usable center fit")
    passing = [
        candidate
        for candidate in usable
        if float(candidate["Q"]) >= q_min and math.isfinite(float(candidate.get("logGBF", float("nan"))))
    ]
    if passing:
        return max(passing, key=lambda candidate: float(candidate["logGBF"]))
    return max(usable, key=lambda candidate: float(candidate["Q"]))


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


def _project_xspace_output(data: EnsembleData, *, source_component: str, output_component: str) -> EnsembleData:
    """Apply the declared x-space projection without losing source provenance."""
    if source_component not in {"re", "im", "both"}:
        raise ValueError("source_component must be re, im, or both")
    if output_component not in {"re", "both"}:
        raise ValueError("output_component must be re or both")
    values = np.real(data.values) if output_component == "re" else np.asarray(data.values)
    attrs = dict(data.attrs)
    for legacy_key in ("component", "matching_component", "resummation_part"):
        attrs.pop(legacy_key, None)
    attrs.update({"source_component": source_component, "output_component": output_component})
    return EnsembleData(
        data.ensemble,
        data.resample,
        list(values),
        data.dims,
        data.coords,
        attrs=attrs,
        name=data.name,
    )


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
    gpd_projection_grid: list[float] | None = None,
    gpd_polarization: str | None = None,
    workers: int = 1,
    show_progress: bool = False,
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
        "source_component",
        "output_scale",
        "q_min",
    }
    if set(transform) != transform_keys or set(tail) != tail_keys or set(scan) != scan_keys:
        raise ValueError("Fourier transform, tail, and scan mappings do not match the native interface")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if observable not in {"PDF", "DA", "GPD"} or not isinstance(phase_transfer_da, bool):
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
    source_component = str(scan["source_component"])
    if source_component not in {"re", "im", "both"}:
        raise ValueError("source_component must be re, im, or both")
    output_component = "both" if data.attrs.get("gpd_completion_mode") == "paired_flow" else "re"
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
    tail_family = _tail_family(observable, hadron, str(scan["sector"]), psi1_flavor_class, psi2_flavor_class)
    requested_gpd_grid = None if gpd_projection_grid is None else np.asarray(gpd_projection_grid, dtype=float)
    if observable == "GPD" and requested_gpd_grid is not None:
        if requested_gpd_grid.ndim != 1 or requested_gpd_grid.size == 0 or np.any(~np.isfinite(requested_gpd_grid)):
            raise ValueError("gpd_projection_grid must be a nonempty finite one-dimensional grid")
        if str(scan["sector"]).lower() != "full":
            x_grid = np.sort(np.unique(np.concatenate([requested_gpd_grid, -requested_gpd_grid]))).tolist()
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
        channel_count = 2 if source_component == "both" else 1
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
            "tail_family": tail_family,
            "power_coordinate_unit": "fm",
            "cg_power_applied": model_id == "cg_nla",
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
                source_component=source_component,
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
        channel_count = 2 if source_component == "both" else 1
        required_points = max(int(math.ceil(len(names) / channel_count)), 2)
        if int(np.count_nonzero(selected_mask)) >= required_points:
            fit_model_specs.extend((order, prior_width) for prior_width in prior_widths)
    if not fit_model_specs:
        fit_model_specs = [(range_order, range_prior_width)]

    parallel = _parallel or _ParallelPool(min(workers, data.n_sample))
    try:
        candidates = []
        model_specs = track(fit_model_specs, label="Fourier models", unit="model", enabled=show_progress)
        for order, prior_width in model_specs:
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
                source_component=source_component,
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
                if source_component == "re"
                else 1j * np.imag(extended.values)
                if source_component == "im"
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
            transformed = _project_xspace_output(
                transformed,
                source_component=source_component,
                output_component=output_component,
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
                    "tail_family": tail_family,
                    "power_coordinate_unit": "fm",
                    "cg_power_applied": selected_model_id == "cg_nla",
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
    if observable == "GPD" and requested_gpd_grid is not None and str(scan["sector"]).lower() != "full":
        evaluated_grid = np.asarray(x_grid, dtype=float)
        direct = np.asarray([int(np.argmin(np.abs(evaluated_grid - value))) for value in requested_gpd_grid])
        reflected = np.asarray([int(np.argmin(np.abs(evaluated_grid + value))) for value in requested_gpd_grid])
        polarization = str(gpd_polarization or data.attrs.get("polarization", "unpolarized")).lower()
        sea_sign = 1.0 if polarization == "helicity" else -1.0
        sector = str(scan["sector"]).lower()
        for candidate in candidates:
            full = np.asarray(candidate["data"].values)
            if sector == "sea":
                projected = sea_sign * full[:, reflected]
            elif sector == "valence":
                projected = full[:, direct] - sea_sign * full[:, reflected]
            elif sector == "singlet":
                projected = full[:, direct] + sea_sign * full[:, reflected]
            else:
                projected = full[:, direct]
            candidate["data"] = EnsembleData(
                candidate["data"].ensemble,
                candidate["data"].resample,
                list(projected),
                ["x"],
                {"x": requested_gpd_grid.tolist()},
                attrs=candidate["data"].attrs,
                name=candidate["data"].name,
            )
    best = _select_fourier_model(candidates, q_min=q_min)
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
            "source_component": source_component,
            "output_component": output_component,
            "output_scale": output_scale,
            "model_average": str(bool(scan["model_average"])).lower(),
            "selected_range": json.dumps([selected_range["z_min_fm"], selected_range["z_max_fm"]]),
            "selected_models": json.dumps([candidate["label"] for candidate in selected]),
            "model_weights": json.dumps(mean_weights.tolist()),
            "phase_transfer_da": str(observable == "DA" and phase_transfer_da).lower(),
            "psi1_flavor_class": psi1_flavor_class,
            "psi2_flavor_class": psi2_flavor_class,
            "tail_family": tail_family,
            "power_coordinate_unit": "fm",
            "cg_power_applied": str(selected_model_id == "cg_nla").lower(),
            "gpd_projection_mode": (
                "post_ft_signed_y"
                if observable == "GPD" and requested_gpd_grid is not None and str(scan["sector"]).lower() != "full"
                else "full_complex"
            ),
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
