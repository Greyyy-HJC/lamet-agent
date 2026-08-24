"""Additive extrapolation basis and sample-level linear fits."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM


def load_data(value: Any) -> EnsembleData:
    """Load one explicitly listed matched-distribution artifact."""
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, Path):
        if value.suffix.lower() != ".nc":
            raise ValueError(f"extrapolation input must be a .nc artifact: {value}")
        return EnsembleData.from_netcdf(value)
    raise TypeError("extrapolation input is neither EnsembleData nor a NetCDF Path")


def basis_terms(attrs: dict[str, object], terms: list[str], physical_mass: float) -> list[float]:
    """Evaluate the authored dimensionless correction basis at one input."""
    a = float(attrs["lattice_spacing_fm"])
    length = float(attrs["L_s"]) * a
    pion_mass = float(attrs["m_pi_gev"])
    momentum = float(attrs["momentum_gev"])
    if a <= 0 or length <= 0 or pion_mass <= 0 or momentum <= 0:
        raise ValueError("extrapolation kinematics must be finite and positive")
    r = pion_mass * pion_mass
    physical_r = physical_mass * physical_mass
    values = []
    for term in terms:
        if term == "a":
            values.append(a)
        elif term == "a2":
            values.append(a**2)
        elif term == "a4":
            values.append(a**4)
        elif term == "ap2":
            values.append((a * momentum) ** 2)
        elif term == "exp_mpi_L":
            values.append(math.exp(-pion_mass * length / HBAR_C_GEV_FM))
        elif term == "exp_sqrt2_mpi_L":
            values.append(math.exp(-math.sqrt(2.0) * pion_mass * length / HBAR_C_GEV_FM))
        elif term == "mpi2":
            values.append(r - physical_r)
        elif term == "mpi4_log_mpi2":
            values.append(r * r * math.log(r) - physical_r * physical_r * math.log(physical_r))
        elif term == "inv_p2":
            values.append(1.0 / momentum**2)
        elif term == "inv_p4":
            values.append(1.0 / momentum**4)
        else:
            raise ValueError(f"unsupported extrapolation term '{term}'")
    return values


def extrapolation_fcn(x: dict[str, object], parameters: dict[str, object]) -> np.ndarray:
    """Evaluate real and imaginary ensemble differences jointly."""
    design = np.asarray(x["design"], dtype=float)
    terms = list(x["terms"])
    x_dependence = dict(x["x_dependence"])
    output_size = int(x["output_size"])
    predicted_re = np.zeros((design.shape[0], output_size), dtype=object)
    predicted_im = np.zeros((design.shape[0], output_size), dtype=object)
    for term_index, term in enumerate(terms):
        basis = design[:, term_index, None]
        coefficient_re = np.asarray(parameters[f"{term}_re"])
        coefficient_im = np.asarray(parameters[f"{term}_im"])
        if not x_dependence[term]:
            coefficient_re = np.full(output_size, coefficient_re.item(), dtype=object)
            coefficient_im = np.full(output_size, coefficient_im.item(), dtype=object)
        predicted_re += basis * coefficient_re[None, :]
        predicted_im += basis * coefficient_im[None, :]
    return np.concatenate([predicted_re.T, predicted_im.T], axis=1).reshape(-1)


def fit_candidate(data: list[EnsembleData], terms: list[str], physical_mass: float, priors: dict[str, dict[str, float]], *, x_range: tuple[float, float], x_dependence: dict[str, bool] | None = None) -> tuple[EnsembleData, dict[str, float]]:
    """Fit the complex physical distribution in one joint lsqfit fit."""
    if not data or any(item.dims != ["x"] for item in data):
        raise ValueError("extrapolation requires one-dimensional x-space inputs")
    x = data[0].coords["x"]
    if any(not np.allclose(item.coords["x"], x, rtol=0.0, atol=1e-12) for item in data[1:]):
        raise ValueError("all extrapolation grids must match by coordinate value")
    if set(priors) != set(terms):
        raise ValueError("candidate priors must contain exactly the selected correction terms")
    x_dependence = {term: True for term in terms} if x_dependence is None else dict(x_dependence)
    if set(x_dependence) != set(terms) or any(not isinstance(value, bool) for value in x_dependence.values()):
        raise ValueError("x_dependence must contain exactly one boolean for every selected term")
    design = np.asarray([[1.0, *basis_terms(item.attrs, terms, physical_mass)] for item in data], dtype=float)
    x_mask = np.asarray([(float(x_range[0]) - 1e-12) <= float(value) <= (float(x_range[1]) + 1e-12) for value in x])
    if not np.any(x_mask):
        raise ValueError("fit_ranges.x does not select any authored x coordinate")
    output_x = [value for value, keep in zip(x, x_mask) if keep]
    if any(item.resample not in {"jackknife", "bootstrap", "raw"} for item in data):
        raise ValueError("covariance-block extrapolation requires raw, jackknife, or bootstrap samples")
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(data):
        resample_id = item.attrs.get("resample_id")
        if not isinstance(resample_id, str) or not resample_id:
            raise ValueError("every extrapolation input requires a nonempty resample_id")
        groups.setdefault(resample_id, []).append(index)
    n_input = len(data)
    for indices in groups.values():
        first = data[indices[0]]
        if any(data[index].resample != first.resample or data[index].n_sample != first.n_sample for index in indices[1:]):
            raise ValueError("inputs in one resample_id block must share mode and sample count")
    sample_error_modes = {str(item.attrs.get("sample_error_mode", "covariance")) for item in data}
    if len(sample_error_modes) != 1:
        raise ValueError("all extrapolation inputs must share one sample_error_mode")
    sample_error_mode = sample_error_modes.pop()
    import gvar as gv
    import lsqfit

    prior_mean = np.asarray([float(priors[term]["mean"]) for term in terms], dtype=float)
    prior_sdev = np.asarray([float(priors[term]["sdev"]) for term in terms], dtype=float)
    if np.any(~np.isfinite(prior_mean)) or np.any(~np.isfinite(prior_sdev)) or np.any(prior_sdev <= 0):
        raise ValueError("selected extrapolation priors are invalid")
    parameter_count = len(terms) + 1
    if n_input <= parameter_count:
        raise ValueError("candidate needs more inputs than fitted real/imaginary coefficients")
    fit_observations = []
    reference_re = []
    reference_im = []
    x_indices = np.flatnonzero(x_mask)
    for x_index in x_indices:
        covariance = np.zeros((2 * n_input, 2 * n_input), dtype=float)
        center = np.zeros(n_input, dtype=complex)
        for indices in groups.values():
            first = data[indices[0]]
            block_samples = np.stack([np.asarray(data[index].values)[:, x_index] for index in indices], axis=1)
            joint_samples = np.concatenate([np.real(block_samples), np.imag(block_samples)], axis=1)
            joint_data = EnsembleData(
                None,
                first.resample,
                list(joint_samples),
                ["observation"],
                {"observation": np.arange(joint_samples.shape[1])},
            )
            average = joint_data.average(sample_error_mode)
            average_center = np.asarray(gv.mean(average), dtype=float)
            group_size = len(indices)
            center[indices] = average_center[:group_size] + 1j * average_center[group_size:]
            block = np.asarray(gv.evalcov(average), dtype=float)
            positions = [*indices, *[n_input + index for index in indices]]
            covariance[np.ix_(positions, positions)] = np.atleast_2d(block)
        np.linalg.cholesky(covariance)
        observed = np.concatenate([np.real(center), np.imag(center)])
        correlated = gv.gvar(observed, covariance)
        fit_observations.extend(
            np.concatenate(
                [correlated[1:n_input] - correlated[0], correlated[n_input + 1 :] - correlated[n_input]]
            )
        )
        reference_re.append(correlated[0])
        reference_im.append(correlated[n_input])
    prior = gv.BufferDict()
    output_size = len(output_x)
    for term, center, width in zip(terms, prior_mean, prior_sdev):
        prior[f"{term}_re"] = gv.gvar(np.full(output_size, center), np.full(output_size, width)) if x_dependence[term] else gv.gvar(center, width)
        prior[f"{term}_im"] = gv.gvar(np.full(output_size, center), np.full(output_size, width)) if x_dependence[term] else gv.gvar(center, width)
    difference_design = design[1:, 1:] - design[0, 1:]
    fit = lsqfit.nonlinear_fit(
        data=({"design": difference_design, "terms": terms, "output_size": output_size, "x_dependence": x_dependence}, np.asarray(fit_observations, dtype=object)),
        fcn=extrapolation_fcn,
        prior=prior,
        maxit=2000,
    )
    physical_re = np.asarray(reference_re, dtype=object)
    physical_im = np.asarray(reference_im, dtype=object)
    for term_index, term in enumerate(terms):
        physical_re -= design[0, term_index + 1] * np.asarray(fit.p[f"{term}_re"])
        physical_im -= design[0, term_index + 1] * np.asarray(fit.p[f"{term}_im"])
    fit_values = np.stack([physical_re, physical_im])
    attrs = dict(data[0].attrs)
    attrs.update({"ensemble": None, "extrapolation_terms": ",".join(terms), "x_dependence": json.dumps(x_dependence, sort_keys=True), "physical_pion_mass_gev": float(physical_mass), "physical_point": "continuum,infinite_volume,infinite_momentum", "units": '{"values":"dimensionless","x":"dimensionless"}'})
    result = EnsembleData(None, "gvar", fit_values, ["component", "x"], {"component": ["real", "imag"], "x": output_x}, attrs=attrs, name="physical_distribution")
    return result, {"chi2": float(fit.chi2), "dof": float(fit.dof), "chi2_dof": float(fit.chi2 / fit.dof), "Q": float(fit.Q), "aic": float(fit.chi2 + 4.0 * parameter_count * output_size)}
