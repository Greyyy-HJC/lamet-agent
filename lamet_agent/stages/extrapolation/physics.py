"""Additive extrapolation basis and sample-level linear fits."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import gvar as gv

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


def basis_terms(attrs: dict[str, object], terms: list[str], physical_mass: float | None) -> list[float]:
    """Evaluate the authored dimensionless correction basis at one input."""
    a = float(attrs["lattice_spacing_fm"])
    length = float(attrs["L_s"]) * a
    pion_mass = float(attrs["m_pi"])
    momentum = float(attrs["momentum_gev"])
    if a <= 0 or length <= 0 or pion_mass <= 0 or momentum <= 0:
        raise ValueError("extrapolation kinematics must be finite and positive")
    r = pion_mass * pion_mass
    physical_r = None if physical_mass is None else physical_mass * physical_mass
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
        elif term == "ap4":
            values.append((a * momentum) ** 4)
        elif term == "exp_mpi_L":
            values.append(math.exp(-pion_mass * length / HBAR_C_GEV_FM))
        elif term == "exp_sqrt2_mpi_L":
            values.append(math.exp(-math.sqrt(2.0) * pion_mass * length / HBAR_C_GEV_FM))
        elif term == "mpi2":
            if physical_r is None:
                raise ValueError("mpi2 requires physical_pion_mass_gev")
            values.append(r - physical_r)
        elif term == "mpi4_log_mpi2":
            if physical_r is None:
                raise ValueError("mpi4_log_mpi2 requires physical_pion_mass_gev")
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


def _ensemble_groups(data: list[EnsembleData]) -> list[list[int]]:
    """Group inputs that share one resampling source; absent provenance stays independent."""
    grouped: dict[tuple[str, object], list[int]] = {}
    for index, item in enumerate(data):
        resample_id = item.attrs.get("resample_id")
        ensemble_id = item.attrs.get("ensemble_id")
        key = (
            ("resample_id", resample_id)
            if resample_id not in {None, ""}
            else ("ensemble_id", ensemble_id)
            if ensemble_id not in {None, ""}
            else ("input", index)
        )
        grouped.setdefault(key, []).append(index)
    return list(grouped.values())


def _grouped_centers_and_covariances(
    values: np.ndarray,
    data: list[EnsembleData],
    sample_error_mode: str,
    *,
    x_covariance: bool,
) -> tuple[np.ndarray, list[np.ndarray] | np.ndarray]:
    """Build centers and covariance with zero correlation between ensemble sources."""
    n_input, _n_sample, n_x = values.shape
    centers = np.empty((n_input, n_x), dtype=float)
    groups = _ensemble_groups(data)
    if not x_covariance:
        covariances = [np.zeros((n_input, n_input), dtype=float) for _ in range(n_x)]
        for group in groups:
            samples = np.moveaxis(values[group], 1, 0)
            for x_index in range(n_x):
                grouped_data = EnsembleData(
                    None,
                    data[0].resample,
                    list(samples[:, :, x_index]),
                    ["input"],
                    {"input": list(range(len(group)))},
                )
                average = grouped_data.average(sample_error_mode)
                centers[group, x_index] = np.asarray(gv.mean(average), dtype=float)
                covariances[x_index][np.ix_(group, group)] = np.asarray(gv.evalcov(average), dtype=float)
        return centers, covariances

    covariance = np.zeros((n_input * n_x, n_input * n_x), dtype=float)
    for group in groups:
        samples = np.moveaxis(values[group], 1, 0).reshape(values.shape[1], -1)
        grouped_data = EnsembleData(
            None,
            data[0].resample,
            list(samples),
            ["observation"],
            {"observation": list(range(samples.shape[1]))},
        )
        average = grouped_data.average(sample_error_mode)
        centers[group] = np.asarray(gv.mean(average), dtype=float).reshape(len(group), n_x)
        indices = np.concatenate([input_index * n_x + np.arange(n_x) for input_index in group])
        covariance[np.ix_(indices, indices)] = np.asarray(gv.evalcov(average), dtype=float)
    return centers, covariance


def _regulated_inverse(matrix: np.ndarray, *, cutoff: float = 1e-12) -> tuple[np.ndarray, float]:
    """Return a symmetric regulated inverse and log determinant."""
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    maximum = float(np.max(eigenvalues))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("fit covariance/information matrix has no positive scale")
    regulated = np.maximum(eigenvalues, maximum * cutoff)
    inverse = (eigenvectors / regulated[None, :]) @ eigenvectors.T
    return inverse, float(np.sum(np.log(regulated)))


def _prepare_block_system(
    design: np.ndarray,
    covariances: list[np.ndarray],
    terms: list[str],
    x_dependence: dict[str, bool],
    local_means: np.ndarray,
    local_sdevs: np.ndarray,
    global_means: np.ndarray,
    global_sdevs: np.ndarray,
) -> dict[str, object]:
    """Factor the block-arrow posterior information matrix."""
    local_terms = [term for term in terms if x_dependence[term]]
    global_terms = [term for term in terms if not x_dependence[term]]
    local_design = np.column_stack(
        [np.ones(design.shape[0]), *[design[:, terms.index(term)] for term in local_terms]]
    )
    global_design = (
        np.column_stack([design[:, terms.index(term)] for term in global_terms])
        if global_terms
        else np.empty((design.shape[0], 0), dtype=float)
    )
    global_precision = 1.0 / global_sdevs**2 if global_terms else np.empty(0, dtype=float)
    global_information = np.diag(global_precision)
    local_inverses = []
    local_couplings = []
    local_information_logdet = 0.0
    covariance_logdet = 0.0
    weights = []
    for x_index, covariance in enumerate(covariances):
        weight, covariance_logdet_x = _regulated_inverse(covariance)
        weights.append(weight)
        covariance_logdet += covariance_logdet_x
        local_precision = 1.0 / local_sdevs[x_index] ** 2
        information = local_design.T @ weight @ local_design + np.diag(local_precision)
        information_inverse, information_logdet = _regulated_inverse(information)
        coupling = local_design.T @ weight @ global_design
        local_inverses.append(information_inverse)
        local_couplings.append(coupling)
        local_information_logdet += information_logdet
        global_information += global_design.T @ weight @ global_design
    schur = global_information.copy()
    for inverse, coupling in zip(local_inverses, local_couplings):
        schur -= coupling.T @ inverse @ coupling
    if global_terms:
        global_inverse, schur_logdet = _regulated_inverse(schur)
    else:
        global_inverse = np.empty((0, 0), dtype=float)
        schur_logdet = 0.0
    return {
        "local_terms": local_terms,
        "global_terms": global_terms,
        "local_design": local_design,
        "global_design": global_design,
        "weights": weights,
        "local_inverses": local_inverses,
        "couplings": local_couplings,
        "global_inverse": global_inverse,
        "local_means": local_means,
        "local_sdevs": local_sdevs,
        "global_means": global_means,
        "global_sdevs": global_sdevs,
        "logdet_information": local_information_logdet + schur_logdet,
        "logdet_covariance": covariance_logdet,
    }


def _solve_block_system(system: dict[str, object], observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve one center or resample RHS using a prepared block-arrow system."""
    local_design = system["local_design"]
    global_design = system["global_design"]
    local_means = system["local_means"]
    local_sdevs = system["local_sdevs"]
    global_means = system["global_means"]
    global_sdevs = system["global_sdevs"]
    local_rhs = []
    local_preliminary = []
    global_rhs = global_means / global_sdevs**2 if global_means.size else np.empty(0, dtype=float)
    for x_index, (weight, inverse, coupling) in enumerate(
        zip(system["weights"], system["local_inverses"], system["couplings"])
    ):
        rhs = local_design.T @ weight @ observations[:, x_index] + local_means[x_index] / local_sdevs[x_index] ** 2
        preliminary = inverse @ rhs
        local_rhs.append(rhs)
        local_preliminary.append(preliminary)
        global_rhs += global_design.T @ weight @ observations[:, x_index] - coupling.T @ preliminary
    global_values = system["global_inverse"] @ global_rhs if global_means.size else np.empty(0, dtype=float)
    local_values = np.asarray(
        [
            inverse @ (rhs - coupling @ global_values)
            for inverse, rhs, coupling in zip(system["local_inverses"], local_rhs, system["couplings"])
        ]
    )
    return local_values, global_values


def _block_posterior_sdevs(system: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal posterior deviations for local and global parameters."""
    global_covariance = system["global_inverse"]
    local_covariances = []
    for inverse, coupling in zip(system["local_inverses"], system["couplings"]):
        bridge = inverse @ coupling
        local_covariances.append(inverse + bridge @ global_covariance @ bridge.T)
    local_sdevs = np.sqrt(np.maximum([np.diag(covariance) for covariance in local_covariances], 0.0))
    global_sdevs = np.sqrt(np.maximum(np.diag(global_covariance), 0.0))
    return np.asarray(local_sdevs), np.asarray(global_sdevs)


def _block_fit_diagnostics(
    system: dict[str, object],
    observations: np.ndarray,
    local_values: np.ndarray,
    global_values: np.ndarray,
) -> dict[str, float]:
    """Compute center Bayesian-fit diagnostics for the block solution."""
    chi2 = 0.0
    for x_index, weight in enumerate(system["weights"]):
        prediction = system["local_design"] @ local_values[x_index] + system["global_design"] @ global_values
        residual = observations[:, x_index] - prediction
        chi2 += float(residual @ weight @ residual)
    chi2 += float(np.sum(((local_values - system["local_means"]) / system["local_sdevs"]) ** 2))
    if global_values.size:
        chi2 += float(np.sum(((global_values - system["global_means"]) / system["global_sdevs"]) ** 2))
    dof = int(observations.size)
    from scipy.special import gammaincc

    prior_logdet = float(2.0 * np.sum(np.log(system["local_sdevs"])))
    if global_values.size:
        prior_logdet += float(2.0 * np.sum(np.log(system["global_sdevs"])))
    log_gbf = -0.5 * (
        chi2
        + dof * math.log(2.0 * math.pi)
        + float(system["logdet_covariance"])
        + prior_logdet
        + float(system["logdet_information"])
    )
    return {
        "chi2": chi2,
        "dof": float(dof),
        "Q": float(gammaincc(dof / 2.0, chi2 / 2.0)),
        "logGBF": log_gbf,
    }


def _full_design_matrix(
    design: np.ndarray,
    terms: list[str],
    x_dependence: dict[str, bool],
    n_x: int,
) -> tuple[np.ndarray, dict[str, slice]]:
    """Build the joint-x linear map and its parameter layout once."""
    layout = {"h0": slice(0, n_x)}
    offset = n_x
    for term in terms:
        size = n_x if x_dependence[term] else 1
        layout[term] = slice(offset, offset + size)
        offset += size
    matrix = np.zeros((design.shape[0] * n_x, offset), dtype=float)
    for input_index in range(design.shape[0]):
        rows = input_index * n_x + np.arange(n_x)
        matrix[rows, np.arange(n_x)] = 1.0
        for term_index, term in enumerate(terms):
            parameter_slice = layout[term]
            if x_dependence[term]:
                matrix[rows, parameter_slice.start + np.arange(n_x)] = design[input_index, term_index]
            else:
                matrix[rows, parameter_slice.start] = design[input_index, term_index]
    return matrix, layout


def _factor_full_covariance(
    covariance: np.ndarray,
    data: list[EnsembleData],
    n_x: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    """Factor independent ensemble-source blocks of the full x covariance."""
    blocks = []
    logdet = 0.0
    for group in _ensemble_groups(data):
        indices = np.concatenate([input_index * n_x + np.arange(n_x) for input_index in group])
        weight, block_logdet = _regulated_inverse(covariance[np.ix_(indices, indices)])
        blocks.append((indices, weight))
        logdet += block_logdet
    return blocks, logdet


def _prepare_full_system(
    design_matrix: np.ndarray,
    covariance_blocks: list[tuple[np.ndarray, np.ndarray]],
    covariance_logdet: float,
    prior_means: np.ndarray,
    prior_sdevs: np.ndarray,
) -> dict[str, object]:
    """Factor one full correlated linear posterior for repeated right-hand sides."""
    from scipy.linalg import cho_factor

    information = np.diag(1.0 / prior_sdevs**2)
    for indices, weight in covariance_blocks:
        block_design = design_matrix[indices]
        information += block_design.T @ weight @ block_design
    factor = cho_factor(information, lower=True, check_finite=False)
    information_logdet = float(2.0 * np.sum(np.log(np.diag(factor[0]))))
    return {
        "design": design_matrix,
        "covariance_blocks": covariance_blocks,
        "covariance_logdet": covariance_logdet,
        "prior_means": prior_means,
        "prior_sdevs": prior_sdevs,
        "factor": factor,
        "information_logdet": information_logdet,
    }


def _solve_full_system(system: dict[str, object], observations: np.ndarray) -> np.ndarray:
    """Solve one center or resample using the prepared full covariance and posterior factor."""
    from scipy.linalg import cho_solve

    design = system["design"]
    rhs = system["prior_means"] / system["prior_sdevs"] ** 2
    for indices, weight in system["covariance_blocks"]:
        rhs += design[indices].T @ weight @ observations[indices]
    return cho_solve(system["factor"], rhs, check_finite=False)


def _full_posterior_sdevs(system: dict[str, object]) -> np.ndarray:
    """Return marginal posterior deviations from one factored information matrix."""
    from scipy.linalg import cho_solve

    size = len(system["prior_means"])
    covariance = cho_solve(system["factor"], np.eye(size), check_finite=False)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _full_fit_diagnostics(
    system: dict[str, object],
    observations: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, float]:
    """Compute center Bayesian-fit diagnostics for the full correlated solution."""
    residual = observations - system["design"] @ parameters
    chi2 = 0.0
    for indices, weight in system["covariance_blocks"]:
        block = residual[indices]
        chi2 += float(block @ weight @ block)
    chi2 += float(np.sum(((parameters - system["prior_means"]) / system["prior_sdevs"]) ** 2))
    dof = int(observations.size)
    from scipy.special import gammaincc

    prior_logdet = float(2.0 * np.sum(np.log(system["prior_sdevs"])))
    log_gbf = -0.5 * (
        chi2
        + dof * math.log(2.0 * math.pi)
        + float(system["covariance_logdet"])
        + prior_logdet
        + float(system["information_logdet"])
    )
    return {
        "chi2": chi2,
        "dof": float(dof),
        "Q": float(gammaincc(dof / 2.0, chi2 / 2.0)),
        "logGBF": log_gbf,
    }


def fit_candidate(
    data: list[EnsembleData],
    terms: list[str],
    physical_mass: float | None,
    priors: dict[str, float],
    *,
    x_range: tuple[float, float],
    x_independent_terms: list[str] | None = None,
    x_covariance: bool = False,
    pdep_gev: list[float] | None = None,
    posterior_prior_error_scale: float = 3.0,
    workers: int = 1,
    _parallel=None,
) -> tuple[EnsembleData, dict[str, float]]:
    """Run the reference joint-x, sample-bearing extrapolation fit."""
    if not data or any(item.dims != ["x"] for item in data):
        raise ValueError("extrapolation requires one-dimensional x-space inputs")
    x_all = np.asarray(data[0].coords["x"], dtype=float)
    if any(not np.allclose(item.coords["x"], x_all, rtol=0.0, atol=1e-12) for item in data[1:]):
        raise ValueError("all extrapolation grids must match by coordinate value")
    mask = (x_all >= float(x_range[0]) - 1e-12) & (x_all <= float(x_range[1]) + 1e-12)
    if not np.any(mask):
        raise ValueError("fit range selects no x coordinate")
    x = x_all[mask]
    if set(priors) != {"mean", "sdev"}:
        raise ValueError("candidate priors must contain exactly mean and sdev")
    prior_center = float(priors["mean"])
    prior_width = float(priors["sdev"])
    if not math.isfinite(prior_center) or not math.isfinite(prior_width) or prior_width <= 0:
        raise ValueError("candidate prior mean and sdev must be finite with positive sdev")
    x_independent_terms = [] if x_independent_terms is None else list(x_independent_terms)
    if len(set(x_independent_terms)) != len(x_independent_terms) or not set(x_independent_terms).issubset(terms):
        raise ValueError("x_independent_terms must be a unique subset of terms")
    x_dependence = {term: term not in x_independent_terms for term in terms}
    if not math.isfinite(posterior_prior_error_scale) or posterior_prior_error_scale <= 0:
        raise ValueError("posterior_prior_error_scale must be finite and positive")
    n_sample = min(item.n_sample for item in data)
    if n_sample < 2 or any(item.resample != data[0].resample for item in data):
        raise ValueError("extrapolation inputs require one shared bootstrap/jackknife mode")
    values = np.stack([np.asarray(item.values)[:n_sample, mask] for item in data], axis=0)
    if np.iscomplexobj(values):
        if not np.allclose(np.imag(values), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("reference extrapolation consumes the real matching channel")
        values = np.real(values)
    design = np.asarray(
        [basis_terms(item.attrs, terms, physical_mass) for item in data],
        dtype=float,
    )
    sample_error_modes = {str(item.attrs.get("sample_error_mode", "covariance")) for item in data}
    if len(sample_error_modes) != 1:
        raise ValueError("all extrapolation inputs must share sample_error_mode")
    sample_error_mode = sample_error_modes.pop()
    centers, covariance = _grouped_centers_and_covariances(
        values,
        data,
        sample_error_mode,
        x_covariance=x_covariance,
    )
    if not x_covariance:
        local_terms = [term for term in terms if x_dependence[term]]
        global_terms = [term for term in terms if not x_dependence[term]]
        local_means = np.full((len(x), 1 + len(local_terms)), prior_center, dtype=float)
        local_sdevs = np.full_like(local_means, prior_width)
        global_means = np.full(len(global_terms), prior_center, dtype=float)
        global_sdevs = np.full(len(global_terms), prior_width, dtype=float)
        center_system = _prepare_block_system(
            design,
            covariance,
            terms,
            x_dependence,
            local_means,
            local_sdevs,
            global_means,
            global_sdevs,
        )
        center_local, center_global = _solve_block_system(center_system, centers)
        fit_diagnostics = _block_fit_diagnostics(center_system, centers, center_local, center_global)
        posterior_local_sdevs, posterior_global_sdevs = _block_posterior_sdevs(center_system)
        sample_system = _prepare_block_system(
            design,
            covariance,
            terms,
            x_dependence,
            center_local,
            posterior_local_sdevs * posterior_prior_error_scale,
            center_global,
            posterior_global_sdevs * posterior_prior_error_scale,
        )

        def parameter_record(local_values: np.ndarray, global_values: np.ndarray) -> dict[str, object]:
            record: dict[str, object] = {"h0": local_values[:, 0]}
            for index, term in enumerate(local_terms, start=1):
                record[term] = local_values[:, index]
            for index, term in enumerate(global_terms):
                record[term] = float(global_values[index])
            return record

        fitted_parameters = []
        for sample_index in range(n_sample):
            sample_local, sample_global = _solve_block_system(sample_system, values[:, sample_index, :])
            fitted_parameters.append(parameter_record(sample_local, sample_global))
        n_failed_samples = 0.0
    else:
        design_matrix, parameter_layout = _full_design_matrix(
            design,
            terms,
            x_dependence,
            len(x),
        )
        covariance_blocks, covariance_logdet = _factor_full_covariance(covariance, data, len(x))
        prior_means = np.full(design_matrix.shape[1], prior_center, dtype=float)
        prior_sdevs = np.full(design_matrix.shape[1], prior_width, dtype=float)
        center_system = _prepare_full_system(
            design_matrix,
            covariance_blocks,
            covariance_logdet,
            prior_means,
            prior_sdevs,
        )
        center_parameters = _solve_full_system(center_system, centers.reshape(-1))
        fit_diagnostics = _full_fit_diagnostics(center_system, centers.reshape(-1), center_parameters)
        posterior_sdevs = _full_posterior_sdevs(center_system)
        sample_system = _prepare_full_system(
            design_matrix,
            covariance_blocks,
            covariance_logdet,
            center_parameters,
            posterior_sdevs * posterior_prior_error_scale,
        )

        def parameter_record(parameters: np.ndarray) -> dict[str, object]:
            record: dict[str, object] = {}
            for name, parameter_slice in parameter_layout.items():
                value = parameters[parameter_slice]
                record[name] = value.copy() if value.size > 1 else float(value[0])
            return record

        flattened = np.moveaxis(values, 1, 0).reshape(n_sample, -1)
        fitted_parameters = [parameter_record(_solve_full_system(sample_system, sample)) for sample in flattened]
        n_failed_samples = 0.0
    samples = [np.asarray(parameters["h0"], dtype=float) for parameters in fitted_parameters]
    parameter_mean: dict[str, object] = {}
    parameter_sdev: dict[str, object] = {}
    for name in ["h0", *terms]:
        values = np.asarray([np.asarray(parameters[name], dtype=float) for parameters in fitted_parameters])
        mean = np.mean(values, axis=0)
        sdev = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        parameter_mean[name] = float(mean) if mean.ndim == 0 else mean.tolist()
        parameter_sdev[name] = float(sdev) if sdev.ndim == 0 else sdev.tolist()
    momentum_dependence: dict[str, dict[str, object]] = {}
    for momentum in pdep_gev or []:
        momentum = float(momentum)
        if not math.isfinite(momentum) or momentum <= 0:
            raise ValueError("pdep_gev values must be finite and positive")
        predictions = []
        for parameters in fitted_parameters:
            prediction = np.asarray(parameters["h0"], dtype=float).copy()
            if "inv_p2" in terms:
                prediction = prediction + np.asarray(parameters["inv_p2"], dtype=float) / momentum**2
            if "inv_p4" in terms:
                prediction = prediction + np.asarray(parameters["inv_p4"], dtype=float) / momentum**4
            predictions.append(prediction)
        values = np.asarray(predictions, dtype=float)
        momentum_dependence[f"{momentum:g}"] = {
            "momentum_gev": momentum,
            "mean": np.mean(values, axis=0).tolist(),
            "sdev": (
                np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1], dtype=float)
            ).tolist(),
        }
    attrs = dict(data[0].attrs)
    attrs.update(
        {
            "ensemble": None,
            "extrapolation_terms": ",".join(terms),
            "x_independent_terms": json.dumps(x_independent_terms),
            "x_dependent_terms": json.dumps([term for term in terms if term not in x_independent_terms]),
            "x_covariance": int(bool(x_covariance)),
            "physical_point": "continuum,infinite_momentum",
            "sample_error_mode": sample_error_mode,
            "posterior_prior_error_scale": float(posterior_prior_error_scale),
            "initial_prior_mean": prior_center,
            "initial_prior_sdev": prior_width,
            "units": '{"values":"dimensionless","x":"dimensionless"}',
        }
    )
    if physical_mass is not None:
        attrs["physical_pion_mass_gev"] = float(physical_mass)
    result = EnsembleData(
        None,
        data[0].resample,
        samples,
        ["x"],
        {"x": x.tolist()},
        attrs=attrs,
        name="extrapolated_distribution",
    )
    parameter_count = len(x) + sum(len(x) if x_dependence[term] else 1 for term in terms)
    return result, {
        "chi2": float(fit_diagnostics["chi2"]),
        "dof": float(fit_diagnostics["dof"]),
        "chi2_dof": float(fit_diagnostics["chi2"] / fit_diagnostics["dof"]),
        "Q": float(fit_diagnostics["Q"]),
        "logGBF": float(fit_diagnostics["logGBF"]),
        "aic": float(fit_diagnostics["chi2"] + 2.0 * parameter_count),
        "n_failed_samples": n_failed_samples,
        "x_covariance": bool(x_covariance),
        "parameter_mean": parameter_mean,
        "parameter_sdev": parameter_sdev,
        "momentum_dependence": momentum_dependence,
    }
