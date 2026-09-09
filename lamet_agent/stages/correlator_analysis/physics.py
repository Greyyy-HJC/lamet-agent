"""Small correlator equations used by candidate tools."""

from __future__ import annotations

import json
from typing import Any, Mapping

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError, nonlinear_fit
from lamet_agent.parallel._pool import _ParallelPool
from lamet_agent.ui import track
from lamet_agent.stages.correlator_analysis._scope import parse_fit_scope


def _state_energies(parameters: Mapping[str, Any], n_states: int, suffix: str = "") -> list[Any]:
    energy = parameters[f"E0{suffix}"]
    energies = []
    for state in range(n_states):
        if state:
            energy = energy + parameters[f"dE{state}{suffix}"]
        energies.append(energy)
    return energies


def _two_point_model(
    times: np.ndarray, parameters: Mapping[str, Any], extent: int, n_states: int, suffix: str = ""
) -> np.ndarray:
    values = 0.0
    for state, energy in enumerate(_state_energies(parameters, n_states, suffix)):
        overlap = parameters[f"z{state}{suffix}"]
        values = values + overlap**2 / (2 * energy) * (np.exp(-energy * times) + np.exp(-energy * (extent - times)))
    return values


def pt2_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one periodic raw two-point correlator."""
    return _two_point_model(
        np.asarray(x["times"]),
        parameters,
        int(x["extent"]),
        int(x["n_states"]),
        str(x.get("suffix", "")),
    )


def _three_point_model(
    times: np.ndarray,
    insertions: np.ndarray,
    parameters: Mapping[str, Any],
    n_states: int,
    form: str,
    component: str,
) -> np.ndarray:
    """Evaluate a raw three-point spectral decomposition."""
    if form == "Breit":
        energies = _state_energies(parameters, n_states)
        value = 0.0
        for source, source_energy in enumerate(energies):
            for sink, sink_energy in enumerate(energies):
                matrix = parameters[f"O{min(source, sink)}{max(source, sink)}_{component}"]
                value = value + matrix * parameters[f"z{source}"] * parameters[f"z{sink}"] * np.exp(
                    -source_energy * (times - insertions)
                ) * np.exp(-sink_energy * insertions) / (2 * source_energy) / (2 * sink_energy)
        return value
    energies_i = _state_energies(parameters, n_states, "_i")
    energies_f = _state_energies(parameters, n_states, "_f")
    value = 0.0
    for sink, sink_energy in enumerate(energies_f):
        for source, source_energy in enumerate(energies_i):
            value = value + parameters[f"O{sink}{source}_{component}"] * parameters[f"z{sink}_f"] * parameters[
                f"z{source}_i"
            ] * np.exp(-sink_energy * (times - insertions)) * np.exp(-source_energy * insertions) / (
                2 * sink_energy
            ) / (2 * source_energy)
    return value


def pt3_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one component of a raw three-point correlator."""
    return _three_point_model(
        np.asarray(x["times"]),
        np.asarray(x["insertions"]),
        parameters,
        int(x["n_states"]),
        str(x["form"]),
        str(x["component"]),
    )


def _ratio_model(
    times: np.ndarray,
    insertions: np.ndarray,
    parameters: Mapping[str, Any],
    extent: int,
    n_states: int,
    form: str,
    component: str,
) -> np.ndarray:
    numerator = _three_point_model(times, insertions, parameters, n_states, form, component)
    if form == "Breit":
        return numerator / _two_point_model(times, parameters, extent, n_states)
    c2_i_t_tau = _two_point_model(times - insertions, parameters, extent, n_states, "_i")
    c2_i_tau = _two_point_model(insertions, parameters, extent, n_states, "_i")
    c2_i_t = _two_point_model(times, parameters, extent, n_states, "_i")
    c2_f_t_tau = _two_point_model(times - insertions, parameters, extent, n_states, "_f")
    c2_f_tau = _two_point_model(insertions, parameters, extent, n_states, "_f")
    c2_f_t = _two_point_model(times, parameters, extent, n_states, "_f")
    return numerator / c2_f_t * gv.sqrt(c2_i_t_tau * c2_f_tau * c2_f_t / (c2_f_t_tau * c2_i_tau * c2_i_t))


def _summed_ratio_model(
    times: np.ndarray, tau_min: int, parameters: Mapping[str, Any], n_states: int, component: str
) -> np.ndarray:
    if n_states == 1:
        return parameters[f"O00_{component}"] * (times - 2 * tau_min + 1) / (2 * parameters["E0"])
    exponential = np.exp(-parameters["dE1"] * times)
    numerator = (
        parameters[f"O00_{component}"]
        * (times - 2 * tau_min + 1)
        * (1 + parameters[f"sum_{component}_excited_coeff"] * exponential)
        + parameters[f"sum_{component}_offset"]
        + parameters[f"sum_{component}_exp_offset"] * exponential
    )
    return numerator / (2 * parameters["E0"] * (1 + parameters["sum_den_exp_coeff"] * exponential))


def matrix_element_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the ordered observation blocks of one joint fit stage."""
    values: list[np.ndarray] = []
    n_states = int(x["n_states"])
    extent = int(x["extent"])
    form = str(x["form"])
    atoms = tuple(x["atoms"])
    components = tuple(x["components"])
    for atom in atoms:
        if atom == "2pt":
            if x.get("family") == "qda" and x.get("denominator_kind") == "qda_z0":
                values.append(_qda_correlator(x["pt2_t"], parameters, extent, n_states, "zprime"))
            elif form == "Breit":
                values.append(_two_point_model(x["pt2_t"], parameters, extent, n_states))
            else:
                values.append(_two_point_model(x["pt2_t"], parameters, extent, n_states, "_i"))
                values.append(_two_point_model(x["pt2_t"], parameters, extent, n_states, "_f"))
        elif atom == "3pt":
            for component in components:
                values.append(_three_point_model(x["ratio_t"], x["ratio_tau"], parameters, n_states, form, component))
        elif atom == "3pt_ratio":
            for component in components:
                values.append(_ratio_model(x["ratio_t"], x["ratio_tau"], parameters, extent, n_states, form, component))
        elif atom == "qda":
            for component in components:
                values.append(_qda_correlator(x["qda_t"], parameters, extent, n_states, component))
        elif atom == "qda_ratio":
            denominator_key = "zprime" if x.get("denominator_kind") == "qda_z0" else None
            denominator = (
                _qda_correlator(x["qda_t"], parameters, extent, n_states, denominator_key)
                if denominator_key is not None
                else _two_point_model(x["qda_t"], parameters, extent, n_states)
            )
            for component in components:
                values.append(_qda_correlator(x["qda_t"], parameters, extent, n_states, component) / denominator)
        elif atom == "FH":
            for component in components:
                after = _summed_ratio_model(x["fh_t"] + x["fh_dt"], int(x["tau_min"]), parameters, n_states, component)
                before = _summed_ratio_model(x["fh_t"], int(x["tau_min"]), parameters, n_states, component)
                values.append((after - before) / x["fh_dt"])
        else:
            raise ValueError(f"unsupported fit-scope atom {atom!r}")
    return np.concatenate([np.atleast_1d(value) for value in values])


def matrix_element_prior(
    n_states: int,
    *,
    form: str,
    scope: str | tuple[str, ...] | list[str],
    components: tuple[str, ...],
    width_scale: float,
    denominator_kind: str = "external_2pt",
) -> gv.BufferDict:
    """Build the explicit spectral and matrix-element prior for one fit model."""
    if n_states < 1 or not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError("n_states and width_scale must be positive")
    atoms = set(scope.split("+")) if isinstance(scope, str) else set(scope)
    if form not in {"Breit", "NonBreit"} or not atoms:
        raise ValueError("unsupported fitting form or scope")
    if form == "NonBreit" and atoms - {"2pt", "3pt", "3pt_ratio"}:
        raise ValueError("NonBreit fitting supports only 2pt, 3pt, and 3pt_ratio")
    prior = gv.BufferDict()
    suffixes = ("",) if form == "Breit" else ("_i", "_f")
    for suffix in suffixes:
        prior[f"log(E0{suffix})"] = gv.gvar(0.0, 3.0 * width_scale)
        for state in range(1, n_states):
            prior[f"log(dE{state}{suffix})"] = gv.gvar(0.0, width_scale)
        for state in range(n_states):
            prior[f"z{state}{suffix}"] = gv.gvar(1.0, 10.0 * width_scale) / 3**state
    if atoms & {"3pt", "3pt_ratio"}:
        if form == "Breit":
            matrix_indices = [(row, column) for row in range(n_states) for column in range(row, n_states)]
        else:
            matrix_indices = [(sink, source) for sink in range(n_states) for source in range(n_states)]
        for row, column in matrix_indices:
            for component in ("re", "im"):
                prior[f"O{row}{column}_{component}"] = gv.gvar(1.0, 10.0 * width_scale)
    if "FH" in atoms:
        for component in ("re", "im"):
            prior.setdefault(f"O00_{component}", gv.gvar(1.0, 10.0 * width_scale))
            if n_states > 1:
                prior[f"sum_{component}_excited_coeff"] = gv.gvar(0.0, 10.0 * width_scale)
                prior[f"sum_{component}_offset"] = gv.gvar(0.0, 10.0 * width_scale)
                prior[f"sum_{component}_exp_offset"] = gv.gvar(0.0, 10.0 * width_scale)
        if n_states > 1:
            prior["sum_den_exp_coeff"] = gv.gvar(0.0, 10.0 * width_scale)
    if denominator_kind == "qda_z0" and "2pt" in atoms:
        for state in range(n_states):
            prior[f"zprime{state}"] = gv.gvar(1.0, 10.0 * width_scale) / 3**state
    if atoms & {"qda", "qda_ratio"}:
        if form != "Breit":
            raise ValueError("qDA fitting supports only Breit kinematics")
        if denominator_kind == "qda_z0" and "2pt" not in atoms:
            for state in range(n_states):
                prior[f"zprime{state}"] = gv.gvar(1.0, 10.0 * width_scale) / 3**state
        for state in range(n_states):
            prior[f"O0{state}_re"] = gv.gvar(1.0, 10.0 * width_scale)
            prior[f"O0{state}_im"] = gv.gvar(0.0, 10.0 * width_scale)
    return prior


def _sample_diagnostic_records(result: Any) -> list[dict[str, float | int]]:
    """Return ordered, JSON-safe quality records for successful sample fits."""
    records: list[dict[str, float | int]] = []
    for sample_index, diagnostics in enumerate(result.sample_diagnostics):
        if diagnostics is None:
            continue
        dof = float(diagnostics["dof"])
        records.append(
            {
                "sample": sample_index,
                "chi2": float(diagnostics["chi2"]),
                "dof": dof,
                "chi2_dof": float(diagnostics["chi2"]) / dof,
                "Q": float(diagnostics["Q"]),
                "logGBF": float(diagnostics["logGBF"]),
            }
        )
    return records


def _gvar_payload(values: Any) -> tuple[list[float], list[float]]:
    return (
        np.asarray(gv.mean(values), dtype=float).reshape(-1).tolist(),
        np.asarray(gv.sdev(values), dtype=float).reshape(-1).tolist(),
    )


def _matrix_sample0_plot_payload(
    *,
    ratios: Mapping[int, np.ndarray],
    z_value: int | float,
    z_index: int,
    posterior: Mapping[str, Any] | None,
    selected_components: tuple[str, ...],
    fit_scope: str,
    fitting_form: str,
    extent: int,
    n_states: int,
    tsep_values: list[int],
    available_tau: np.ndarray,
    tau_min: int,
    ensemble: Any,
    resample: str,
    sample_error_mode: str,
) -> dict[str, Any] | None:
    """Build serializable sample-0 data and posterior bands without refitting."""
    if posterior is None:
        return None
    plots: list[dict[str, Any]] = []
    if fit_scope in {"3pt_ratio", "3pt_ratio+FH"}:
        for component in selected_components:
            series = []
            for tsep in tsep_values:
                mask = (available_tau >= tau_min) & (available_tau <= tsep - tau_min)
                values = ratios[tsep][:, mask, z_index]
                selected = np.real(values) if component == "re" else np.imag(values)
                average = EnsembleData(
                    ensemble,
                    resample,
                    list(selected),
                    ["tau"],
                    {"tau": available_tau[mask].tolist()},
                ).average(sample_error_mode)
                # The fitted ratio uses the periodic two-point denominator.
                # For the diagnostic figure, restore the legacy forward-
                # denominator convention so its asymptotic band is directly
                # comparable with O00/(2 E0).
                correction_energy = posterior["E0_f"] if fitting_form == "NonBreit" else posterior["E0"]
                denominator_correction = 1.0 + gv.exp(-correction_energy * (float(extent) - 2.0 * float(tsep)))
                plotted_data = (
                    gv.gvar(
                        np.asarray(selected[0], dtype=float),
                        np.asarray(gv.evalcov(average), dtype=float),
                    )
                    * denominator_correction
                )
                fit_tau = np.linspace(float(tau_min) - 0.5, float(tsep - tau_min) + 0.5, 200)
                fit_values = (
                    _ratio_model(
                        np.full_like(fit_tau, float(tsep)),
                        fit_tau,
                        posterior,
                        extent,
                        n_states,
                        fitting_form,
                        component,
                    )
                    * denominator_correction
                )
                data_mean, data_sdev = _gvar_payload(plotted_data)
                fit_mean, fit_sdev = _gvar_payload(fit_values)
                series.append(
                    {
                        "label": rf"$t_{{\mathrm{{sep}}}}={tsep}\,a$",
                        "x": (available_tau[mask].astype(float) - float(tsep) / 2.0).tolist(),
                        "y": data_mean,
                        "yerr": data_sdev,
                        "fit_x": (fit_tau - float(tsep) / 2.0).tolist(),
                        "fit_mean": fit_mean,
                        "fit_sdev": fit_sdev,
                    }
                )
            if fitting_form == "NonBreit":
                sign = -1.0 if float(gv.mean(posterior["z0_i"] * posterior["z0_f"])) < 0.0 else 1.0
                plateau = sign * posterior[f"O00_{component}"] / (posterior["E0_i"] + posterior["E0_f"])
            else:
                plateau = posterior[f"O00_{component}"] / (2.0 * posterior["E0"])
            plots.append(
                {
                    "kind": "pt3_ratio",
                    "component": component,
                    "series": series,
                    "plateau_mean": float(gv.mean(plateau)),
                    "plateau_sdev": float(gv.sdev(plateau)),
                }
            )
    if fit_scope in {"FH", "3pt_ratio+FH"}:
        summed = []
        for tsep in tsep_values:
            mask = (available_tau >= tau_min) & (available_tau <= tsep - tau_min)
            summed.append(np.sum(ratios[tsep][:, mask, z_index], axis=1))
        summed_values = np.stack(summed, axis=1)
        dt = np.diff(np.asarray(tsep_values, dtype=float))
        differences = np.diff(summed_values, axis=1) / dt[None, :]
        fh_t = np.asarray(tsep_values[:-1], dtype=float)
        if fh_t.size == 1 or not np.allclose(dt, dt[0], rtol=0.0, atol=1e-12):
            fit_t = fh_t
            fit_dt = dt
        else:
            fit_t = np.linspace(float(fh_t.min()), float(fh_t.max()), 200)
            fit_dt = np.full_like(fit_t, float(dt[0]))
        for component in selected_components:
            selected = np.real(differences) if component == "re" else np.imag(differences)
            average = EnsembleData(
                ensemble,
                resample,
                list(selected),
                ["tsep"],
                {"tsep": fh_t.tolist()},
            ).average(sample_error_mode)
            after = _summed_ratio_model(fit_t + fit_dt, tau_min, posterior, n_states, component)
            before = _summed_ratio_model(fit_t, tau_min, posterior, n_states, component)
            fit_mean, fit_sdev = _gvar_payload((after - before) / fit_dt)
            plateau = posterior[f"O00_{component}"] / (2.0 * posterior["E0"])
            plots.append(
                {
                    "kind": "fh",
                    "component": component,
                    "series": [
                        {
                            "label": "FH",
                            "x": fh_t.tolist(),
                            "y": np.asarray(selected[0], dtype=float).tolist(),
                            "yerr": np.asarray(gv.sdev(average), dtype=float).tolist(),
                            "fit_x": fit_t.tolist(),
                            "fit_mean": fit_mean,
                            "fit_sdev": fit_sdev,
                        }
                    ],
                    "plateau_mean": float(gv.mean(plateau)),
                    "plateau_sdev": float(gv.sdev(plateau)),
                }
            )
    return {"z": float(z_value), "plots": plots} if plots else None


def _momentum(data: EnsembleData, name: str) -> tuple[int, int, int]:
    value = json.loads(str(data.attrs[name]))
    return tuple(int(component) for component in value)


def _optional_momentum(data: EnsembleData, name: str) -> tuple[int, int, int] | None:
    value = data.attrs.get(name)
    if value is None:
        return None
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        momentum = tuple(int(component) for component in decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return momentum if len(momentum) == 3 else None


def fit_matrix_element_samples(
    correlators: Mapping[str, EnsembleData],
    *,
    fitting_form: str,
    fit_scope: list[str] | tuple[str, ...],
    components: str,
    tmin: int,
    tmax: int,
    tsep_values: list[int],
    tau_min: int,
    n_states: int,
    prior_width: float,
    correlator_rescale: float,
    svdcut: float,
    posterior_prior_error_scale: float,
    sample_error_mode: str = "covariance",
    workers: int,
    tune_z: int | float | None = None,
    fit_samples: bool = True,
    show_progress: bool = False,
    _parallel: _ParallelPool | None = None,
) -> tuple[EnsembleData | None, dict[str, Any]]:
    """Fit one ordinary three-point scope pipeline."""
    pipeline = parse_fit_scope(fit_scope)
    if pipeline.is_qda or pipeline.is_spectrum or not pipeline.needs_pt3_data:
        raise ValueError("ordinary matrix-element fitting requires a three-point or FH fit_scope")
    if fitting_form not in {"Breit", "NonBreit"}:
        raise ValueError("fitting_form must be Breit or NonBreit")
    if fitting_form == "NonBreit" and pipeline.atom_set - {"2pt", "3pt", "3pt_ratio"}:
        raise ValueError("NonBreit fitting supports only 2pt, 3pt, and 3pt_ratio")
    if n_states > 2 and "FH" in pipeline.atom_set:
        raise ValueError("FH fitting supports at most two states")
    selected_components = {"real": ("re",), "imag": ("im",), "both": ("re", "im")}.get(components)
    if selected_components is None:
        raise ValueError("components must be real, imag, or both")
    if not np.isfinite(correlator_rescale) or correlator_rescale <= 0:
        raise ValueError("correlator_rescale must be finite and positive")
    if not isinstance(fit_samples, bool):
        raise TypeError("fit_samples must be a boolean")
    if fit_samples and tune_z is not None:
        raise ValueError("tune_z is only valid for a sample-average candidate fit")
    two_points = [value for value in correlators.values() if value.attrs.get("correlator_type") == "two_point"]
    three_points = [value for value in correlators.values() if value.attrs.get("correlator_type") == "three_point"]
    if len(three_points) != 1:
        raise ValueError("matrix-element model fitting requires exactly one three-point correlator")
    three_point = three_points[0]
    if three_point.dims != ["tsep", "tau", "z"]:
        raise ValueError("three-point data must have exactly tsep, tau, and z dimensions")
    source_momentum = _momentum(three_point, "source_momentum")
    sink_momentum = _momentum(three_point, "sink_momentum")
    initial_matches = [
        value for value in two_points if value.dims == ["t"] and _momentum(value, "sink_momentum") == source_momentum
    ]
    final_matches = [
        value for value in two_points if value.dims == ["t"] and _momentum(value, "sink_momentum") == sink_momentum
    ]
    if len(initial_matches) != 1 or len(final_matches) != 1:
        raise ValueError("two-point inputs must uniquely match the three-point source and sink momenta")
    initial, final = initial_matches[0], final_matches[0]
    if fitting_form == "Breit" and (source_momentum != sink_momentum or initial is not final):
        raise ValueError("Breit fitting requires equal source and sink momentum")
    if fitting_form == "NonBreit" and source_momentum == sink_momentum:
        raise ValueError("NonBreit fitting requires distinct source and sink momentum")
    aligned = [initial, final, three_point]
    if any(value.resample != three_point.resample or value.n_sample != three_point.n_sample for value in aligned):
        raise ValueError("all correlators must share resampling mode and sample count")
    resample_ids = {value.attrs.get("resample_id") for value in aligned}
    if len(resample_ids) != 1 or None in resample_ids:
        raise ValueError("all correlators must share one nonempty resample_id")
    extent = int(initial.ensemble.L_t)
    if extent < 1:
        raise ValueError("matrix-element fitting requires the temporal extent")
    times = np.asarray(initial.coords["t"], dtype=int)
    pt2_mask = (times >= tmin) & (times < tmax)
    if np.count_nonzero(pt2_mask) < 2 * n_states:
        raise ValueError("the two-point window must contain at least 2*n_states points")
    available_tseps = np.asarray(three_point.coords["tsep"], dtype=int)
    available_tau = np.asarray(three_point.coords["tau"], dtype=int)
    if not tsep_values or any(np.count_nonzero(available_tseps == tsep) != 1 for tsep in tsep_values):
        raise ValueError("every selected tsep must occur exactly once")

    initial_values = np.asarray(initial.values)
    final_values = np.asarray(final.values)
    three_values = np.asarray(three_point.values)
    ratios: dict[int, np.ndarray] = {}
    for tsep in tsep_values:
        time_index = np.flatnonzero(times == tsep)
        tsep_index = np.flatnonzero(available_tseps == tsep)
        if time_index.size != 1 or tsep_index.size != 1:
            raise ValueError(f"missing exact two-/three-point coordinate for tsep={tsep}")
        three_slice = three_values[:, tsep_index[0], :, :]
        if fitting_form == "Breit":
            ratios[tsep] = three_slice / initial_values[:, time_index[0], None, None]
        else:
            valid_tau = available_tau <= tsep
            reflected = tsep - available_tau[valid_tau]
            if any(np.count_nonzero(times == value) != 1 for value in reflected):
                raise ValueError("NonBreit ratio needs exact t and tsep-t two-point coordinates")
            reflected_indices = np.asarray([np.flatnonzero(times == value)[0] for value in reflected])
            tau_time_indices = np.asarray([np.flatnonzero(times == value)[0] for value in available_tau[valid_tau]])
            correction = (
                initial_values[:, reflected_indices]
                * final_values[:, tau_time_indices]
                * final_values[:, time_index[0], None]
                / (
                    final_values[:, reflected_indices]
                    * initial_values[:, tau_time_indices]
                    * initial_values[:, time_index[0], None]
                )
            )
            ratio = np.zeros_like(three_slice)
            ratio[:, valid_tau, :] = (
                three_slice[:, valid_tau, :]
                / final_values[:, time_index[0], None, None]
                * np.sqrt(correction[:, :, None])
            )
            ratios[tsep] = ratio

    z_values = list(three_point.coords["z"])
    if tune_z is None:
        z_indices = list(range(len(z_values)))
    else:
        tune_matches = np.flatnonzero(
            np.isclose(np.asarray(z_values, dtype=float), float(tune_z), rtol=0.0, atol=1e-12)
        )
        if tune_matches.size != 1:
            raise ValueError("lsqfit.tune_z must name exactly one available z coordinate")
        z_indices = [int(tune_matches[0])]
    fitted_samples = np.zeros((three_point.n_sample, len(z_values)), dtype=complex)
    center_metrics = []
    sample_failures = []
    parallel = _parallel or _ParallelPool(min(workers, three_point.n_sample))
    try:
        fit_indices = track(
            z_indices,
            label="Matrix-element fits",
            unit="z",
            enabled=fit_samples and show_progress,
        )
        for z_index in fit_indices:
            z_value = z_values[z_index]
            ratio_t: list[int] = []
            ratio_tau: list[int] = []
            selected_ratio: list[np.ndarray] = []
            selected_raw: list[np.ndarray] = []
            for tsep in tsep_values:
                tau_mask = (available_tau >= tau_min) & (available_tau <= tsep - tau_min)
                ratio_t.extend([tsep] * int(np.count_nonzero(tau_mask)))
                ratio_tau.extend(available_tau[tau_mask].tolist())
                selected_ratio.append(ratios[tsep][:, tau_mask, z_index])
                tsep_index = int(np.flatnonzero(available_tseps == tsep)[0])
                selected_raw.append(three_values[:, tsep_index, tau_mask, z_index])
            summed = [np.sum(values, axis=1) for values in selected_ratio]
            summed_values = np.stack(summed, axis=1)
            differences = np.diff(summed_values, axis=1) / np.diff(np.asarray(tsep_values, dtype=float))[None, :]

            previous_posterior: Mapping[str, Any] | None = None
            stage_diagnostics: list[dict[str, Any]] = []
            result = None
            fit_prior = None
            observations = None
            for stage_index, atoms in enumerate(pipeline.stages):
                x: dict[str, Any] = {
                    "n_states": n_states,
                    "extent": extent,
                    "form": fitting_form,
                    "atoms": atoms,
                    "components": selected_components,
                    "tau_min": tau_min,
                    "family": "ordinary",
                    "pt2_t": times[pt2_mask],
                    "ratio_t": np.asarray(ratio_t, dtype=float),
                    "ratio_tau": np.asarray(ratio_tau, dtype=float),
                    "fh_t": np.asarray(tsep_values[:-1], dtype=float),
                    "fh_dt": np.diff(np.asarray(tsep_values, dtype=float)),
                }
                pieces: list[np.ndarray] = []
                for atom in atoms:
                    if atom == "2pt":
                        pieces.append(np.real(initial_values[:, pt2_mask]) * correlator_rescale)
                        if fitting_form == "NonBreit":
                            pieces.append(np.real(final_values[:, pt2_mask]) * correlator_rescale)
                    elif atom in {"3pt", "3pt_ratio"}:
                        source_values = selected_raw if atom == "3pt" else selected_ratio
                        for component in selected_components:
                            component_parts = [
                                np.real(values) if component == "re" else np.imag(values) for values in source_values
                            ]
                            scale = correlator_rescale if atom == "3pt" else 1.0
                            pieces.append(np.concatenate(component_parts, axis=1) * scale)
                    elif atom == "FH":
                        for component in selected_components:
                            pieces.append(np.real(differences) if component == "re" else np.imag(differences))
                observations = np.concatenate(pieces, axis=1)
                fit_data = EnsembleData(
                    initial.ensemble,
                    initial.resample,
                    list(observations),
                    ["observation"],
                    {"observation": list(range(observations.shape[1]))},
                )
                fit_prior = matrix_element_prior(
                    n_states,
                    form=fitting_form,
                    scope=atoms,
                    components=selected_components,
                    width_scale=prior_width,
                )
                if previous_posterior is not None:
                    for key in fit_prior:
                        if key in previous_posterior:
                            value = previous_posterior[key]
                            fit_prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * posterior_prior_error_scale)
                is_final = stage_index == len(pipeline.stages) - 1
                result = nonlinear_fit(
                    (x, fit_data),
                    matrix_element_fcn,
                    fit_prior,
                    workers=workers,
                    sample_prior_scale=posterior_prior_error_scale * prior_width,
                    sample_error_mode=sample_error_mode,
                    mode="resamples" if is_final and fit_samples else "center",
                    tolerate_sample_failures=is_final and fit_samples,
                    capture_sample_posteriors=(0,) if is_final and fit_samples else (),
                    _parallel=parallel if is_final else None,
                    svdcut=svdcut,
                    maxit=10000,
                )
                previous_posterior = result.p
                stage_diagnostics.append(
                    {
                        "stage": "+".join(atoms),
                        "chi2": result.chi2,
                        "dof": result.dof,
                        "chi2_dof": result.chi2 / result.dof,
                        "Q": result.Q,
                        "logGBF": result.logGBF,
                        "n_data": int(observations.shape[1]),
                    }
                )
            if result is None or fit_prior is None or observations is None:
                raise RuntimeError("fit_scope produced no ordinary fit stage")
            n_params = sum(int(np.size(gv.mean(fit_prior[key]))) for key in fit_prior)
            energy_keys = ("E0_i", "E0_f") if fitting_form == "NonBreit" else ("E0",)
            energy_summary = {}
            for energy_key in energy_keys:
                energy_summary[energy_key] = float(result.pmean[energy_key])
                energy_samples = (
                    [float(parameters[energy_key]) if parameters is not None else None for parameters in result.samples]
                    if fit_samples
                    else []
                )
                finite_energy_samples = [value for value in energy_samples if value is not None]
                energy_summary[f"{energy_key}_sdev"] = (
                    float(
                        gv.sdev(
                            EnsembleData(
                                initial.ensemble,
                                initial.resample,
                                [[value] for value in finite_energy_samples],
                                ["energy"],
                                {"energy": [0]},
                            ).average(sample_error_mode)[0]
                        )
                    )
                    if len(finite_energy_samples) == len(energy_samples) and len(energy_samples) > 1
                    else None
                )
                energy_summary[f"{energy_key}_samples"] = energy_samples
            sample_diagnostics = _sample_diagnostic_records(result) if fit_samples else []
            plot_scope = (
                "3pt_ratio+FH"
                if "FH" in pipeline.final_stage and pipeline.atom_set & {"3pt", "3pt_ratio"}
                else "FH"
                if "FH" in pipeline.final_stage
                else "3pt_ratio"
            )
            sample0_plot = (
                _matrix_sample0_plot_payload(
                    ratios=ratios,
                    z_value=z_value,
                    z_index=z_index,
                    posterior=result.sample_posteriors[0] if result.sample_posteriors else None,
                    selected_components=selected_components,
                    fit_scope=plot_scope,
                    fitting_form=fitting_form,
                    extent=extent,
                    n_states=n_states,
                    tsep_values=tsep_values,
                    available_tau=available_tau,
                    tau_min=tau_min,
                    ensemble=three_point.ensemble,
                    resample=three_point.resample,
                    sample_error_mode=sample_error_mode,
                )
                if fit_samples
                else None
            )
            center_metrics.append(
                {
                    "z": z_value,
                    "chi2": result.chi2,
                    "dof": result.dof,
                    "chi2_dof": result.chi2 / result.dof,
                    "Q": result.Q,
                    "logGBF": result.logGBF,
                    "n_data": int(observations.shape[1]),
                    "n_params": n_params,
                    "n_failed_samples": 0,
                    "sample_diagnostics": sample_diagnostics,
                    "sample0_plot": sample0_plot,
                    "stages": stage_diagnostics,
                    **energy_summary,
                }
            )
            if not fit_samples:
                continue
            for sample_index, parameters in enumerate(result.samples):
                if parameters is None:
                    fitted_samples[sample_index, z_index] = np.nan + 1j * np.nan
                    sample_failures.append(
                        {"z": z_value, "sample": sample_index, "error": result.sample_errors[sample_index]}
                    )
                    continue
                try:
                    real = (
                        float(parameters["O00_re"] / (parameters["E0_f"] + parameters["E0_i"]))
                        if fitting_form == "NonBreit" and "re" in selected_components
                        else float(parameters["O00_re"] / (2 * parameters["E0"]))
                        if "re" in selected_components
                        else 0.0
                    )
                    imag = (
                        float(parameters["O00_im"] / (parameters["E0_f"] + parameters["E0_i"]))
                        if fitting_form == "NonBreit" and "im" in selected_components
                        else float(parameters["O00_im"] / (2 * parameters["E0"]))
                        if "im" in selected_components
                        else 0.0
                    )
                    if fitting_form == "NonBreit" and float(parameters["z0_f"] * parameters["z0_i"]) < 0:
                        real, imag = -real, -imag
                    if not np.isfinite(real) or not np.isfinite(imag):
                        raise FloatingPointError("non-finite fitted matrix element")
                except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError) as exc:
                    fitted_samples[sample_index, z_index] = np.nan + 1j * np.nan
                    sample_failures.append(
                        {"z": z_value, "sample": sample_index, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                fitted_samples[sample_index, z_index] = real + 1j * imag
            z_failure_count = sum(failure["z"] == z_value for failure in sample_failures)
            if z_failure_count == three_point.n_sample:
                raise FitNumericalError(f"all sample fits failed at z={z_value}")
            center_metrics[-1]["n_failed_samples"] = z_failure_count
    finally:
        if _parallel is None:
            parallel.close()
    output = None
    if fit_samples:
        attrs = dict(three_point.attrs)
        attrs.update(
            {
                "method": "lsqfit",
                "fitting_form": fitting_form,
                "fit_scope": json.dumps(pipeline.as_list()),
                "n_states": n_states,
                "tmin": tmin,
                "tmax": tmax,
                "tau_min": tau_min,
                "correlator_rescale": correlator_rescale,
                "sample_error_mode": sample_error_mode,
            }
        )
        output = EnsembleData(
            three_point.ensemble,
            three_point.resample,
            list(fitted_samples),
            ["z"],
            {"z": z_values},
            attrs=attrs,
            name="bare_matrix_element",
        )
    tuning_fit = center_metrics[0] if not fit_samples and tune_z is not None else None
    diagnostics = {
        "fit_scope": pipeline.as_list(),
        "fitting_form": fitting_form,
        "workers": workers,
        "min_Q": min(record["Q"] for record in center_metrics),
        "max_chi2_dof": max(record["chi2_dof"] for record in center_metrics),
        "n_failed_samples": len(sample_failures),
        "sample_failures": sample_failures,
        "fits": center_metrics,
    }
    if tuning_fit is not None:
        diagnostics.update(
            {
                "tune_z": tuning_fit["z"],
                "chi2": tuning_fit["chi2"],
                "dof": tuning_fit["dof"],
                "chi2_dof": tuning_fit["chi2_dof"],
                "Q": tuning_fit["Q"],
                "logGBF": tuning_fit["logGBF"],
                "n_data": tuning_fit["n_data"],
                "n_params": tuning_fit["n_params"],
            }
        )
    return output, diagnostics


def spectrum_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the periodic two-point spectral model."""
    return _two_point_model(x["times"], parameters, int(x["extent"]), int(x["n_states"]))


def _qda_correlator(
    times: np.ndarray,
    parameters: Mapping[str, Any],
    extent: int,
    n_states: int,
    matrix_key: str,
) -> np.ndarray:
    values = 0.0
    for state, energy in enumerate(_state_energies(parameters, n_states)):
        matrix = parameters[f"zprime{state}"] if matrix_key == "zprime" else parameters[f"O0{state}_{matrix_key}"]
        values = values + parameters[f"z{state}"] / (2 * energy) * matrix * (
            np.exp(-energy * times) + np.exp(-energy * (extent - times))
        )
    return values


def qda_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one component of a raw qDA correlator."""
    component = str(x["component"])
    if component not in {"re", "im"}:
        raise ValueError("qDA component must be re or im")
    return _qda_correlator(
        np.asarray(x["times"]),
        parameters,
        int(x["extent"]),
        int(x["n_states"]),
        component,
    )


def qda_ratio_fcn(x: Mapping[str, Any], parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the selected qDA components divided by their local two-point model."""
    times = x["times"]
    n_states = int(x.get("n_states", 1))
    extent = int(x["extent"])
    denominator = (
        _qda_correlator(times, parameters, extent, n_states, "zprime")
        if x.get("denominator_kind", "qda_z0") == "qda_z0"
        else _two_point_model(times, parameters, extent, n_states)
    )
    components = tuple(x.get("components", ("re", "im")))
    return np.concatenate(
        [_qda_correlator(times, parameters, extent, n_states, component) / denominator for component in components]
    )


def _qda_correlator_rescale(values: np.ndarray) -> float:
    """Return a deterministic power-of-ten scale for a local qDA correlator."""
    absolute = np.abs(np.asarray(values, dtype=float)).reshape(-1)
    usable = absolute[np.isfinite(absolute) & (absolute > 0.0)]
    if not usable.size:
        raise ValueError("qDA local denominator has no finite nonzero real values in the fit window")
    exponent = -int(np.floor(np.log10(float(np.median(usable)))))
    scale = float(10.0**exponent)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("qDA local-denominator rescale is outside the finite float range")
    return scale


def fit_spectrum_samples(
    values: np.ndarray,
    times: np.ndarray,
    n_states: int,
    *,
    extent: int,
    resample: str,
    prior_means: dict[str, float],
    prior_widths: dict[str, float],
    sample_error_mode: str = "covariance",
    workers: int = 1,
    _parallel: _ParallelPool | None = None,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Perform one correlated, prior-constrained multi-state spectral fit."""
    samples = np.asarray(values)
    times = np.asarray(times, dtype=float)
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
        raise ValueError("spectrum fitting requires a positive temporal extent")
    if samples.ndim != 2 or samples.shape[1] != times.size or samples.shape[0] < 2 or times.size < 2 * n_states:
        raise ValueError("spectrum fit requires a two-dimensional sample/time array with at least 2*n_states times")
    if np.iscomplexobj(samples):
        if not np.allclose(np.imag(samples), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("direct spectrum fitting requires real two-point data")
        samples = np.real(samples)
    samples = np.asarray(samples, dtype=float)
    names = [*[f"E{index}" for index in range(n_states)], *[f"z{index}" for index in range(n_states)]]
    if set(prior_means) != set(names) or set(prior_widths) != set(names):
        raise ValueError(f"spectrum priors must contain exactly {names}")
    mean = np.asarray([float(prior_means[name]) for name in names])
    widths = np.asarray([float(prior_widths[name]) for name in names])
    energies = mean[:n_states]
    if (
        np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0)
        or np.any(energies <= 0)
        or np.any(np.diff(energies) <= 0)
        or np.any(mean[n_states:] <= 0)
    ):
        raise ValueError("spectrum priors require ordered positive energies, positive overlaps, and positive widths")
    if resample not in {"raw", "jackknife", "bootstrap"}:
        raise ValueError("spectrum fitting requires raw, jackknife, or bootstrap samples")
    prior = gv.BufferDict()
    prior["log(E0)"] = gv.log(gv.gvar(mean[0], widths[0]))
    for state in range(1, n_states):
        gap = mean[state] - mean[state - 1]
        prior[f"log(dE{state})"] = gv.log(gv.gvar(gap, np.hypot(widths[state], widths[state - 1])))
    for state in range(n_states):
        prior[f"z{state}"] = gv.gvar(mean[n_states + state], widths[n_states + state])
    fit_data = EnsembleData(None, resample, list(samples), ["t"], {"t": times.tolist()})
    result = nonlinear_fit(
        ({"times": times, "n_states": n_states, "extent": extent}, fit_data),
        spectrum_fcn,
        prior,
        workers=workers,
        sample_error_mode=sample_error_mode,
        maxit=2000,
        _parallel=_parallel,
    )
    fitted_samples = []
    for parameters in result.samples:
        fitted_energies = [float(parameters["E0"])]
        for state in range(1, n_states):
            fitted_energies.append(fitted_energies[-1] + float(parameters[f"dE{state}"]))
        fitted_samples.append(np.asarray(fitted_energies))
    return fitted_samples, {
        "chi2": result.chi2,
        "dof": float(result.dof),
        "chi2_dof": result.chi2 / result.dof,
        "Q": result.Q,
        "logGBF": result.logGBF,
        "aic": result.chi2 + 4.0 * n_states,
    }


def _qda_two_point_source(records: list[EnsembleData], source: EnsembleData) -> tuple[EnsembleData | None, str]:
    """Prefer one compatible explicit two-point correlator, else use qDA z=0."""
    explicit = [value for value in records if value.attrs.get("correlator_type") == "two_point"]
    if not explicit:
        return None, "qda_z0"
    if len(explicit) != 1:
        raise ValueError("qDA fitting accepts at most one explicit two-point input")
    source_momentum = _optional_momentum(source, "sink_momentum")
    compatible = [
        value
        for value in explicit
        if value.dims == ["t"]
        and value.resample == source.resample
        and value.n_sample == source.n_sample
        and value.attrs.get("resample_id") == source.attrs.get("resample_id")
        and _optional_momentum(value, "sink_momentum") == source_momentum
    ]
    if len(compatible) != 1:
        raise ValueError("the explicit qDA two-point input is incompatible with the qDA correlator")
    return compatible[0], "external_2pt"


def fit_qda_samples(
    correlators: Mapping[str, EnsembleData],
    *,
    fit_scope: list[str] | tuple[str, ...],
    components: str,
    tmin: int,
    tmax: int,
    n_states: int,
    prior_width: float,
    svdcut: float,
    posterior_prior_error_scale: float,
    sample_error_mode: str = "covariance",
    workers: int = 1,
    tune_z: int | float | None = None,
    fit_samples: bool = True,
    show_progress: bool = False,
    _parallel: _ParallelPool | None = None,
) -> tuple[EnsembleData | None, dict[str, Any]]:
    """Fit one raw/ratio qDA scope pipeline and return normalized samples."""
    pipeline = parse_fit_scope(fit_scope)
    if not pipeline.is_qda:
        raise ValueError("qDA fitting requires a qda or qda_ratio atom")
    selected_components = {"real": ("re",), "imag": ("im",), "both": ("re", "im")}.get(components)
    if selected_components is None:
        raise ValueError("components must be real, imag, or both")
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ValueError("qDA fitting requires a positive state count")
    if not np.isfinite(prior_width) or prior_width <= 0:
        raise ValueError("qDA prior_width must be finite and positive")
    if fit_samples and tune_z is not None:
        raise ValueError("tune_z is only valid for sample-average qDA tuning")
    if not fit_samples and tune_z is None:
        raise ValueError("sample-average qDA tuning requires tune_z")
    records = [value for value in correlators.values() if isinstance(value, EnsembleData)]
    sources = [value for value in records if value.attrs.get("correlator_type") == "qda"]
    if len(sources) != 1 or sources[0].dims != ["t", "z"]:
        raise ValueError("qDA fitting requires exactly one qDA correlator with dimensions ['t', 'z']")
    source = sources[0]
    if source.ensemble is None:
        raise ValueError("qDA fitting requires the temporal extent")
    extent = int(source.ensemble.L_t)
    t = np.asarray(source.coords["t"], dtype=float)
    z = np.asarray(source.coords["z"], dtype=float)
    selected = np.flatnonzero((t >= tmin) & (t < tmax))
    if selected.size < 2 * n_states:
        raise ValueError("qDA fit window must contain at least 2*n_states times")
    source_values = np.asarray(source.values)
    two_point, denominator_kind = _qda_two_point_source(records, source)
    if two_point is None:
        origin = np.flatnonzero(np.isclose(z, 0.0, rtol=0.0, atol=1e-12))
        if origin.size != 1:
            raise ValueError("qDA fitting without an explicit two-point input requires one unique z=0 coordinate")
        origin_index: int | None = int(origin[0])
        denominator_values = source_values[:, :, origin_index]
    else:
        origin_index = None
        pt2_t = np.asarray(two_point.coords["t"], dtype=float)
        indices = []
        for value in t:
            matches = np.flatnonzero(np.isclose(pt2_t, value, rtol=0.0, atol=1e-12))
            if matches.size != 1:
                raise ValueError("explicit qDA two-point input must cover every qDA time exactly once")
            indices.append(int(matches[0]))
        denominator_values = np.asarray(two_point.values)[:, indices]
    if np.any(denominator_values[:, selected] == 0):
        raise ValueError("qDA denominator contains zero values in the fit window")
    raw_scale = _qda_correlator_rescale(np.real(denominator_values[:, selected]))
    plot_upper = float(extent) / 2.0
    plot_selected = np.flatnonzero((t >= 0.0) & (t <= plot_upper))
    plot_selected = plot_selected[np.all(denominator_values[:, plot_selected] != 0, axis=0)]
    if plot_selected.size == 0:
        plot_selected = selected
    window_ratios = source_values[:, selected, :] / denominator_values[:, selected, None]
    plot_ratios = source_values[:, plot_selected, :] / denominator_values[:, plot_selected, None]
    output_values = np.ones((source.n_sample, len(z)), dtype=complex) if fit_samples else None
    if tune_z is None:
        z_indices = list(range(len(z)))
        if origin_index is not None:
            z_indices.remove(origin_index)
    else:
        matches = np.flatnonzero(np.isclose(z, float(tune_z), rtol=0.0, atol=1e-12))
        if matches.size != 1 or (origin_index is not None and int(matches[0]) == origin_index):
            raise ValueError("tune_z must name one fitted non-denominator qDA separation")
        z_indices = [int(matches[0])]

    fit_metrics: list[dict[str, Any]] = []
    sample_failures: list[dict[str, Any]] = []
    parallel = _parallel or _ParallelPool(min(workers, source.n_sample))
    try:
        fit_indices = track(z_indices, label="qDA fits", unit="z", enabled=fit_samples and show_progress)
        for z_index in fit_indices:
            previous_posterior: Mapping[str, Any] | None = None
            stage_diagnostics: list[dict[str, Any]] = []
            result = None
            fit_prior = None
            observations = None
            for stage_index, atoms in enumerate(pipeline.stages):
                x = {
                    "n_states": n_states,
                    "extent": extent,
                    "form": "Breit",
                    "atoms": atoms,
                    "components": selected_components,
                    "family": "qda",
                    "denominator_kind": denominator_kind,
                    "pt2_t": t[selected],
                    "qda_t": t[selected],
                }
                pieces: list[np.ndarray] = []
                for atom in atoms:
                    if atom == "2pt":
                        pieces.append(np.real(denominator_values[:, selected]) * raw_scale)
                    elif atom in {"qda", "qda_ratio"}:
                        selected_values = (
                            source_values[:, selected, z_index] if atom == "qda" else window_ratios[:, :, z_index]
                        )
                        for component in selected_components:
                            values = np.real(selected_values) if component == "re" else np.imag(selected_values)
                            pieces.append(values * (raw_scale if atom == "qda" else 1.0))
                observations = np.concatenate(pieces, axis=1)
                fit_data = EnsembleData(
                    source.ensemble,
                    source.resample,
                    list(observations),
                    ["observation"],
                    {"observation": list(range(observations.shape[1]))},
                )
                fit_prior = matrix_element_prior(
                    n_states,
                    form="Breit",
                    scope=atoms,
                    components=selected_components,
                    width_scale=prior_width,
                    denominator_kind=denominator_kind,
                )
                if previous_posterior is not None:
                    for key in fit_prior:
                        if key in previous_posterior:
                            value = previous_posterior[key]
                            fit_prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * posterior_prior_error_scale)
                is_final = stage_index == len(pipeline.stages) - 1
                result = nonlinear_fit(
                    (x, fit_data),
                    matrix_element_fcn,
                    fit_prior,
                    workers=workers,
                    sample_prior_scale=(
                        None if is_final and atoms == ("qda_ratio",) else posterior_prior_error_scale * prior_width
                    ),
                    sample_error_mode=sample_error_mode,
                    mode="resamples" if is_final and fit_samples else "center",
                    tolerate_sample_failures=is_final and fit_samples,
                    capture_sample_posteriors=(0,) if is_final and fit_samples else (),
                    _parallel=parallel if is_final else None,
                    svdcut=svdcut,
                    maxit=10000,
                )
                previous_posterior = result.p
                stage_diagnostics.append(
                    {
                        "stage": "+".join(atoms),
                        "chi2": result.chi2,
                        "dof": result.dof,
                        "chi2_dof": result.chi2 / result.dof,
                        "Q": result.Q,
                        "logGBF": result.logGBF,
                        "n_data": int(observations.shape[1]),
                    }
                )
            if result is None or fit_prior is None or observations is None:
                raise RuntimeError("fit_scope produced no qDA fit stage")
            denominator_key = "zprime0" if denominator_kind == "qda_z0" else "z0"
            energy_samples = [float(p["E0"]) if p is not None else None for p in result.samples] if fit_samples else []
            finite_energy_samples = [value for value in energy_samples if value is not None]
            energy_sdev = (
                float(
                    gv.sdev(
                        EnsembleData(
                            source.ensemble,
                            source.resample,
                            [[value] for value in finite_energy_samples],
                            ["energy"],
                            {"energy": [0]},
                        ).average(sample_error_mode)[0]
                    )
                )
                if len(finite_energy_samples) == len(energy_samples) and len(energy_samples) > 1
                else None
            )
            sample0_plot = None
            if fit_samples and result.sample_posteriors and result.sample_posteriors[0] is not None:
                posterior = result.sample_posteriors[0]
                fit_t = (
                    np.asarray([float(t[selected][0]) - 0.5, float(t[selected][-1]) + 0.5])
                    if n_states == 1
                    else np.linspace(float(t[selected][0]) - 0.5, float(t[selected][-1]) + 0.5, 200)
                )
                fit_ratio = qda_ratio_fcn(
                    {
                        "times": fit_t,
                        "n_states": n_states,
                        "extent": extent,
                        "denominator_kind": denominator_kind,
                        "components": selected_components,
                    },
                    posterior,
                )
                plots = []
                for component_index, component in enumerate(selected_components):
                    plot_samples = (
                        np.real(plot_ratios[:, :, z_index])
                        if component == "re"
                        else np.imag(plot_ratios[:, :, z_index])
                    )
                    average = EnsembleData(
                        source.ensemble,
                        source.resample,
                        list(plot_samples),
                        ["t"],
                        {"t": t[plot_selected].tolist()},
                    ).average(sample_error_mode)
                    curve = fit_ratio[component_index * fit_t.size : (component_index + 1) * fit_t.size]
                    plateau = posterior[f"O00_{component}"] / posterior[denominator_key]
                    fit_mean, fit_sdev = _gvar_payload(curve)
                    plots.append(
                        {
                            "kind": "qda_ratio",
                            "component": component,
                            "series": [
                                {
                                    "label": "qDA ratio",
                                    "x": t[plot_selected].astype(float).tolist(),
                                    "y": np.asarray(plot_samples[0], dtype=float).tolist(),
                                    "yerr": np.asarray(gv.sdev(average), dtype=float).tolist(),
                                    "fit_x": fit_t.tolist(),
                                    "fit_mean": fit_mean,
                                    "fit_sdev": fit_sdev,
                                }
                            ],
                            "plateau_mean": float(gv.mean(plateau)),
                            "plateau_sdev": float(gv.sdev(plateau)),
                        }
                    )
                sample0_plot = {"z": float(z[z_index]), "plots": plots}
            fit_metrics.append(
                {
                    "z": float(z[z_index]),
                    "chi2": result.chi2,
                    "dof": result.dof,
                    "chi2_dof": result.chi2 / result.dof,
                    "Q": result.Q,
                    "logGBF": result.logGBF,
                    "E0": float(result.pmean["E0"]),
                    "E0_sdev": energy_sdev,
                    "E0_samples": energy_samples,
                    "n_data": int(observations.shape[1]),
                    "n_params": sum(int(np.size(gv.mean(value))) for value in fit_prior.values()),
                    "sample_diagnostics": _sample_diagnostic_records(result) if fit_samples else [],
                    "sample0_plot": sample0_plot,
                    "stages": stage_diagnostics,
                }
            )
            if output_values is not None:
                for sample_index, parameters in enumerate(result.samples):
                    if parameters is None:
                        output_values[sample_index, z_index] = np.nan + 1j * np.nan
                        sample_failures.append({"z": float(z[z_index]), "sample": sample_index})
                        continue
                    real = (
                        float(parameters["O00_re"] / parameters[denominator_key])
                        if "re" in selected_components
                        else 0.0
                    )
                    imag = (
                        float(parameters["O00_im"] / parameters[denominator_key])
                        if "im" in selected_components
                        else 0.0
                    )
                    output_values[sample_index, z_index] = real + 1j * imag
    finally:
        if _parallel is None:
            parallel.close()
    diagnostics: dict[str, Any] = {
        "fit_scope": pipeline.as_list(),
        "denominator_kind": denominator_kind,
        "correlator_rescale": raw_scale,
        "min_Q": min(record["Q"] for record in fit_metrics),
        "max_chi2_dof": max(record["chi2_dof"] for record in fit_metrics),
        "n_failed_samples": len(sample_failures),
        "sample_failures": sample_failures,
        "fits": fit_metrics,
        "n_data": int(fit_metrics[0]["n_data"]),
        "n_params": int(fit_metrics[0]["n_params"]),
    }
    if tune_z is not None:
        fit = fit_metrics[0]
        quality_keys = ("z", "chi2", "dof", "chi2_dof", "Q", "logGBF", "n_data", "n_params")
        diagnostics.update({key: fit[key] for key in quality_keys})
        diagnostics["tune_z"] = diagnostics.pop("z")
    if output_values is None:
        return None, diagnostics
    attrs = dict(source.attrs)
    attrs.update(
        {
            "method": "lsqfit",
            "fit_scope": json.dumps(pipeline.as_list()),
            "n_states": n_states,
            "tmin": tmin,
            "tmax": tmax,
            "denominator_kind": denominator_kind,
            "sample_error_mode": sample_error_mode,
        }
    )
    published_values = (
        output_values.real if components == "real" else output_values.imag if components == "imag" else output_values
    )
    output = EnsembleData(
        source.ensemble,
        source.resample,
        list(published_values),
        ["z"],
        {"z": z.tolist()},
        attrs=attrs,
        name="bare_matrix_element",
    )
    return output, diagnostics


def matrix_element_samples(
    correlators: dict[str, object],
    *,
    method: str,
    tmin: int,
    tmax: int,
    tau_min: int | None,
    lsqfit: Mapping[str, Any] | None = None,
    sample_error_mode: str = "covariance",
    workers: int = 1,
    tune_z: int | float | None = None,
    fit_samples: bool = True,
    show_progress: bool = False,
    n_states: int = 1,
    prior_width: float = 1.0,
    fit_scope: list[str] | tuple[str, ...] = ("qda_ratio",),
    _parallel: _ParallelPool | None = None,
) -> tuple[np.ndarray | None, list[float], dict[str, Any]]:
    """Extract ratio/summation-style matrix-element samples from correlators.

    Three-point data are reduced using the declared coordinates, never by
    assuming an axis position. Ratio paths average ``C3/C2`` over
    the selected insertion window; summation fits the slope of the summed
    ratio versus ``tsep``. qDA coordinate-space two-point numerators are
    divided by the aligned ``z=0`` nonlocal denominator, then their real and
    imaginary ratios are fitted together on the selected window. One-state
    fits remain a constant plateau; multi-state fits use the periodic spectral
    ratio. Independent qDA fits use only the ratio, joint fits include the
    local denominator, and chained fits propagate a denominator-spectrum
    posterior to the ratio fit.
    """
    from lamet_agent.data import EnsembleData

    records = [value for value in correlators.values() if isinstance(value, EnsembleData)]
    source = next((value for value in records if value.attrs.get("correlator_type") in {"three_point", "qda"}), None)
    if source is None:
        raise ValueError("matrix-element fitting requires a three-point or qDA correlator")
    if source.attrs.get("correlator_type") == "qda":
        if method != "qda" or lsqfit is None:
            raise ValueError("qDA fitting requires method='qda' and lsqfit settings")
        component = {"re": "real", "im": "imag", "both": "both"}[str(lsqfit.get("component", "both"))]
        data, diagnostics = fit_qda_samples(
            {key: value for key, value in correlators.items() if isinstance(value, EnsembleData)},
            fit_scope=fit_scope,
            components=component,
            tmin=tmin,
            tmax=tmax,
            n_states=n_states,
            prior_width=prior_width,
            svdcut=float(lsqfit["svdcut"]),
            posterior_prior_error_scale=float(lsqfit["posterior_prior_error_scale"]),
            sample_error_mode=sample_error_mode,
            workers=workers,
            tune_z=tune_z,
            fit_samples=fit_samples,
            show_progress=show_progress,
            _parallel=_parallel,
        )
        return (
            np.asarray(data.values) if data is not None else None,
            [float(value) for value in source.coords["z"]],
            diagnostics,
        )
    if "tau" not in source.dims:
        array = np.asarray(source.values)
        physical_dims = [dim for dim in source.dims if dim in {"z", "x"}]
        if not physical_dims:
            return np.asarray([np.asarray(sample).reshape(-1).mean() for sample in array]), [0.0], {}
        z_dim = physical_dims[0]
        z_axis = source.array.dims.index(z_dim)
        moved = np.moveaxis(array, z_axis - 1, -1)
        if moved.ndim > 2:
            moved = moved.reshape(moved.shape[0], -1, moved.shape[-1]).mean(axis=1)
        return moved, [float(value) for value in source.coords[z_dim]], {}
    if "tsep" not in source.dims:
        authored_tsep = source.attrs.get("source_sink_separation")
        if authored_tsep is None:
            raise ValueError("three-point matrix elements require tsep or source_sink_separation")
        tsep = np.asarray([authored_tsep], dtype=float)
        tsep_axis = None
    else:
        tsep = np.asarray(source.coords["tsep"], dtype=float)
        tsep_axis = source.array.dims.index("tsep") - 1
    tau = np.asarray(source.coords["tau"], dtype=float)
    tau_axis = source.array.dims.index("tau") - 1
    z_dim = "z" if "z" in source.dims else None
    z_axis = source.array.dims.index(z_dim) - 1 if z_dim is not None else None
    two_point = next((value for value in records if value.attrs.get("correlator_type") == "two_point"), None)
    if two_point is None or "t" not in two_point.dims:
        raise ValueError("matrix-element ratio paths require a two-point correlator")
    c2_time = np.asarray(two_point.coords["t"], dtype=float)
    c2_axis = two_point.array.dims.index("t") - 1
    outputs: list[np.ndarray] = []
    used_tseps: list[float] = []
    for sample_index in range(source.n_sample):
        c3 = np.asarray(source.values[sample_index])
        if tsep_axis is not None:
            axes = [tsep_axis, tau_axis] + [axis for axis in range(c3.ndim) if axis not in {tsep_axis, tau_axis}]
            c3 = np.transpose(c3, axes)
            if z_axis is not None:
                c3 = np.moveaxis(c3, axes.index(z_axis), -1)
        else:
            c3 = np.expand_dims(c3, axis=0)
            shifted_tau_axis = tau_axis + 1
            axes = [0, shifted_tau_axis] + [axis for axis in range(c3.ndim) if axis not in {0, shifted_tau_axis}]
            c3 = np.transpose(c3, axes)
            if z_axis is not None:
                c3 = np.moveaxis(c3, axes.index(z_axis + 1), -1)
        if z_dim is None:
            c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).mean(axis=-1)
        else:
            c3 = c3.reshape(c3.shape[0], c3.shape[1], -1, c3.shape[-1]).mean(axis=2)
        c2 = np.asarray(two_point.values[sample_index])
        c2 = np.moveaxis(c2, c2_axis, 0)
        if c2.ndim > 1:
            c2 = c2.reshape(c2.shape[0], -1).mean(axis=1)
        series: list[np.ndarray] = []
        series_t: list[float] = []
        for tsep_index, tsep_value in enumerate(tsep):
            if tsep_value < tmin or tsep_value > tmax:
                continue
            tau_mask = (tau >= (tau_min if tau_min is not None else tmin)) & (
                tau <= tsep_value - (tau_min if tau_min is not None else 0)
            )
            if not np.any(tau_mask):
                continue
            c2_indices = np.where(np.isclose(c2_time, tsep_value, rtol=0.0, atol=1e-12))[0]
            if c2_indices.size != 1:
                raise ValueError(f"two-point data must contain exactly one entry at tsep={tsep_value}")
            if c2[c2_indices[0]] == 0:
                raise ValueError(f"two-point denominator is zero at tsep={tsep_value}")
            ratio_values = c3[tsep_index, tau_mask] / c2[c2_indices[0]]
            series.append(np.sum(ratio_values, axis=0) if method == "summation" else np.mean(ratio_values, axis=0))
            series_t.append(float(tsep_value))
        if not series:
            raise ValueError("matrix-element window has no usable three-point points")
        stacked = np.stack(series, axis=0)
        if method == "summation":
            if len(series_t) < 2:
                raise ValueError("summation requires at least two tsep values")
            coefficients = np.polyfit(np.asarray(series_t), stacked, 1)
            outputs.append(coefficients[0])
        else:
            outputs.append(np.mean(stacked, axis=0))
        used_tseps.extend(series_t)
    return np.asarray(outputs), ([float(value) for value in source.coords["z"]] if z_dim is not None else [0.0]), {}
