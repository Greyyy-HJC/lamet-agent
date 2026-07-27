"""Extrapolation stage tools."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Callable

import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import xarray as xr

from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FONT_SIZE, default_plot
from lamet_agent.core.resampling import sample_mean_and_sdev, sample_value_with_error


def _scaled_prior(pmean: gv.BufferDict, psdev: gv.BufferDict, scale: float) -> gv.BufferDict:
    prior = gv.BufferDict()
    for key in pmean:
        prior[key] = gv.gvar(float(pmean[key]), max(float(psdev[key]) * max(float(scale), 0.0), 1e-8))
    return prior


def _fit_extrapolation_one(design: np.ndarray, y_data, *, p0: np.ndarray | None = None, prior: gv.BufferDict | None = None) -> tuple[np.ndarray, gv.BufferDict | None, gv.BufferDict | None, bool, float, int, float, float]:
    n_param = int(design.shape[1])
    start = np.zeros(n_param, dtype=float) if p0 is None else np.asarray(p0, dtype=float)

    def fcn(x_design: np.ndarray, p: gv.BufferDict):
        coeff = [p[f"c{idx}"] for idx in range(n_param)]
        return sum(x_design[:, idx] * coeff[idx] for idx in range(n_param))

    fit_prior = prior
    if fit_prior is None:
        fit_prior = gv.BufferDict()
        for idx, value in enumerate(start):
            fit_prior[f"c{idx}"] = gv.gvar(float(value), 3.0)
    try:
        fit = lsqfit.nonlinear_fit(
            data=(design, y_data),
            fcn=fcn,
            p0={f"c{idx}": float(start[idx]) for idx in range(n_param)},
            prior=fit_prior,
            maxit=2000,
            svdcut=1e-12,
            fitter="scipy_least_squares",
        )
        params = np.asarray([float(fit.pmean[f"c{idx}"]) for idx in range(n_param)], dtype=float)
        return params, fit.pmean, fit.psdev, bool(np.isfinite(fit.chi2)), float(fit.chi2), int(fit.dof), float(fit.Q), float(fit.logGBF)
    except (FloatingPointError, RuntimeError, ValueError, OverflowError, AssertionError):
        return start, None, None, False, float("inf"), max(1, int(design.shape[0] - n_param)), 0.0, float("-inf")


def _save_fit_info(path: Path, *, x: np.ndarray, parameter_labels: list[str], fit_params: np.ndarray, fit_chi2: np.ndarray, fit_dof: np.ndarray, fit_q: np.ndarray, fit_log_gbf: np.ndarray, mean_fit_params: np.ndarray, mean_fit_chi2: np.ndarray, mean_fit_dof: np.ndarray, mean_fit_q: np.ndarray, attrs: dict[str, str]) -> None:
    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    mean_fit_chi2_dof = mean_fit_chi2 / np.maximum(mean_fit_dof, 1)
    dataset = xr.Dataset(
        {
            "fit_params": (("resample", "x", "parameter"), fit_params),
            "fit_chi2": (("resample", "x"), fit_chi2),
            "fit_dof": (("resample", "x"), fit_dof),
            "fit_chi2_dof": (("resample", "x"), fit_chi2_dof),
            "fit_q": (("resample", "x"), fit_q),
            "fit_log_gbf": (("resample", "x"), fit_log_gbf),
            "mean_fit_params": (("x", "parameter"), mean_fit_params),
            "mean_fit_chi2": (("x",), mean_fit_chi2),
            "mean_fit_dof": (("x",), mean_fit_dof),
            "mean_fit_chi2_dof": (("x",), mean_fit_chi2_dof),
            "mean_fit_q": (("x",), mean_fit_q),
            "fit_chi2_dof_center": (("x",), np.mean(fit_chi2_dof, axis=0)),
        },
        coords={"resample": list(range(fit_params.shape[0])), "x": x.tolist(), "parameter": parameter_labels},
        attrs=attrs,
    )
    dataset.to_netcdf(path, format="NETCDF4")


def _sample_batches(n_samples: int, workers: int) -> list[list[int]]:
    return [batch.tolist() for batch in np.array_split(np.arange(n_samples), min(int(workers), int(n_samples))) if batch.size]


def _fit_extrapolation_sample_batch(payload: bytes, sample_indices: list[int]) -> list[dict[str, Any]]:
    context = gv.loads(payload)
    results: list[dict[str, Any]] = []
    for isample in sample_indices:
        sample_y_data = sample_value_with_error(
            context["samples"][:, isample],
            context["fit_samples"],
            mode=context["resample_mode"],
            sample_error_mode=context["sample_error_mode"],
        )
        params, _pmean, _psdev, ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(
            context["design"],
            sample_y_data,
            p0=context["mean_params"],
            prior=context["sample_prior"],
        )
        results.append(
            {
                "sample": int(isample),
                "params": params,
                "success": bool(ok),
                "chi2": float(chi2),
                "dof": int(dof),
                "q_value": float(q_value),
                "log_gbf": float(log_gbf),
            }
        )
    return results


def _fit_extrapolation_global_sample_batch(payload: bytes, sample_indices: list[int]) -> list[dict[str, Any]]:
    context = gv.loads(payload)
    results: list[dict[str, Any]] = []
    fit_samples = context["fit_samples"]
    for isample in sample_indices:
        values = context["samples"][:, isample, :].reshape(-1)
        sample_y_data = sample_value_with_error(
            values,
            fit_samples,
            mode=context["resample_mode"],
            sample_error_mode=context["sample_error_mode"],
        )
        params, _pmean, _psdev, ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(
            context["design"],
            sample_y_data,
            p0=context["mean_params"],
            prior=context["sample_prior"],
        )
        results.append(
            {
                "sample": int(isample),
                "params": params,
                "success": bool(ok),
                "chi2": float(chi2),
                "dof": int(dof),
                "q_value": float(q_value),
                "log_gbf": float(log_gbf),
            }
        )
    return results


def run_extrapolation(
    store: dict[str, Any],
    *,
    lightcone: str = "lightcone",
    allow_order_a: list[int] | None = None,
    allow_order_1overp: list[int] | None = None,
    allow_order_ap: list[int] | None = None,
    fitting_param_xdep: list[bool] | None = None,
    pdep_gev: list[float] | None = None,
    sample_error_mode: str = "covariance",
    posterior_prior_error_scale: float = 3.0,
    workers: int = 1,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "extrapolated_distribution",
) -> dict[str, Any]:
    """Fit matched light-cone data to the IMF and/or continuum limit."""
    workers = max(1, int(workers))
    inputs = [
        item if isinstance(item, EnsembleData) else EnsembleData.from_netcdf(item.path)
        for item in list(store.get(lightcone, []))
    ]
    ensembles = {str(data.attrs.get("ensemble") or (data.ensemble.id if data.ensemble else "")) for data in inputs}
    momenta_gev = {float(data.attrs.get("momentum_gev")) for data in inputs}
    stage_dir = Path(artifacts_dir or ".")
    stem = Path(save_path) if save_path is not None else stage_dir / "extrapolation"
    stem.parent.mkdir(parents=True, exist_ok=True)
    is_systematics = stem.parent.name == "sym"

    if len(ensembles) == 1 and len(momenta_gev) == 1:
        warning = "Input from perturbative_matching cannot have only one ensemble and one momentum."
        result = {"out": out, "warning": warning, "mode": "insufficient_input", "n_inputs": len(inputs)}
        store["output"] = result
        return result

    x = np.asarray(inputs[0].coords["x"], dtype=float)
    n_sample = min(data.n_sample for data in inputs)
    input_samples = []
    for data in inputs:
        x_in = np.asarray(data.coords["x"], dtype=float)
        values = np.asarray(data.values, dtype=float)[:n_sample]
        if not np.array_equal(x_in, x):
            values = np.asarray([np.interp(x, x_in, sample) for sample in values], dtype=float)
        input_samples.append(values)
    samples = np.asarray(input_samples, dtype=float)

    use_a = len(ensembles) > 1
    use_p = any(
        len(
            {
                float(data.attrs.get("momentum_gev"))
                for data in inputs
                if str(data.attrs.get("ensemble") or (data.ensemble.id if data.ensemble else "")) == ensemble
            }
        )
        > 1
        for ensemble in ensembles
    )
    mode = "IMF+Continuum Extrapolation" if use_a and use_p else "IMF Extrapolation" if use_p else "Continuum Extrapolation"

    allow_order_a = [2] if allow_order_a is None else [int(value) for value in allow_order_a]
    allow_order_1overp = [2] if allow_order_1overp is None else [int(value) for value in allow_order_1overp]
    allow_order_ap = [] if allow_order_ap is None else [int(value) for value in allow_order_ap]
    fitting_param_xdep = [False, True, False] if fitting_param_xdep is None else [bool(value) for value in fitting_param_xdep]
    a_xdep = bool(fitting_param_xdep[0]) if fitting_param_xdep else True
    p_xdep = bool(fitting_param_xdep[1]) if len(fitting_param_xdep) > 1 else True
    include_ap = bool(fitting_param_xdep[2]) if len(fitting_param_xdep) > 2 else False
    a_powers = allow_order_a if use_a else []
    p_powers = allow_order_1overp if use_p else []
    ap_powers = allow_order_ap if include_ap and use_a and use_p else []
    rows = []
    for data in inputs:
        a = float(data.attrs["lattice_spacing_fm"])
        p = float(data.attrs["momentum_gev"])
        row = [1.0]
        row.extend(a**i for i in a_powers)
        row.extend(1.0 / p**power for power in p_powers)
        row.extend((a * p) ** power for power in ap_powers)
        rows.append(row)
    design = np.asarray(rows, dtype=float)
    parameter_labels = [
        "h0",
        *(f"c_a_{power}" for power in a_powers),
        *(f"c_p_{power}" for power in p_powers),
        *(f"c_ap_{power}" for power in ap_powers),
    ]
    n_param = int(design.shape[1])
    coeff_samples = np.empty((n_sample, n_param, len(x)), dtype=float)
    fit_chi2 = np.empty((n_sample, len(x)), dtype=float)
    fit_dof = np.empty((n_sample, len(x)), dtype=int)
    fit_q = np.empty((n_sample, len(x)), dtype=float)
    fit_log_gbf = np.empty((n_sample, len(x)), dtype=float)
    mean_fit_params = np.empty((len(x), n_param), dtype=float)
    mean_fit_chi2 = np.empty(len(x), dtype=float)
    mean_fit_dof = np.empty(len(x), dtype=int)
    mean_fit_q = np.empty(len(x), dtype=float)
    mean_fit_log_gbf = np.empty(len(x), dtype=float)
    sample_executor = ProcessPoolExecutor(max_workers=min(workers, n_sample)) if workers > 1 else None
    try:
        if a_xdep and p_xdep:
            with tqdm(total=n_sample * len(x), desc="extrapolation sample fits", leave=False) as sample_progress:
                for ix in range(len(x)):
                    fit_samples = samples[:, :, ix].T
                    mean, _sdev = sample_mean_and_sdev(fit_samples, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
                    mean_y_data = sample_value_with_error(mean, fit_samples, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
                    mean_params, mean_pmean, mean_psdev, mean_ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(design, mean_y_data)
                    sample_prior = None
                    if mean_ok and mean_pmean is not None and mean_psdev is not None:
                        sample_prior = _scaled_prior(mean_pmean, mean_psdev, posterior_prior_error_scale)
                        mean_params, _pmean, _psdev, mean_ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(
                            design, mean_y_data, p0=mean_params, prior=sample_prior
                        )
                    mean_fit_params[ix] = mean_params
                    mean_fit_chi2[ix] = chi2
                    mean_fit_dof[ix] = dof
                    mean_fit_q[ix] = q_value
                    mean_fit_log_gbf[ix] = log_gbf
                    payload = gv.dumps(
                        {
                            "design": design,
                            "samples": samples[:, :, ix],
                            "fit_samples": fit_samples,
                            "resample_mode": inputs[0].resample,
                            "sample_error_mode": sample_error_mode,
                            "mean_params": mean_params,
                            "sample_prior": sample_prior,
                        }
                    )
                    batches = _sample_batches(n_sample, workers)
                    if sample_executor is None:
                        sample_results = _fit_extrapolation_sample_batch(payload, batches[0])
                    else:
                        futures = [sample_executor.submit(_fit_extrapolation_sample_batch, payload, batch) for batch in batches]
                        sample_results = []
                        for future in as_completed(futures):
                            batch_results = future.result()
                            sample_results.extend(batch_results)
                            sample_progress.update(len(batch_results))
                    if sample_executor is None:
                        sample_progress.update(len(sample_results))
                    for item in sorted(sample_results, key=lambda value: value["sample"]):
                        isample = int(item["sample"])
                        params = np.asarray(item["params"], dtype=float)
                        chi2 = float(item["chi2"])
                        dof = int(item["dof"])
                        q_value = float(item["q_value"])
                        log_gbf = float(item["log_gbf"])
                        if not item["success"]:
                            params = mean_params
                            chi2 = mean_fit_chi2[ix]
                            dof = mean_fit_dof[ix]
                            q_value = mean_fit_q[ix]
                            log_gbf = mean_fit_log_gbf[ix]
                        coeff_samples[isample, :, ix] = params
                        fit_chi2[isample, ix] = chi2
                        fit_dof[isample, ix] = dof
                        fit_q[isample, ix] = q_value
                        fit_log_gbf[isample, ix] = log_gbf
        else:
            n_input = len(inputs)
            internal_labels = [("h0", None, ix) for ix in range(len(x))]
            internal_labels.extend(("a", index, ix if a_xdep else None) for index in range(len(a_powers)) for ix in (range(len(x)) if a_xdep else [None]))
            internal_labels.extend(("p", index, ix if p_xdep else None) for index in range(len(p_powers)) for ix in (range(len(x)) if p_xdep else [None]))
            internal_labels.extend(("ap", index, ix) for index in range(len(ap_powers)) for ix in range(len(x)))
            global_design = np.zeros((n_input * len(x), len(internal_labels)), dtype=float)
            for input_index in range(n_input):
                for ix in range(len(x)):
                    row_index = input_index * len(x) + ix
                    for column, (kind, power_index, x_index) in enumerate(internal_labels):
                        if kind == "h0":
                            global_design[row_index, column] = 1.0 if x_index == ix else 0.0
                        elif kind == "a" and (x_index is None or x_index == ix):
                            global_design[row_index, column] = design[input_index, 1 + int(power_index)]
                        elif kind == "p" and (x_index is None or x_index == ix):
                            global_design[row_index, column] = design[input_index, 1 + len(a_powers) + int(power_index)]
                        elif kind == "ap" and x_index == ix:
                            global_design[row_index, column] = design[input_index, 1 + len(a_powers) + len(p_powers) + int(power_index)]
            fit_samples = np.moveaxis(samples, 1, 0).reshape(n_sample, -1)
            mean, _sdev = sample_mean_and_sdev(fit_samples, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
            mean_y_data = sample_value_with_error(mean, fit_samples, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
            mean_params, mean_pmean, mean_psdev, mean_ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(global_design, mean_y_data)
            sample_prior = None
            if mean_ok and mean_pmean is not None and mean_psdev is not None:
                sample_prior = _scaled_prior(mean_pmean, mean_psdev, posterior_prior_error_scale)
                mean_params, _pmean, _psdev, mean_ok, chi2, dof, q_value, log_gbf = _fit_extrapolation_one(
                    global_design, mean_y_data, p0=mean_params, prior=sample_prior
                )
            x_iterator = range(len(x))
            if len(x) > 1:
                x_iterator = tqdm(x_iterator, desc="extrapolation x diagnostics", leave=False)
            fit_samples_by_x = [samples[:, :, ix].T for ix in range(len(x))]
            for ix in x_iterator:
                external = [mean_params[ix]]
                for index in range(len(a_powers)):
                    target = ("a", index, ix if a_xdep else None)
                    external.append(mean_params[internal_labels.index(target)])
                for index in range(len(p_powers)):
                    target = ("p", index, ix if p_xdep else None)
                    external.append(mean_params[internal_labels.index(target)])
                for index in range(len(ap_powers)):
                    external.append(mean_params[internal_labels.index(("ap", index, ix))])
                external_array = np.asarray(external, dtype=float)
                fit_samples_x = fit_samples_by_x[ix]
                mean_x, _sdev_x = sample_mean_and_sdev(fit_samples_x, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
                mean_y_data_x = sample_value_with_error(mean_x, fit_samples_x, mode=inputs[0].resample, sample_error_mode=sample_error_mode)
                residual = gv.mean(mean_y_data_x) - design @ external_array
                cov = gv.evalcov(mean_y_data_x)
                mean_fit_params[ix] = external_array
                mean_fit_chi2[ix] = float(residual @ np.linalg.pinv(cov) @ residual)
                mean_fit_dof[ix] = max(1, int(n_input - n_param))
                mean_fit_q[ix] = q_value
                mean_fit_log_gbf[ix] = log_gbf
            payload = gv.dumps(
                {
                    "design": global_design,
                    "samples": samples,
                    "fit_samples": fit_samples,
                    "resample_mode": inputs[0].resample,
                    "sample_error_mode": sample_error_mode,
                    "mean_params": mean_params,
                    "sample_prior": sample_prior,
                }
            )
            batches = _sample_batches(n_sample, workers)
            if sample_executor is None:
                sample_results = _fit_extrapolation_global_sample_batch(payload, batches[0])
            else:
                futures = [sample_executor.submit(_fit_extrapolation_global_sample_batch, payload, batch) for batch in batches]
                sample_results = []
                with tqdm(total=n_sample, desc="extrapolation sample fits", leave=False) as sample_progress:
                    for future in as_completed(futures):
                        batch_results = future.result()
                        sample_results.extend(batch_results)
                        sample_progress.update(len(batch_results))
            sample_iterator = sorted(sample_results, key=lambda value: value["sample"])
            for item in sample_iterator:
                isample = int(item["sample"])
                params = np.asarray(item["params"], dtype=float)
                if not item["success"]:
                    params = mean_params
                for ix in range(len(x)):
                    external = [params[ix]]
                    for index in range(len(a_powers)):
                        external.append(params[internal_labels.index(("a", index, ix if a_xdep else None))])
                    for index in range(len(p_powers)):
                        external.append(params[internal_labels.index(("p", index, ix if p_xdep else None))])
                    for index in range(len(ap_powers)):
                        external.append(params[internal_labels.index(("ap", index, ix))])
                    external_array = np.asarray(external, dtype=float)
                    fit_samples_x = fit_samples_by_x[ix]
                    sample_y_data_x = sample_value_with_error(
                        samples[:, isample, ix],
                        fit_samples_x,
                        mode=inputs[0].resample,
                        sample_error_mode=sample_error_mode,
                    )
                    residual = gv.mean(sample_y_data_x) - design @ external_array
                    cov = gv.evalcov(sample_y_data_x)
                    coeff_samples[isample, :, ix] = external_array
                    fit_chi2[isample, ix] = float(residual @ np.linalg.pinv(cov) @ residual) if item["success"] else mean_fit_chi2[ix]
                    fit_dof[isample, ix] = max(1, int(n_input - n_param)) if item["success"] else mean_fit_dof[ix]
                    fit_q[isample, ix] = float(item["q_value"]) if item["success"] else mean_fit_q[ix]
                    fit_log_gbf[isample, ix] = float(item["log_gbf"]) if item["success"] else mean_fit_log_gbf[ix]
    finally:
        if sample_executor is not None:
            sample_executor.shutdown()
    extrapolated = coeff_samples[:, 0, :]
    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    finite_fit_chi2_dof = fit_chi2_dof[np.isfinite(fit_chi2_dof)]
    chi2_dof = float(np.mean(mean_fit_chi2 / np.maximum(mean_fit_dof, 1)))
    chi2_dof_min = float(np.min(finite_fit_chi2_dof)) if finite_fit_chi2_dof.size else float("nan")
    chi2_dof_max = float(np.max(finite_fit_chi2_dof)) if finite_fit_chi2_dof.size else float("nan")
    ca_label = "c_a_i(x)" if a_xdep else "c_a_i"
    cp_label = "c_p_j(x)" if p_xdep else "c_p_j"
    cap_label = "c_ap_k(x)"

    attrs = {
        "mode": mode,
        "model": f"h(x,Pz,a)=h0(x)+sum_i {ca_label} a^i+sum_j {cp_label}/Pz^j" + (f"+sum_k {cap_label} a^k Pz^k" if ap_powers else ""),
        "fit_chi2_dof_mean": str(chi2_dof),
        "fit_chi2_dof_min": str(chi2_dof_min),
        "fit_chi2_dof_max": str(chi2_dof_max),
        "fit_chi2_dof_source": "sample_level_fits",
        "allow_order_a": json.dumps(a_powers),
        "allow_order_1overp": json.dumps(p_powers),
        "allow_order_ap": json.dumps(ap_powers),
        "fitting_param_xdep": json.dumps([a_xdep, p_xdep, include_ap]),
        "input_jobs": ",".join(str(data.attrs.get("job_id", "")) for data in inputs),
        "input_ensembles": ",".join(str(data.attrs.get("ensemble") or (data.ensemble.id if data.ensemble else "")) for data in inputs),
        "input_momenta_gev": ",".join(f"{float(data.attrs['momentum_gev']):.12g}" for data in inputs),
        "parameter_labels": json.dumps(parameter_labels),
        "design_matrix": json.dumps(design.tolist()),
        "resample_mode": inputs[0].resample,
        "sample_error_mode": str(sample_error_mode),
        "posterior_prior_error_scale": str(float(posterior_prior_error_scale)),
        "workers": str(int(workers)),
    }
    output = EnsembleData(
        EnsembleInfo("", mode, 0.0, 0.0, 0, 0, 0.0),
        inputs[0].resample,
        [extrapolated[i] for i in range(n_sample)],
        dims=("x",),
        coords={"x": x.tolist()},
        attrs=attrs,
        name="extrapolated_distribution",
    )
    artifact = stem.with_suffix(".nc")
    coords = {"resample": list(range(n_sample)), "x": x.tolist()}
    data_vars = {
        "extrapolated_distribution": (("resample", "x"), extrapolated),
    }
    offset = 1
    for index, power in enumerate(a_powers):
        data_vars[f"c_a_{power}"] = (("resample", "x"), coeff_samples[:, offset + index, :])
    offset += len(a_powers)
    for index, power in enumerate(p_powers):
        data_vars[f"c_p_{power}"] = (("resample", "x"), coeff_samples[:, offset + index, :])
    offset += len(p_powers)
    for index, power in enumerate(ap_powers):
        data_vars[f"c_ap_{power}"] = (("resample", "x"), coeff_samples[:, offset + index, :])
    dataset = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    dataset["extrapolated_distribution"].attrs.update(attrs)
    dataset["extrapolated_distribution"].attrs["ensemble"] = json.dumps(output.ensemble._asdict())
    dataset["extrapolated_distribution"].attrs["resample"] = output.resample
    dataset.to_netcdf(artifact, format="NETCDF4", auto_complex=True)
    fit_info_artifact = stem.with_name(f"{stem.name}_fit_info").with_suffix(".nc")
    _save_fit_info(
        fit_info_artifact,
        x=x,
        parameter_labels=parameter_labels,
        fit_params=np.moveaxis(coeff_samples, 1, 2),
        fit_chi2=fit_chi2,
        fit_dof=fit_dof,
        fit_q=fit_q,
        fit_log_gbf=fit_log_gbf,
        mean_fit_params=mean_fit_params,
        mean_fit_chi2=mean_fit_chi2,
        mean_fit_dof=mean_fit_dof,
        mean_fit_q=mean_fit_q,
        attrs=attrs,
    )
    store[out] = output
    store["output"] = output

    fig, ax = default_plot()
    for index, data in enumerate(inputs):
        xi = np.asarray(data.coords["x"], dtype=float)
        mean = np.asarray(data.mean, dtype=float)
        err = np.asarray(data.sdev, dtype=float)
        a = float(data.attrs["lattice_spacing_fm"])
        p = float(data.attrs["momentum_gev"])
        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        ax.plot(xi, mean, color=color, linewidth=1.0, label=f"a={a:.2f} fm, p={p:.2f} GeV")
        ax.fill_between(xi, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
    mean = np.asarray(output.mean, dtype=float)
    err = np.asarray(output.sdev, dtype=float)
    if mode == "IMF+Continuum Extrapolation":
        mode_label = r"$a\rightarrow0,\ p\rightarrow\infty$"
    elif mode == "IMF Extrapolation":
        mode_label = r"$p\rightarrow\infty$"
    else:
        mode_label = r"$a\rightarrow0$"
    ax.plot(x, mean, color="0.45", linewidth=1.4, label=mode_label)
    ax.fill_between(x, mean - err, mean + err, color="0.75", alpha=0.60, linewidth=0)
    ax.set_xlabel(r"$x$", **FONT_SIZE)
    ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
    ax.set_title("Extrapolation", **FONT_SIZE)
    ax.text(
        0.03,
        0.95,
        rf"$\chi^2/\mathrm{{dof}}={chi2_dof:.3g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=12)
    fig.tight_layout()
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(pdf, bbox_inches="tight", transparent=True)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    chi2_pdf = chi2_svg = adep_pdf = adep_svg = pdep_pdf = pdep_svg = None
    if not is_systematics:
        chi2_plot_values = np.where(np.isfinite(fit_chi2_dof), fit_chi2_dof, np.nan)
        chi2_x_mean = np.nanmean(chi2_plot_values, axis=0)
        chi2_x_err = np.nanstd(chi2_plot_values, axis=0, ddof=1) if chi2_plot_values.shape[0] > 1 else np.zeros_like(chi2_x_mean)
        fig, ax = default_plot()
        ax.axhline(1.0, color="black", linewidth=1.0)
        ax.errorbar(x, chi2_x_mean, yerr=chi2_x_err, color="0.35", **ERRORBAR_STYLE)
        ax.set_xlabel(r"$x$", **FONT_SIZE)
        ax.set_ylabel(r"$\chi^2/\mathrm{dof}$", **FONT_SIZE)
        fig.tight_layout()
        chi2_pdf = stem.with_name("chi2_xdep").with_suffix(".pdf")
        chi2_svg = stem.with_name("chi2_xdep").with_suffix(".svg")
        fig.savefig(chi2_pdf, bbox_inches="tight", transparent=True)
        fig.savefig(chi2_svg, bbox_inches="tight")
        plt.close(fig)
        if use_a:
            fig, ax = default_plot()
            for index, a in enumerate(sorted({float(data.attrs["lattice_spacing_fm"]) for data in inputs})):
                row = [1.0, *(a**power for power in a_powers), *(0.0 for _power in p_powers), *(0.0 for _power in ap_powers)]
                values = np.einsum("p,spx->sx", np.asarray(row, dtype=float), coeff_samples)
                mean = np.mean(values, axis=0)
                err = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
                color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
                ax.plot(x, mean, color=color, linewidth=1.0, label=f"a={a:.2f} fm")
                ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, output.mean, color="0.45", linewidth=1.4, label=r"$a\rightarrow0$")
            ax.fill_between(x, output.mean - output.sdev, output.mean + output.sdev, color="0.75", alpha=0.60, linewidth=0)
            ax.set_xlabel(r"$x$", **FONT_SIZE)
            ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
            ax.set_title("Extrapolation, lattice spacing dependence", **FONT_SIZE)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=12)
            fig.tight_layout()
            adep_pdf = stem.with_name("extrapolate_adep").with_suffix(".pdf")
            adep_svg = stem.with_name("extrapolate_adep").with_suffix(".svg")
            fig.savefig(adep_pdf, bbox_inches="tight", transparent=True)
            fig.savefig(adep_svg, bbox_inches="tight")
            plt.close(fig)
        if use_p and pdep_gev:
            fig, ax = default_plot()
            for index, p in enumerate([float(value) for value in pdep_gev]):
                row = [1.0, *(0.0 for _power in a_powers), *(1.0 / p**power for power in p_powers), *(0.0 for _power in ap_powers)]
                values = np.einsum("p,spx->sx", np.asarray(row, dtype=float), coeff_samples)
                mean = np.mean(values, axis=0)
                err = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
                color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
                ax.plot(x, mean, color=color, linewidth=1.0, label=f"p={p:.2f} GeV")
                ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, output.mean, color="0.45", linewidth=1.4, label=r"$p\rightarrow\infty$")
            ax.fill_between(x, output.mean - output.sdev, output.mean + output.sdev, color="0.75", alpha=0.60, linewidth=0)
            ax.set_xlabel(r"$x$", **FONT_SIZE)
            ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
            ax.set_title("Extrapolation, momentum dependence", **FONT_SIZE)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=12)
            fig.tight_layout()
            pdep_pdf = stem.with_name("extrapolate_pdep").with_suffix(".pdf")
            pdep_svg = stem.with_name("extrapolate_pdep").with_suffix(".svg")
            fig.savefig(pdep_pdf, bbox_inches="tight", transparent=True)
            fig.savefig(pdep_svg, bbox_inches="tight")
            plt.close(fig)

    result = {
        "out": out,
        "mode": mode,
        "model": attrs["model"],
        "artifact": str(artifact),
        "fit_info_artifact": str(fit_info_artifact),
        "plot": str(pdf),
        "plot_image": str(svg),
        "n_inputs": len(inputs),
        "n_points": int(len(x)),
        "n_sample": int(n_sample),
        "n_parameters": int(coeff_samples.shape[1]),
        "chi2_dof": chi2_dof,
        "chi2_dof_min": chi2_dof_min,
        "chi2_dof_max": chi2_dof_max,
        "chi2_dof_source": "sample_level_fits",
        "allow_order_a": a_powers,
        "allow_order_1overp": p_powers,
        "allow_order_ap": ap_powers,
        "fitting_param_xdep": [a_xdep, p_xdep, include_ap],
        "use_lattice_spacing_dependence": use_a,
        "use_momentum_dependence": use_p,
        "pdep_gev": [float(value) for value in pdep_gev] if pdep_gev else [],
        "sample_error_mode": str(sample_error_mode),
        "posterior_prior_error_scale": float(posterior_prior_error_scale),
        "workers": int(workers),
    }
    if chi2_pdf is not None and chi2_svg is not None:
        result.update({"chi2_xdep_plot": str(chi2_pdf), "chi2_xdep_plot_image": str(chi2_svg)})
    if adep_pdf is not None and adep_svg is not None:
        result.update({"adep_plot": str(adep_pdf), "adep_plot_image": str(adep_svg)})
    if pdep_pdf is not None and pdep_svg is not None:
        result.update({"pdep_plot": str(pdep_pdf), "pdep_plot_image": str(pdep_svg)})
    return result


def run_systematics_budget(
    store: dict[str, Any],
    *,
    main: str = "main",
    zs: str = "zs",
    lambda_extrapolation: str = "lambda_extrapolation",
    lamet_scale: str = "lamet_scale",
    other_extrapolations: str = "other_extrapolations",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "systematics_budget",
) -> dict[str, Any]:
    """Build a Fig.10-style systematic-error budget from extrapolated outputs."""
    stage_dir = Path(artifacts_dir or ".")
    stem = Path(save_path) if save_path is not None else stage_dir / "systematics_budget"
    stem.parent.mkdir(parents=True, exist_ok=True)
    main_data = store[main]
    x = np.asarray(main_data.coords["x"], dtype=float)
    main_mean = np.asarray(main_data.mean, dtype=float)
    stat_sdev = np.asarray(main_data.sdev, dtype=float)
    components: dict[str, np.ndarray] = {}
    for key in (zs, lambda_extrapolation, lamet_scale, other_extrapolations):
        values = store.get(key, [])
        variants = values if isinstance(values, list) else ([] if values in {None, ""} else [values])
        if not variants:
            components[key] = np.zeros_like(main_mean)
            continue
        aligned_values = []
        for variant in variants:
            variant_x = np.asarray(variant.coords["x"], dtype=float)
            variant_mean = np.asarray(variant.mean, dtype=float)
            aligned_values.append(variant_mean if np.array_equal(variant_x, x) else np.interp(x, variant_x, variant_mean))
        aligned_stack = np.asarray(aligned_values, dtype=float)
        if aligned_stack.shape[0] == 1:
            components[key] = np.abs(aligned_stack[0] - main_mean)
        else:
            components[key] = np.max(aligned_stack, axis=0) - np.min(aligned_stack, axis=0)
    total_sys = np.sqrt(
        components[zs] ** 2
        + components[lambda_extrapolation] ** 2
        + components[lamet_scale] ** 2
        + components[other_extrapolations] ** 2
    )
    total_err = np.sqrt(stat_sdev**2 + total_sys**2)

    payload = {
        "x": np.asarray(x, dtype=float).tolist(),
        "stat_sdev": np.asarray(stat_sdev, dtype=float).tolist(),
        "zs": np.asarray(components[zs], dtype=float).tolist(),
        "lambda_extrapolation": np.asarray(components[lambda_extrapolation], dtype=float).tolist(),
        "lamet_scale": np.asarray(components[lamet_scale], dtype=float).tolist(),
        "other_extrapolations": np.asarray(components[other_extrapolations], dtype=float).tolist(),
        "total_systematic_error": np.asarray(total_sys, dtype=float).tolist(),
        "total_error": np.asarray(total_err, dtype=float).tolist(),
    }
    artifact = stem.with_suffix(".json")
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if np.count_nonzero((x >= 0.0) & (x <= 1.0)) >= 6:
        x_plot = np.arange(0.0, 1.0001, 0.1, dtype=float)
        x_plot = x_plot[(x_plot >= float(np.min(x))) & (x_plot <= float(np.max(x)))]
    else:
        x_plot = np.asarray(x, dtype=float)
    if x_plot.size == 0:
        x_plot = np.asarray(x, dtype=float)
    total_err_plot = np.interp(x_plot, x, total_err)
    total_sys_plot = np.interp(x_plot, x, total_sys)
    zs_plot = np.interp(x_plot, x, components[zs])
    lambda_plot = np.interp(x_plot, x, components[lambda_extrapolation])
    mu_plot = np.interp(x_plot, x, components[lamet_scale])
    other_plot = np.interp(x_plot, x, components[other_extrapolations])
    spacing = 0.1 if x_plot.size > 1 else 0.08
    group_width = 0.078 if x_plot.size > 1 else 0.062
    bar_width = group_width / 5.0
    fig, ax = default_plot()
    ax.bar(x_plot, total_err_plot, width=0.92 * group_width, color="0.82", label="total error", zorder=0)
    ax.bar(x_plot, total_sys_plot, width=0.72 * group_width, color="0.65", label="total systematic error", zorder=1)
    ax.bar(x_plot - 1.5 * bar_width, zs_plot, width=bar_width, color="tab:blue", label=r"sys : $z_s$ uncertainty", zorder=2)
    ax.bar(x_plot - 0.5 * bar_width, lambda_plot, width=bar_width, color="tab:orange", label=r"sys : $\lambda$ extrapolation", zorder=2)
    ax.bar(x_plot + 0.5 * bar_width, mu_plot, width=bar_width, color="tab:green", label="sys : LaMET scale", zorder=2)
    ax.bar(x_plot + 1.5 * bar_width, other_plot, width=bar_width, color="tab:red", label="sys : other extrapolations", zorder=2)
    ax.set_xlabel(r"$x$", **FONT_SIZE)
    ax.set_ylabel(r"$\Delta f(x)$", **FONT_SIZE)
    ax.set_title("Error Analysis", **FONT_SIZE)
    ax.set_xticks(np.arange(0.0, 1.0001, 0.5, dtype=float))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=12)
    fig.tight_layout()
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(pdf, bbox_inches="tight", transparent=True)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = default_plot()
    ax.fill_between(x, main_mean - total_err, main_mean + total_err, color="0.88", alpha=0.60, linewidth=0)
    ax.fill_between(x, main_mean - stat_sdev, main_mean + stat_sdev, color="0.75", alpha=0.60, linewidth=0)
    ax.plot(x, main_mean, color="0.45", linewidth=1.2)
    ax.set_xlabel(r"$x$", **FONT_SIZE)
    ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
    ax.set_title("Extrapolation", **FONT_SIZE)
    fig.tight_layout()
    final_pdf = stem.with_name("ex_final").with_suffix(".pdf")
    final_svg = stem.with_name("ex_final").with_suffix(".svg")
    final_nc = stem.with_name("ex_final").with_suffix(".nc")
    fig.savefig(final_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(final_svg, bbox_inches="tight")
    plt.close(fig)
    xr.Dataset(
        {
            "central": (("x",), np.asarray(main_mean, dtype=float)),
            "stat_sdev": (("x",), np.asarray(stat_sdev, dtype=float)),
            "total_systematic_error": (("x",), np.asarray(total_sys, dtype=float)),
            "total_error": (("x",), np.asarray(total_err, dtype=float)),
        },
        coords={"x": np.asarray(x, dtype=float)},
    ).to_netcdf(final_nc)

    result = {
        "out": out,
        "operation": "systematics_budget",
        "artifact": str(artifact),
        "plot": str(pdf),
        "plot_image": str(svg),
        "final_artifact": str(final_nc),
        "final_plot": str(final_pdf),
        "final_plot_image": str(final_svg),
        "n_points": int(x.size),
        "plot_points": int(x_plot.size),
        "sources": ["zs", "lambda_extrapolation", "lamet_scale", "other_extrapolations"],
    }
    store["output"] = result
    return result


STAGE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "run_extrapolation": run_extrapolation,
    "run_systematics_budget": run_systematics_budget,
}
