"""Internal qDA-ratio tuning and grid-fit implementation.

The public correlator tools remain ``tune_bare_matrix`` and
``fit_bare_matrix_grid``. This module supplies their ``fit_scope='qda_ratio'``
branch while reusing the shared spectral priors, fit constructor, selection,
resampling, and output contracts from :mod:`correlator.functions`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
from typing import Any

import gvar as gv
import numpy as np
import matplotlib.pyplot as plt

from lamet_agent.core.plotting import (
    COLOR_CYCLE,
    ERRORBAR_STYLE,
    FONT_SIZE,
    LEGEND_SETS,
    default_plot,
    plot_qda_ratio_fit_on_data,
)
from lamet_agent.core.resampling import sample_mean_and_sdev, samples_to_gvar
from lamet_agent.core.tools import log_nonlinear_fit_quality, resolve_plot_save_path, setup_logger

from .functions import (
    NUMERICAL_FIT_ERRORS,
    _bare_matrix_element_from_fit,
    _bare_matrix_element_mean_for_part,
    _bare_records_to_ensemble,
    _check_mode,
    _check_rescale,
    _energy_summary,
    _fit_data_count,
    _fit_summary,
    _fit_usable,
    _loggbf_weights,
    _normalise_fit_scope,
    _normalise_prior_width,
    _normalise_pt2_windows,
    _normalise_strategy,
    _p0_from_fit,
    _parts,
    _prior_parameter_count,
    _read_2pt,
    _recenter,
    _record,
    _resample_pt2,
    _scaled_prior,
    _scope_prior_with_width,
    _vary_prior_width,
    _weighted_model_sdev,
    _with_fit_size_metadata,
    _anchor_qda_pt2_prior,
    fit_matrix_element,
    fit_two_point,
    qda_pt2_prior,
    qda_ratio_fcn,
    select_data_window,
)


def _plot_sample0_qda_ratio(
    *,
    ratio_re: np.ndarray,
    ratio_im: np.ndarray,
    fit: Any,
    nstate: int,
    tmin: int,
    tmax: int,
    Lt: int,
    strategy: str,
    momentum: str,
    bT: int,
    bz: int,
    part: str,
    qda_denominator_mode: str,
    log_dir: Path,
) -> dict[str, str]:
    """Write sample-0 qDA ratio data and posterior bands on the main process."""
    plot_t = np.arange(0, Lt // 2 + 1, dtype=int)
    fit_t = np.arange(tmin, tmax, dtype=int)
    stem = log_dir / (
        f"{strategy}_qda_ratio_fit_{momentum}_bT{int(bT)}_bz{int(bz)}_sample0"
    )
    figures = plot_qda_ratio_fit_on_data(
        plot_t,
        np.asarray(ratio_re, dtype=object)[plot_t],
        np.asarray(ratio_im, dtype=object)[plot_t],
        fit_t=fit_t,
        fit_real=qda_ratio_fcn(
            fit_t,
            fit.p,
            Lt,
            nstate=nstate,
            part="re",
            qda_denominator_mode=qda_denominator_mode,
        ),
        fit_imag=qda_ratio_fcn(
            fit_t,
            fit.p,
            Lt,
            nstate=nstate,
            part="im",
            qda_denominator_mode=qda_denominator_mode,
        ),
        components=_parts(part),
        fit_label=f"{nstate}-state sample-0 fit",
        title=f"{momentum}, bT={int(bT)}, bz={int(bz)}",
        save_path=stem,
    )
    for figure, _axis in figures.values():
        plt.close(figure)
    output: dict[str, str] = {}
    for component in figures:
        output[f"qda_ratio_{component}_pdf"] = str(
            stem.with_name(f"{stem.name}_qda_ratio_{component}.pdf")
        )
        output[f"qda_ratio_{component}_svg"] = str(
            stem.with_name(f"{stem.name}_qda_ratio_{component}.svg")
        )
    return output

def _ratio_samples(
    numerator_samples: np.ndarray,
    denominator_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.divide(
        numerator_samples,
        denominator_samples,
        out=np.zeros_like(numerator_samples),
        where=denominator_samples != 0,
    )
    return np.real(ratio), np.imag(ratio)


def _ratio_samples_to_gvar(
    samples: np.ndarray,
    *,
    mode: str,
    sample_error_mode: str,
) -> np.ndarray:
    """Convert ratio samples, regularizing only exactly deterministic data."""
    values = samples_to_gvar(
        samples, mode=mode, sample_error_mode=sample_error_mode
    )
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.shape[0] > 0 and np.allclose(
        sample_array,
        np.broadcast_to(sample_array[0], sample_array.shape),
        rtol=1e-14,
        atol=1e-14,
    ):
        mean = np.asarray(np.mean(sample_array, axis=0), dtype=float)
        floor = np.maximum(np.abs(mean), 1.0) * 1e-8
        return gv.gvar(mean, floor)
    return values


def _validate_denominator_selectors(
    qda_denominator_mode: str,
    *,
    pt2_bT: int | None,
    pt2_bz: int | None,
) -> None:
    if qda_denominator_mode == "local_local":
        if pt2_bT is not None or pt2_bz is not None:
            raise ValueError(
                "local_local qDA denominators must not declare pt2_bT or pt2_bz"
            )
        return
    if qda_denominator_mode == "nonlocal_bz0":
        if pt2_bT is None or pt2_bz != 0:
            raise ValueError(
                "nonlocal_bz0 qDA denominators require pt2_bT and pt2_bz=0"
            )
        return
    raise ValueError(
        "qda_denominator_mode must be 'local_local' or 'nonlocal_bz0'"
    )


def _average_records(
    *,
    pt2_gv: np.ndarray,
    ratio_re: np.ndarray,
    ratio_im: np.ndarray,
    windows: list[dict[str, int]],
    strategies: list[str],
    nstates: list[int],
    prior_widths: list[float],
    Lt: int,
    part: str,
    svdcut: float,
    scale: float,
    qda_denominator_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit every qDA strategy/model/window candidate on ensemble averages."""
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for strategy_value, nstate, width, window in product(
        strategies, nstates, prior_widths, windows
    ):
        strategy, _ = _normalise_strategy(strategy_value)
        template = _scope_prior_with_width(
            "Breit",
            nstate,
            "qda_ratio",
            strategy,
            width,
            qda_denominator_mode=qda_denominator_mode,
        )
        fit_prior = _scope_prior_with_width(
            "Breit",
            nstate,
            "qda_ratio",
            strategy,
            width,
            qda_denominator_mode=qda_denominator_mode,
        )
        metadata = {
            "tmin": int(window["tmin"]),
            "tmax": int(window["tmax"]),
            "nstate": int(nstate),
            "prior_width": float(width),
            "fit_strategy": strategy,
            "fit_scope": "qda_ratio",
            "part": part,
            "correlator_rescale": scale,
        }
        metadata = _with_fit_size_metadata(
            metadata,
            n_data=_fit_data_count(
                metadata,
                strategy=strategy,
                fit_scope="qda_ratio",
                part=part,
                fitting_form="Breit",
            ),
            n_params=_prior_parameter_count(template),
        )
        try:
            pt2_fit = None
            if strategy == "chained":
                pt2_fit = fit_two_point(
                    pt2_gv,
                    metadata["tmin"],
                    metadata["tmax"],
                    Lt,
                    nstate=nstate,
                    svdcut=svdcut,
                    rescale=scale,
                    prior=_vary_prior_width(
                        qda_pt2_prior(
                            nstate,
                            qda_denominator_mode=qda_denominator_mode,
                        ),
                        width,
                    ),
                    qda_denominator_mode=qda_denominator_mode,
                )
                _anchor_qda_pt2_prior(
                    fit_prior,
                    pt2_fit,
                    nstate,
                    qda_denominator_mode=qda_denominator_mode,
                )
            fit = fit_matrix_element(
                ratio_re,
                ratio_im,
                None,
                None,
                Lt,
                strategy=strategy,
                fit_scope="qda_ratio",
                fitting_form="Breit",
                pt2_gv=pt2_gv if strategy == "joint" else None,
                tmin=metadata["tmin"],
                tmax=metadata["tmax"],
                nstate=nstate,
                part=part,
                svdcut=svdcut,
                rescale=scale,
                prior=fit_prior,
                qda_denominator_mode=qda_denominator_mode,
            )
            usable, reason = _fit_usable(fit, template)
            if not usable:
                rejected.append({**metadata, "reason": reason})
                continue
            records.append(_record(fit, pt2_fit=pt2_fit, **metadata))
        except NUMERICAL_FIT_ERRORS as exc:
            rejected.append({**metadata, "reason": str(exc)})
    return records, rejected


def _fit_sample_batch(payload: bytes, sample_indices: list[int]) -> list[dict[str, Any]]:
    """Fit a process batch of recentered qDA-ratio samples."""
    context = gv.loads(payload)
    output: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        records: list[dict[str, Any]] = []
        logs: list[dict[str, Any]] = []
        try:
            ratio_re = _recenter(
                context["ratio_re_samples"][sample_index], context["ratio_re_gv"]
            )
            ratio_im = _recenter(
                context["ratio_im_samples"][sample_index], context["ratio_im_gv"]
            )
            pt2_sample = None
            if context["strategy"] == "joint":
                pt2_sample = _recenter(
                    context["pt2_samples"][sample_index], context["pt2_gv"]
                )
            for candidate_index, candidate in enumerate(context["candidates"]):
                fit = fit_matrix_element(
                    ratio_re,
                    ratio_im,
                    None,
                    None,
                    context["Lt"],
                    strategy=context["strategy"],
                    fit_scope="qda_ratio",
                    fitting_form="Breit",
                    pt2_gv=pt2_sample,
                    tmin=context["tmin"],
                    tmax=context["tmax"],
                    nstate=candidate["nstate"],
                    part=context["part"],
                    svdcut=context["svdcut"],
                    rescale=context["scale"],
                    prior=candidate["prior"],
                    p0=candidate["p0"],
                    qda_denominator_mode=context["qda_denominator_mode"],
                )
                usable, reason = _fit_usable(fit, candidate["template"])
                if not usable:
                    logs.append(
                        {
                            "kind": "rejected",
                            "nstate": candidate["nstate"],
                            "prior_width": candidate["prior_width"],
                            "reason": reason,
                        }
                    )
                    continue
                records.append(
                    _record(
                        fit,
                        candidate_index=candidate_index,
                        nstate=candidate["nstate"],
                        prior_width=candidate["prior_width"],
                    )
                )
                logs.append(
                    {
                        "kind": "fit",
                        "nstate": candidate["nstate"],
                        "prior_width": candidate["prior_width"],
                        "Q": float(fit.Q),
                        "chi2": float(fit.chi2),
                        "dof": int(fit.dof),
                        "logGBF": float(fit.logGBF),
                    }
                )
            if not records:
                raise ValueError("all qda_ratio candidate fits failed")
            weights = (
                _loggbf_weights(records)
                if context["model_average"] and len(records) > 1
                else np.asarray([1.0])
            )
            real_values = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        record["fit"].p,
                        output_part="re",
                        fit_part=context["part"],
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=context["qda_denominator_mode"],
                    )
                    for record in records
                ]
            )
            imag_values = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        record["fit"].p,
                        output_part="im",
                        fit_part=context["part"],
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=context["qda_denominator_mode"],
                    )
                    for record in records
                ]
            )
            full_weights = np.zeros(len(context["candidates"]), dtype=float)
            for weight, record in zip(weights, records):
                full_weights[int(record["candidate_index"])] = float(weight)
            plot_payload = None
            if sample_index == 0:
                plot_record = records[int(np.argmax(weights))]
                plot_payload = gv.dumps(
                    {
                        "fit": plot_record["fit"],
                        "ratio_re": ratio_re,
                        "ratio_im": ratio_im,
                        "nstate": int(plot_record["nstate"]),
                        "prior_width": float(plot_record["prior_width"]),
                    }
                )
            output.append(
                {
                    "sample": int(sample_index),
                    "real": float(np.sum(weights * real_values)),
                    "imag": float(np.sum(weights * imag_values)),
                    "candidate_weights": full_weights.tolist(),
                    "logs": logs,
                    "plot_payload": plot_payload,
                    "error": None,
                }
            )
        except NUMERICAL_FIT_ERRORS as exc:
            output.append(
                {
                    "sample": int(sample_index),
                    "real": float("nan"),
                    "imag": float("nan"),
                    "candidate_weights": [0.0] * len(context["candidates"]),
                    "logs": logs,
                    "plot_payload": None,
                    "error": str(exc),
                }
            )
    return output


def _load_denominator(
    *,
    pt2_path: str,
    source_operator: str,
    sink_operator: str,
    momentum: str,
    temporal_extent: int | None,
    pt2_bT: int | None,
    pt2_bz: int | None,
    mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int,
    sample_error_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    denominator = _read_2pt(
        pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=momentum,
        temporal_extent=temporal_extent,
        bT=pt2_bT,
        bz=pt2_bz,
    )
    pt2_samples, denominator_samples, indices = _resample_pt2(
        denominator,
        mode=mode,
        n_boot=n_boot,
        seed=seed,
        bin_size=bin_size,
    )
    pt2_gv = samples_to_gvar(
        pt2_samples, mode=mode, sample_error_mode=sample_error_mode
    )
    return denominator, pt2_samples, denominator_samples, indices, pt2_gv


def _load_ratio(
    *,
    z: int,
    denominator_shape: tuple[int, ...],
    denominator_samples: np.ndarray,
    indices: np.ndarray | None,
    qda_path: str,
    qda_source_operator: str,
    qda_sink_operator: str,
    momentum: str,
    bT: int,
    temporal_extent: int | None,
    mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int,
    sample_error_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numerator = _read_2pt(
        qda_path,
        source_operator=qda_source_operator,
        sink_operator=qda_sink_operator,
        momentum=momentum,
        temporal_extent=temporal_extent,
        bT=bT,
        bz=z,
    )
    if numerator.shape != denominator_shape:
        raise ValueError(
            f"qda numerator shape mismatch at z={z}: {numerator.shape} != {denominator_shape}"
        )
    _, numerator_samples, _ = _resample_pt2(
        numerator,
        mode=mode,
        n_boot=n_boot,
        seed=seed,
        bin_size=bin_size,
        indices=indices,
    )
    sample_re, sample_im = _ratio_samples(numerator_samples, denominator_samples)
    ratio_re = _ratio_samples_to_gvar(
        sample_re, mode=mode, sample_error_mode=sample_error_mode
    )
    ratio_im = _ratio_samples_to_gvar(
        sample_im, mode=mode, sample_error_mode=sample_error_mode
    )
    return sample_re, sample_im, ratio_re, ratio_im


def tune_qda_ratio(
    store: dict[str, Any],
    *,
    pt2_path: str,
    qda_path: str | None,
    momentum: str | None,
    source_operator: str,
    sink_operator: str,
    qda_source_operator: str | None,
    qda_sink_operator: str | None,
    qda_denominator_mode: str,
    pt2_bT: int | None,
    pt2_bz: int | None,
    bT: int,
    tune_z_values: list[int] | None,
    bz: list[int] | None,
    temporal_extent: int | None,
    pt2_windows: list[dict[str, int]] | None,
    fit_strategies: list[str] | None,
    fit_strategy: str | None,
    nstate_values: list[int] | None,
    nstate: int | None,
    prior_width: float | list[float] | None,
    svdcut: float,
    correlator_rescale: float,
    resample_mode: str,
    sample_error_mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int,
    part: str,
    q_min: float,
    out: str,
) -> dict[str, Any]:
    """Tune qDA windows/models across representative qDA bz values."""
    if momentum is None:
        raise ValueError("qda_ratio jobs require scalar params.momentum")
    if qda_path is None or qda_source_operator is None or qda_sink_operator is None:
        raise ValueError("qda_ratio jobs require one nonlocal qDA 2pt correlator")
    _validate_denominator_selectors(
        qda_denominator_mode, pt2_bT=pt2_bT, pt2_bz=pt2_bz
    )
    z_list = [int(value) for value in (bz or [])]
    if not z_list:
        raise ValueError("the qDA 2pt correlator must declare a non-empty bz grid")
    tune_list = list(
        dict.fromkeys(
            int(value)
            for value in (tune_z_values or [])
        )
    )
    if not tune_list or any(value not in z_list for value in tune_list):
        raise ValueError(
            "tune_z_values must contain values from the qda_ratio bz grid"
        )
    denominator, pt2_samples, denominator_samples, indices, pt2_gv = _load_denominator(
        pt2_path=pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=momentum,
        temporal_extent=temporal_extent,
        pt2_bT=pt2_bT,
        pt2_bz=pt2_bz,
        mode=resample_mode,
        n_boot=n_boot,
        seed=seed,
        bin_size=bin_size,
        sample_error_mode=sample_error_mode,
    )
    del pt2_samples
    Lt = int(denominator.shape[1])
    windows = _normalise_pt2_windows(pt2_windows, Lt=Lt)
    strategies = fit_strategies or ([fit_strategy] if fit_strategy else ["joint"])
    states = [
        int(value)
        for value in (nstate_values or ([nstate] if nstate is not None else [2]))
    ]
    widths = _normalise_prior_width(prior_width)
    records_by_z: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for z in tune_list:
        _, _, ratio_re, ratio_im = _load_ratio(
            z=z,
            denominator_shape=denominator.shape,
            denominator_samples=denominator_samples,
            indices=indices,
            qda_path=qda_path,
            qda_source_operator=qda_source_operator,
            qda_sink_operator=qda_sink_operator,
            momentum=momentum,
            bT=bT,
            temporal_extent=temporal_extent,
            mode=resample_mode,
            n_boot=n_boot,
            seed=seed,
            bin_size=bin_size,
            sample_error_mode=sample_error_mode,
        )
        records, rejected_z = _average_records(
            pt2_gv=pt2_gv,
            ratio_re=ratio_re,
            ratio_im=ratio_im,
            windows=windows,
            strategies=strategies,
            nstates=states,
            prior_widths=widths,
            Lt=Lt,
            part=part,
            svdcut=svdcut,
            scale=correlator_rescale,
            qda_denominator_mode=qda_denominator_mode,
        )
        rejected.extend({**item, "z": z} for item in rejected_z)
        records_by_z[z] = {
            (
                record["fit_strategy"],
                record["nstate"],
                record["prior_width"],
                record["tmin"],
                record["tmax"],
            ): record
            for record in records
        }
    common_keys = set.intersection(
        *(set(records) for records in records_by_z.values())
    )
    if not common_keys:
        raise ValueError("no qda_ratio fit candidate succeeded at every tune z")
    candidates: list[dict[str, Any]] = []
    primary_records: list[dict[str, Any]] = []
    for key in sorted(common_keys):
        diagnostics = {
            str(z): _fit_summary(records_by_z[z][key], fallback=False, index=0)
            for z in tune_list
        }
        primary = records_by_z[tune_list[0]][key]
        primary_records.append(primary)
        candidates.append(
            {
                "index": len(candidates),
                "fit_strategy": key[0],
                "fit_scope": "qda_ratio",
                "nstate": key[1],
                "prior_width": key[2],
                "tmin": key[3],
                "tmax": key[4],
                "tune_z_diagnostics": diagnostics,
                "feasible_at_all_tune_z": True,
                "min_Q": min(item["Q"] for item in diagnostics.values()),
                "worst_chi2_dof": max(
                    item["chi2_dof"] for item in diagnostics.values()
                ),
                "bare_re": str(
                    _bare_matrix_element_from_fit(
                        primary["fit"].p,
                        part="re",
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=qda_denominator_mode,
                    )
                ),
                "bare_im": str(
                    _bare_matrix_element_from_fit(
                        primary["fit"].p,
                        part="im",
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=qda_denominator_mode,
                    )
                ),
            }
        )
    best_index, fallback = select_data_window(primary_records, q_min=q_min)
    robust_index = min(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index]["min_Q"],
            candidates[index]["worst_chi2_dof"],
        ),
    )
    store[out] = primary_records
    return {
        "out": out,
        "fit_strategies": [_normalise_strategy(value)[0] for value in strategies],
        "fit_scopes": ["qda_ratio"],
        "nstate_values": states,
        "prior_width": widths,
        "tune_z_values": tune_list,
        "primary_tune_z": tune_list[0],
        "tune_z": tune_list[0],
        "allowed_bz": z_list,
        "Lt": Lt,
        "n_cfg": int(denominator.shape[0]),
        "correlator_rescale": correlator_rescale,
        "fitting_form": "Breit",
        "qda_denominator_mode": qda_denominator_mode,
        "candidates": candidates,
        "rejected": rejected,
        "recommended_index": best_index,
        "recommended_fallback_no_q_passing": fallback,
        "recommended_window": _fit_summary(
            primary_records[best_index], fallback=fallback, index=best_index
        ),
        "recommended_robust_index": robust_index,
        "recommended_robust_window": _fit_summary(
            primary_records[robust_index], fallback=False, index=robust_index
        ),
        "tuning_diagnostic_pdfs": {},
    }


def fit_qda_ratio_grid(
    store: dict[str, Any],
    *,
    pt2_path: str,
    qda_path: str | None,
    bz: list[int],
    ensemble: str,
    tag: str,
    momentum: str | None,
    source_operator: str,
    sink_operator: str,
    qda_source_operator: str | None,
    qda_sink_operator: str | None,
    qda_denominator_mode: str,
    pt2_bT: int | None,
    pt2_bz: int | None,
    bz_direction: str | None,
    bT: int,
    pt2_window: dict[str, int] | None,
    pt2_windows: list[dict[str, int]] | None,
    resample_mode: str,
    sample_error_mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int,
    svdcut: float,
    part: str,
    q_min: float,
    nstate_values: list[int],
    fit_strategy: str,
    prior_width: list[float],
    posterior_prior_error_scale: float,
    correlator_rescale: float,
    model_average: bool,
    tune_z: int | None,
    job_id: str | None,
    hadron: str | None,
    gfix: str | None,
    volume: str | None,
    lattice_spacing_fm: float | None,
    momentum_gev: float | None,
    temporal_extent: int | None,
    save_path: str | None,
    log_dir: str | Path | None,
    log_path: str | Path | None,
    artifacts_dir: str | Path | None,
    workers: int,
) -> dict[str, Any]:
    """Fit qDA ratios with the shared correlator fit and artifact contracts."""
    if momentum is None:
        raise ValueError("qda_ratio jobs require scalar params.momentum")
    if qda_path is None or qda_source_operator is None or qda_sink_operator is None:
        raise ValueError("qda_ratio jobs require one nonlocal qDA 2pt correlator")
    _validate_denominator_selectors(
        qda_denominator_mode, pt2_bT=pt2_bT, pt2_bz=pt2_bz
    )
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    strategy, _ = _normalise_strategy(fit_strategy)
    mode = _check_mode(resample_mode)
    scale = _check_rescale(correlator_rescale)
    z_list = [int(value) for value in bz]
    if not z_list:
        raise ValueError("the qDA 2pt correlator must declare a non-empty bz grid")
    tune_z_value = int(tune_z) if tune_z is not None else z_list[0]
    if tune_z_value not in z_list:
        raise ValueError("tune_z must be present in the qda_ratio bz grid")
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_log = (
        Path(log_path)
        if log_path is not None
        else fit_log_dir / f"{ensemble}_{tag}_{momentum}_{strategy}_qda_ratio.log"
    )
    logger = setup_logger(resolved_log, logger_name="qda_ratio_logger")
    resolved_save = resolve_plot_save_path(
        save_path, artifacts_dir=out_dir, default_stem=tag or "qda_ratio"
    )
    denominator, pt2_samples, denominator_samples, indices, pt2_gv = _load_denominator(
        pt2_path=pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=momentum,
        temporal_extent=temporal_extent,
        pt2_bT=pt2_bT,
        pt2_bz=pt2_bz,
        mode=mode,
        n_boot=n_boot,
        seed=seed,
        bin_size=bin_size,
        sample_error_mode=sample_error_mode,
    )
    n_cfg, Lt = denominator.shape
    n_samples = int(pt2_samples.shape[0])
    windows = _normalise_pt2_windows(pt2_windows, Lt=Lt)
    if pt2_window is not None:
        windows = [
            {"tmin": int(pt2_window["tmin"]), "tmax": int(pt2_window["tmax"])}
        ]
    tune_sample_re, tune_sample_im, tune_ratio_re, tune_ratio_im = _load_ratio(
        z=tune_z_value,
        denominator_shape=denominator.shape,
        denominator_samples=denominator_samples,
        indices=indices,
        qda_path=qda_path,
        qda_source_operator=qda_source_operator,
        qda_sink_operator=qda_sink_operator,
        momentum=momentum,
        bT=bT,
        temporal_extent=temporal_extent,
        mode=mode,
        n_boot=n_boot,
        seed=seed,
        bin_size=bin_size,
        sample_error_mode=sample_error_mode,
    )
    del tune_sample_re, tune_sample_im
    tune_records, tune_rejected = _average_records(
        pt2_gv=pt2_gv,
        ratio_re=tune_ratio_re,
        ratio_im=tune_ratio_im,
        windows=windows,
        strategies=[strategy],
        nstates=nstate_values,
        prior_widths=prior_width,
        Lt=Lt,
        part=part,
        svdcut=svdcut,
        scale=scale,
        qda_denominator_mode=qda_denominator_mode,
    )
    if not tune_records:
        raise ValueError("all qda_ratio shared-window tuning fits failed")
    tune_index, tune_fallback = select_data_window(tune_records, q_min=q_min)
    chosen = tune_records[tune_index]
    shared_window = {"tmin": int(chosen["tmin"]), "tmax": int(chosen["tmax"])}
    logger.info(
        "selected qda_ratio window t=[%s,%s)",
        shared_window["tmin"],
        shared_window["tmax"],
    )
    sample_batches = [
        batch.tolist()
        for batch in np.array_split(
            np.arange(n_samples), min(int(workers), n_samples)
        )
        if batch.size
    ]
    executor = ProcessPoolExecutor(max_workers=int(workers)) if workers > 1 else None
    z_records: list[dict[str, Any]] = []
    z_report: list[dict[str, Any]] = []
    energy_fit = chosen.get("pt2_fit") or chosen["fit"]
    try:
        from tqdm import tqdm
    except ImportError:
        z_iterator = z_list
    else:
        z_iterator = tqdm(
            z_list,
            desc=f"fit qDA ratio {ensemble} {momentum}",
        )
    try:
        for z in z_iterator:
            sample_re, sample_im, ratio_re, ratio_im = _load_ratio(
                z=z,
                denominator_shape=denominator.shape,
                denominator_samples=denominator_samples,
                indices=indices,
                qda_path=qda_path,
                qda_source_operator=qda_source_operator,
                qda_sink_operator=qda_sink_operator,
                momentum=momentum,
                bT=bT,
                temporal_extent=temporal_extent,
                mode=mode,
                n_boot=n_boot,
                seed=seed,
                bin_size=bin_size,
                sample_error_mode=sample_error_mode,
            )
            average_records, rejected = _average_records(
                pt2_gv=pt2_gv,
                ratio_re=ratio_re,
                ratio_im=ratio_im,
                windows=[shared_window],
                strategies=[strategy],
                nstates=nstate_values,
                prior_widths=prior_width,
                Lt=Lt,
                part=part,
                svdcut=svdcut,
                scale=scale,
                qda_denominator_mode=qda_denominator_mode,
            )
            if not average_records:
                raise ValueError(f"all qda_ratio sample-average fits failed for z={z}")
            fallback = False
            if not model_average:
                selected_index, fallback = select_data_window(
                    average_records, q_min=q_min
                )
                average_records = [average_records[selected_index]]
            average_weights = (
                _loggbf_weights(average_records)
                if model_average and len(average_records) > 1
                else np.asarray([1.0])
            )
            average_real = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        record["fit"].p,
                        output_part="re",
                        fit_part=part,
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=qda_denominator_mode,
                    )
                    for record in average_records
                ]
            )
            average_imag = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        record["fit"].p,
                        output_part="im",
                        fit_part=part,
                        fitting_form="Breit",
                        fit_scope="qda_ratio",
                        qda_denominator_mode=qda_denominator_mode,
                    )
                    for record in average_records
                ]
            )
            real_sys = (
                _weighted_model_sdev(average_real, average_weights)
                if model_average and "re" in _parts(part)
                else 0.0
            )
            imag_sys = (
                _weighted_model_sdev(average_imag, average_weights)
                if model_average and "im" in _parts(part)
                else 0.0
            )
            candidates = []
            for record in average_records:
                template = _scope_prior_with_width(
                    "Breit",
                    int(record["nstate"]),
                    "qda_ratio",
                    strategy,
                    float(record["prior_width"]),
                    qda_denominator_mode=qda_denominator_mode,
                )
                candidates.append(
                    {
                        "nstate": int(record["nstate"]),
                        "prior_width": float(record["prior_width"]),
                        "template": template,
                        "prior": _scaled_prior(
                            record["fit"],
                            template,
                            error_scale=posterior_prior_error_scale,
                            prior_width=float(record["prior_width"]),
                        ),
                        "p0": _p0_from_fit(record["fit"], template),
                    }
                )
                log_nonlinear_fit_quality(
                    record["fit"],
                    kind="sample-average qda_ratio",
                    label=(
                        f"z={z} t=[{shared_window['tmin']},{shared_window['tmax']}) "
                        f"nstate={record['nstate']} prior_width={record['prior_width']}"
                    ),
                    logger=logger,
                    q_min=q_min,
                )
            payload = gv.dumps(
                {
                    "pt2_samples": pt2_samples,
                    "pt2_gv": pt2_gv,
                    "ratio_re_samples": sample_re,
                    "ratio_im_samples": sample_im,
                    "ratio_re_gv": ratio_re,
                    "ratio_im_gv": ratio_im,
                    "strategy": strategy,
                    "Lt": Lt,
                    "tmin": shared_window["tmin"],
                    "tmax": shared_window["tmax"],
                    "part": part,
                    "svdcut": svdcut,
                    "scale": scale,
                    "model_average": model_average,
                    "candidates": candidates,
                    "qda_denominator_mode": qda_denominator_mode,
                }
            )
            if executor is None:
                sample_results = _fit_sample_batch(payload, sample_batches[0])
            else:
                futures = [
                    executor.submit(_fit_sample_batch, payload, batch)
                    for batch in sample_batches
                ]
                sample_results = [
                    item for future in futures for item in future.result()
                ]
            sample_results.sort(key=lambda item: item["sample"])
            real_samples = np.asarray(
                [item["real"] for item in sample_results], dtype=float
            )
            imag_samples = np.asarray(
                [item["imag"] for item in sample_results], dtype=float
            )
            failures = [item for item in sample_results if item["error"] is not None]
            if not np.any(np.isfinite(real_samples)):
                raise ValueError(f"all qda_ratio resampled fits failed for z={z}")
            sample0_paths: dict[str, str] = {}
            sample0_result = next(
                (item for item in sample_results if item["sample"] == 0), None
            )
            if sample0_result is not None and sample0_result["plot_payload"] is not None:
                plot_data = gv.loads(sample0_result["plot_payload"])
                sample0_paths = _plot_sample0_qda_ratio(
                    ratio_re=plot_data["ratio_re"],
                    ratio_im=plot_data["ratio_im"],
                    fit=plot_data["fit"],
                    nstate=int(plot_data["nstate"]),
                    tmin=shared_window["tmin"],
                    tmax=shared_window["tmax"],
                    Lt=Lt,
                    strategy=strategy,
                    momentum=momentum,
                    bT=bT,
                    bz=z,
                    part=part,
                    qda_denominator_mode=qda_denominator_mode,
                    log_dir=fit_log_dir,
                )
            best_average = average_records[int(np.argmax(average_weights))]
            z_records.append(
                {
                    "z": z,
                    "real_samples": real_samples,
                    "imag_samples": imag_samples,
                    "real_sys_sdev": real_sys,
                    "imag_sys_sdev": imag_sys,
                    "window": _fit_summary(
                        best_average, fallback=fallback, index=0
                    ),
                    "sample0_plot_paths": sample0_paths,
                }
            )
            z_report.append(
                {
                    "z": z,
                    "window": _fit_summary(
                        best_average, fallback=fallback, index=0
                    ),
                    "rejected_fit_models": rejected,
                    "n_failed_samples": len(failures),
                    "sample_failures": failures[:10],
                    "real_sys_sdev": real_sys,
                    "imag_sys_sdev": imag_sys,
                    "sample0_plot_paths": sample0_paths,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown()
    sorted_records = sorted(z_records, key=lambda item: item["z"])
    summary_z = [int(item["z"]) for item in sorted_records]
    real_means: list[float] = []
    real_errors: list[float] = []
    imag_means: list[float] = []
    imag_errors: list[float] = []
    output_rows: list[dict[str, Any]] = []
    for record in sorted_records:
        real_mean, real_error = sample_mean_and_sdev(
            np.asarray(record["real_samples"]),
            mode=mode,
            sample_error_mode=sample_error_mode,
        )
        imag_mean, imag_error = sample_mean_and_sdev(
            np.asarray(record["imag_samples"]),
            mode=mode,
            sample_error_mode=sample_error_mode,
        )
        record.update(
            real_mean=float(real_mean),
            imag_mean=float(imag_mean),
            real_stat_sdev=float(real_error),
            imag_stat_sdev=float(imag_error),
        )
        real_means.append(float(real_mean))
        real_errors.append(float(real_error))
        imag_means.append(float(imag_mean))
        imag_errors.append(float(imag_error))
        output_rows.append(
            {
                "z": record["z"],
                "real_mean": float(real_mean),
                "real_stat_sdev": float(real_error),
                "real_sys_sdev": float(record["real_sys_sdev"]),
                "imag_mean": float(imag_mean),
                "imag_stat_sdev": float(imag_error),
                "imag_sys_sdev": float(record["imag_sys_sdev"]),
                "n_failed_samples": int(
                    np.count_nonzero(~np.isfinite(record["real_samples"]))
                ),
            }
        )
    figure, axis = default_plot()
    if "re" in _parts(part):
        axis.errorbar(
            summary_z,
            real_means,
            real_errors,
            label="Re",
            color=COLOR_CYCLE[0],
            **ERRORBAR_STYLE,
        )
    if "im" in _parts(part):
        axis.errorbar(
            summary_z,
            imag_means,
            imag_errors,
            label="Im",
            color=COLOR_CYCLE[1],
            marker="s",
            **ERRORBAR_STYLE,
        )
    axis.set_xlabel(r"$z/a$", **FONT_SIZE)
    denominator_label = "z'_0" if qda_denominator_mode == "nonlocal_bz0" else "z_0"
    axis.set_ylabel(
        rf"Bare matrix element $O_{{00}}/{denominator_label}$", **FONT_SIZE
    )
    p_label = "n/a" if momentum_gev is None else f"{float(momentum_gev):.2f}"
    axis.set_title(
        rf"{ensemble} $p={p_label}\,\mathrm{{GeV}}$ bare matrix elements",
        **FONT_SIZE,
    )
    axis.legend(**LEGEND_SETS)
    figure.tight_layout()
    pdf_path = f"{resolved_save}.pdf"
    svg_path = f"{resolved_save}.svg"
    figure.savefig(pdf_path, bbox_inches="tight", transparent=True)
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    bare_data = _bare_records_to_ensemble(
        z_records,
        resample_mode=mode,
        attrs={
            "ensemble": ensemble,
            "tag": tag,
            "fitting_form": "Breit",
            "fit_scope": "qda_ratio",
            "fit_strategy": strategy,
            "fit_mode": f"{strategy}_2pt_qda_ratio",
            "qda_denominator_mode": qda_denominator_mode,
            "coord_unit": "lattice",
            "bz_direction": bz_direction,
            "momentum": momentum,
            "bT": bT,
            "resample_mode": mode,
            "sample_error_mode": sample_error_mode,
            "average_method": sample_error_mode,
            "part": part,
            "component": part,
            "job_id": job_id,
            "volume": volume,
            "lattice_spacing_fm": lattice_spacing_fm,
            "momentum_gev": momentum_gev,
            "model_average": model_average,
            "nstate_values": json.dumps(nstate_values),
            "prior_width": json.dumps(prior_width),
            "posterior_prior_error_scale": posterior_prior_error_scale,
            "hadron": hadron,
            "gfix": gfix,
            "workers": int(workers),
        },
    )
    artifact_path = f"{resolved_save}.nc"
    bare_data.to_netcdf(artifact_path)
    energy = _energy_summary(
        fit=energy_fit,
        key="E0",
        momentum=momentum,
        momentum_gev=momentum_gev,
        lattice_spacing_fm=lattice_spacing_fm,
        channel="qda_denominator",
        pt2_path=pt2_path,
        ensemble=ensemble,
        hadron=hadron,
        gfix=gfix,
        volume=volume,
        source_operator=source_operator,
        sink_operator=sink_operator,
        fitting_form="Breit",
        job_id=job_id,
    )
    store["bare_matrix_element_data"] = bare_data
    store["bare_matrix_element_netcdf"] = artifact_path
    store["output"] = bare_data
    shared_spec = {
        "fit_scope": "qda_ratio",
        "fit_strategy": strategy,
        "tmin": shared_window["tmin"],
        "tmax": shared_window["tmax"],
        "pt2_window": f"[{shared_window['tmin']},{shared_window['tmax']})",
        "pt3_window": "not used",
        "n_data": int(chosen["n_data"]),
        "n_params": int(chosen["n_params"]),
    }
    return {
        "artifact": artifact_path,
        "netcdf_path": artifact_path,
        "plot_pdf": pdf_path,
        "plot_svg": svg_path,
        "n_z": len(z_records),
        "n_sample": bare_data.n_sample,
        "outputs": output_rows,
        "fitting_form": "Breit",
        "fit_scope": "qda_ratio",
        "fit_strategy": strategy,
        "fit_mode": f"{strategy}_2pt_qda_ratio",
        "qda_denominator_mode": qda_denominator_mode,
        "model_average": model_average,
        "selection_rule": (
            "qda_ratio shared pt2 window "
            f"(fallback_no_q_passing={tune_fallback})"
        ),
        "shared_window_specs": [shared_spec],
        "tuning_log_path": str(resolved_log),
        "sample_log_path": str(resolved_log),
        "correlator_rescale": scale,
        "resample_mode": mode,
        "sample_error_mode": sample_error_mode,
        "n_samples": n_samples,
        "workers": int(workers),
        "bz": z_list,
        "tune_z": tune_z_value,
        "z_fits": z_report,
        "pt2_energies": [energy] if energy is not None else [],
        "momentum_gev": momentum_gev,
        "component": part,
        "nstate_values": nstate_values,
        "prior_width": prior_width,
        "window_candidates": [
            _fit_summary(record, fallback=False, index=index)
            for index, record in enumerate(tune_records)
        ],
        "rejected_windows": tune_rejected,
    }
