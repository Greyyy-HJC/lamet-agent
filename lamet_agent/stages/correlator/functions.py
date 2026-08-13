"""Correlator-analysis stage tools.

Purpose:
- provide one agentic tool set for 2pt ground-state, 3pt/FH, and qDA-ratio analysis
- the agent drives strategy: inspect the 2pt scale, tune one shared fit setting on
  sample-average data, then apply that setting to every bootstrap/jackknife sample

Expected inputs:
- local 2pt HDF5: ``source_operator/sink_operator/momentum`` with shape (Lt, n_cfg)
- qDA 2pt HDF5: ``source_operator/sink_operator/momentum/bT*/bz*``
- 3pt HDF5: ``source_operator/sink_operator/current_operator/momentum/tsep*/bT*/bz*``
  with shape (tsep+1, n_cfg)
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- fit diagnostics for the agent to judge candidate windows
- bare matrix-element NetCDF, fit-on-data PDFs, split fit logs, and a summary PDF
  under ``artifacts/``

Example usage:
- from lamet_agent.stages.correlator.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["inspect_correlator_scale"](store, pt2_path="data/2pt.h5", momentum="PX0PY0PZ0")
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import gvar as gv
import h5py
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

np.seterr(over="ignore")

from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.manifest import HBAR_C_GEV_FM
from lamet_agent.core.plotting import (
    COLOR_CYCLE,
    ERRORBAR_STYLE,
    FONT_SIZE,
    LEGEND_SETS,
    default_plot,
    plot_fh_fit_on_data,
    plot_pt2_fit_on_data,
    plot_pt2_meff_on_data,
    plot_pt3_ratio_fit_on_data,
    plot_qda_ratio_fit_on_data,
    plot_sample_fit_quality_cdf,
    plot_sample_fit_quality_chi2,
)
from lamet_agent.core.resampling import (
    resample_config_samples,
    sample_mean_and_sdev,
    sample_value_with_error,
    samples_to_gvar,
)
from lamet_agent.core.tools import (
    log_nonlinear_fit_quality,
    resolve_plot_save_path,
    setup_logger,
)

# 2pt ground-state posteriors anchor the chained 3pt prior with widened errors.
PT2_PRIOR_ERROR_SCALE = 3.0
NUMERICAL_FIT_ERRORS = (FloatingPointError, RuntimeError, ValueError, OverflowError)


# --- physics models and priors ----------------------------------------------


def _state_key(name: str, state: int | None = None, suffix: str = "") -> str:
    if state is None:
        return f"{name}{suffix}"
    return f"{name}{state}{suffix}"


def _state_energies(p: dict, nstate: int, suffix: str = "") -> list[Any]:
    energy = p[_state_key("E0", suffix=suffix)]
    energies = []
    for state in range(nstate):
        if state > 0:
            energy = energy + p[_state_key("dE", state, suffix)]
        energies.append(energy)
    return energies


def _energy_summary(
    *,
    fit: lsf.nonlinear_fit | None,
    key: str,
    momentum: str | None,
    momentum_gev: float | None,
    lattice_spacing_fm: float | None,
    channel: str,
    pt2_path: str | None,
    ensemble: str,
    hadron: str | None,
    gfix: str | None,
    volume: str | None,
    source_operator: str,
    sink_operator: str,
    fitting_form: str,
    job_id: str | None,
    E0_lattice_samples: list[float] | np.ndarray | None = None,
    resample_mode: str | None = None,
    sample_error_mode: str | None = None,
    workers: int = 1,
) -> dict[str, Any] | None:
    if fit is None or key not in fit.p or momentum is None or momentum_gev is None or lattice_spacing_fm is None:
        return None
    e_lattice = fit.p[key]
    scale = HBAR_C_GEV_FM / float(lattice_spacing_fm)
    e_mean = float(gv.mean(e_lattice)) * scale
    e_sdev = float(gv.sdev(e_lattice)) * scale
    e_samples = np.asarray(E0_lattice_samples, dtype=float) * scale if E0_lattice_samples is not None else np.asarray([], dtype=float)
    e2_samples = e_samples**2
    finite_e2 = e2_samples[np.isfinite(e2_samples)]
    if finite_e2.size > 1 and resample_mode:
        e2_mean, e2_sdev = sample_mean_and_sdev(finite_e2, mode=resample_mode, sample_error_mode=sample_error_mode or "covariance")
        e2_mean = float(e2_mean)
        e2_sdev = float(e2_sdev)
    elif finite_e2.size == 1:
        e2_mean = float(finite_e2[0])
        e2_sdev = 0.0
    else:
        e2_mean = e_mean**2
        e2_sdev = abs(2.0 * e_mean * e_sdev)
    return {
        "job_id": job_id,
        "pt2_path": pt2_path,
        "ensemble": ensemble,
        "hadron": hadron,
        "gfix": gfix,
        "volume": volume,
        "lattice_spacing_fm": float(lattice_spacing_fm),
        "source_operator": source_operator,
        "sink_operator": sink_operator,
        "fitting_form": fitting_form,
        "channel": channel,
        "momentum": momentum,
        "p_gev": float(momentum_gev),
        "p2_gev2": float(momentum_gev) ** 2,
        "E0_lattice_mean": float(gv.mean(e_lattice)),
        "E0_lattice_sdev": float(gv.sdev(e_lattice)),
        "E0_gev_mean": e_mean,
        "E0_gev_sdev": e_sdev,
        "E0_gev2_mean": e2_mean,
        "E0_gev2_sdev": e2_sdev,
        "E0_gev_samples": json.dumps(e_samples.tolist()),
        "resample_mode": resample_mode or "",
        "sample_error_mode": sample_error_mode or "",
        "workers": int(workers),
    }


def _optional_float(value: Any) -> float | None:
    """Preserve an unestimated quantity as ``None`` instead of reporting zero."""
    return None if value is None else float(value)


def _fit_dispersion_sample_batch(payload: bytes, sample_indices: list[int]) -> list[tuple[int, list[float]]]:
    context = gv.loads(payload)
    design = context["design"]
    samples = context["samples"]
    template = context["template"]
    p0 = context["p0"]
    prior = context["prior"]

    def fcn(x_design: np.ndarray, p: gv.BufferDict):
        return p["m2"] + p["k2"] * x_design[:, 0] + p["k3"] * x_design[:, 1]

    out = []
    for sample in sample_indices:
        y = sample_value_with_error(
            samples[sample],
            template,
            mode=context["resample_mode"],
            sample_error_mode=context["sample_error_mode"],
        )
        try:
            fit = lsf.nonlinear_fit(
                data=(design, y),
                fcn=fcn,
                p0=p0,
                prior=prior,
                maxit=2000,
                svdcut=1e-12,
                fitter="scipy_least_squares",
            )
            out.append((int(sample), [float(fit.pmean[key]) for key in ("m2", "k2", "k3")]))
        except (*NUMERICAL_FIT_ERRORS, AssertionError):
            out.append((int(sample), [float(p0[key]) for key in ("m2", "k2", "k3")]))
    return out


def write_correlator_energy_artifacts(records: list[dict[str, Any]], stage_dir: Path) -> dict[str, str]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("pt2_path"),
            record.get("ensemble"),
            record.get("hadron"),
            record.get("gfix"),
            record.get("volume"),
            record.get("lattice_spacing_fm"),
            record.get("source_operator"),
            record.get("sink_operator"),
            record.get("fitting_form"),
            record.get("channel"),
            record.get("momentum"),
        )
        deduped.setdefault(key, record)
    rows = list(deduped.values())
    if not rows:
        return {}

    stage_dir.mkdir(parents=True, exist_ok=True)
    row = np.arange(len(rows), dtype=int)
    numeric_keys = [
        "lattice_spacing_fm",
        "p_gev",
        "p2_gev2",
        "E0_lattice_mean",
        "E0_lattice_sdev",
        "E0_gev_mean",
        "E0_gev_sdev",
        "E0_gev2_mean",
        "E0_gev2_sdev",
        "workers",
    ]
    text_keys = [
        "job_id",
        "pt2_path",
        "ensemble",
        "hadron",
        "gfix",
        "volume",
        "source_operator",
        "sink_operator",
        "fitting_form",
        "channel",
        "momentum",
        "E0_gev_samples",
        "resample_mode",
        "sample_error_mode",
    ]
    data_vars = {
        key: ("row", np.asarray([float(item.get(key, np.nan)) for item in rows], dtype=float))
        for key in numeric_keys
    }
    data_vars.update({key: ("row", np.asarray([str(item.get(key, "")) for item in rows], dtype=object)) for key in text_keys})
    dataset = xr.Dataset(data_vars=data_vars, coords={"row": row})
    dataset.attrs.update(
        {
            "description": "Ground-state energies from final 2pt fit posteriors used in correlator_analysis.",
            "energy_unit": "GeV",
            "momentum_unit": "GeV",
            "dispersion_y": "E0_gev^2 from final 2pt fit posterior samples",
            "fit_model": "E0^2 = m^2 + k2 P^2 + k3 P^4 a^2",
        }
    )
    e0_path = stage_dir / "dispersion_relation.nc"
    dataset.to_netcdf(e0_path)

    fig, ax = default_plot()
    labels = sorted({str(item.get("ensemble", "")) for item in rows})
    for index, label in enumerate(labels):
        group = [item for item in rows if str(item.get("ensemble", "")) == label]
        group.sort(key=lambda item: (float(item["p2_gev2"]), str(item.get("channel", "")), str(item.get("momentum", ""))))
        ax.errorbar(
            [float(item["p2_gev2"]) for item in group],
            [float(item["E0_gev2_mean"]) for item in group],
            yerr=[float(item["E0_gev2_sdev"]) for item in group],
            label=label or "ensemble",
            color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            **ERRORBAR_STYLE,
        )
    p2_max = max(float(item["p2_gev2"]) for item in rows)
    p2_line = np.linspace(0.0, p2_max * 1.05 if p2_max > 0.0 else 1.0, 200)
    ax.plot(p2_line, p2_line, color="0.65", linestyle="--", linewidth=1.0, label=r"$E^2=p^2$")
    for index, label in enumerate(labels):
        group = [item for item in rows if str(item.get("ensemble", "")) == label]
        fit_group = []
        for item in group:
            e_samples = np.asarray(json.loads(str(item.get("E0_gev_samples") or "[]")), dtype=float)
            samples = e_samples**2
            if samples.size:
                fit_group.append((item, samples))
        if len(fit_group) < 2:
            continue
        n_sample = min(samples.size for _item, samples in fit_group)
        y_samples = np.asarray([samples[:n_sample] for _item, samples in fit_group], dtype=float).T
        y_samples = y_samples[np.all(np.isfinite(y_samples), axis=1)]
        if y_samples.shape[0] < 2:
            continue
        p2 = np.asarray([float(item["p2_gev2"]) for item, _samples in fit_group], dtype=float)
        a2 = np.asarray([(float(item["lattice_spacing_fm"]) / HBAR_C_GEV_FM) ** 2 for item, _samples in fit_group], dtype=float)
        design = np.column_stack([p2, p2**2 * a2])
        mode = str(fit_group[0][0].get("resample_mode") or "bootstrap")
        sample_error_mode = str(fit_group[0][0].get("sample_error_mode") or "covariance")
        mean, _ = sample_mean_and_sdev(y_samples, mode=mode, sample_error_mode=sample_error_mode, axis=0)
        y_data = sample_value_with_error(mean, y_samples, mode=mode, sample_error_mode=sample_error_mode, axis=0)

        def fcn(x_design: np.ndarray, p: gv.BufferDict):
            return p["m2"] + p["k2"] * x_design[:, 0] + p["k3"] * x_design[:, 1]

        prior = gv.BufferDict({"m2": gv.gvar(float(np.nanmin(mean)), 10.0), "k2": gv.gvar(1.0, 10.0), "k3": gv.gvar(0.0, 10.0)})
        p0 = {"m2": float(np.nanmin(mean)), "k2": 1.0, "k3": 0.0}
        fit = lsf.nonlinear_fit(data=(design, y_data), fcn=fcn, p0=p0, prior=prior, maxit=2000, svdcut=1e-12, fitter="scipy_least_squares")
        p0 = {key: float(fit.pmean[key]) for key in ("m2", "k2", "k3")}
        prior = gv.BufferDict({key: gv.gvar(p0[key], max(float(fit.psdev[key]) * 3.0, 1e-8)) for key in ("m2", "k2", "k3")})
        workers = max(int(item.get("workers", 1) or 1) for item in group)
        batches = [batch.tolist() for batch in np.array_split(np.arange(n_sample), min(workers, n_sample)) if batch.size]
        payload = gv.dumps({"design": design, "samples": y_samples, "template": y_samples, "p0": p0, "prior": prior, "resample_mode": mode, "sample_error_mode": sample_error_mode})
        if workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, n_sample)) as executor:
                fit_results = [item for future in [executor.submit(_fit_dispersion_sample_batch, payload, batch) for batch in batches] for item in future.result()]
        else:
            fit_results = _fit_dispersion_sample_batch(payload, batches[0])
        params = np.asarray([values for _sample, values in sorted(fit_results)], dtype=float)
        p4a2_line = p2_line**2 * float(np.mean(a2))
        curves = params[:, 0, None] + params[:, 1, None] * p2_line[None, :] + params[:, 2, None] * p4a2_line[None, :]
        band_mean, band_sdev = sample_mean_and_sdev(curves, mode=mode, sample_error_mode=sample_error_mode, axis=0)
        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        ax.plot(p2_line, band_mean, color=color, linewidth=1.0, alpha=0.75, label="_nolegend_")
        ax.fill_between(p2_line, band_mean - band_sdev, band_mean + band_sdev, color=color, alpha=0.18, linewidth=0.0, label="_nolegend_")
    ax.set_xlabel(r"$p^2\,[\mathrm{GeV}^2]$", **FONT_SIZE)
    ax.set_ylabel(r"$E_0^2\,[\mathrm{GeV}^2]$", **FONT_SIZE)
    ax.set_title("Dispersion relation", **FONT_SIZE)
    ax.legend(**{**LEGEND_SETS, "loc": "upper left"})
    fig.tight_layout()
    pdf_path = stage_dir / "dispersion_relation.pdf"
    svg_path = stage_dir / "dispersion_relation.svg"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "E0_artifact": str(e0_path),
        "dispersion_relation_plot": str(pdf_path),
        "dispersion_relation_image": str(svg_path),
    }


def _job_sample_quality_series(
    jobs: list[dict[str, Any]], key: str
) -> list[tuple[str, np.ndarray]]:
    series: list[tuple[str, np.ndarray]] = []
    for item in jobs:
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        values = np.asarray(result.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        series.append((str(item.get("job_id") or "job"), values))
    return series


def write_correlator_sample_quality_artifacts(
    jobs: list[dict[str, Any]], stage_dir: Path
) -> dict[str, str]:
    """Write stage-level CDF/histogram SVGs of per-sample fit quality."""
    q_series = _job_sample_quality_series(jobs, "sample_fit_Q")
    chi2_series = _job_sample_quality_series(jobs, "sample_fit_chi2_dof")
    if not q_series and not chi2_series:
        return {}
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if q_series:
        q_stem = stage_dir / "sample_fit_quality_Q"
        fig, _ = plot_sample_fit_quality_cdf(q_series, save_path=q_stem)
        plt.close(fig)
        artifacts["sample_fit_quality_Q_plot"] = str(q_stem.with_suffix(".pdf"))
        artifacts["sample_fit_quality_Q_image"] = str(q_stem.with_suffix(".svg"))
    if chi2_series:
        chi2_stem = stage_dir / "sample_fit_quality_chi2"
        fig, _ = plot_sample_fit_quality_chi2(chi2_series, save_path=chi2_stem)
        plt.close(fig)
        artifacts["sample_fit_quality_chi2_plot"] = str(chi2_stem.with_suffix(".pdf"))
        artifacts["sample_fit_quality_chi2_image"] = str(chi2_stem.with_suffix(".svg"))
    return artifacts


def _pt2_re_fcn_with_suffix(t: np.ndarray, p: dict, Lt: int, nstate: int = 2, suffix: str = "") -> np.ndarray:
    """Real part of an n-state two-point correlator, with optional parameter suffix."""
    energies = _state_energies(p, nstate, suffix)
    val = 0.0
    for state, energy in enumerate(energies):
        z = p[_state_key("z", state, suffix)]
        val = val + z**2 / (2 * energy) * (np.exp(-energy * t) + np.exp(-energy * (Lt - t)))
    return val


def pt2_re_fcn(t: np.ndarray, p: dict, Lt: int, nstate: int = 2) -> np.ndarray:
    """Real part of the n-state two-point correlator (symmetric about Lt/2)."""
    return _pt2_re_fcn_with_suffix(t, p, Lt, nstate=nstate)


def qda_mixed_pt2_re_fcn(
    t: np.ndarray, p: dict, Lt: int, nstate: int = 2
) -> np.ndarray:
    """qDA ``bz=0`` two-point function with distinct source/sink overlaps."""
    energies = _state_energies(p, nstate)
    value = 0.0
    for state, energy in enumerate(energies):
        value = value + (
            p[f"z{state}"]
            * p[f"zprime{state}"]
            / (2 * energy)
            * (np.exp(-energy * t) + np.exp(-energy * (Lt - t)))
        )
    return value


def _qda_denominator_fcn(
    t: np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int,
    qda_denominator_mode: str,
) -> np.ndarray:
    if qda_denominator_mode == "local":
        return pt2_re_fcn(t, p, Lt, nstate=nstate)
    if qda_denominator_mode == "nonlocal_bz0":
        return qda_mixed_pt2_re_fcn(t, p, Lt, nstate=nstate)
    raise ValueError(
        "qda_denominator_mode must be 'local' or 'nonlocal_bz0'"
    )


def pt3_ratio_fcn(
    t: np.ndarray,
    tau: np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "re",
) -> np.ndarray:
    """Real (``part='re'``) or imaginary (``part='im'``) n-state 3pt/2pt ratio."""
    energies = _state_energies(p, nstate)

    numerator = 0.0
    for src, src_e in enumerate(energies):
        for snk, snk_e in enumerate(energies):
            matrix_element = p[f"O{min(src, snk)}{max(src, snk)}_{part}"]
            numerator = numerator + (
                matrix_element
                * p[f"z{src}"]
                * p[f"z{snk}"]
                * np.exp(-src_e * (t - tau))
                * np.exp(-snk_e * tau)
                / (2 * src_e)
                / (2 * snk_e)
            )
    return numerator / pt2_re_fcn(t, p, Lt, nstate=nstate)


def qda_fcn(
    t: np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "re",
) -> np.ndarray:
    """Real or imaginary qDA two-point numerator spectral decomposition."""
    energies = _state_energies(p, nstate)
    value = 0.0
    for state, energy in enumerate(energies):
        value = value + (
            p[f"z{state}"]
            * p[f"O0{state}_{part}"]
            / (2 * energy)
            * (np.exp(-energy * t) + np.exp(-energy * (Lt - t)))
        )
    return value


def qda_ratio_fcn(
    t: np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "re",
    qda_denominator_mode: str = "local",
) -> np.ndarray:
    """qDA numerator divided by the selected two-point model."""
    return qda_fcn(t, p, Lt, nstate=nstate, part=part) / _qda_denominator_fcn(
        t,
        p,
        Lt,
        nstate=nstate,
        qda_denominator_mode=qda_denominator_mode,
    )


def pt3_nonbreit_ratio_fcn(
    t: np.ndarray,
    tau: np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "re",
) -> np.ndarray:
    """Non-forward raw ratio model without the external kinematic prefactor.

    The data ratio deliberately omits 2*sqrt(E0_f*E0_i)/(E0_f+E0_i). The final
    matrix element is therefore extracted as O00/(E0_f+E0_i), which reduces to
    O00/(2*E0) in the forward limit.
    """
    energies_i = _state_energies(p, nstate, "_i")
    energies_f = _state_energies(p, nstate, "_f")
    numerator = 0.0
    for snk, snk_e in enumerate(energies_f):
        for src, src_e in enumerate(energies_i):
            matrix_element = p[f"O{snk}{src}_{part}"]
            numerator = numerator + (
                matrix_element
                * p[_state_key("z", snk, "_f")]
                * p[_state_key("z", src, "_i")]
                * np.exp(-snk_e * (t - tau))
                * np.exp(-src_e * tau)
                / (2 * snk_e)
                / (2 * src_e)
            )
    c2_i_ts_tau = _pt2_re_fcn_with_suffix(t - tau, p, Lt, nstate=nstate, suffix="_i")
    c2_i_tau = _pt2_re_fcn_with_suffix(tau, p, Lt, nstate=nstate, suffix="_i")
    c2_i_t = _pt2_re_fcn_with_suffix(t, p, Lt, nstate=nstate, suffix="_i")
    c2_f_ts_tau = _pt2_re_fcn_with_suffix(t - tau, p, Lt, nstate=nstate, suffix="_f")
    c2_f_tau = _pt2_re_fcn_with_suffix(tau, p, Lt, nstate=nstate, suffix="_f")
    c2_f_t = _pt2_re_fcn_with_suffix(t, p, Lt, nstate=nstate, suffix="_f")
    ratio_factor = (c2_i_ts_tau * c2_f_tau * c2_f_t) / (c2_f_ts_tau * c2_i_tau * c2_i_t)
    return numerator / c2_f_t * gv.sqrt(ratio_factor)


def pt2_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for an n-state two-point fit."""
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(1, 10)
    for state in range(1, nstate):
        prior[f"log(dE{state})"] = gv.gvar(0, 1)
    for state in range(nstate):
        prior[f"z{state}"] = gv.gvar(1, 10) / 3**state
    return prior


def qda_pt2_prior(
    nstate: int = 2, *, qda_denominator_mode: str = "local"
) -> gv.BufferDict:
    """Two-point prior for the selected qDA denominator overlap structure."""
    prior = pt2_prior(nstate)
    if qda_denominator_mode == "nonlocal_bz0":
        for state in range(nstate):
            prior[f"zprime{state}"] = gv.gvar(1, 10) / 3**state
    elif qda_denominator_mode != "local":
        raise ValueError(
            "qda_denominator_mode must be 'local' or 'nonlocal_bz0'"
        )
    return prior


def _pt2_prior_with_suffix(nstate: int, suffix: str) -> gv.BufferDict:
    prior = gv.BufferDict()
    prior[_state_key("E0", suffix=suffix)] = gv.gvar(1, 10)
    for state in range(1, nstate):
        prior[f"log({_state_key('dE', state, suffix)})"] = gv.gvar(0, 1)
    for state in range(nstate):
        prior[_state_key("z", state, suffix)] = gv.gvar(1, 10) / 3**state
    return prior


def pt3_ratio_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for an n-state 3pt/2pt ratio fit (adds O_ij matrix elements)."""
    prior = pt2_prior(nstate)
    for row in range(nstate):
        for col in range(row, nstate):
            prior[f"O{row}{col}_re"] = gv.gvar(1, 10)
            prior[f"O{row}{col}_im"] = gv.gvar(1, 10)
    return prior


def qda_ratio_prior(
    nstate: int = 2, *, qda_denominator_mode: str = "local"
) -> gv.BufferDict:
    """Two-point spectral prior extended by qDA source-to-state amplitudes."""
    prior = qda_pt2_prior(
        nstate, qda_denominator_mode=qda_denominator_mode
    )
    for state in range(nstate):
        prior[f"O0{state}_re"] = gv.gvar(1, 10)
        prior[f"O0{state}_im"] = gv.gvar(0, 10)
    return prior


def pt3_nonbreit_ratio_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for a non-forward ratio with separate initial/final spectra."""
    prior = gv.BufferDict()
    prior.update(_pt2_prior_with_suffix(nstate, "_i"))
    prior.update(_pt2_prior_with_suffix(nstate, "_f"))
    for snk in range(nstate):
        for src in range(nstate):
            prior[f"O{snk}{src}_re"] = gv.gvar(1, 10)
            prior[f"O{snk}{src}_im"] = gv.gvar(1, 10)
    return prior


def _fh_extra_prior(nstate: int = 2) -> gv.BufferDict:
    """Nuisance priors for the FH finite-difference summed-ratio model."""
    if nstate > 2:
        raise ValueError("FH fits currently support nstate <= 2")
    prior = gv.BufferDict()
    if nstate == 1:
        return prior
    for part in ("re", "im"):
        prior[f"sum_{part}_excited_coeff"] = gv.gvar(0, 10)
        prior[f"sum_{part}_offset"] = gv.gvar(0, 10)
        prior[f"sum_{part}_exp_offset"] = gv.gvar(0, 10)
    prior["sum_den_exp_coeff"] = gv.gvar(0, 10)
    return prior


def fh_prior(nstate: int = 2) -> gv.BufferDict:
    """Broad priors for an FH-only fit."""
    if nstate > 2:
        raise ValueError("FH fits currently support nstate <= 2")
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(1, 10)
    for state in range(1, nstate):
        prior[f"log(dE{state})"] = gv.gvar(0, 1)
    prior["O00_re"] = gv.gvar(1, 10)
    prior["O00_im"] = gv.gvar(1, 10)
    prior.update(_fh_extra_prior(nstate))
    return prior


def _joint_fh_prior(nstate: int = 2) -> gv.BufferDict:
    """FH prior with 2pt overlap parameters for simultaneous 2pt+FH fits."""
    prior = pt2_prior(nstate)
    prior["O00_re"] = gv.gvar(1, 10)
    prior["O00_im"] = gv.gvar(1, 10)
    prior.update(_fh_extra_prior(nstate))
    return prior


def _ratio_fh_prior(nstate: int = 2) -> gv.BufferDict:
    """Ratio prior extended with the FH summed-ratio nuisance parameters."""
    prior = pt3_ratio_prior(nstate)
    prior.update(_fh_extra_prior(nstate))
    return prior


def sum_ratio_fcn(
    t: np.ndarray,
    tau_cut: int,
    p: dict,
    *,
    nstate: int = 2,
    part: str = "re",
) -> np.ndarray:
    """Summed-ratio ansatz used to define the FH finite difference."""
    if nstate > 2:
        raise ValueError("summed-ratio fit functions currently support nstate <= 2")
    e0 = p["E0"]
    if nstate == 1:
        return p[f"O00_{part}"] * (t - 2 * tau_cut + 1) / (2 * e0)
    d_e1 = p["dE1"]
    exp_term = np.exp(-d_e1 * t)
    numerator = (
        p[f"O00_{part}"]
        * (t - 2 * tau_cut + 1)
        * (1 + p[f"sum_{part}_excited_coeff"] * exp_term)
        + p[f"sum_{part}_offset"]
        + p[f"sum_{part}_exp_offset"] * exp_term
    )
    denominator = 2 * e0 * (1 + p["sum_den_exp_coeff"] * exp_term)
    return numerator / denominator


def fh_fcn(
    t: np.ndarray,
    tau_cut: int,
    p: dict,
    *,
    nstate: int = 2,
    part: str = "re",
    dt: int | float = 1,
) -> np.ndarray:
    """FH ansatz from neighboring summed-ratio finite differences."""
    if nstate > 2:
        raise ValueError("FH fits currently support nstate <= 2")
    if nstate == 1:
        return p[f"O00_{part}"] / (2 * p["E0"]) + np.asarray(t, dtype=float) * 0
    return (
        sum_ratio_fcn(np.asarray(t, dtype=float) + dt, tau_cut, p, nstate=nstate, part=part)
        - sum_ratio_fcn(np.asarray(t, dtype=float), tau_cut, p, nstate=nstate, part=part)
    ) / dt


# --- fit constructors --------------------------------------------------------


def _check_rescale(correlator_rescale: float) -> float:
    scale = float(correlator_rescale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"correlator_rescale must be positive and finite, got {correlator_rescale!r}")
    return scale


def _parts(part: str) -> tuple[str, ...]:
    if part == "both":
        return ("re", "im")
    if part in ("re", "im"):
        return (part,)
    raise ValueError("part must be 're', 'im', or 'both'")


def _normalise_fitting_form(value: str | None) -> str:
    form = "Breit" if value is None else str(value)
    if form not in {"Breit", "NonBreit"}:
        raise ValueError("fitting_form must be 'Breit' or 'NonBreit'")
    return form


def _normalise_fit_scope(value: str | None) -> tuple[str, str]:
    raw = "3pt_ratio" if value is None else str(value).strip().lower().replace(" ", "")
    if raw == "3pt_ratio":
        return "3pt_ratio", "3pt_ratio"
    if raw == "fh":
        return "FH", "fh"
    if raw == "3pt_ratio+fh":
        return "3pt_ratio+FH", "3pt_ratio_fh"
    if raw == "qda_ratio":
        return "qda_ratio", "qda_ratio"
    raise ValueError(
        "fit_scope must be '3pt_ratio', 'FH', '3pt_ratio+FH', or 'qda_ratio', "
        f"got {value!r}"
    )


def _validate_scope_form(scope: str, fitting_form: str) -> None:
    if "FH" in scope and fitting_form == "NonBreit":
        raise ValueError("fit_scope values containing 'FH' currently require fitting_form='Breit'")
    if scope == "qda_ratio" and fitting_form != "Breit":
        raise ValueError("fit_scope='qda_ratio' requires fitting_form='Breit'")


def _ratio_points(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
) -> tuple[list[int], list[int], list, list]:
    """Flatten ratio data over ``tsep_ls`` and ``tau in [tau_cut, tsep+1-tau_cut)``."""
    ts: list[int] = []
    taus: list[int] = []
    re_vals: list = []
    im_vals: list = []
    for tsep in tsep_ls:
        if tsep not in ratio_re or tsep not in ratio_im:
            raise KeyError(f"ratio data missing tsep {tsep}")
        tau_range = range(tau_cut, tsep + 1 - tau_cut)
        if len(tau_range) == 0:
            raise ValueError(f"empty tau window for tsep {tsep} with tau_cut {tau_cut}")
        re_row = np.asarray(ratio_re[tsep], dtype=object)
        im_row = np.asarray(ratio_im[tsep], dtype=object)
        for tau in tau_range:
            ts.append(tsep)
            taus.append(tau)
            re_vals.append(re_row[tau])
            im_vals.append(im_row[tau])
    return ts, taus, re_vals, im_vals


def _summed_ratio_samples(ratio: dict[int, np.ndarray], tsep_ls: list[int], tau_cut: int) -> dict[int, np.ndarray]:
    """Sum ratio samples over tau in [tau_cut, tsep - tau_cut]."""
    summed: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        if tsep not in ratio:
            raise KeyError(f"ratio data missing tsep {tsep}")
        row = np.asarray(ratio[tsep], dtype=object)
        start = int(tau_cut)
        stop = int(tsep) - int(tau_cut) + 1
        if row.shape[-1] < stop:
            raise ValueError(f"requested tau upper bound exceeds available tau range for tsep={tsep}")
        if start >= stop:
            raise ValueError(f"tau_cut={tau_cut} leaves no tau points for tsep={tsep}")
        summed[tsep] = np.sum(row[..., start:stop], axis=-1)
    return summed


def _fh_samples_from_ratios(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build FH samples by finite-differencing adjacent summed ratios."""
    tseps = [int(tsep) for tsep in tsep_ls]
    if len(tseps) < 2:
        raise ValueError("FH construction requires at least two tsep values")
    sum_re = _summed_ratio_samples(ratio_re, tseps, tau_cut)
    sum_im = _summed_ratio_samples(ratio_im, tseps, tau_cut)
    fh_re_cols = []
    fh_im_cols = []
    for t0, t1 in zip(tseps[:-1], tseps[1:]):
        dt = t1 - t0
        if dt <= 0:
            raise ValueError("tsep values must be strictly increasing for FH construction")
        fh_re_cols.append((sum_re[t1] - sum_re[t0]) / dt)
        fh_im_cols.append((sum_im[t1] - sum_im[t0]) / dt)
    return np.stack(fh_re_cols, axis=-1), np.stack(fh_im_cols, axis=-1)


def _fh_dt(tsep_ls: list[int]) -> int | float:
    if len(tsep_ls) < 2:
        raise ValueError("FH fit requires at least two tsep values")
    return int(tsep_ls[1]) - int(tsep_ls[0])


def fit_two_point(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    *,
    nstate: int = 2,
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict | None = None,
    p0: dict[str, float] | None = None,
    qda_denominator_mode: str = "local",
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator over ``[tmin, tmax)`` with an n-state ansatz."""
    fit_t = np.arange(tmin, tmax, dtype=int)
    fit_y = np.asarray(pt2_gv)[fit_t] * rescale
    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(fit_t, fit_y),
        prior=(
            prior
            if prior is not None
            else qda_pt2_prior(
                nstate, qda_denominator_mode=qda_denominator_mode
            )
        ),
        fcn=lambda t, p: _qda_denominator_fcn(
            t,
            p,
            Lt,
            nstate=nstate,
            qda_denominator_mode=qda_denominator_mode,
        ),
        svdcut=svdcut,
        maxit=10000,
        **kwargs,
    )


def fit_matrix_element(
    ratio_re: dict[int, np.ndarray] | np.ndarray,
    ratio_im: dict[int, np.ndarray] | np.ndarray,
    tsep_ls: list[int] | None,
    tau_cut: int | None,
    Lt: int,
    *,
    strategy: str,
    fit_scope: str,
    fitting_form: str,
    pt2_gv: np.ndarray | None = None,
    pt2_f_gv: np.ndarray | None = None,
    tmin: int | None = None,
    tmax: int | None = None,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
    qda_denominator_mode: str = "local",
) -> lsf.nonlinear_fit:
    """Fit the configured 2pt, ratio, and FH observables with one numeric core."""
    if strategy not in {"joint", "chained", "independent"}:
        raise ValueError(f"unsupported fit strategy {strategy!r}")
    if fit_scope not in {"3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"}:
        raise ValueError(f"unsupported fit scope {fit_scope!r}")
    if fitting_form not in {"Breit", "NonBreit"}:
        raise ValueError(f"unsupported fitting form {fitting_form!r}")
    _validate_scope_form(fit_scope, fitting_form)
    parts = _parts(part)
    x_data: dict[str, np.ndarray] = {}
    y_data: dict[str, Any] = {}

    # joint fits include one forward 2pt channel or separate initial/final channels
    if strategy == "joint":
        if pt2_gv is None or tmin is None or tmax is None:
            raise ValueError("joint fits require pt2_gv, tmin, and tmax")
        fit_t = np.arange(tmin, tmax, dtype=int)
        x_data["pt2_t"] = fit_t
        if fitting_form == "NonBreit":
            if pt2_f_gv is None:
                raise ValueError("NonBreit joint fits require pt2_f_gv")
            y_data["pt2_i"] = np.asarray(pt2_gv)[fit_t] * rescale
            y_data["pt2_f"] = np.asarray(pt2_f_gv)[fit_t] * rescale
        else:
            y_data["pt2"] = np.asarray(pt2_gv)[fit_t] * rescale

    # ratio and FH channels share the same component selection and matrix elements
    if fit_scope in {"3pt_ratio", "3pt_ratio+FH"}:
        if not isinstance(ratio_re, dict) or not isinstance(ratio_im, dict) or tsep_ls is None or tau_cut is None:
            raise ValueError("3pt ratio fits require tsep-indexed ratio data and tau_cut")
        ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)
        x_data["ratio_t"] = np.asarray(ts, dtype=float)
        x_data["ratio_tau"] = np.asarray(taus, dtype=float)
        if "re" in parts:
            y_data["ratio_re"] = re_vals
        if "im" in parts:
            y_data["ratio_im"] = im_vals
    elif fit_scope == "qda_ratio":
        if isinstance(ratio_re, dict) or isinstance(ratio_im, dict) or tmin is None or tmax is None:
            raise ValueError("qda_ratio fits require array ratio data and a 2pt window")
        qda_t = np.arange(tmin, tmax, dtype=int)
        x_data["qda_t"] = qda_t
        if "re" in parts:
            y_data["qda_re"] = np.asarray(ratio_re, dtype=object)[qda_t]
        if "im" in parts:
            y_data["qda_im"] = np.asarray(ratio_im, dtype=object)[qda_t]

    dt: int | float | None = None
    if "FH" in fit_scope:
        if not isinstance(ratio_re, dict) or not isinstance(ratio_im, dict) or tsep_ls is None or tau_cut is None:
            raise ValueError("FH fits require tsep-indexed ratio data and tau_cut")
        fh_re, fh_im = _fh_samples_from_ratios(ratio_re, ratio_im, tsep_ls, tau_cut)
        x_data["fh_t"] = np.asarray(tsep_ls[:-1], dtype=float)
        dt = _fh_dt(tsep_ls)
        if "re" in parts:
            y_data["fh_re"] = np.asarray(fh_re, dtype=object)
        if "im" in parts:
            y_data["fh_im"] = np.asarray(fh_im, dtype=object)

    # evaluate exactly the channels assembled above from the shared posterior
    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        if strategy == "joint":
            if fitting_form == "NonBreit":
                values["pt2_i"] = _pt2_re_fcn_with_suffix(
                    x["pt2_t"], p, Lt, nstate=nstate, suffix="_i"
                )
                values["pt2_f"] = _pt2_re_fcn_with_suffix(
                    x["pt2_t"], p, Lt, nstate=nstate, suffix="_f"
                )
            else:
                values["pt2"] = _qda_denominator_fcn(
                    x["pt2_t"],
                    p,
                    Lt,
                    nstate=nstate,
                    qda_denominator_mode=(
                        qda_denominator_mode
                        if fit_scope == "qda_ratio"
                        else "local"
                    ),
                )
        if fit_scope in {"3pt_ratio", "3pt_ratio+FH"}:
            ratio_fcn = pt3_nonbreit_ratio_fcn if fitting_form == "NonBreit" else pt3_ratio_fcn
            for component in parts:
                values[f"ratio_{component}"] = ratio_fcn(
                    x["ratio_t"],
                    x["ratio_tau"],
                    p,
                    Lt,
                    nstate=nstate,
                    part=component,
                )
        elif fit_scope == "qda_ratio":
            for component in parts:
                values[f"qda_{component}"] = qda_ratio_fcn(
                    x["qda_t"],
                    p,
                    Lt,
                    nstate=nstate,
                    part=component,
                    qda_denominator_mode=qda_denominator_mode,
                )
        if "FH" in fit_scope:
            for component in parts:
                values[f"fh_{component}"] = fh_fcn(
                    x["fh_t"],
                    tau_cut,
                    p,
                    nstate=nstate,
                    part=component,
                    dt=dt if dt is not None else 1,
                )
        return values

    # hand the complete channel dictionary to one nonlinear fit
    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_data, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


# --- fit records, selection, and model averaging ----------------------------


def _record(fit: lsf.nonlinear_fit, **meta: Any) -> dict[str, Any]:
    """Wrap a fit with its window metadata and quality metrics."""
    record = dict(meta)
    record.update(
        chi2_dof=float(fit.chi2 / fit.dof),
        Q=float(fit.Q),
        logGBF=float(fit.logGBF),
        fit=fit,
    )
    return record


def _selected_record_quality(
    records: list[dict[str, Any]], weights: np.ndarray
) -> tuple[float, float]:
    """Return Q and chi2/dof for the highest-weight (or only) usable record."""
    primary = records[int(np.argmax(np.asarray(weights, dtype=float)))]
    return float(primary["Q"]), float(primary["chi2_dof"])


def _append_finite_sample_quality(
    q_values: list[float],
    chi2_values: list[float],
    result: dict[str, Any],
) -> None:
    """Keep selected-sample Q and chi2/dof when the nonlinear fit returned."""
    if result.get("error") is not None:
        return
    try:
        q_value = float(result["Q"])
        chi2_dof = float(result["chi2_dof"])
    except (KeyError, TypeError, ValueError):
        return
    if np.isfinite(q_value) and np.isfinite(chi2_dof):
        q_values.append(q_value)
        chi2_values.append(chi2_dof)


def select_best(records: list[dict[str, Any]], *, q_min: float = 0.05) -> tuple[int, bool]:
    """Pick max logGBF among Q-passing windows; otherwise the max-Q window."""
    if not records:
        raise ValueError("no fit windows to select from")
    passing = [i for i, rec in enumerate(records) if rec["Q"] >= q_min]
    if passing:
        return max(passing, key=lambda i: records[i]["logGBF"]), False
    return max(range(len(records)), key=lambda i: records[i]["Q"]), True


def _loggbf_weights(records: list[dict[str, Any]]) -> np.ndarray:
    log_gbf = np.array([rec["logGBF"] for rec in records], dtype=float)
    weights = np.exp(log_gbf - np.max(log_gbf))
    return weights / np.sum(weights)


DEFAULT_PRIOR_WIDTH = [0.5, 1.0, 2.0]


def _normalise_prior_width(prior_width: float | list[float] | tuple[float, ...] | None) -> list[float]:
    """Return positive prior-width factors for fit-function scans."""
    if prior_width is None:
        values = DEFAULT_PRIOR_WIDTH
    elif isinstance(prior_width, (list, tuple)):
        values = list(prior_width)
    else:
        values = [prior_width]
    widths = [float(value) for value in values]
    if not widths:
        raise ValueError("prior_width must contain at least one value")
    if any((not np.isfinite(width)) or width <= 0.0 for width in widths):
        raise ValueError(f"prior_width values must be positive and finite, got {prior_width!r}")
    return widths


def _vary_prior_width(prior: gv.BufferDict, prior_width: float) -> gv.BufferDict:
    """Copy a prior while multiplying every parameter width by ``prior_width``."""
    width = float(prior_width)
    varied = gv.BufferDict()
    for key in prior:
        value = prior[key]
        varied[key] = gv.gvar(gv.mean(value), gv.sdev(value) * width)
    return varied


def bayesian_average(values: np.ndarray, weights: np.ndarray) -> gv.GVar:
    """Combine fit values with statistical and systematic spread (BMA)."""
    mean = np.sum(weights * gv.mean(values))
    var = np.sum(weights * (gv.sdev(values) ** 2 + gv.mean(values) ** 2)) - mean**2
    return gv.gvar(mean, np.sqrt(var))


def _weighted_model_sdev(values: np.ndarray, weights: np.ndarray, *, center: float | None = None) -> float:
    """Weighted spread of model central values around their combined mean."""
    vals = np.asarray(values, dtype=float)
    wgt = np.asarray(weights, dtype=float)
    finite = np.isfinite(vals) & np.isfinite(wgt)
    if not np.any(finite):
        return float("nan")
    vals = vals[finite]
    wgt = wgt[finite]
    total = float(np.sum(wgt))
    if total <= 0:
        return float("nan")
    wgt = wgt / total
    avg = float(np.sum(wgt * vals)) if center is None else float(center)
    return float(np.sqrt(np.sum(wgt * (vals - avg) ** 2)))


DATA_WINDOW_CHI2_DOF_TOLERANCE = 0.25


def _prior_parameter_count(prior: gv.BufferDict) -> int:
    """Count scalar fit parameters represented by a prior BufferDict."""
    return int(sum(np.size(gv.mean(prior[key])) for key in prior))


def _ratio_data_count(tsep_ls: list[int], tau_cut: int) -> int:
    """Count 3pt ratio tau points before real/imag component expansion."""
    count = 0
    for tsep in tsep_ls:
        count += max(int(tsep) + 1 - 2 * int(tau_cut), 0)
    return int(count)


def _fit_data_count(
    spec: dict[str, Any],
    *,
    strategy: str,
    fit_scope: str,
    part: str,
    fitting_form: str,
) -> int:
    """Count data points implied by a correlator fit window."""
    components = len(_parts(part))
    pt2_points = max(int(spec["tmax"]) - int(spec["tmin"]), 0)
    tsep_ls = [int(t) for t in spec.get("tsep_ls", [])]
    tau_cut = int(spec.get("tau_cut", 0))
    ratio_points = _ratio_data_count(tsep_ls, tau_cut)
    fh_points = max(len(tsep_ls) - 1, 0)

    total = 0
    if strategy == "joint":
        total += pt2_points * (2 if fitting_form == "NonBreit" else 1)
    if fit_scope in {"3pt_ratio", "3pt_ratio+FH"}:
        total += ratio_points * components
    if fit_scope == "qda_ratio":
        total += pt2_points * components
    if "FH" in fit_scope:
        total += fh_points * components
    return int(total)


def _with_fit_size_metadata(
    metadata: dict[str, Any],
    *,
    n_data: int,
    n_params: int,
) -> dict[str, Any]:
    """Attach determinedness metadata used by data-window selection."""
    return {
        **metadata,
        "n_data": int(n_data),
        "n_params": int(n_params),
        "dof_is_positive": int(n_data) > int(n_params),
    }


def select_data_window(
    records: list[dict[str, Any]],
    *,
    q_min: float = 0.05,
    chi2_dof_tolerance: float = DATA_WINDOW_CHI2_DOF_TOLERANCE,
) -> tuple[int, bool]:
    """Select a data window without comparing raw logGBF across data sets."""
    if not records:
        raise ValueError("no fit windows to select from")
    overdetermined = [
        i
        for i, rec in enumerate(records)
        if int(rec.get("n_data", 0)) > int(rec.get("n_params", 0))
        and np.isfinite(float(rec.get("chi2_dof", np.inf)))
    ]
    if not overdetermined:
        raise ValueError("no overdetermined fit windows to select from")

    passing = [i for i in overdetermined if float(records[i]["Q"]) >= q_min]
    candidate_indices = passing or overdetermined
    fallback = not bool(passing)
    best_chi2 = min(float(records[i]["chi2_dof"]) for i in candidate_indices)
    comparable = [
        i
        for i in candidate_indices
        if float(records[i]["chi2_dof"]) <= best_chi2 + float(chi2_dof_tolerance)
    ]
    return max(
        comparable,
        key=lambda i: (
            int(records[i]["n_data"]),
            -float(records[i]["chi2_dof"]),
            float(records[i]["Q"]),
        ),
    ), fallback


def _window_candidate_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    """Stable identity for one bare-matrix window candidate."""
    tsep_ls = meta.get("tsep_ls")
    return (
        str(meta.get("fit_strategy", "")),
        str(meta.get("fit_scope", "")),
        int(meta.get("nstate", 0)),
        float(meta.get("prior_width", 0.0)),
        int(meta.get("tmin", -1)),
        int(meta.get("tmax", -1)),
        tuple(int(t) for t in tsep_ls) if tsep_ls is not None else (),
        int(meta.get("tau_cut", -1)),
    )


def _summarise_cross_z_feasibility(
    per_z: dict[int, dict[str, Any]],
    tune_z_values: list[int],
) -> dict[str, Any]:
    """Aggregate cross-z feasibility metrics for one window candidate."""
    failure_reasons: dict[str, str] = {}
    for z in tune_z_values:
        diag = per_z.get(z)
        if diag is None or not diag.get("usable"):
            reason = "missing" if diag is None else str(diag.get("reason", "unknown"))
            failure_reasons[str(z)] = reason
    usable_diags = [per_z[z] for z in tune_z_values if per_z.get(z, {}).get("usable")]
    feasible = not failure_reasons
    min_q = min(float(diag["Q"]) for diag in usable_diags) if usable_diags else None
    worst_chi2 = max(float(diag["chi2_dof"]) for diag in usable_diags) if usable_diags else None
    if not feasible and failure_reasons:
        bottleneck_z = min(int(z) for z in failure_reasons)
    elif usable_diags:
        usable_z = [z for z in tune_z_values if per_z.get(z, {}).get("usable")]
        bottleneck_z = min(usable_z, key=lambda z: float(per_z[z]["Q"]))
    else:
        bottleneck_z = None
    tune_z_diagnostics = {str(z): per_z[z] for z in tune_z_values if z in per_z}
    return {
        "tune_z_diagnostics": tune_z_diagnostics,
        "feasible_at_all_tune_z": feasible,
        "bottleneck_z": bottleneck_z,
        "min_Q": min_q,
        "worst_chi2_dof": worst_chi2,
        "failure_reasons": failure_reasons,
    }


def _fit_usable(
    fit: lsf.nonlinear_fit,
    template: gv.BufferDict,
    *,
    sdev_floor: float = 1e-12,
    e0_floor: float = 1e-4,
) -> tuple[bool, str | None]:
    """Reject non-finite or numerically degenerate posteriors before sample fits."""
    for key in template:
        if key not in fit.p:
            return False, f"missing posterior {key}"
        mean = float(gv.mean(fit.p[key]))
        sdev = float(gv.sdev(fit.p[key]))
        if not np.isfinite(mean) or not np.isfinite(sdev):
            return False, f"non-finite posterior {key}"
        if sdev <= sdev_floor:
            return False, f"degenerate posterior {key}"
    for key in ("E0", "E0_i", "E0_f"):
        if key in fit.p and float(gv.mean(fit.p[key])) <= e0_floor:
            return False, f"non-physical {key}"
    return True, None


def _scaled_prior(
    fit: lsf.nonlinear_fit, template: gv.BufferDict, *, error_scale: float, prior_width: float = 1.0
) -> gv.BufferDict:
    """Use a fit posterior as a prior with inflated uncertainties."""
    prior = gv.BufferDict()
    for key in template:
        value = fit.p[key] if key in fit.p else template[key]
        prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * error_scale * float(prior_width))
    return prior


def _p0_from_fit(fit: lsf.nonlinear_fit, prior: gv.BufferDict) -> dict[str, float]:
    p0: dict[str, float] = {}
    for key in prior:
        value = fit.p[key] if key in fit.p else prior[key]
        p0[key] = float(gv.mean(value))
    return p0


def _anchor_pt2_prior(prior: gv.BufferDict, pt2_fit: lsf.nonlinear_fit, suffix: str = "") -> None:
    """Pin E0 and z0 of a ratio prior to widened 2pt posteriors (chained mode)."""
    for key in ("E0", "z0"):
        value = pt2_fit.p[key]
        prior[_state_key(key, suffix=suffix)] = gv.gvar(gv.mean(value), gv.sdev(value) * PT2_PRIOR_ERROR_SCALE)


def _anchor_qda_pt2_prior(
    prior: gv.BufferDict,
    pt2_fit: lsf.nonlinear_fit,
    nstate: int,
    *,
    qda_denominator_mode: str = "local",
) -> None:
    """Anchor the complete qDA spectrum to widened two-point posteriors."""
    keys = ["E0", *(f"log(dE{state})" for state in range(1, nstate)), *(f"z{state}" for state in range(nstate))]
    if qda_denominator_mode == "nonlocal_bz0":
        keys.extend(f"zprime{state}" for state in range(nstate))
    for key in keys:
        if key in prior and key in pt2_fit.p:
            value = pt2_fit.p[key]
            prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * PT2_PRIOR_ERROR_SCALE)


def _anchor_fh_energy_prior(prior: gv.BufferDict, pt2_fit: lsf.nonlinear_fit, nstate: int) -> None:
    """Pin FH energy priors to widened 2pt posteriors in chained mode."""
    for key in ("E0", *(f"log(dE{state})" for state in range(1, nstate))):
        if key in prior and key in pt2_fit.p:
            value = pt2_fit.p[key]
            prior[key] = gv.gvar(gv.mean(value), gv.sdev(value) * PT2_PRIOR_ERROR_SCALE)


def _overlaps(p: dict, nstate: int, rescale: float) -> dict[str, gv.GVar]:
    """Physical overlaps z_state / sqrt(rescale) for tuning logs."""
    overlap_rescale = np.sqrt(rescale)
    diag: dict[str, gv.GVar] = {}
    physical: list[gv.GVar] = []
    for state in range(nstate):
        key = f"z{state}"
        if key in p:
            value = p[key] / overlap_rescale
            physical.append(value)
            diag[f"{key}_physical"] = value
    if len(physical) >= 2 and gv.mean(physical[0]) != 0.0:
        diag["z1_over_z0_physical"] = physical[1] / physical[0]
    return diag


def _bare_matrix_element_from_fit(
    p: dict,
    *,
    part: str,
    fitting_form: str,
    fit_scope: str = "3pt_ratio",
    qda_denominator_mode: str = "local",
) -> Any:
    if fit_scope == "qda_ratio":
        denominator_overlap = (
            "zprime0" if qda_denominator_mode == "nonlocal_bz0" else "z0"
        )
        return p[f"O00_{part}"] / p[denominator_overlap]
    if fitting_form == "NonBreit":
        overlap_sign = -1.0 if gv.mean(p["z0_f"] * p["z0_i"]) < 0.0 else 1.0
        return overlap_sign * p[f"O00_{part}"] / (p["E0_f"] + p["E0_i"])
    return p[f"O00_{part}"] / (2 * p["E0"])


def _bare_matrix_element_mean_for_part(
    p: dict,
    *,
    output_part: str,
    fit_part: str,
    fitting_form: str,
    fit_scope: str = "3pt_ratio",
    qda_denominator_mode: str = "local",
) -> float:
    """Return zero for the component that was intentionally excluded from the fit."""
    if output_part not in _parts(fit_part):
        return 0.0
    return float(
        gv.mean(
            _bare_matrix_element_from_fit(
                p,
                part=output_part,
                fitting_form=fitting_form,
                fit_scope=fit_scope,
                qda_denominator_mode=qda_denominator_mode,
            )
        )
    )


def _ratio_prior_template(fitting_form: str, nstate: int) -> gv.BufferDict:
    if fitting_form == "NonBreit":
        return pt3_nonbreit_ratio_prior(nstate)
    return pt3_ratio_prior(nstate)


def _scope_prior_template(
    fitting_form: str,
    nstate: int,
    fit_scope: str,
    strategy: str,
    *,
    qda_denominator_mode: str = "local",
) -> gv.BufferDict:
    _validate_scope_form(fit_scope, fitting_form)
    if fit_scope == "3pt_ratio":
        return _ratio_prior_template(fitting_form, nstate)
    if fit_scope == "FH":
        return _joint_fh_prior(nstate) if strategy == "joint" else fh_prior(nstate)
    if fit_scope == "3pt_ratio+FH":
        return _ratio_fh_prior(nstate)
    if fit_scope == "qda_ratio":
        return qda_ratio_prior(
            nstate, qda_denominator_mode=qda_denominator_mode
        )
    raise ValueError(f"unsupported fit_scope {fit_scope!r}")


def _scope_prior_with_width(
    fitting_form: str,
    nstate: int,
    fit_scope: str,
    strategy: str,
    prior_width: float,
    *,
    qda_denominator_mode: str = "local",
) -> gv.BufferDict:
    return _vary_prior_width(
        _scope_prior_template(
            fitting_form,
            nstate,
            fit_scope,
            strategy,
            qda_denominator_mode=qda_denominator_mode,
        ),
        prior_width,
    )


def _fit_summary(rec: dict[str, Any], *, fallback: bool, index: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "index": index,
        "chi2_dof": float(rec["chi2_dof"]),
        "Q": float(rec["Q"]),
        "logGBF": float(rec["logGBF"]),
        "fallback_no_q_passing": bool(fallback),
    }
    for key in (
        "tmin",
        "tmax",
        "tsep_ls",
        "tau_cut",
        "nstate",
        "prior_width",
        "part",
        "fit_scope",
        "correlator_rescale",
        "n_data",
        "n_params",
        "dof_is_positive",
    ):
        if key in rec:
            summary[key] = rec[key]
    fit = rec.get("fit")
    if fit is not None and "nstate" in rec:
        for key, value in _overlaps(fit.p, rec["nstate"], rec.get("correlator_rescale", 1.0)).items():
            summary[key] = str(value)
    return summary


# --- data IO and resampling --------------------------------------------------


def _read_2pt(
    path: str,
    *,
    source_operator: str,
    sink_operator: str,
    momentum: str,
    temporal_extent: int | None = None,
    bT: int | None = None,
    bz: int | None = None,
) -> np.ndarray:
    """Read one 2pt dataset as a complex (n_cfg, Lt) array."""
    dset = f"{source_operator}/{sink_operator}/{momentum}"
    if (bT is None) != (bz is None):
        raise ValueError("qDA 2pt reads require bT and bz together")
    candidates = [dset]
    if bT is not None and bz is not None:
        candidates = [f"{dset}/bT{int(bT)}/bz{int(bz)}"]
        if "_nonlocal" in sink_operator:
            candidates.append(
                f"{source_operator}/{sink_operator}_bT{int(bT)}_bz{int(bz)}/{momentum}"
            )
        elif "_nonlocal" in source_operator:
            candidates.append(
                f"{source_operator}_bT{int(bT)}_bz{int(bz)}/{sink_operator}/{momentum}"
            )
    with h5py.File(path, "r") as h5f:
        selected = next((candidate for candidate in candidates if candidate in h5f), None)
        if selected is None:
            raise KeyError(
                f"none of the expected 2pt datasets exist in {path}: {candidates}"
            )
        data = np.swapaxes(np.asarray(h5f[selected]), 0, 1)
    if temporal_extent is not None and data.shape[1] != int(temporal_extent):
        raise ValueError(f"{path}:{selected} has Lt={data.shape[1]}, expected {temporal_extent} from manifest volume")
    return data


def _read_3pt(
    path: str,
    *,
    source_operator: str,
    sink_operator: str,
    current_operator: str,
    momentum: str,
    bT: int,
    bz: int,
    tsep: int,
) -> np.ndarray:
    """Read one 3pt slice as a complex (n_cfg, tsep+1) array."""
    dset = (
        f"{source_operator}/{sink_operator}/{current_operator}/{momentum}/"
        f"tsep{tsep}/bT{bT}/bz{bz}"
    )
    with h5py.File(path, "r") as h5f:
        data = np.swapaxes(np.asarray(h5f[dset]), 0, 1)
    if data.shape[1] != tsep + 1:
        raise ValueError(f"{path}:{dset} has ntau={data.shape[1]}, expected {tsep + 1} for tsep={tsep}")
    return data


def _resample_pt2(
    pt2_complex: np.ndarray,
    *,
    mode: str,
    n_boot: int,
    seed: int | None,
    bin_size: int = 1,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return real 2pt samples, complex 2pt samples, and shared bootstrap indices."""
    re_samples, indices = resample_config_samples(
        np.real(pt2_complex), mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
    )
    im_samples, _ = resample_config_samples(
        np.imag(pt2_complex), mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
    )
    return re_samples, re_samples + 1j * im_samples, indices


def _ratio_samples(pt2_complex_samples: np.ndarray, pt3_samples: np.ndarray, tsep: int) -> tuple[np.ndarray, np.ndarray]:
    ratio = pt3_samples / pt2_complex_samples[:, tsep][:, None]
    return np.real(ratio), np.imag(ratio)


def _non_forward_ratio_samples(
    pt2_i_complex_samples: np.ndarray,
    pt2_f_complex_samples: np.ndarray,
    pt3_samples: np.ndarray,
    tsep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the non-forward ratio without 2*sqrt(E0_f*E0_i)/(E0_f+E0_i).

    That kinematic factor is intentionally left to the fit/output stage, where
    O00 is converted to O00/(E0_f+E0_i).
    """
    tau = np.arange(tsep + 1, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = (
            pt2_i_complex_samples[:, tsep - tau]
            * pt2_f_complex_samples[:, tau]
            * pt2_f_complex_samples[:, tsep][:, None]
        ) / (
            pt2_f_complex_samples[:, tsep - tau]
            * pt2_i_complex_samples[:, tau]
            * pt2_i_complex_samples[:, tsep][:, None]
        )
        ratio = pt3_samples / pt2_f_complex_samples[:, tsep][:, None] * np.sqrt(correction)
    return np.real(ratio), np.imag(ratio)


def _recenter(mean: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Reuse ``template`` covariance with a replacement mean vector for one sample."""
    return gv.gvar(np.asarray(mean, dtype=float), gv.evalcov(template))


def _check_mode(resample_mode: str) -> str:
    mode = str(resample_mode)
    if mode not in ("bs", "jk"):
        raise ValueError(f"resample_mode must be 'bs' or 'jk', got {resample_mode!r}")
    return mode


def _normalise_pt3_paths(pt3_paths: dict[str, str] | list[str], *, tsep_ls: list[int]) -> dict[int, str]:
    if isinstance(pt3_paths, dict):
        return {int(key): str(value) for key, value in pt3_paths.items()}
    if len(pt3_paths) != len(tsep_ls):
        raise ValueError("pt3_paths list length must match tsep_ls")
    return {int(tsep): str(path) for tsep, path in zip(tsep_ls, pt3_paths)}


# --- window grids ------------------------------------------------------------

DEFAULT_MAX_PT2_WINDOWS = 6
AUTO_MAX_PT2_WINDOWS = 16
AUTO_PT2_TMAX_LIMIT = 4
AUTO_PT2_TMIN_LIMIT = 4
AUTO_MAX_PT3_TAU_CUTS = 3
AUTO_PT2_SNR_MIN = 1.0
AUTO_PT2_TAIL_POINTS = 2


def _normalise_pt2_windows(windows: list[dict[str, int]] | None, *, Lt: int) -> list[dict[str, int]]:
    if windows is not None:
        return [{"tmin": int(w["tmin"]), "tmax": int(w["tmax"])} for w in windows]
    quarter = max(Lt // 4, 1)
    tmins = list(range(2, quarter - 3))
    return [{"tmin": tmin, "tmax": quarter} for tmin in tmins[:DEFAULT_MAX_PT2_WINDOWS]]


def _evenly_spaced_integers(start: int, stop: int, *, limit: int) -> list[int]:
    """Return up to ``limit`` ordered integers spanning an inclusive interval."""
    if stop < start or limit < 1:
        return []
    if stop - start + 1 <= limit:
        return list(range(start, stop + 1))
    return sorted({int(round(value)) for value in np.linspace(start, stop, limit)})


def _evenly_subsample(items: list[Any], *, limit: int) -> list[Any]:
    """Keep up to ``limit`` items with even index spacing (first and last included)."""
    if limit < 1 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    indices = _evenly_spaced_integers(0, len(items) - 1, limit=limit)
    return [items[index] for index in indices]


def _pt2_snr_endpoint(pt2_gv: np.ndarray, *, Lt: int) -> tuple[int | None, dict[str, Any]]:
    """Find the exclusive first-half endpoint through two points past SNR >= 1."""
    half = max(int(Lt) // 2, 1)
    mean = np.asarray(gv.mean(pt2_gv), dtype=float)
    sdev = np.asarray(gv.sdev(pt2_gv), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.divide(
            np.abs(mean),
            sdev,
            out=np.where(np.abs(mean) > 0.0, np.full_like(mean, np.inf), np.zeros_like(mean)),
            where=sdev > 0.0,
        )
    eligible = np.arange(2, min(half, len(snr)), dtype=int)
    stable = eligible[(~np.isnan(snr[eligible])) & (snr[eligible] >= AUTO_PT2_SNR_MIN)]
    if stable.size == 0:
        return None, {
            "last_snr_passing_t": None,
            "last_valid_t": None,
            "stable_tmax": None,
            "fallback_reason": "no first-half 2pt point at t>=2 has SNR >= 1",
        }
    last = int(stable[-1])
    # Zero-padded (or otherwise degenerate) tail points have sdev == 0 and
    # would make every fit window that includes them numerically singular, so
    # the tail-point extension must never reach past the last usable point.
    valid = eligible[np.isfinite(mean[eligible]) & (sdev[eligible] > 0.0)]
    last_valid = int(valid[-1]) if valid.size else last
    endpoint = min(half, last + 1 + AUTO_PT2_TAIL_POINTS, last_valid + 1)
    return endpoint, {
        "last_snr_passing_t": last,
        "last_valid_t": last_valid,
        "stable_tmax": int(endpoint),
        "fallback_reason": None,
    }


def _auto_pt2_windows(
    pt2_gv: np.ndarray,
    *,
    Lt: int,
    nstate_values: list[int],
    pt2_f_gv: np.ndarray | None = None,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Generate a bounded, data-driven first-half 2pt window scan."""
    states = [int(value) for value in nstate_values]
    if not states or any(value < 1 for value in states):
        raise ValueError("automatic pt2 window scan requires positive nstate values")
    minimum_points = max(4, 2 * max(states) + 1)
    endpoint, primary_diag = _pt2_snr_endpoint(pt2_gv, Lt=Lt)
    channel_diags: dict[str, Any] = {"initial": primary_diag}
    fallback_reasons: list[str] = []
    if endpoint is None:
        fallback_reasons.append(str(primary_diag["fallback_reason"]))
    if pt2_f_gv is not None:
        final_endpoint, final_diag = _pt2_snr_endpoint(pt2_f_gv, Lt=Lt)
        channel_diags["final"] = final_diag
        if final_endpoint is None:
            fallback_reasons.append(str(final_diag["fallback_reason"]))
        endpoints = [value for value in (endpoint, final_endpoint) if value is not None]
        endpoint = min(endpoints) if len(endpoints) == 2 else None

    used_fallback = endpoint is None
    if endpoint is None:
        endpoint = min(max(int(Lt) // 4, 1), max(int(Lt) // 2, 1))

    tmax_min = 2 + minimum_points
    tmax_values = _evenly_spaced_integers(tmax_min, int(endpoint), limit=AUTO_PT2_TMAX_LIMIT)
    windows: list[dict[str, int]] = []
    for tmax in tmax_values:
        for tmin in _evenly_spaced_integers(2, tmax - minimum_points, limit=AUTO_PT2_TMIN_LIMIT):
            windows.append({"tmin": int(tmin), "tmax": int(tmax)})
    windows = _evenly_subsample(windows, limit=AUTO_MAX_PT2_WINDOWS)
    if not windows:
        reason = "; ".join(fallback_reasons) or (
            f"stable first-half endpoint tmax={endpoint} leaves fewer than {minimum_points} fit points"
        )
        raise ValueError(
            "automatic pt2 window scan could not generate a legal window "
            f"({reason}); provide explicit pt2_windows"
        )
    return windows, {
        "source": "automatic",
        "snr_threshold": AUTO_PT2_SNR_MIN,
        "tail_points": AUTO_PT2_TAIL_POINTS,
        "minimum_points": minimum_points,
        "tmax_limit": AUTO_PT2_TMAX_LIMIT,
        "tmin_limit": AUTO_PT2_TMIN_LIMIT,
        "tmax_values": [int(value) for value in tmax_values],
        "stable_tmax": int(endpoint),
        "used_fallback": used_fallback,
        "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else None,
        "channels": channel_diags,
        "pt2_windows": windows,
    }


def _resolve_pt2_windows(
    windows: list[dict[str, int]] | None,
    *,
    Lt: int,
    pt2_gv: np.ndarray,
    nstate_values: list[int],
    pt2_f_gv: np.ndarray | None = None,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Use explicit 2pt windows exactly, or generate automatic candidates."""
    if windows is not None:
        resolved = _normalise_pt2_windows(windows, Lt=Lt)
        return resolved, {
            "source": "explicit",
            "used_fallback": False,
            "fallback_reason": None,
            "pt2_windows": resolved,
        }
    return _auto_pt2_windows(
        pt2_gv,
        Lt=Lt,
        nstate_values=nstate_values,
        pt2_f_gv=pt2_f_gv,
    )


def _normalise_pt3_windows(
    windows: list[dict[str, Any]] | None,
    *,
    tsep_ls: list[int],
    tau_cuts: list[int] | None,
) -> list[dict[str, Any]]:
    if windows is not None:
        return [
            {"tsep_ls": [int(t) for t in w.get("tsep_ls", tsep_ls)], "tau_cut": int(w["tau_cut"])}
            for w in windows
        ]
    cuts = [int(cut) for cut in (tau_cuts if tau_cuts is not None else [1, 2, 3, 4])]
    return [{"tsep_ls": list(tsep_ls), "tau_cut": cut} for cut in cuts]


def _auto_pt3_windows(
    *,
    tsep_ls: list[int],
    fit_scopes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate bounded contiguous-tsep and non-empty insertion windows."""
    tseps = sorted({int(value) for value in tsep_ls})
    if not tseps:
        raise ValueError("automatic pt3 window scan requires at least one tsep")
    requires_fh = any("FH" in str(scope) for scope in fit_scopes)
    minimum_tseps = 2 if requires_fh else 1
    raw_subsets = [tseps]
    if len(tseps) > minimum_tseps:
        raw_subsets.extend((tseps[1:], tseps[:-1]))
    subsets: list[list[int]] = []
    for subset in raw_subsets:
        if len(subset) >= minimum_tseps and subset not in subsets:
            subsets.append(subset)

    windows: list[dict[str, Any]] = []
    for subset in subsets:
        max_cut = min(int(tsep) // 2 for tsep in subset)
        cuts = _evenly_spaced_integers(1, max_cut, limit=AUTO_MAX_PT3_TAU_CUTS)
        for cut in cuts:
            if all(int(tsep) + 1 - 2 * int(cut) >= 1 for tsep in subset):
                windows.append({"tsep_ls": list(subset), "tau_cut": int(cut)})
    if not windows:
        raise ValueError(
            "automatic pt3 window scan could not leave a non-empty insertion window; "
            "provide explicit pt3_windows or pt3_tau_cuts"
        )
    return windows, {
        "source": "automatic",
        "minimum_tseps": minimum_tseps,
        "minimum_insertion_points": 1,
        "pt3_windows": windows,
        "used_fallback": False,
        "fallback_reason": None,
    }


def _resolve_pt3_windows(
    windows: list[dict[str, Any]] | None,
    *,
    tsep_ls: list[int],
    tau_cuts: list[int] | None,
    fit_scopes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply explicit 3pt precedence, otherwise generate automatic candidates."""
    if windows is not None:
        resolved = _normalise_pt3_windows(windows, tsep_ls=tsep_ls, tau_cuts=tau_cuts)
        return resolved, {
            "source": "explicit_pt3_windows",
            "used_fallback": False,
            "fallback_reason": None,
            "pt3_windows": resolved,
        }
    if tau_cuts is not None:
        resolved = _normalise_pt3_windows(None, tsep_ls=tsep_ls, tau_cuts=tau_cuts)
        return resolved, {
            "source": "explicit_pt3_tau_cuts",
            "used_fallback": False,
            "fallback_reason": None,
            "pt3_windows": resolved,
        }
    return _auto_pt3_windows(tsep_ls=tsep_ls, fit_scopes=fit_scopes)


# --- plotting helpers (sample-average tuning and per-sample diagnostics) -----


def _pt2_band(rec: dict[str, Any], Lt: int) -> tuple[np.ndarray, np.ndarray]:
    fit_t = np.arange(rec["tmin"], rec["tmax"], dtype=int)
    fit_gv = pt2_re_fcn(fit_t, rec["fit"].p, Lt, nstate=rec["nstate"]) / float(rec.get("correlator_rescale", 1.0))
    return fit_t, fit_gv


def _ratio_bands(rec: dict[str, Any], Lt: int, *, fitting_form: str = "Breit") -> list[dict[str, Any]]:
    bands = []
    tau_cut = rec["tau_cut"]
    nstate = rec["nstate"]
    p = rec["fit"].p
    for i, tsep in enumerate(rec["tsep_ls"]):
        fit_tau = np.linspace(tau_cut - 0.5, tsep - tau_cut + 0.5, 200)
        fit_t = np.full_like(fit_tau, float(tsep))
        if fitting_form == "NonBreit":
            fit_re = pt3_nonbreit_ratio_fcn(fit_t, fit_tau, p, Lt, nstate=nstate, part="re")
            fit_im = pt3_nonbreit_ratio_fcn(fit_t, fit_tau, p, Lt, nstate=nstate, part="im")
        else:
            fit_re = pt3_ratio_fcn(fit_t, fit_tau, p, Lt, nstate=nstate, part="re")
            fit_im = pt3_ratio_fcn(fit_t, fit_tau, p, Lt, nstate=nstate, part="im")
        bands.append(
            {
                "tsep": tsep,
                "tau_cut": tau_cut,
                "fit_tau": fit_tau,
                "fit_re": fit_re,
                "fit_im": fit_im,
                "label": rf"$t_{{\mathrm{{sep}}}}$={tsep}",
                "color": COLOR_CYCLE[i % len(COLOR_CYCLE)],
            }
        )
    return bands


def _plot_sample0_ratio(
    *,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    rec: dict[str, Any],
    Lt: int,
    log_dir: Path,
    ensemble: str,
    tag: str,
    momentum: str,
    z: int,
    fit_label: str,
    fitting_form: str = "Breit",
    part: str = "both",
) -> dict[str, str]:
    stem = log_dir / f"{ensemble}_{tag}_{fit_label}_{momentum}_z{z}_sample0"
    plotted_parts = _parts(part)
    p = rec["fit"].p
    plateau_ref_re = (
        _bare_matrix_element_from_fit(p, part="re", fitting_form=fitting_form)
        if "re" in plotted_parts
        else None
    )
    plateau_ref_im = (
        _bare_matrix_element_from_fit(p, part="im", fitting_form=fitting_form)
        if "im" in plotted_parts
        else None
    )
    denominator_energy = p["E0_f"] if fitting_form == "NonBreit" else p["E0"]
    figures = plot_pt3_ratio_fit_on_data(
        ratio_re,
        ratio_im,
        denominator_correction_energy=denominator_energy,
        denominator_correction_Lt=Lt,
        window_bands=[{"record_label": fit_label, "bands": _ratio_bands(rec, Lt, fitting_form=fitting_form), "fit": rec["fit"]}],
        plateau_ref_re=plateau_ref_re,
        plateau_ref_im=plateau_ref_im,
        plateau_label=r"Sample-0 fit bare matrix element",
        save_path=stem,
    )
    for fig, _ax in figures:
        plt.close(fig)
    paths = {
        "re": stem.with_name(f"{stem.name}_pt3_ratio_re.pdf"),
        "im": stem.with_name(f"{stem.name}_pt3_ratio_im.pdf"),
    }
    for component, path in paths.items():
        if component not in plotted_parts:
            path.unlink(missing_ok=True)
            path.with_suffix(".svg").unlink(missing_ok=True)
    output = {}
    for component, path in paths.items():
        if component in plotted_parts:
            output[f"ratio_{component}_pdf"] = str(path)
            output[f"ratio_{component}_svg"] = str(path.with_suffix(".svg"))
    return output


def _fh_bands(rec: dict[str, Any]) -> list[dict[str, Any]]:
    tsep_fit = np.asarray(rec["tsep_ls"][:-1], dtype=float)
    if tsep_fit.size == 0:
        raise ValueError("FH plot requires at least two tsep values")
    if tsep_fit.size == 1:
        fit_t = tsep_fit
    else:
        fit_t = np.linspace(float(np.min(tsep_fit)), float(np.max(tsep_fit)), 200)
    p = rec["fit"].p
    nstate = rec["nstate"]
    tau_cut = rec["tau_cut"]
    dt = _fh_dt(rec["tsep_ls"])
    return [
        {
            "fit_t": fit_t,
            "fit_re": fh_fcn(fit_t, tau_cut, p, nstate=nstate, part="re", dt=dt),
            "fit_im": fh_fcn(fit_t, tau_cut, p, nstate=nstate, part="im", dt=dt),
            "color": COLOR_CYCLE[0],
        }
    ]


def _plot_sample0_fh(
    *,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    rec: dict[str, Any],
    log_dir: Path,
    ensemble: str,
    tag: str,
    momentum: str,
    z: int,
    fit_label: str,
    part: str = "both",
) -> dict[str, str]:
    stem = log_dir / f"{ensemble}_{tag}_{fit_label}_{momentum}_z{z}_sample0"
    plotted_parts = _parts(part)
    fh_re, fh_im = _fh_samples_from_ratios(ratio_re, ratio_im, rec["tsep_ls"], rec["tau_cut"])
    p = rec["fit"].p
    figures = plot_fh_fit_on_data(
        fh_re,
        fh_im,
        tsep_ls=rec["tsep_ls"],
        window_bands=_fh_bands(rec),
        plateau_ref_re=p["O00_re"] / (2 * p["E0"]) if "re" in plotted_parts else None,
        plateau_ref_im=p["O00_im"] / (2 * p["E0"]) if "im" in plotted_parts else None,
        plateau_label=r"Sample-0 fit bare matrix element",
        save_path=stem,
    )
    for fig, _ax in figures:
        plt.close(fig)
    paths = {
        "re": stem.with_name(f"{stem.name}_fh_re.pdf"),
        "im": stem.with_name(f"{stem.name}_fh_im.pdf"),
    }
    for component, path in paths.items():
        if component not in plotted_parts:
            path.unlink(missing_ok=True)
            path.with_suffix(".svg").unlink(missing_ok=True)
    output = {}
    for component, path in paths.items():
        if component in plotted_parts:
            output[f"fh_{component}_pdf"] = str(path)
            output[f"fh_{component}_svg"] = str(path.with_suffix(".svg"))
    return output


def _plot_sample0_pt2(
    *,
    pt2_sample: np.ndarray,
    rec: dict[str, Any],
    Lt: int,
    log_dir: Path,
    ensemble: str,
    tag: str,
    momentum: str,
    fit_label: str,
) -> dict[str, str]:
    stem = log_dir / f"{ensemble}_{tag}_{fit_label}_{momentum}_sample0"
    fit_t, fit_gv = _pt2_band(rec, Lt)
    fig, _ax = plot_pt2_meff_on_data(
        pt2_sample,
        boundary="none",
        fit_bands=[{"fit_t": fit_t, "fit_gv": fit_gv, "label": f"2pt t=[{rec['tmin']},{rec['tmax']})", "color": COLOR_CYCLE[0]}],
        E0_band=rec["fit"].p["E0"],
        E0_label=r"Sample-0 fit $E_0$",
        t_max=Lt // 4,
        save_path=stem,
    )
    plt.close(fig)
    return {
        "meff_pdf": str(stem.with_name(f"{stem.name}_meff.pdf")),
        "meff_svg": str(stem.with_name(f"{stem.name}_meff.svg")),
    }


def _bare_records_to_ensemble(
    records: list[dict[str, Any]],
    *,
    resample_mode: str,
    attrs: dict[str, Any],
) -> EnsembleData:
    z_values: list[int] = []
    samples_by_z: list[np.ndarray] = []
    n_sample: int | None = None
    for rec in sorted(records, key=lambda item: item["z"]):
        real = np.asarray(rec["real_samples"], dtype=float)
        imag = np.asarray(rec["imag_samples"], dtype=float)
        if real.shape != imag.shape:
            raise ValueError(f"real/imag sample shape mismatch for z={rec['z']}")
        if n_sample is None:
            n_sample = int(real.shape[0])
        elif real.shape[0] != n_sample:
            raise ValueError(f"sample count mismatch for z={rec['z']}: {real.shape[0]} != {n_sample}")
        z_values.append(int(rec["z"]))
        samples_by_z.append(real + 1j * imag)
    if not samples_by_z:
        raise ValueError("no bare matrix-element records were produced")

    samples = np.stack(samples_by_z, axis=1)
    values = [samples[idx] for idx in range(samples.shape[0])]
    sorted_records = sorted(records, key=lambda item: item["z"])
    bare_attrs = dict(attrs)
    bare_attrs.update(
        {
            "bare_re_mean": json.dumps([float(rec["real_mean"]) for rec in sorted_records]),
            "bare_im_mean": json.dumps([float(rec["imag_mean"]) for rec in sorted_records]),
            "bare_re_stat_sdev": json.dumps([float(rec["real_stat_sdev"]) for rec in sorted_records]),
            "bare_im_stat_sdev": json.dumps([float(rec["imag_stat_sdev"]) for rec in sorted_records]),
            "bare_re_sys_sdev": json.dumps([_optional_float(rec.get("real_sys_sdev")) for rec in sorted_records]),
            "bare_im_sys_sdev": json.dumps([_optional_float(rec.get("imag_sys_sdev")) for rec in sorted_records]),
            "bare_re_sys_status": (
                "not estimated"
                if any(rec.get("real_sys_sdev") is None for rec in sorted_records)
                else "estimated"
            ),
            "bare_im_sys_status": (
                "not estimated"
                if any(rec.get("imag_sys_sdev") is None for rec in sorted_records)
                else "estimated"
            ),
        }
    )
    return EnsembleData(
        ensemble=EnsembleInfo(
            "",
            str(bare_attrs.get("ensemble", "")),
            1.0,
            1.0,
            1,
            1,
            0.0,
        ),
        resample={"bs": "bootstrap", "jk": "jackknife"}.get(resample_mode, resample_mode),
        values=values,
        dims=("z",),
        coords={"z": z_values},
        attrs={key: str(value) for key, value in bare_attrs.items() if value is not None},
        name="bare_matrix_element",
    )


# --- tool 1: inspect the 2pt scale ------------------------------------------


def inspect_correlator_scale(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_windows: list[dict[str, int]] | None = None,
    source_operator: str = "g5",
    sink_operator: str = "g5",
    momentum: str = "PX0PY0PZ0",
    temporal_extent: int | None = None,
    pt2_bT: int | None = None,
    pt2_bz: int | None = None,
    nstate: int | list[int] = 2,
    selectors: dict[str, Any] | None = None,
    out: str = "correlator_scale_inspection",
) -> dict[str, Any]:
    """Report 2pt magnitudes so the agent can choose a power-of-ten correlator_rescale."""
    if selectors is not None:
        source_operator = str(selectors.get("source_operator") or source_operator)
        sink_operator = str(selectors.get("sink_operator") or sink_operator)
        momentum = str(selectors.get("momentum") or momentum)
    pt2_real = np.real(
        _read_2pt(
            pt2_path,
            source_operator=source_operator,
            sink_operator=sink_operator,
            momentum=momentum,
            temporal_extent=temporal_extent,
            bT=pt2_bT,
            bz=pt2_bz,
        )
    )
    n_cfg, Lt = pt2_real.shape
    pt2_mean = np.mean(pt2_real, axis=0)
    pt2_sem = (
        np.std(pt2_real, axis=0, ddof=1) / np.sqrt(n_cfg)
        if n_cfg > 1
        else np.zeros(Lt, dtype=float)
    )
    pt2_gv = gv.gvar(pt2_mean, pt2_sem)
    states = [int(value) for value in (nstate if isinstance(nstate, list) else [nstate])]
    windows, auto_window_scan = _resolve_pt2_windows(
        pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        nstate_values=states,
    )
    window_stats = []
    for window in windows:
        values = np.abs(pt2_real[:, window["tmin"] : window["tmax"]]).reshape(-1)
        nonzero = values[values > 0.0]
        window_stats.append(
            {
                "tmin": window["tmin"],
                "tmax": window["tmax"],
                "median_abs": float(np.median(values)),
                "max_abs": float(np.max(values)),
                "min_abs_nonzero": float(np.min(nonzero)) if nonzero.size else None,
            }
        )
    result = {
        "out": out,
        "pt2_path": pt2_path,
        "source_operator": source_operator,
        "sink_operator": sink_operator,
        "momentum": momentum,
        "n_cfg": int(n_cfg),
        "Lt": int(Lt),
        "windows": window_stats,
        "auto_window_scan": {"pt2": auto_window_scan},
        "target_typical_abs_range": [0.0001, 0.01],
    }
    store[out] = result
    return result


# --- tool 2: tune the 2pt ground state (2pt-only path) ----------------------


def tune_ground_state(
    store: dict[str, Any],
    *,
    pt2_path: str,
    source_operator: str = "g5",
    sink_operator: str = "g5",
    momentum: str = "PX0PY0PZ0",
    temporal_extent: int | None = None,
    pt2_windows: list[dict[str, int]] | None = None,
    nstate: int = 2,
    svdcut: float = 1e-2,
    correlator_rescale: float = 1.0,
    resample_mode: str = "jk",
    sample_error_mode: str = "covariance",
    n_boot: int = 200,
    seed: int | None = 1984,
    bin_size: int = 1,
    window_indices: list[int] | None = None,
    model_average: bool = True,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "pt2_tune",
) -> dict[str, Any]:
    """Fit 2pt windows on sample-average data; return diagnostics and write a plot.

    With ``window_indices`` and ``model_average`` the tool also stores
    ``E0_avg`` / ``z0_avg`` (single window when one index is given) for reporting.
    """
    # validate agent-facing settings and load the resampled 2pt ensemble
    mode = _check_mode(resample_mode)
    scale = _check_rescale(correlator_rescale)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"

    pt2_complex = _read_2pt(
        pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=momentum,
        temporal_extent=temporal_extent,
    )
    n_cfg, Lt = pt2_complex.shape
    re_samples, _ = resample_config_samples(np.real(pt2_complex), mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size)
    pt2_gv = samples_to_gvar(re_samples, mode=mode, sample_error_mode=sample_error_mode)
    store["Lt"] = int(Lt)

    # fit every candidate window and retain numerical failures as diagnostics
    windows, pt2_scan = _resolve_pt2_windows(
        pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        nstate_values=[int(nstate)],
    )
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for window in windows:
        try:
            fit = fit_two_point(pt2_gv, window["tmin"], window["tmax"], Lt, nstate=nstate, svdcut=svdcut, rescale=scale)
            records.append(_record(fit, tmin=window["tmin"], tmax=window["tmax"], nstate=nstate, correlator_rescale=scale))
        except NUMERICAL_FIT_ERRORS as exc:
            rejected.append({**window, "reason": str(exc)})
    if not records:
        raise ValueError("all 2pt windows failed: " + "; ".join(str(item) for item in rejected[:5]))
    store["pt2_scan"] = records

    # model-average the selected windows on sample-average data
    selected = window_indices if window_indices is not None else list(range(len(records)))
    chosen = [records[i] for i in selected]
    e0_avg = None
    z0_avg = None
    if model_average and chosen:
        weights = _loggbf_weights(chosen)
        e0_avg = bayesian_average(np.array([rec["fit"].p["E0"] for rec in chosen], dtype=object), weights)
        z0_avg = bayesian_average(np.array([rec["fit"].p["z0"] for rec in chosen], dtype=object), weights)
        store["E0_avg"] = e0_avg
        store["z0_avg"] = z0_avg

    # write the combined correlator and effective-mass diagnostics
    bands = [
        {"fit_t": np.arange(rec["tmin"], rec["tmax"], dtype=int), "fit_gv": _pt2_band(rec, Lt)[1],
         "label": f"t=[{rec['tmin']},{rec['tmax']})", "color": COLOR_CYCLE[i % len(COLOR_CYCLE)]}
        for i, rec in enumerate(chosen)
    ]
    resolved_save = resolve_plot_save_path(save_path, artifacts_dir=out_dir, default_stem="pt2_tune")
    tune_tmax = max((rec["tmax"] for rec in chosen), default=0)
    meff_t_max = max(Lt // 4, tune_tmax)
    figures = plot_pt2_fit_on_data(
        pt2_gv,
        fit_bands=bands,
        E0_band=e0_avg,
        t_max=meff_t_max,
        save_path=resolved_save,
    )
    for fig, _ax in figures:
        plt.close(fig)

    return {
        "out": out,
        "Lt": int(Lt),
        "n_cfg": int(n_cfg),
        "n_samples": int(re_samples.shape[0]),
        "windows": [
            {
                "index": i,
                "tmin": rec["tmin"],
                "tmax": rec["tmax"],
                "Q": rec["Q"],
                "chi2_dof": rec["chi2_dof"],
                "logGBF": rec["logGBF"],
                "E0": str(rec["fit"].p["E0"]),
                "z0": str(rec["fit"].p["z0"]),
            }
            for i, rec in enumerate(records)
        ],
        "rejected": rejected,
        "auto_window_scan": {"pt2": pt2_scan},
        "E0_avg": str(e0_avg) if e0_avg is not None else None,
        "z0_avg": str(z0_avg) if z0_avg is not None else None,
        "c2pt_pdf": f"{resolved_save}_c2pt.pdf",
        "meff_pdf": f"{resolved_save}_meff.pdf",
    }


# --- shared sample-average scan for the bare matrix ---------------------------


def _normalise_strategy(value: str | None) -> tuple[str, str]:
    raw = "joint" if value is None else str(value).strip().lower()
    if raw in ("joint", "joint_2pt_ratio", "joint-fit"):
        return "joint", "joint_2pt_ratio"
    if raw in ("chained", "chained_2pt_ratio", "chain"):
        return "chained", "chained_2pt_ratio"
    if raw in ("independent", "independent_ratio"):
        return "independent", "independent_ratio"
    raise ValueError(
        f"fit_strategy must be 'joint', 'chained', or 'independent', got {value!r}"
    )


def _scan_average(
    specs: list[dict[str, Any]],
    *,
    strategy: str,
    fit_scope: str,
    pt2_gv: np.ndarray,
    pt2_f_gv: np.ndarray | None,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    pt2_best: dict[str, Any] | None,
    pt2_f_best: dict[str, Any] | None,
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    scale: float,
    fitting_form: str,
    prior_width: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit every candidate window on sample-average data; drop unusable posteriors."""
    # count parameters once because only the data window changes within this scan
    template = _scope_prior_with_width(fitting_form, nstate, fit_scope, strategy, prior_width)
    n_params = _prior_parameter_count(template)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for spec in specs:
        n_data = _fit_data_count(
            spec,
            strategy=strategy,
            fit_scope=fit_scope,
            part=part,
            fitting_form=fitting_form,
        )
        size_metadata = _with_fit_size_metadata(spec, n_data=n_data, n_params=n_params)
        # chained fits anchor their matrix-element prior to the selected 2pt posterior
        try:
            fit_prior = _scope_prior_with_width(
                fitting_form,
                nstate,
                fit_scope,
                strategy,
                prior_width,
            )
            if strategy == "chained":
                if fitting_form == "NonBreit":
                    _anchor_pt2_prior(fit_prior, pt2_best["fit"], suffix="_i")
                    _anchor_pt2_prior(fit_prior, (pt2_f_best or pt2_best)["fit"], suffix="_f")
                else:
                    if fit_scope in {"3pt_ratio", "3pt_ratio+FH"}:
                        _anchor_pt2_prior(fit_prior, pt2_best["fit"])
                    if "FH" in fit_scope:
                        _anchor_fh_energy_prior(fit_prior, pt2_best["fit"], nstate)
            fit = fit_matrix_element(
                ratio_re,
                ratio_im,
                spec["tsep_ls"],
                spec["tau_cut"],
                Lt,
                strategy=strategy,
                fit_scope=fit_scope,
                fitting_form=fitting_form,
                pt2_gv=pt2_gv,
                pt2_f_gv=pt2_f_gv,
                tmin=spec["tmin"],
                tmax=spec["tmax"],
                nstate=nstate,
                part=part,
                svdcut=svdcut,
                rescale=scale,
                prior=fit_prior,
            )
            usable, reason = _fit_usable(fit, fit_prior)
            if not usable:
                rejected.append({**size_metadata, "nstate": nstate, "prior_width": prior_width, "reason": reason})
                continue
            records.append(
                _record(
                    fit,
                    nstate=nstate,
                    prior_width=float(prior_width),
                    part=part,
                    fit_scope=fit_scope,
                    correlator_rescale=scale,
                    **size_metadata,
                )
            )
        except NUMERICAL_FIT_ERRORS as exc:
            rejected.append({**size_metadata, "nstate": nstate, "prior_width": prior_width, "reason": str(exc)})
    return records, rejected


def _candidate_specs(
    *,
    strategy: str,
    pt2_window_specs: list[dict[str, int]],
    pt3_window_specs: list[dict[str, Any]],
    pt2_best: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Cartesian window candidates: joint/independent pair 2pt windows with ratio windows."""
    if strategy in {"joint", "independent"}:
        return [
            {"tmin": w["tmin"], "tmax": w["tmax"], "tsep_ls": p["tsep_ls"], "tau_cut": p["tau_cut"]}
            for w, p in product(pt2_window_specs, pt3_window_specs)
        ]
    if pt2_best is None:
        raise ValueError("chained candidate specs require a selected 2pt window")
    tmin = pt2_best["tmin"]
    tmax = pt2_best["tmax"]
    return [{"tmin": tmin, "tmax": tmax, "tsep_ls": p["tsep_ls"], "tau_cut": p["tau_cut"]} for p in pt3_window_specs]


# --- tool 3: tune the bare matrix on sample-average data ---------------------


def tune_bare_matrix(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_out_path: str | None = None,
    qda_path: str | None = None,
    pt3_paths: dict[str, str] | list[str] | None = None,
    tsep_ls: list[int] | None = None,
    momentum: str | None = None,
    initial_momentum: str | None = None,
    final_momentum: str | None = None,
    fitting_form: str = "Breit",
    tune_z_values: list[int] | None = None,
    z_values: list[int] | None = None,
    source_operator: str = "g5",
    sink_operator: str = "g5",
    qda_source_operator: str | None = None,
    qda_sink_operator: str | None = None,
    qda_denominator_mode: str = "local",
    pt2_bT: int | None = None,
    pt2_bz: int | None = None,
    current_operator: str = "gT_nonlocal",
    bT: int = 0,
    temporal_extent: int | None = None,
    pt2_windows: list[dict[str, int]] | None = None,
    pt3_windows: list[dict[str, Any]] | None = None,
    pt3_tau_cuts: list[int] | None = None,
    fit_scope_values: list[str] | None = None,
    fit_scope: str | None = None,
    fit_strategies: list[str] | None = None,
    nstate_values: list[int] | None = None,
    fit_strategy: str | None = None,
    nstate: int | None = None,
    prior_width: float | list[float] | None = None,
    svdcut: float = 1e-2,
    correlator_rescale: float = 1.0,
    resample_mode: str = "jk",
    sample_error_mode: str = "covariance",
    n_boot: int = 200,
    seed: int | None = 1984,
    bin_size: int = 1,
    part: str = "both",
    q_min: float = 0.05,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "bare_tune",
) -> dict[str, Any]:
    """Scan bare-matrix fit windows on sample-average data at multiple tune z values.

    Returns ranked candidate diagnostics with cross-z feasibility summaries so
    the agent can choose one shared window to pass to ``fit_bare_matrix_grid``.
    """
    # normalize agent-facing parameters and validate the requested tune-z grid
    form = _normalise_fitting_form(fitting_form)
    scale = _check_rescale(correlator_rescale)
    mode = _check_mode(resample_mode)
    requested_scopes = fit_scope_values or ([fit_scope] if fit_scope is not None else ["3pt_ratio"])
    normalised_scopes = [_normalise_fit_scope(value)[0] for value in requested_scopes]
    if normalised_scopes == ["qda_ratio"]:
        return tune_qda_ratio(
            store,
            pt2_path=pt2_path,
            qda_path=qda_path,
            momentum=momentum,
            source_operator=source_operator,
            sink_operator=sink_operator,
            qda_source_operator=qda_source_operator,
            qda_sink_operator=qda_sink_operator,
            qda_denominator_mode=qda_denominator_mode,
            pt2_bT=pt2_bT,
            pt2_bz=pt2_bz,
            bT=bT,
            tune_z_values=tune_z_values,
            bz=z_values,
            temporal_extent=temporal_extent,
            pt2_windows=pt2_windows,
            fit_strategies=fit_strategies,
            fit_strategy=fit_strategy,
            nstate_values=nstate_values,
            nstate=nstate,
            prior_width=prior_width,
            svdcut=svdcut,
            correlator_rescale=scale,
            resample_mode=mode,
            sample_error_mode=sample_error_mode,
            n_boot=n_boot,
            seed=seed,
            bin_size=bin_size,
            part=part,
            q_min=q_min,
            out=out,
        )
    if "qda_ratio" in normalised_scopes:
        raise ValueError("qda_ratio cannot be mixed with 3pt/FH fit_scope values")
    if tsep_ls is None or pt3_paths is None:
        raise ValueError("3pt/FH tuning requires pt3_paths and tsep_ls")
    tseps = [int(t) for t in tsep_ls]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    if form == "Breit":
        if momentum is None:
            raise ValueError("momentum is required for Breit correlator fits")
        initial_momentum = final_momentum = three_point_momentum = momentum
    else:
        if initial_momentum is None or final_momentum is None:
            raise ValueError("initial_momentum and final_momentum are required for NonBreit correlator fits")
        three_point_momentum = final_momentum
    if not z_values:
        raise ValueError("z_values must be provided for tune_bare_matrix validation")
    allowed_z = [int(z) for z in z_values]
    if not tune_z_values:
        raise ValueError("tune_z_values is required and must be non-empty")
    allowed_z_set = set(allowed_z)
    if not allowed_z_set:
        raise ValueError("no allowed z values are available from the 3pt correlators")
    tune_z_list: list[int] = []
    seen_tune_z: set[int] = set()
    for raw_z in tune_z_values:
        tune_z = int(raw_z)
        if tune_z not in allowed_z_set:
            raise ValueError(f"tune_z value {tune_z} is not in the job z grid {sorted(allowed_z_set)}")
        if tune_z not in seen_tune_z:
            seen_tune_z.add(tune_z)
            tune_z_list.append(tune_z)
    tune_z_list.sort()
    primary_tune_z = tune_z_list[0]

    # load and resample the initial/final 2pt correlators with shared indices
    pt2_complex = _read_2pt(
        pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=initial_momentum,
        temporal_extent=temporal_extent,
    )
    n_cfg, Lt = pt2_complex.shape
    re_samples, pt2_complex_samples, indices = _resample_pt2(
        pt2_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size
    )
    pt2_gv = samples_to_gvar(re_samples, mode=mode, sample_error_mode=sample_error_mode)
    pt2_f_gv = None
    pt2_f_complex_samples = pt2_complex_samples
    if form == "NonBreit":
        pt2_f_complex = _read_2pt(
            pt2_out_path or pt2_path,
            source_operator=source_operator,
            sink_operator=sink_operator,
            momentum=final_momentum,
            temporal_extent=temporal_extent,
        )
        if pt2_f_complex.shape != pt2_complex.shape:
            raise ValueError(f"initial/final 2pt shape mismatch: {pt2_complex.shape} != {pt2_f_complex.shape}")
        re_f_samples, pt2_f_complex_samples, _ = _resample_pt2(
            pt2_f_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
        )
        pt2_f_gv = samples_to_gvar(re_f_samples, mode=mode, sample_error_mode=sample_error_mode)

    # resolve the fit-model search first because automatic windows depend on its size
    strategies = fit_strategies or ([fit_strategy] if fit_strategy is not None else ["joint"])
    scopes = fit_scope_values or ([fit_scope] if fit_scope is not None else ["3pt_ratio"])
    states = nstate_values or ([nstate] if nstate is not None else [2])
    prior_widths = _normalise_prior_width(prior_width)
    pt2_window_specs, pt2_scan = _resolve_pt2_windows(
        pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        pt2_f_gv=pt2_f_gv,
        nstate_values=[int(value) for value in states],
    )
    pt3_window_specs, pt3_scan = _resolve_pt3_windows(
        pt3_windows,
        tsep_ls=tseps,
        tau_cuts=pt3_tau_cuts,
        fit_scopes=[str(value) for value in scopes],
    )
    auto_window_scan = {"pt2": pt2_scan, "pt3": pt3_scan}

    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    all_rejected: list[dict[str, Any]] = []
    for tune_z in tune_z_list:
        # load and resample every tsep with the same ensemble indices before forming ratios
        ratio_re: dict[int, np.ndarray] = {}
        ratio_im: dict[int, np.ndarray] = {}
        for tsep in tseps:
            pt3 = _read_3pt(
                paths_by_tsep[tsep],
                source_operator=source_operator,
                sink_operator=sink_operator,
                current_operator=current_operator,
                momentum=three_point_momentum,
                bT=bT,
                bz=tune_z,
                tsep=tsep,
            )
            pt3_samples, _ = resample_config_samples(
                pt3,
                mode=mode,
                n_boot=n_boot,
                seed=seed,
                bin_size=bin_size,
                indices=indices,
            )
            if form == "NonBreit":
                re_samples, im_samples = _non_forward_ratio_samples(
                    pt2_complex_samples,
                    pt2_f_complex_samples,
                    pt3_samples,
                    tsep,
                )
            else:
                re_samples, im_samples = _ratio_samples(pt2_complex_samples, pt3_samples, tsep)
            ratio_re[tsep] = samples_to_gvar(
                re_samples,
                mode=mode,
                sample_error_mode=sample_error_mode,
            )
            ratio_im[tsep] = samples_to_gvar(
                im_samples,
                mode=mode,
                sample_error_mode=sample_error_mode,
            )
        # scan all configured strategies, scopes, state counts, priors, and windows
        records: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for strategy_value in strategies:
            strategy, _ = _normalise_strategy(strategy_value)
            for scope_value in scopes:
                scope, _ = _normalise_fit_scope(scope_value)
                _validate_scope_form(scope, form)
                for nstate_value in states:
                    nstate_int = int(nstate_value)
                    if "FH" in scope and nstate_int > 2:
                        rejected.append(
                            {
                                "fit_strategy": strategy,
                                "fit_scope": scope,
                                "nstate": nstate_int,
                                "reason": "FH fits currently support nstate <= 2",
                            }
                        )
                        continue
                    for prior_width_value in prior_widths:
                        pt2_best = None
                        pt2_f_best = None
                        if strategy == "chained":
                            pt2_records: list[dict[str, Any]] = []
                            pt2_f_records: list[dict[str, Any]] = []
                            pt2_prior_template = _vary_prior_width(
                                pt2_prior(nstate_int),
                                prior_width_value,
                            )
                            for window in pt2_window_specs:
                                try:
                                    fit = fit_two_point(
                                        pt2_gv,
                                        window["tmin"],
                                        window["tmax"],
                                        Lt,
                                        nstate=nstate_int,
                                        svdcut=svdcut,
                                        rescale=scale,
                                        prior=pt2_prior_template,
                                    )
                                    pt2_records.append(
                                        _record(
                                            fit,
                                            tmin=window["tmin"],
                                            tmax=window["tmax"],
                                            nstate=nstate_int,
                                            prior_width=prior_width_value,
                                            correlator_rescale=scale,
                                        )
                                    )
                                except NUMERICAL_FIT_ERRORS as exc:
                                    rejected.append(
                                        {
                                            **window,
                                            "fit_strategy": strategy,
                                            "fit_scope": scope,
                                            "nstate": nstate_int,
                                            "prior_width": prior_width_value,
                                            "reason": str(exc),
                                        }
                                    )
                                if form == "NonBreit" and pt2_f_gv is not None:
                                    try:
                                        fit_f = fit_two_point(
                                            pt2_f_gv,
                                            window["tmin"],
                                            window["tmax"],
                                            Lt,
                                            nstate=nstate_int,
                                            svdcut=svdcut,
                                            rescale=scale,
                                            prior=pt2_prior_template,
                                        )
                                        pt2_f_records.append(
                                            _record(
                                                fit_f,
                                                tmin=window["tmin"],
                                                tmax=window["tmax"],
                                                nstate=nstate_int,
                                                prior_width=prior_width_value,
                                                correlator_rescale=scale,
                                            )
                                        )
                                    except NUMERICAL_FIT_ERRORS as exc:
                                        rejected.append(
                                            {
                                                **window,
                                                "fit_strategy": strategy,
                                                "fit_scope": scope,
                                                "nstate": nstate_int,
                                                "prior_width": prior_width_value,
                                                "reason": str(exc),
                                            }
                                        )
                            if not pt2_records:
                                continue
                            pt2_best = pt2_records[select_best(pt2_records, q_min=q_min)[0]]
                            if form == "NonBreit":
                                if not pt2_f_records:
                                    continue
                                pt2_f_best = pt2_f_records[
                                    select_best(pt2_f_records, q_min=q_min)[0]
                                ]

                        specs = _candidate_specs(
                            strategy=strategy,
                            pt2_window_specs=pt2_window_specs,
                            pt3_window_specs=pt3_window_specs,
                            pt2_best=pt2_best,
                        )
                        found, failed = _scan_average(
                            specs,
                            strategy=strategy,
                            fit_scope=scope,
                            pt2_gv=pt2_gv,
                            pt2_f_gv=pt2_f_gv,
                            ratio_re=ratio_re,
                            ratio_im=ratio_im,
                            pt2_best=pt2_best,
                            pt2_f_best=pt2_f_best,
                            Lt=Lt,
                            nstate=nstate_int,
                            part=part,
                            svdcut=svdcut,
                            scale=scale,
                            fitting_form=form,
                            prior_width=prior_width_value,
                        )
                        for record in found:
                            record["fit_strategy"] = strategy
                            record["fit_scope"] = scope
                        records.extend(found)
                        rejected.extend(
                            {
                                **item,
                                "fit_strategy": strategy,
                                "fit_scope": scope,
                                "nstate": nstate_int,
                            }
                            for item in failed
                        )

        # merge the per-z results by stable window identity for robust selection
        for record in records:
            key = _window_candidate_key(record)
            entry = by_key.setdefault(
                key,
                {"per_z": {}, "meta": record, "primary_record": None},
            )
            entry["per_z"][tune_z] = {
                "Q": float(record["Q"]),
                "chi2_dof": float(record["chi2_dof"]),
                "logGBF": float(record["logGBF"]),
                "n_data": int(record["n_data"]),
                "n_params": int(record["n_params"]),
                "dof_is_positive": bool(record["dof_is_positive"]),
                "usable": True,
            }
            if tune_z == primary_tune_z:
                entry["primary_record"] = record
                entry["meta"] = record
        for item in rejected:
            if "tmin" not in item or "tsep_ls" not in item or "tau_cut" not in item:
                continue
            key = _window_candidate_key(item)
            entry = by_key.setdefault(
                key,
                {"per_z": {}, "meta": item, "primary_record": None},
            )
            entry["per_z"][tune_z] = {
                "Q": float(item["Q"]) if "Q" in item and np.isfinite(float(item["Q"])) else None,
                "chi2_dof": (
                    float(item["chi2_dof"])
                    if "chi2_dof" in item and np.isfinite(float(item["chi2_dof"]))
                    else None
                ),
                "logGBF": (
                    float(item["logGBF"])
                    if "logGBF" in item and np.isfinite(float(item["logGBF"]))
                    else None
                ),
                "n_data": int(item.get("n_data", 0)),
                "n_params": int(item.get("n_params", 0)),
                "dof_is_positive": bool(item.get("dof_is_positive", False)),
                "usable": False,
                "reason": str(item.get("reason", "unknown")),
            }
        all_rejected.extend(rejected)

    # assemble JSON-safe candidates with cross-z feasibility diagnostics
    primary_fit_records: list[dict[str, Any] | None] = []
    candidates: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(by_key.keys())):
        entry = by_key[key]
        cross_z_summary = _summarise_cross_z_feasibility(entry["per_z"], tune_z_list)
        primary_record = entry.get("primary_record")
        if primary_record is not None:
            fit_parameters = primary_record["fit"].p
            candidate = {
                "index": index,
                "fit_strategy": primary_record["fit_strategy"],
                "fit_scope": primary_record["fit_scope"],
                "nstate": primary_record["nstate"],
                "prior_width": primary_record["prior_width"],
                "tmin": primary_record["tmin"],
                "tmax": primary_record["tmax"],
                "tsep_ls": primary_record["tsep_ls"],
                "tau_cut": primary_record["tau_cut"],
                "Q": primary_record["Q"],
                "chi2_dof": primary_record["chi2_dof"],
                "logGBF": primary_record["logGBF"],
                "n_data": primary_record["n_data"],
                "n_params": primary_record["n_params"],
                "dof_is_positive": primary_record["dof_is_positive"],
                "bare_re": str(
                    _bare_matrix_element_from_fit(fit_parameters, part="re", fitting_form=form)
                ),
                "bare_im": str(
                    _bare_matrix_element_from_fit(fit_parameters, part="im", fitting_form=form)
                ),
                **cross_z_summary,
            }
            if form == "Breit":
                candidate["O00_re_over_2E0"] = candidate["bare_re"]
                candidate["O00_im_over_2E0"] = candidate["bare_im"]
        else:
            metadata = entry["meta"]
            primary_diagnostic = cross_z_summary["tune_z_diagnostics"].get(
                str(primary_tune_z),
                {},
            )
            candidate = {
                "index": index,
                "fit_strategy": metadata.get("fit_strategy"),
                "fit_scope": metadata.get("fit_scope"),
                "nstate": metadata.get("nstate"),
                "prior_width": metadata.get("prior_width"),
                "tmin": metadata.get("tmin"),
                "tmax": metadata.get("tmax"),
                "tsep_ls": metadata.get("tsep_ls"),
                "tau_cut": metadata.get("tau_cut"),
                "Q": primary_diagnostic.get("Q"),
                "chi2_dof": primary_diagnostic.get("chi2_dof"),
                "logGBF": primary_diagnostic.get("logGBF"),
                "n_data": primary_diagnostic.get("n_data", metadata.get("n_data", 0)),
                "n_params": primary_diagnostic.get("n_params", metadata.get("n_params", 0)),
                "dof_is_positive": primary_diagnostic.get(
                    "dof_is_positive",
                    metadata.get("dof_is_positive", False),
                ),
                **cross_z_summary,
            }
            if form == "Breit":
                candidate["O00_re_over_2E0"] = "n/a"
                candidate["O00_im_over_2E0"] = "n/a"
        candidates.append(candidate)
        primary_fit_records.append(primary_record)

    # choose the primary-z recommendation and the all-z robust recommendation
    selectable = [(index, record) for index, record in enumerate(primary_fit_records) if record is not None]
    if not selectable:
        raise ValueError("all bare-matrix tuning windows failed: " + "; ".join(str(item) for item in all_rejected[:5]))

    selectable_indices, selectable_records = zip(*selectable)
    local_best_index, fallback = select_data_window(list(selectable_records), q_min=q_min)
    best_index = selectable_indices[local_best_index]
    best = primary_fit_records[best_index]
    assert best is not None

    for index, candidate in enumerate(candidates):
        candidate["index"] = index

    store[out] = [record for record in primary_fit_records if record is not None]
    feasible_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.get("feasible_at_all_tune_z") and primary_fit_records[index] is not None
    ]
    robust_index = None
    if feasible_indices:
        feasible_records = [primary_fit_records[index] for index in feasible_indices]
        local_robust_index, _ = select_data_window(feasible_records, q_min=q_min)
        robust_index = feasible_indices[local_robust_index]
    robust_window = None
    if robust_index is not None:
        robust_record = primary_fit_records[robust_index]
        assert robust_record is not None
        robust_window = _fit_summary(robust_record, fallback=False, index=robust_index)

    result = {
        "out": out,
        "fit_strategies": strategies,
        "fit_scopes": scopes,
        "nstate_values": states,
        "prior_width": prior_widths,
        "tune_z_values": tune_z_list,
        "primary_tune_z": primary_tune_z,
        "tune_z": primary_tune_z,
        "allowed_z_values": sorted(allowed_z),
        "Lt": int(Lt),
        "n_cfg": int(n_cfg),
        "correlator_rescale": scale,
        "fitting_form": form,
        "candidates": candidates,
        "rejected": all_rejected,
        "recommended_index": best_index,
        "recommended_fallback_no_q_passing": fallback,
        "recommended_window": _fit_summary(best, fallback=fallback, index=best_index),
        "recommended_robust_index": robust_index,
        "recommended_robust_window": robust_window,
        "tuning_diagnostic_pdfs": {},
        "auto_window_scan": auto_window_scan,
    }
    store["_correlator_tuning_summary"] = {
        "pt2_path": str(pt2_path),
        "fit_scopes": [str(value) for value in scopes],
        "auto_window_scan": auto_window_scan,
        "recommended_robust_window": robust_window,
    }
    return result


# --- tool 4: apply one shared setting to all samples and z -------------------


def _fit_correlator_sample_batch(payload: bytes, sample_indices: list[int]) -> list[dict[str, Any]]:
    """Fit one batch of correlator samples without logging or plotting."""
    # deserialize once per process batch so gvar correlations remain intact
    context = gv.loads(payload)
    results: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        logs: list[dict[str, Any]] = []
        try:
            re_vals: list[float] = []
            im_vals: list[float] = []
            sample_records: list[dict[str, Any]] = []
            first_fit = None
            first_rre = first_rim = None
            first_meta = None
            # refit every configured model against this sample's recentered means
            for candidate_index, candidate in enumerate(context["candidates"]):
                common = context["common"]
                spec = candidate["spec"]
                rre = {
                    tsep: _recenter(
                        common["samples_re"][tsep][sample_index],
                        common["ratio_re"][tsep],
                    )
                    for tsep in common["tseps"]
                }
                rim = {
                    tsep: _recenter(
                        common["samples_im"][tsep][sample_index],
                        common["ratio_im"][tsep],
                    )
                    for tsep in common["tseps"]
                }
                pt2_sample = None
                pt2_f_sample = None
                if common["strategy"] == "joint":
                    pt2_sample = _recenter(
                        common["pt2_samples"][sample_index],
                        common["pt2_gv"],
                    )
                    if common["fitting_form"] == "NonBreit":
                        pt2_f_samples = (
                            common["pt2_f_samples"]
                            if common["pt2_f_samples"] is not None
                            else common["pt2_samples"]
                        )
                        pt2_f_template = (
                            common["pt2_f_gv"]
                            if common["pt2_f_gv"] is not None
                            else common["pt2_gv"]
                        )
                        pt2_f_sample = _recenter(
                            pt2_f_samples[sample_index],
                            pt2_f_template,
                        )
                fit = fit_matrix_element(
                    rre,
                    rim,
                    spec["tsep_ls"],
                    spec["tau_cut"],
                    common["Lt"],
                    strategy=common["strategy"],
                    fit_scope=common["fit_scope"],
                    fitting_form=common["fitting_form"],
                    pt2_gv=pt2_sample,
                    pt2_f_gv=pt2_f_sample,
                    tmin=spec["tmin"],
                    tmax=spec["tmax"],
                    nstate=candidate["nstate"],
                    part=common["part"],
                    svdcut=common["svdcut"],
                    rescale=common["scale"],
                    prior=candidate["prior"],
                    p0=candidate["p0"],
                )
                usable, reason = _fit_usable(fit, candidate["template"])
                if not usable:
                    if not context["model_average"]:
                        raise ValueError(str(reason))
                    logs.append(
                        {
                            "kind": "rejected",
                            "nstate": candidate["nstate"],
                            "prior_width": candidate["prior_width"],
                            "reason": str(reason),
                        }
                    )
                    continue
                sample_record = _record(
                    fit,
                    candidate_index=candidate_index,
                    nstate=candidate["nstate"],
                    prior_width=candidate["prior_width"],
                    part=context["part"],
                    fit_scope=context["scope"],
                    correlator_rescale=context["scale"],
                    **candidate["spec"],
                )
                sample_records.append(sample_record)
                logs.append(
                    {
                        "kind": "quality",
                        "spec": candidate["spec"],
                        "nstate": candidate["nstate"],
                        "prior_width": candidate["prior_width"],
                        "Q": float(fit.Q),
                        "chi2": float(fit.chi2),
                        "dof": int(fit.dof),
                        "logGBF": float(fit.logGBF),
                    }
                )
                re_vals.append(
                    _bare_matrix_element_mean_for_part(
                        fit.p,
                        output_part="re",
                        fit_part=context["part"],
                        fitting_form=context["form"],
                    )
                )
                im_vals.append(
                    _bare_matrix_element_mean_for_part(
                        fit.p,
                        output_part="im",
                        fit_part=context["part"],
                        fitting_form=context["form"],
                    )
                )
                if first_fit is None:
                    first_fit, first_rre, first_rim = fit, rre, rim
                    first_meta = {
                        **candidate["spec"],
                        "nstate": candidate["nstate"],
                        "prior_width": candidate["prior_width"],
                    }
            # combine usable models and preserve sample-0 data for main-process plotting
            if not sample_records:
                raise ValueError("all fit-function candidates failed")
            sample_weights = _loggbf_weights(sample_records) if context["model_average"] else np.ones(1, dtype=float)
            candidate_weights = np.zeros(len(context["candidates"]), dtype=float)
            for weight, sample_record in zip(sample_weights, sample_records):
                candidate_weights[int(sample_record["candidate_index"])] += float(weight)
            selected_q, selected_chi2_dof = _selected_record_quality(sample_records, sample_weights)
            e0_pairs = [(weight, record) for weight, record in zip(sample_weights, sample_records) if "E0" in record["fit"].p]
            e0_i_pairs = [(weight, record) for weight, record in zip(sample_weights, sample_records) if "E0_i" in record["fit"].p]
            e0_f_pairs = [(weight, record) for weight, record in zip(sample_weights, sample_records) if "E0_f" in record["fit"].p]
            plot_payload = None
            if sample_index == 0:
                plot_payload = gv.dumps(
                    {"fit": first_fit, "rre": first_rre, "rim": first_rim, "meta": first_meta}
                )
            results.append(
                {
                    "sample": sample_index,
                    "real": float(np.sum(sample_weights * np.asarray(re_vals))),
                    "imag": float(np.sum(sample_weights * np.asarray(im_vals))),
                    "Q": selected_q,
                    "chi2_dof": selected_chi2_dof,
                    "E0_lattice": float(sum(weight * float(gv.mean(record["fit"].p["E0"])) for weight, record in e0_pairs) / sum(weight for weight, _record in e0_pairs)) if e0_pairs else float("nan"),
                    "E0_i_lattice": float(sum(weight * float(gv.mean(record["fit"].p["E0_i"])) for weight, record in e0_i_pairs) / sum(weight for weight, _record in e0_i_pairs)) if e0_i_pairs else float("nan"),
                    "E0_f_lattice": float(sum(weight * float(gv.mean(record["fit"].p["E0_f"])) for weight, record in e0_f_pairs) / sum(weight for weight, _record in e0_f_pairs)) if e0_f_pairs else float("nan"),
                    "candidate_weights": candidate_weights,
                    "logs": logs,
                    "plot_payload": plot_payload,
                    "error": None,
                }
            )
        # one pathological sample must not discard the rest of the resampled ensemble
        except NUMERICAL_FIT_ERRORS as exc:
            results.append(
                {
                    "sample": sample_index,
                    "logs": logs,
                    "error": str(exc),
                }
            )
    return results


def fit_bare_matrix_grid(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_out_path: str | None = None,
    qda_path: str | None = None,
    pt3_paths: dict[str, str] | list[str] | None = None,
    tsep_ls: list[int] | None = None,
    z_values: list[int],
    ensemble: str,
    tag: str,
    momentum: str | None = None,
    initial_momentum: str | None = None,
    final_momentum: str | None = None,
    fitting_form: str = "Breit",
    hadron: str | None = None,
    gfix: str | None = None,
    source_operator: str = "g5",
    sink_operator: str = "g5",
    qda_source_operator: str | None = None,
    qda_sink_operator: str | None = None,
    qda_denominator_mode: str = "local",
    pt2_bT: int | None = None,
    pt2_bz: int | None = None,
    current_operator: str = "gT_nonlocal",
    distribution_type: str = "unpolarized",
    bz_direction: str,
    bT: int = 0,
    pt2_window: dict[str, int] | None = None,
    pt3_window: dict[str, Any] | None = None,
    pt3_tau_cut: int | None = None,
    pt2_windows: list[dict[str, int]] | None = None,
    pt3_windows: list[dict[str, Any]] | None = None,
    pt3_tau_cuts: list[int] | None = None,
    model_average: bool = False,
    tune_z: int | None = None,
    fit_strategy: str = "joint",
    fit_scope: str = "3pt_ratio",
    nstate: int | list[int] = 2,
    nstate_values: list[int] | None = None,
    prior_width: float | list[float] | None = None,
    resample_mode: str = "bs",
    sample_error_mode: str = "covariance",
    n_boot: int = 200,
    seed: int | None = 1984,
    bin_size: int = 1,
    svdcut: float = 1e-2,
    part: str = "both",
    q_min: float = 0.05,
    posterior_prior_error_scale: float = 3.0,
    correlator_rescale: float = 1.0,
    job_id: str | None = None,
    volume: str | None = None,
    lattice_spacing_fm: float | None = None,
    momentum_gev: float | None = None,
    initial_momentum_gev: float | None = None,
    final_momentum_gev: float | None = None,
    temporal_extent: int | None = None,
    save_path: str | None = None,
    log_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "bare_matrix_grid",
    workers: int = 1,
) -> dict[str, Any]:
    """Apply one shared window to all samples and z, then export bare matrix elements.

    Window/tau-cut choices are tuned once on sample-average data (for ``tune_z``)
    and used for every z and every resampled sample. ``model_average=True`` varies
    fit-function choices (nstate and prior_width) within that fixed data window.
    """
    del out
    # validate agent-facing fit, resampling, and worker settings once at the tool edge
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    form = _normalise_fitting_form(fitting_form)
    if form == "Breit":
        if momentum is None:
            raise ValueError("momentum is required for Breit correlator fits")
        initial_momentum = final_momentum = three_point_momentum = momentum
        initial_momentum_gev = final_momentum_gev = momentum_gev
    else:
        if initial_momentum is None or final_momentum is None:
            raise ValueError("initial_momentum and final_momentum are required for NonBreit correlator fits")
        three_point_momentum = final_momentum
        momentum_gev = initial_momentum_gev
    strategy, _ = _normalise_strategy(fit_strategy)
    scope, scope_label = _normalise_fit_scope(fit_scope)
    _validate_scope_form(scope, form)
    raw_states = nstate_values if nstate_values is not None else (nstate if isinstance(nstate, list) else [nstate])
    fit_nstates = [int(value) for value in raw_states]
    if not fit_nstates:
        raise ValueError("nstate_values must contain at least one value")
    if "FH" in scope and any(value > 2 for value in fit_nstates):
        raise ValueError("FH fits currently support nstate <= 2")
    primary_nstate = fit_nstates[0]
    prior_widths = _normalise_prior_width(prior_width)
    if scope == "qda_ratio":
        return fit_qda_ratio_grid(
            store,
            pt2_path=pt2_path,
            qda_path=qda_path,
            bz=z_values,
            ensemble=ensemble,
            tag=tag,
            momentum=momentum,
            source_operator=source_operator,
            sink_operator=sink_operator,
            qda_source_operator=qda_source_operator,
            qda_sink_operator=qda_sink_operator,
            qda_denominator_mode=qda_denominator_mode,
            pt2_bT=pt2_bT,
            pt2_bz=pt2_bz,
            bz_direction=bz_direction,
            bT=bT,
            pt2_window=pt2_window,
            pt2_windows=pt2_windows,
            resample_mode=resample_mode,
            sample_error_mode=sample_error_mode,
            n_boot=n_boot,
            seed=seed,
            bin_size=bin_size,
            svdcut=svdcut,
            part=part,
            q_min=q_min,
            nstate_values=fit_nstates,
            fit_strategy=strategy,
            prior_width=prior_widths,
            posterior_prior_error_scale=posterior_prior_error_scale,
            correlator_rescale=correlator_rescale,
            model_average=model_average,
            tune_z=tune_z,
            job_id=job_id,
            hadron=hadron,
            gfix=gfix,
            volume=volume,
            lattice_spacing_fm=lattice_spacing_fm,
            momentum_gev=momentum_gev,
            temporal_extent=temporal_extent,
            save_path=save_path,
            log_dir=log_dir,
            log_path=log_path,
            artifacts_dir=artifacts_dir,
            workers=workers,
        )
    if tsep_ls is None or pt3_paths is None:
        raise ValueError("3pt/FH grid fits require pt3_paths and tsep_ls")
    fit_mode = (
        f"independent_{scope_label}"
        if strategy == "independent"
        else f"{strategy}_2pt_{scope_label}"
    )
    scale = _check_rescale(correlator_rescale)
    mode = _check_mode(resample_mode)
    fitted_parts = _parts(part)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    if log_path is not None:
        base_log_path = Path(log_path)
        log_suffix = base_log_path.suffix or ".log"
        tuning_log_path = base_log_path.with_name(f"{base_log_path.stem}_tuning{log_suffix}")
        sample_log_path = base_log_path.with_name(f"{base_log_path.stem}_samples{log_suffix}")
    else:
        log_stem = f"{ensemble}_{tag}_{three_point_momentum}_bT{bT}_{fit_mode}"
        tuning_log_path = fit_log_dir / f"{log_stem}_tuning.log"
        sample_log_path = fit_log_dir / f"{log_stem}_samples.log"
    tuning_logger = setup_logger(tuning_log_path, logger_name="correlator_tuning_logger")
    sample_logger = setup_logger(sample_log_path, logger_name="correlator_sample_logger")
    tuning_logger.info("Starting %s bare matrix grid fit (model_average=%s)", fit_mode, model_average)
    tuning_logger.info("ensemble=%s tag=%s momentum=%s rescale=%s", ensemble, tag, three_point_momentum, scale)

    # resolve input grids, selectors, and shared output/log locations
    tseps = [int(t) for t in tsep_ls]
    if pt3_window is None and pt3_tau_cut is not None:
        pt3_window = {"tsep_ls": tseps, "tau_cut": int(pt3_tau_cut)}
    z_list = [int(z) for z in z_values]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    missing = [tsep for tsep in tseps if tsep not in paths_by_tsep]
    if missing:
        raise ValueError(f"pt3_paths missing tsep entries: {missing}")
    tune_z_value = z_list[0] if tune_z is None else int(tune_z)
    # load and resample the initial/final 2pt ensembles with identical indices
    pt2_complex = _read_2pt(
        pt2_path,
        source_operator=source_operator,
        sink_operator=sink_operator,
        momentum=initial_momentum,
        temporal_extent=temporal_extent,
    )
    n_cfg, Lt = pt2_complex.shape
    pt2_samples, pt2_complex_samples, indices = _resample_pt2(pt2_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size)
    pt2_gv = samples_to_gvar(pt2_samples, mode=mode, sample_error_mode=sample_error_mode)
    pt2_f_samples = None
    pt2_f_gv = None
    pt2_f_complex_samples = pt2_complex_samples
    if form == "NonBreit":
        pt2_f_complex = _read_2pt(
            pt2_out_path or pt2_path,
            source_operator=source_operator,
            sink_operator=sink_operator,
            momentum=final_momentum,
            temporal_extent=temporal_extent,
        )
        if pt2_f_complex.shape != pt2_complex.shape:
            raise ValueError(f"initial/final 2pt shape mismatch: {pt2_complex.shape} != {pt2_f_complex.shape}")
        pt2_f_samples, pt2_f_complex_samples, _ = _resample_pt2(
            pt2_f_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
        )
        pt2_f_gv = samples_to_gvar(pt2_f_samples, mode=mode, sample_error_mode=sample_error_mode)
    n_samples = int(pt2_samples.shape[0])
    effective_pt2_windows = [pt2_window] if pt2_window is not None else pt2_windows
    pt2_window_specs, pt2_scan = _resolve_pt2_windows(
        effective_pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        pt2_f_gv=pt2_f_gv,
        nstate_values=fit_nstates,
    )
    effective_pt3_windows = [pt3_window] if pt3_window is not None else pt3_windows
    pt3_window_specs, pt3_scan = _resolve_pt3_windows(
        effective_pt3_windows,
        tsep_ls=tseps,
        tau_cuts=pt3_tau_cuts,
        fit_scopes=[scope],
    )
    auto_window_scan = {"pt2": pt2_scan, "pt3": pt3_scan}
    tuning_summary = store.get("_correlator_tuning_summary")
    if (
        isinstance(tuning_summary, dict)
        and tuning_summary.get("pt2_path") == str(pt2_path)
        and isinstance(tuning_summary.get("auto_window_scan"), dict)
    ):
        auto_window_scan = tuning_summary["auto_window_scan"]
    tuning_logger.info("auto_window_scan=%s", json.dumps(auto_window_scan, sort_keys=True))
    tuning_logger.info("Lt=%s n_cfg=%s mode=%s n_samples=%s", Lt, n_cfg, mode, n_samples)

    # ratio loading is reused for the tune z and every production z
    def read_ratios(z: int):
        samples_re: dict[int, np.ndarray] = {}
        samples_im: dict[int, np.ndarray] = {}
        gv_re: dict[int, np.ndarray] = {}
        gv_im: dict[int, np.ndarray] = {}
        for tsep in tseps:
            pt3 = _read_3pt(
                paths_by_tsep[tsep],
                source_operator=source_operator,
                sink_operator=sink_operator,
                current_operator=current_operator,
                momentum=three_point_momentum,
                bT=bT,
                bz=z,
                tsep=tsep,
            )
            if pt3.shape[0] != n_cfg:
                raise ValueError(f"3pt n_cfg mismatch for z={z}, tsep={tsep}: {pt3.shape[0]} != {n_cfg}")
            pt3_samples, _ = resample_config_samples(pt3, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices)
            if form == "NonBreit":
                samples_re[tsep], samples_im[tsep] = _non_forward_ratio_samples(
                    pt2_complex_samples, pt2_f_complex_samples, pt3_samples, tsep
                )
            else:
                samples_re[tsep], samples_im[tsep] = _ratio_samples(pt2_complex_samples, pt3_samples, tsep)
            gv_re[tsep] = samples_to_gvar(samples_re[tsep], mode=mode, sample_error_mode=sample_error_mode)
            gv_im[tsep] = samples_to_gvar(samples_im[tsep], mode=mode, sample_error_mode=sample_error_mode)
        return samples_re, samples_im, gv_re, gv_im

    # chained mode: fit 2pt once and reuse the same 2pt posterior as a ratio anchor.
    pt2_best = None
    pt2_f_best = None
    sample0_pt2_paths: dict[str, str] = {}
    if strategy == "chained":
        pt2_records: list[dict[str, Any]] = []
        pt2_f_records: list[dict[str, Any]] = []
        pt2_prior_template = _vary_prior_width(pt2_prior(primary_nstate), 1.0)
        pt2_n_params = _prior_parameter_count(pt2_prior_template)
        for window in pt2_window_specs:
            pt2_size_metadata = _with_fit_size_metadata(
                window,
                n_data=max(int(window["tmax"]) - int(window["tmin"]), 0),
                n_params=pt2_n_params,
            )
            try:
                fit = fit_two_point(
                    pt2_gv, window["tmin"], window["tmax"], Lt,
                    nstate=primary_nstate, svdcut=svdcut, rescale=scale, prior=pt2_prior_template,
                )
                pt2_records.append(
                    _record(
                        fit,
                        nstate=primary_nstate,
                        prior_width=1.0,
                        correlator_rescale=scale,
                        **pt2_size_metadata,
                    )
                )
            except NUMERICAL_FIT_ERRORS as exc:
                tuning_logger.info("2pt window %s rejected: %s", window, exc)
            if form == "NonBreit" and pt2_f_gv is not None:
                try:
                    fit_f = fit_two_point(
                        pt2_f_gv, window["tmin"], window["tmax"], Lt,
                        nstate=primary_nstate, svdcut=svdcut, rescale=scale, prior=pt2_prior_template,
                    )
                    pt2_f_records.append(
                        _record(
                            fit_f,
                            nstate=primary_nstate,
                            prior_width=1.0,
                            correlator_rescale=scale,
                            **pt2_size_metadata,
                        )
                    )
                except NUMERICAL_FIT_ERRORS as exc:
                    tuning_logger.info("final 2pt window %s rejected: %s", window, exc)
        pt2_window_matched = False
        if pt2_window is not None:
            matching_pt2_records = [
                rec for rec in pt2_records
                if rec["tmin"] == int(pt2_window["tmin"]) and rec["tmax"] == int(pt2_window["tmax"])
            ]
            if matching_pt2_records:
                pt2_records = matching_pt2_records
                pt2_window_matched = True
        if pt2_window_matched:
            pt2_best_index, pt2_fallback = 0, False
        else:
            pt2_best_index, pt2_fallback = select_data_window(pt2_records, q_min=q_min)
        pt2_best = pt2_records[pt2_best_index]
        if form == "NonBreit":
            if pt2_window_matched:
                pt2_f_records = [
                    rec for rec in pt2_f_records
                    if rec["tmin"] == int(pt2_window["tmin"]) and rec["tmax"] == int(pt2_window["tmax"])
                ] or pt2_f_records
                pt2_f_best_index = 0
            else:
                pt2_f_best_index, _ = select_data_window(pt2_f_records, q_min=q_min)
            pt2_f_best = pt2_f_records[pt2_f_best_index]
        tuning_logger.info("selected 2pt window t=[%s,%s) Q=%.4g", pt2_best["tmin"], pt2_best["tmax"], pt2_best["Q"])
        try:
            pt2_sample0 = _recenter(pt2_samples[0], pt2_gv)
            rec0 = _record(
                fit_two_point(
                    pt2_sample0, pt2_best["tmin"], pt2_best["tmax"], Lt,
                    nstate=primary_nstate, svdcut=svdcut, rescale=scale,
                    prior=pt2_prior_template, p0=_p0_from_fit(pt2_best["fit"], pt2_prior_template),
                ),
                tmin=pt2_best["tmin"], tmax=pt2_best["tmax"], nstate=primary_nstate, prior_width=1.0,
                correlator_rescale=scale,
            )
            sample0_pt2_paths = _plot_sample0_pt2(
                pt2_sample=pt2_sample0,
                rec=rec0,
                Lt=Lt,
                log_dir=fit_log_dir,
                ensemble=ensemble,
                tag=tag,
                momentum=momentum,
                fit_label="chained_fit",
            )
        except NUMERICAL_FIT_ERRORS as exc:
            sample_logger.info("Bad chained 2pt sample=0: %s", exc)

    # resolve the shared window setting once, on the representative tune_z.
    tune_samples_re, tune_samples_im, tune_gv_re, tune_gv_im = read_ratios(tune_z_value)
    candidate_specs = _candidate_specs(
        strategy=strategy, pt2_window_specs=pt2_window_specs, pt3_window_specs=pt3_window_specs, pt2_best=pt2_best
    )
    explicit_spec = None
    if pt3_window is not None:
        tmin = int(pt2_window["tmin"]) if pt2_window is not None else (pt2_best["tmin"] if pt2_best else pt2_window_specs[0]["tmin"])
        tmax = int(pt2_window["tmax"]) if pt2_window is not None else (pt2_best["tmax"] if pt2_best else pt2_window_specs[0]["tmax"])
        explicit_spec = {
            "tmin": tmin,
            "tmax": tmax,
            "tsep_ls": [int(t) for t in pt3_window.get("tsep_ls", tseps)],
            "tau_cut": int(pt3_window["tau_cut"]),
        }
        explicit_template = _scope_prior_with_width(form, primary_nstate, scope, strategy, 1.0)
        explicit_spec = _with_fit_size_metadata(
            explicit_spec,
            n_data=_fit_data_count(
                explicit_spec,
                strategy=strategy,
                fit_scope=scope,
                part=part,
                fitting_form=form,
            ),
            n_params=_prior_parameter_count(explicit_template),
        )

    if explicit_spec is not None:
        shared_specs = [explicit_spec]
        selection_rule = "single fixed window provided by the agent"
    else:
        tune_records, _ = _scan_average(
            candidate_specs, strategy=strategy, fit_scope=scope, pt2_gv=pt2_gv, pt2_f_gv=pt2_f_gv,
            ratio_re=tune_gv_re, ratio_im=tune_gv_im, pt2_best=pt2_best, pt2_f_best=pt2_f_best,
            Lt=Lt, nstate=primary_nstate, part=part, svdcut=svdcut, scale=scale, fitting_form=form,
            prior_width=1.0,
        )
        if not tune_records:
            raise ValueError("all shared-window tuning fits failed on tune_z")
        best_index, fallback = select_data_window(tune_records, q_min=q_min)
        chosen = tune_records[best_index]
        shared_specs = [
            {
                "tmin": chosen["tmin"],
                "tmax": chosen["tmax"],
                "tsep_ls": chosen["tsep_ls"],
                "tau_cut": chosen["tau_cut"],
                "n_data": chosen["n_data"],
                "n_params": chosen["n_params"],
                "dof_is_positive": chosen["dof_is_positive"],
            }
        ]
        selection_rule = (
            f"auto-selected single data window on z={tune_z_value} "
            f"(Q>={q_min}, n_data>n_params, fallback_no_q_passing={fallback}, "
            f"chi2_dof_tolerance={DATA_WINDOW_CHI2_DOF_TOLERANCE})"
        )
    tuning_logger.info("shared setting (%s): %s", selection_rule, shared_specs)
    z_records: list[dict[str, Any]] = []
    z_report: list[dict[str, Any]] = []
    sample_fit_Q: list[float] = []
    sample_fit_chi2_dof: list[float] = []
    energy_record: dict[str, Any] | None = None
    E0_lattice_samples: list[float] | None = None
    E0_i_lattice_samples: list[float] | None = None
    E0_f_lattice_samples: list[float] | None = None

    try:
        from tqdm import tqdm
    except ImportError:
        z_iterator = z_list
    else:
        z_iterator = tqdm(
            z_list,
            desc=f"fit bare matrix {ensemble} {three_point_momentum}",
        )

    # reuse one process pool across all z values; serial mode follows the same batch path
    sample_executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        # fit every z with the shared data window and the configured fit models
        for z in z_iterator:
            tuning_logger.info("=== z=%s ===", z)
            if z == tune_z_value:
                samples_re, samples_im, gv_re, gv_im = tune_samples_re, tune_samples_im, tune_gv_re, tune_gv_im
            else:
                samples_re, samples_im, gv_re, gv_im = read_ratios(z)

            candidate_records: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for nstate_value in fit_nstates:
                for prior_width_value in prior_widths:
                    found, failed = _scan_average(
                        shared_specs, strategy=strategy, fit_scope=scope, pt2_gv=pt2_gv, pt2_f_gv=pt2_f_gv,
                        ratio_re=gv_re, ratio_im=gv_im, pt2_best=pt2_best, pt2_f_best=pt2_f_best,
                        Lt=Lt, nstate=nstate_value, part=part, svdcut=svdcut, scale=scale, fitting_form=form,
                        prior_width=prior_width_value,
                    )
                    candidate_records.extend(found)
                    rejected.extend(failed)
            if not candidate_records:
                raise ValueError(f"shared window failed on sample-average for z={z}: {rejected[:3]}")

            best_avg_index, fallback = select_best(candidate_records, q_min=q_min)
            selected_avg_record = candidate_records[best_avg_index]
            if z == tune_z_value:
                energy_record = selected_avg_record
            avg_records = candidate_records if model_average else [selected_avg_record]
            fit_model_candidates = [
                {
                    "index": i,
                    "nstate": int(rec["nstate"]),
                    "prior_width": float(rec["prior_width"]),
                    "Q": float(rec["Q"]),
                    "chi2_dof": float(rec["chi2_dof"]),
                    "logGBF": float(rec["logGBF"]),
                    "n_data": int(rec["n_data"]),
                    "n_params": int(rec["n_params"]),
                    "dof_is_positive": bool(rec["dof_is_positive"]),
                }
                for i, rec in enumerate(candidate_records)
            ]
            avg_weights = _loggbf_weights(avg_records)
            avg_re_vals = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        rec["fit"].p,
                        output_part="re",
                        fit_part=part,
                        fitting_form=form,
                    )
                    for rec in avg_records
                ],
                dtype=float,
            )
            avg_im_vals = np.asarray(
                [
                    _bare_matrix_element_mean_for_part(
                        rec["fit"].p,
                        output_part="im",
                        fit_part=part,
                        fitting_form=form,
                    )
                    for rec in avg_records
                ],
                dtype=float,
            )
            avg_re_mean = float(np.sum(avg_weights * avg_re_vals))
            avg_im_mean = float(np.sum(avg_weights * avg_im_vals))
            real_sys_sdev = (
                _weighted_model_sdev(avg_re_vals, avg_weights, center=avg_re_mean)
                if model_average and "re" in fitted_parts
                else None
            )
            imag_sys_sdev = (
                _weighted_model_sdev(avg_im_vals, avg_weights, center=avg_im_mean)
                if model_average and "im" in fitted_parts
                else None
            )
            templates = [
                _scope_prior_with_width(form, int(rec["nstate"]), scope, strategy, float(rec["prior_width"]))
                for rec in avg_records
            ]
            priors = [
                (
                    _scaled_prior(
                        rec["fit"],
                        template,
                        error_scale=posterior_prior_error_scale,
                        prior_width=float(rec["prior_width"]),
                    ),
                    _p0_from_fit(rec["fit"], template),
                )
                for rec, template in zip(avg_records, templates)
            ]
            for rec in avg_records:
                log_nonlinear_fit_quality(
                    rec["fit"], kind=f"sample-average {fit_mode}",
                    label=(
                        f"z={z} t=[{rec['tmin']},{rec['tmax']}) tau_cut={rec['tau_cut']} "
                        f"nstate={rec['nstate']} prior_width={rec['prior_width']}"
                    ),
                    logger=tuning_logger, q_min=q_min,
                )

            real_samples = np.full(n_samples, np.nan)
            imag_samples = np.full(n_samples, np.nan)
            failures: list[dict[str, Any]] = []
            sample0_paths: dict[str, str] = {}
            common = dict(
                strategy=strategy, fit_scope=scope, pt2_samples=pt2_samples, pt2_gv=pt2_gv,
                pt2_f_samples=pt2_f_samples, pt2_f_gv=pt2_f_gv,
                samples_re=samples_re, samples_im=samples_im,
                ratio_re=gv_re, ratio_im=gv_im, tseps=tseps, Lt=Lt, part=part, svdcut=svdcut, scale=scale,
                fitting_form=form,
            )
            weight_sums = np.zeros(len(avg_records), dtype=float)
            weight_counts = 0
            candidates = [
                {
                    "spec": {
                        "tmin": rec["tmin"],
                        "tmax": rec["tmax"],
                        "tsep_ls": rec["tsep_ls"],
                        "tau_cut": rec["tau_cut"],
                    },
                    "nstate": int(rec["nstate"]),
                    "prior_width": float(rec["prior_width"]),
                    "template": template,
                    "prior": prior,
                    "p0": p0,
                }
                for rec, template, (prior, p0) in zip(avg_records, templates, priors)
            ]
            payload = gv.dumps(
                {
                    "common": common,
                    "candidates": candidates,
                    "model_average": model_average,
                    "part": part,
                    "form": form,
                    "scope": scope,
                    "scale": scale,
                }
            )
            sample_batches = [
                batch.tolist()
                for batch in np.array_split(np.arange(n_samples), min(workers, n_samples))
                if batch.size
            ]
            if sample_executor is None:
                sample_results = _fit_correlator_sample_batch(payload, sample_batches[0])
            else:
                futures = [
                    sample_executor.submit(_fit_correlator_sample_batch, payload, batch)
                    for batch in sample_batches
                ]
                sample_results = [item for future in futures for item in future.result()]
            parallel_results = sorted(sample_results, key=lambda item: item["sample"])
            if z == tune_z_value and E0_lattice_samples is None:
                values = [float(item.get("E0_lattice", np.nan)) for item in parallel_results]
                E0_lattice_samples = values if np.any(np.isfinite(values)) else None
            if z == tune_z_value and E0_i_lattice_samples is None:
                values = [float(item.get("E0_i_lattice", np.nan)) for item in parallel_results]
                E0_i_lattice_samples = values if np.any(np.isfinite(values)) else None
            if z == tune_z_value and E0_f_lattice_samples is None:
                values = [float(item.get("E0_f_lattice", np.nan)) for item in parallel_results]
                E0_f_lattice_samples = values if np.any(np.isfinite(values)) else None
            for result in parallel_results:
                sample_index = int(result["sample"])
                for log_item in result["logs"]:
                    if log_item["kind"] == "rejected":
                        sample_logger.info(
                            "Rejected %s z=%s sample=%s nstate=%s prior_width=%s: %s",
                            fit_mode,
                            z,
                            sample_index,
                            log_item["nstate"],
                            log_item["prior_width"],
                            log_item["reason"],
                        )
                        continue
                    spec = log_item["spec"]
                    log_nonlinear_fit_quality(
                        SimpleNamespace(
                            Q=log_item["Q"],
                            chi2=log_item["chi2"],
                            dof=log_item["dof"],
                            logGBF=log_item["logGBF"],
                        ),
                        kind=f"sample ground-state {fit_mode}",
                        label=(
                            f"z={z} sample={sample_index} t=[{spec['tmin']},{spec['tmax']}) "
                            f"tseps={spec['tsep_ls']} tau_cut={spec['tau_cut']} "
                            f"nstate={log_item['nstate']} prior_width={log_item['prior_width']}"
                        ),
                        logger=sample_logger,
                        q_min=q_min,
                    )
                if result["error"] is not None:
                    failures.append({"sample": sample_index, "error": result["error"]})
                    sample_logger.info("Bad %s z=%s sample=%s: %s", fit_mode, z, sample_index, result["error"])
                    continue
                _append_finite_sample_quality(sample_fit_Q, sample_fit_chi2_dof, result)
                real_samples[sample_index] = float(result["real"])
                imag_samples[sample_index] = float(result["imag"])
                weight_sums += np.asarray(result["candidate_weights"], dtype=float)
                weight_counts += 1
                if result["plot_payload"] is not None:
                    plot_data = gv.loads(result["plot_payload"])
                    rec0 = _record(
                        plot_data["fit"],
                        **plot_data["meta"],
                        part=part,
                        fit_scope=scope,
                        correlator_rescale=scale,
                    )
                    if scope != "FH":
                        sample0_paths.update(
                            _plot_sample0_ratio(
                                ratio_re=plot_data["rre"],
                                ratio_im=plot_data["rim"],
                                rec=rec0,
                                Lt=Lt,
                                log_dir=fit_log_dir,
                                ensemble=ensemble,
                                tag=tag,
                                momentum=momentum,
                                z=z,
                                fit_label=f"{strategy}_{scope_label}_fit",
                                fitting_form=form,
                                part=part,
                            )
                        )
                    if "FH" in scope:
                        sample0_paths.update(
                            _plot_sample0_fh(
                                ratio_re=plot_data["rre"],
                                ratio_im=plot_data["rim"],
                                rec=rec0,
                                log_dir=fit_log_dir,
                                ensemble=ensemble,
                                tag=tag,
                                momentum=momentum,
                                z=z,
                                fit_label=f"{strategy}_{scope_label}_fit",
                                part=part,
                            )
                        )
            if not np.any(np.isfinite(real_samples)):
                raise ValueError(f"all resampled fits failed for z={z}")
            real_mean_arr, real_sdev_arr = sample_mean_and_sdev(real_samples, mode=mode, sample_error_mode=sample_error_mode)
            imag_mean_arr, imag_sdev_arr = sample_mean_and_sdev(imag_samples, mode=mode, sample_error_mode=sample_error_mode)
            real_mean = float(real_mean_arr)
            real_sdev = float(real_sdev_arr)
            imag_mean = float(imag_mean_arr)
            imag_sdev = float(imag_sdev_arr)
            sample_logger.info("summary z=%s real=%s +/- %s imag=%s +/- %s failed=%s", z, real_mean, real_sdev, imag_mean, imag_sdev, len(failures))

            mean_weights = (weight_sums / weight_counts).tolist() if weight_counts else [float("nan")] * len(avg_records)
            window_summary = _fit_summary(selected_avg_record, fallback=fallback, index=best_avg_index)
            z_records.append(
                {
                    "z": z,
                    "real_samples": real_samples,
                    "imag_samples": imag_samples,
                    "real_sys_sdev": real_sys_sdev,
                    "imag_sys_sdev": imag_sys_sdev,
                    "window": window_summary,
                    "fit_model_candidates": fit_model_candidates,
                    "fit_model_weights": mean_weights,
                    "sample0_plot_paths": sample0_paths,
                }
            )
            z_report.append(
                {
                    "z": z,
                    "window": window_summary,
                    "rejected_windows": rejected,
                    "rejected_fit_models": rejected,
                    "fit_model_candidates": fit_model_candidates,
                    "fit_model_weights": mean_weights,
                    "selected_fit_model": window_summary,
                    "n_failed_samples": len(failures),
                    "sample_failures": failures[:10],
                    "real_sys_sdev": real_sys_sdev,
                    "imag_sys_sdev": imag_sys_sdev,
                    "sample0_plot_paths": sample0_paths,
                }
            )

    finally:
        if sample_executor is not None:
            sample_executor.shutdown()

    if form == "NonBreit":
        initial_fit = pt2_best["fit"] if pt2_best is not None else (energy_record or {}).get("fit")
        final_fit = pt2_f_best["fit"] if pt2_f_best is not None else (energy_record or {}).get("fit")
        initial_energy_key = "E0" if pt2_best is not None else "E0_i"
        final_energy_key = "E0" if pt2_f_best is not None else "E0_f"
        t_gev2 = (
            None
            if initial_momentum_gev is None or final_momentum_gev is None
            else (float(final_momentum_gev) - float(initial_momentum_gev)) ** 2
        )
        denominator = (
            None
            if initial_momentum_gev is None or final_momentum_gev is None
            else float(initial_momentum_gev) + float(final_momentum_gev)
        )
        xi = None if denominator in (None, 0.0) else (float(initial_momentum_gev) - float(final_momentum_gev)) / denominator
        t_label = "n/a" if t_gev2 is None else f"{t_gev2:.2f}"
        xi_label = "n/a" if xi is None else f"{xi:.2f}"
        plot_title = rf"{ensemble} $t={t_label}\,\mathrm{{GeV}}^2$, $\xi={xi_label}$ bare matrix elements"
    else:
        p_label = "n/a" if momentum_gev is None else f"{float(momentum_gev):.2f}"
        plot_title = rf"{ensemble} $p={p_label}\,\mathrm{{GeV}}$ bare matrix elements"
    # assemble the terminal artifact, summary plot, and JSON-safe observation
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_save = resolve_plot_save_path(
        save_path,
        artifacts_dir=out_dir,
        default_stem="bare_matrix_elements",
    )
    summary_z: list[int] = []
    real_means: list[float] = []
    real_errors: list[float] = []
    imag_means: list[float] = []
    imag_errors: list[float] = []
    output_rows: list[dict[str, Any]] = []
    for record in sorted(z_records, key=lambda item: item["z"]):
        real = np.asarray(record["real_samples"], dtype=float)
        imag = np.asarray(record["imag_samples"], dtype=float)
        real_mean_array, real_error_array = sample_mean_and_sdev(
            real,
            mode=mode,
            sample_error_mode=sample_error_mode,
        )
        imag_mean_array, imag_error_array = sample_mean_and_sdev(
            imag,
            mode=mode,
            sample_error_mode=sample_error_mode,
        )
        real_mean = float(real_mean_array)
        real_error = float(real_error_array)
        imag_mean = float(imag_mean_array)
        imag_error = float(imag_error_array)
        record.update(
            real_mean=real_mean,
            imag_mean=imag_mean,
            real_stat_sdev=real_error,
            imag_stat_sdev=imag_error,
        )
        summary_z.append(int(record["z"]))
        real_means.append(real_mean)
        real_errors.append(real_error)
        imag_means.append(imag_mean)
        imag_errors.append(imag_error)
        output_rows.append(
            {
                "z": record["z"],
                "n_samples": int(real.shape[0]),
                "n_failed_samples": int(
                    np.count_nonzero(~np.isfinite(real) | ~np.isfinite(imag))
                ),
                "real_mean": real_mean,
                "real_sdev": real_error,
                "real_stat_sdev": real_error,
                "real_sys_sdev": _optional_float(record.get("real_sys_sdev")),
                "imag_mean": imag_mean,
                "imag_sdev": imag_error,
                "imag_stat_sdev": imag_error,
                "imag_sys_sdev": _optional_float(record.get("imag_sys_sdev")),
                "window": record["window"],
                "sample0_plot_paths": record.get("sample0_plot_paths", {}),
            }
        )

    plotted_parts = _parts(part)
    figure, axis = default_plot()
    if "re" in plotted_parts:
        axis.errorbar(
            summary_z,
            real_means,
            real_errors,
            label="Re",
            color=COLOR_CYCLE[0],
            **ERRORBAR_STYLE,
        )
    if "im" in plotted_parts:
        axis.errorbar(
            summary_z,
            imag_means,
            imag_errors,
            label="Im",
            color=COLOR_CYCLE[1],
            marker="s",
            **ERRORBAR_STYLE,
        )
    matrix_element_label = (
        r"Bare matrix element $O_{00}/(E_{0}^{i}+E_{0}^{f})$"
        if form == "NonBreit"
        else r"Bare matrix element $O_{00}/(2E_0)$"
    )
    axis.set_xlabel(r"$z/a$", **FONT_SIZE)
    axis.set_ylabel(matrix_element_label, **FONT_SIZE)
    axis.set_title(plot_title, **FONT_SIZE)
    axis.set_ylim(-0.2, 1.2)
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
            "bz_direction": bz_direction,
            "momentum": three_point_momentum,
            "bT": bT,
            "resample_mode": mode,
            "sample_error_mode": sample_error_mode,
            "average_method": sample_error_mode,
            "part": part,
            "job_id": job_id,
            "volume": volume,
            "lattice_spacing_fm": lattice_spacing_fm,
            "momentum_gev": momentum_gev,
            "initial_momentum_gev": initial_momentum_gev,
            "final_momentum_gev": final_momentum_gev,
            **({"t_gev2": t_gev2, "xi": xi} if form == "NonBreit" and t_gev2 is not None and xi is not None else {}),
            "fitting_form": form,
            "fit_scope": scope,
            "nstate_values": json.dumps(fit_nstates),
            "prior_width": json.dumps(prior_widths),
            "initial_momentum": initial_momentum,
            "final_momentum": final_momentum,
            "hadron": hadron,
            "gfix": gfix,
            "current_operator": current_operator,
            "distribution_type": distribution_type,
            "workers": workers,
        },
    )
    artifact_path = f"{resolved_save}.nc"
    bare_data.to_netcdf(artifact_path)
    output = {
        "artifact": artifact_path,
        "netcdf_path": artifact_path,
        "plot_pdf": pdf_path,
        "plot_svg": svg_path,
        "n_z": len(z_records),
        "n_sample": bare_data.n_sample,
        "outputs": output_rows,
    }
    pt2_energies = []
    if form == "NonBreit":
        initial_energy = _energy_summary(
            fit=initial_fit,
            key=initial_energy_key,
            momentum=initial_momentum,
            momentum_gev=initial_momentum_gev,
            lattice_spacing_fm=lattice_spacing_fm,
            channel="initial",
            pt2_path=pt2_path,
            ensemble=ensemble,
            hadron=hadron,
            gfix=gfix,
            volume=volume,
            source_operator=source_operator,
            sink_operator=sink_operator,
            fitting_form=form,
            job_id=job_id,
            E0_lattice_samples=E0_i_lattice_samples,
            resample_mode=mode,
            sample_error_mode=sample_error_mode,
            workers=workers,
        )
        final_energy = _energy_summary(
            fit=final_fit,
            key=final_energy_key,
            momentum=final_momentum,
            momentum_gev=final_momentum_gev,
            lattice_spacing_fm=lattice_spacing_fm,
            channel="final",
            pt2_path=pt2_out_path or pt2_path,
            ensemble=ensemble,
            hadron=hadron,
            gfix=gfix,
            volume=volume,
            source_operator=source_operator,
            sink_operator=sink_operator,
            fitting_form=form,
            job_id=job_id,
            E0_lattice_samples=E0_f_lattice_samples,
            resample_mode=mode,
            sample_error_mode=sample_error_mode,
            workers=workers,
        )
        pt2_energies.extend(item for item in (initial_energy, final_energy) if item is not None)
    else:
        energy_fit = pt2_best["fit"] if pt2_best is not None else (energy_record or {}).get("fit")
        energy = _energy_summary(
            fit=energy_fit,
            key="E0",
            momentum=momentum,
            momentum_gev=momentum_gev,
            lattice_spacing_fm=lattice_spacing_fm,
            channel="breit",
            pt2_path=pt2_path,
            ensemble=ensemble,
            hadron=hadron,
            gfix=gfix,
            volume=volume,
            source_operator=source_operator,
            sink_operator=sink_operator,
            fitting_form=form,
            job_id=job_id,
            E0_lattice_samples=E0_lattice_samples,
            resample_mode=mode,
            sample_error_mode=sample_error_mode,
            workers=workers,
        )
        if energy is not None:
            pt2_energies.append(energy)
    store["bare_matrix_element_data"] = bare_data
    store["bare_matrix_element_netcdf"] = output["netcdf_path"]
    store["output"] = bare_data
    return {
        **output,
        "fit_strategy": strategy,
        "fit_scope": scope,
        "fit_mode": fit_mode,
        "fitting_form": form,
        "model_average": model_average,
        "nstate_values": fit_nstates,
        "prior_width": prior_widths,
        "selection_rule": selection_rule,
        "shared_window_specs": shared_specs,
        "tuning_log_path": str(tuning_log_path),
        "sample_log_path": str(sample_log_path),
        "correlator_rescale": scale,
        "resample_mode": mode,
        "n_samples": n_samples,
        "workers": int(workers),
        "z_values": z_list,
        "tune_z": tune_z_value,
        "z_fits": z_report,
        "sample_fit_Q": sample_fit_Q,
        "sample_fit_chi2_dof": sample_fit_chi2_dof,
        "sample0_pt2_plot_paths": sample0_pt2_paths,
        "pt2_energies": pt2_energies,
        "momentum_gev": momentum_gev,
        "initial_momentum_gev": initial_momentum_gev,
        "final_momentum_gev": final_momentum_gev,
        "t_gev2": t_gev2 if form == "NonBreit" else None,
        "xi": xi if form == "NonBreit" else None,
        "auto_window_scan": auto_window_scan,
    }



# --- qDA-ratio tuning and grid fits -----------------------------------------

def _split_fit_log_paths(
    *,
    log_path: str | Path | None,
    log_dir: Path,
    log_stem: str,
) -> tuple[Path, Path]:
    """Return the standard tuning and per-sample fit-log paths."""
    if log_path is not None:
        base_log_path = Path(log_path)
        log_suffix = base_log_path.suffix or ".log"
        return (
            base_log_path.with_name(f"{base_log_path.stem}_tuning{log_suffix}"),
            base_log_path.with_name(f"{base_log_path.stem}_samples{log_suffix}"),
        )
    return (
        log_dir / f"{log_stem}_tuning.log",
        log_dir / f"{log_stem}_samples.log",
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
    ensemble: str,
    tag: str,
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
        f"{ensemble}_{tag}_{strategy}_qda_ratio_fit_{momentum}_bT{int(bT)}_bz{int(bz)}_sample0"
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
        title=f"{ensemble}, {momentum}, bT={int(bT)}, bz={int(bz)}",
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


def _qda_ratio_samples(
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
    if qda_denominator_mode == "local":
        if pt2_bT is not None or pt2_bz is not None:
            raise ValueError(
                "local qDA denominators must not declare pt2_bT or pt2_bz"
            )
        return
    if qda_denominator_mode == "nonlocal_bz0":
        if pt2_bT is None or pt2_bz != 0:
            raise ValueError(
                "nonlocal_bz0 qDA denominators require pt2_bT and pt2_bz=0"
            )
        return
    raise ValueError(
        "qda_denominator_mode must be 'local' or 'nonlocal_bz0'"
    )


def _qda_fit_z_list(
    z_values: list[int],
    *,
    qda_denominator_mode: str,
    label: str = "bz",
) -> tuple[list[int], bool]:
    """Return z values to fit; drop z=0 only for ``nonlocal_bz0``.

    With the nonlocal bz=0 denominator fallback the ratio at z=0 is identically
    one, so fitting that point is skipped and later reinjected as ME=1.
    """
    values = [int(z) for z in z_values]
    if qda_denominator_mode != "nonlocal_bz0":
        return values, False
    fit_values = [z for z in values if z != 0]
    if not fit_values:
        raise ValueError(
            f"nonlocal_bz0 qDA requires at least one nonzero {label} value"
        )
    return fit_values, True


def _assigned_unity_z0_record(n_sample: int) -> dict[str, Any]:
    """Synthetic bare-ME record for nonlocal_bz0 z=0 (ratio identically one)."""
    ones = np.ones(int(n_sample), dtype=float)
    zeros = np.zeros(int(n_sample), dtype=float)
    return {
        "z": 0,
        "real_samples": ones,
        "imag_samples": zeros,
        "real_sys_sdev": 0.0,
        "imag_sys_sdev": 0.0,
        "window": None,
        "sample0_plot_paths": {},
    }


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
            selected_q, selected_chi2_dof = _selected_record_quality(records, weights)
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
                    "Q": selected_q,
                    "chi2_dof": selected_chi2_dof,
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


def _fit_qda_pt2_energy_sample_batch(payload: bytes, sample_indices: list[int]) -> list[tuple[int, float]]:
    context = gv.loads(payload)
    out: list[tuple[int, float]] = []
    for sample_index in sample_indices:
        try:
            pt2_sample = _recenter(context["pt2_samples"][sample_index], context["pt2_gv"])
            fit = fit_two_point(
                pt2_sample,
                context["tmin"],
                context["tmax"],
                context["Lt"],
                nstate=context["nstate"],
                svdcut=context["svdcut"],
                rescale=context["scale"],
                prior=context["prior"],
                p0=context["p0"],
                qda_denominator_mode=context["qda_denominator_mode"],
            )
            out.append((int(sample_index), float(gv.mean(fit.p["E0"]))))
        except NUMERICAL_FIT_ERRORS:
            out.append((int(sample_index), float("nan")))
    return out


def _qda_fit_mode_label(strategy: str) -> str:
    """Return the log/token label for a qDA fit strategy."""
    if strategy == "independent":
        return "independent_qda_ratio"
    return f"{strategy}_2pt_qda_ratio"


def _log_qda_sample_results(
    sample_results: list[dict[str, Any]],
    *,
    z: int,
    strategy: str,
    shared_window: dict[str, int],
    logger: Any,
    q_min: float,
) -> None:
    """Write every qDA sample fit, rejection, and failure to the sample log."""
    fit_mode = _qda_fit_mode_label(strategy)
    for result in sample_results:
        sample_index = int(result["sample"])
        for log_item in result["logs"]:
            if log_item["kind"] == "rejected":
                logger.info(
                    "Rejected %s z=%s sample=%s "
                    "nstate=%s prior_width=%s: %s",
                    fit_mode,
                    z,
                    sample_index,
                    log_item["nstate"],
                    log_item["prior_width"],
                    log_item["reason"],
                )
                continue
            log_nonlinear_fit_quality(
                SimpleNamespace(
                    Q=log_item["Q"],
                    chi2=log_item["chi2"],
                    dof=log_item["dof"],
                    logGBF=log_item["logGBF"],
                ),
                kind=f"sample ground-state {fit_mode}",
                label=(
                    f"z={z} sample={sample_index} "
                    f"t=[{shared_window['tmin']},{shared_window['tmax']}) "
                    f"nstate={log_item['nstate']} "
                    f"prior_width={log_item['prior_width']}"
                ),
                logger=logger,
                q_min=q_min,
            )
        if result["error"] is not None:
            logger.info(
                "Bad %s z=%s sample=%s: %s",
                fit_mode,
                z,
                sample_index,
                result["error"],
            )


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
    sample_re, sample_im = _qda_ratio_samples(numerator_samples, denominator_samples)
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
    tune_list, skipped_z0_fit = _qda_fit_z_list(
        tune_list,
        qda_denominator_mode=qda_denominator_mode,
        label="tune_z",
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
    strategies = fit_strategies or ([fit_strategy] if fit_strategy else ["joint"])
    states = [
        int(value)
        for value in (nstate_values or ([nstate] if nstate is not None else [2]))
    ]
    windows, pt2_scan = _resolve_pt2_windows(
        pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        nstate_values=states,
    )
    auto_window_scan = {
        "pt2": pt2_scan,
        "pt3": {"source": "not_used", "pt3_windows": []},
    }
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
    succeeded_counts_by_z = {
        str(z): len(records_by_z[z]) for z in tune_list
    }
    common_keys = set.intersection(
        *(set(records) for records in records_by_z.values())
    )
    if not common_keys:
        store[out] = []
        result = {
            "out": out,
            "status": "no_common_feasible_candidate",
            "retry_hint": (
                "No shared (strategy, nstate, prior_width, tmin, tmax) window "
                "succeeded at every tune_z. Retry tune_bare_matrix with a "
                "narrower tune_z_values list: keep the minimum nonzero z and "
                "one mid-range z; drop the largest tune z first. Do not guess "
                "a primary-z-only window for fit_bare_matrix_grid."
                + (
                    " For nonlocal_bz0, do not include z=0 in tune_z_values."
                    if skipped_z0_fit
                    else ""
                )
            ),
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
            "skipped_z0_fit": skipped_z0_fit,
            "succeeded_counts_by_z": succeeded_counts_by_z,
            "candidates": [],
            "rejected": rejected,
            "recommended_index": None,
            "recommended_fallback_no_q_passing": None,
            "recommended_window": None,
            "recommended_robust_index": None,
            "recommended_robust_window": None,
            "tuning_diagnostic_pdfs": {},
            "auto_window_scan": auto_window_scan,
        }
        store["_correlator_tuning_summary"] = {
            "pt2_path": str(pt2_path),
            "fit_scopes": ["qda_ratio"],
            "auto_window_scan": auto_window_scan,
            "recommended_robust_window": None,
            "status": result["status"],
        }
        return result
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
    result = {
        "out": out,
        "status": "ok",
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
        "skipped_z0_fit": skipped_z0_fit,
        "succeeded_counts_by_z": succeeded_counts_by_z,
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
        "auto_window_scan": auto_window_scan,
    }
    store["_correlator_tuning_summary"] = {
        "pt2_path": str(pt2_path),
        "fit_scopes": ["qda_ratio"],
        "auto_window_scan": auto_window_scan,
        "recommended_robust_window": result["recommended_robust_window"],
    }
    return result


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
    fit_z_list, skipped_z0_fit = _qda_fit_z_list(
        z_list,
        qda_denominator_mode=qda_denominator_mode,
        label="bz",
    )
    tune_z_value = int(tune_z) if tune_z is not None else fit_z_list[0]
    if tune_z_value not in z_list:
        raise ValueError("tune_z must be present in the qda_ratio bz grid")
    if skipped_z0_fit and tune_z_value == 0:
        tune_z_value = fit_z_list[0]
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    tuning_log_path, sample_log_path = _split_fit_log_paths(
        log_path=log_path,
        log_dir=fit_log_dir,
        log_stem=f"{ensemble}_{tag}_{momentum}_{strategy}_qda_ratio",
    )
    tuning_logger = setup_logger(
        tuning_log_path, logger_name="qda_ratio_tuning_logger"
    )
    sample_logger = setup_logger(
        sample_log_path, logger_name="qda_ratio_sample_logger"
    )
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
    effective_pt2_windows = [pt2_window] if pt2_window is not None else pt2_windows
    windows, pt2_scan = _resolve_pt2_windows(
        effective_pt2_windows,
        Lt=Lt,
        pt2_gv=pt2_gv,
        nstate_values=nstate_values,
    )
    auto_window_scan = {
        "pt2": pt2_scan,
        "pt3": {"source": "not_used", "pt3_windows": []},
    }
    tuning_summary = store.get("_correlator_tuning_summary")
    if (
        isinstance(tuning_summary, dict)
        and tuning_summary.get("pt2_path") == str(pt2_path)
        and isinstance(tuning_summary.get("auto_window_scan"), dict)
    ):
        auto_window_scan = tuning_summary["auto_window_scan"]
    tuning_logger.info("auto_window_scan=%s", json.dumps(auto_window_scan, sort_keys=True))
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
    tuning_logger.info(
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
    sample_fit_Q: list[float] = []
    sample_fit_chi2_dof: list[float] = []
    energy_fit = chosen.get("pt2_fit")
    if energy_fit is None and strategy == "independent":
        energy_fit = fit_two_point(
            pt2_gv,
            shared_window["tmin"],
            shared_window["tmax"],
            Lt,
            nstate=int(chosen["nstate"]),
            svdcut=svdcut,
            rescale=scale,
            prior=_vary_prior_width(
                qda_pt2_prior(
                    int(chosen["nstate"]),
                    qda_denominator_mode=qda_denominator_mode,
                ),
                float(chosen["prior_width"]),
            ),
            qda_denominator_mode=qda_denominator_mode,
        )
    if energy_fit is None:
        energy_fit = chosen["fit"]
    try:
        from tqdm import tqdm
    except ImportError:
        z_iterator = fit_z_list
    else:
        z_iterator = tqdm(
            fit_z_list,
            desc=f"fit qDA ratio {ensemble} {momentum}",
        )
    try:
        energy_template = qda_pt2_prior(
            int(chosen["nstate"]),
            qda_denominator_mode=qda_denominator_mode,
        )
        energy_prior = _scaled_prior(
            energy_fit,
            energy_template,
            error_scale=posterior_prior_error_scale,
            prior_width=float(chosen["prior_width"]),
        )
        energy_payload = gv.dumps(
            {
                "pt2_samples": pt2_samples,
                "pt2_gv": pt2_gv,
                "tmin": shared_window["tmin"],
                "tmax": shared_window["tmax"],
                "Lt": Lt,
                "nstate": int(chosen["nstate"]),
                "svdcut": svdcut,
                "scale": scale,
                "prior": energy_prior,
                "p0": _p0_from_fit(energy_fit, energy_template),
                "qda_denominator_mode": qda_denominator_mode,
            }
        )
        if executor is None:
            energy_results = _fit_qda_pt2_energy_sample_batch(energy_payload, sample_batches[0])
        else:
            futures = [
                executor.submit(_fit_qda_pt2_energy_sample_batch, energy_payload, batch)
                for batch in sample_batches
            ]
            energy_results = [item for future in futures for item in future.result()]
        E0_lattice_samples = [value for _sample, value in sorted(energy_results)]
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
                else None
            )
            imag_sys = (
                _weighted_model_sdev(average_imag, average_weights)
                if model_average and "im" in _parts(part)
                else None
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
                    logger=tuning_logger,
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
            _log_qda_sample_results(
                sample_results,
                z=z,
                strategy=strategy,
                shared_window=shared_window,
                logger=sample_logger,
                q_min=q_min,
            )
            real_samples = np.asarray(
                [item["real"] for item in sample_results], dtype=float
            )
            imag_samples = np.asarray(
                [item["imag"] for item in sample_results], dtype=float
            )
            failures = [item for item in sample_results if item["error"] is not None]
            for item in sample_results:
                _append_finite_sample_quality(sample_fit_Q, sample_fit_chi2_dof, item)
            if not np.any(np.isfinite(real_samples)):
                raise ValueError(f"all qda_ratio resampled fits failed for z={z}")
            real_mean, real_sdev = sample_mean_and_sdev(
                real_samples, mode=mode, sample_error_mode=sample_error_mode
            )
            imag_mean, imag_sdev = sample_mean_and_sdev(
                imag_samples, mode=mode, sample_error_mode=sample_error_mode
            )
            sample_logger.info(
                "summary z=%s real=%s +/- %s imag=%s +/- %s failed=%s",
                z,
                float(real_mean),
                float(real_sdev),
                float(imag_mean),
                float(imag_sdev),
                len(failures),
            )
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
                    ensemble=ensemble,
                    tag=tag,
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
    if skipped_z0_fit:
        z_records.append(_assigned_unity_z0_record(n_samples))
        z_report.append(
            {
                "z": 0,
                "source": "assigned_unity_nonlocal_bz0",
                "window": None,
                "rejected_fit_models": [],
                "n_failed_samples": 0,
                "sample_failures": [],
                "real_sys_sdev": 0.0,
                "imag_sys_sdev": 0.0,
                "sample0_plot_paths": {},
            }
        )
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
                "real_sys_sdev": _optional_float(record["real_sys_sdev"]),
                "imag_mean": float(imag_mean),
                "imag_stat_sdev": float(imag_error),
                "imag_sys_sdev": _optional_float(record["imag_sys_sdev"]),
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
            "skipped_z0_fit": skipped_z0_fit,
            "assigned_z0_unity": skipped_z0_fit,
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
        E0_lattice_samples=E0_lattice_samples,
        resample_mode=mode,
        sample_error_mode=sample_error_mode,
        workers=workers,
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
        "skipped_z0_fit": skipped_z0_fit,
        "assigned_z0_unity": skipped_z0_fit,
        "model_average": model_average,
        "selection_rule": (
            "qda_ratio shared pt2 window "
            f"(fallback_no_q_passing={tune_fallback})"
        ),
        "shared_window_specs": [shared_spec],
        "tuning_log_path": str(tuning_log_path),
        "sample_log_path": str(sample_log_path),
        "correlator_rescale": scale,
        "resample_mode": mode,
        "sample_error_mode": sample_error_mode,
        "n_samples": n_samples,
        "workers": int(workers),
        "bz": z_list,
        "tune_z": tune_z_value,
        "z_fits": z_report,
        "sample_fit_Q": sample_fit_Q,
        "sample_fit_chi2_dof": sample_fit_chi2_dof,
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
        "auto_window_scan": auto_window_scan,
    }


STAGE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_correlator_scale": inspect_correlator_scale,
    "tune_ground_state": tune_ground_state,
    "tune_bare_matrix": tune_bare_matrix,
    "fit_bare_matrix_grid": fit_bare_matrix_grid,
}
