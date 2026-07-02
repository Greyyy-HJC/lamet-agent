"""Correlator-analysis stage tools.

Purpose:
- provide a small, agentic tool set for 2pt ground-state and 3pt/2pt ratio analysis
- the agent drives strategy: inspect the 2pt scale, tune one shared fit setting on
  sample-average data, then apply that setting to every bootstrap/jackknife sample

Expected inputs:
- 2pt HDF5: ``source_sink/gamma/momentum`` with shape (Lt, n_cfg)
- 3pt HDF5: ``source_sink/gamma/momentum/b_dir/eta/bT*/bz*`` with shape (tsep+1, n_cfg)
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

from itertools import product
import json
from pathlib import Path
from typing import Any, Callable

import gvar as gv
import h5py
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np

np.seterr(over="ignore")

from lamet_agent.core.data import EnsembleData, EnsembleInfo
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
)
from lamet_agent.core.resampling import (
    resample_config_samples,
    sample_mean_err,
    samples_to_gvar,
)
from lamet_agent.core.tools import (
    log_nonlinear_fit_quality,
    resolve_plot_save_path,
    setup_logger,
)

# 2pt ground-state posteriors anchor the chained 3pt prior with widened errors.
PT2_PRIOR_ERROR_SCALE = 3.0


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


def fh_re_fcn(t: np.ndarray, tau_cut: int, p: dict, *, nstate: int = 2, dt: int | float = 1) -> np.ndarray:
    """Real FH finite-difference fit function."""
    return fh_fcn(t, tau_cut, p, nstate=nstate, part="re", dt=dt)


def fh_im_fcn(t: np.ndarray, tau_cut: int, p: dict, *, nstate: int = 2, dt: int | float = 1) -> np.ndarray:
    """Imaginary FH finite-difference fit function."""
    return fh_fcn(t, tau_cut, p, nstate=nstate, part="im", dt=dt)


def asymptotic_ratio(o00: gv.GVar, E0: gv.GVar, *, tsep: int, Lt: int) -> gv.GVar:
    """Ground-state ratio plateau at symmetric tau (wrap-aware)."""
    forward = gv.exp(-E0 * float(tsep))
    backward = gv.exp(-E0 * (float(Lt) - float(tsep)))
    return o00 * forward / (2 * E0 * (forward + backward))


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
    raw = "ratio" if value is None else str(value).strip().lower().replace(" ", "")
    if raw in {"ratio", "pt3_ratio", "3pt_ratio"}:
        return "ratio", "ratio"
    if raw in {"fh", "feynman-hellmann", "feynman_hellmann"}:
        return "FH", "fh"
    if raw in {"ratio+fh", "fh+ratio", "ratio_fh", "fh_ratio", "joint_ratio_fh"}:
        return "ratio+FH", "ratio_fh"
    raise ValueError(f"fit_scope must be 'ratio', 'FH', or 'ratio+FH', got {value!r}")


def _validate_scope_form(scope: str, fitting_form: str) -> None:
    if "FH" in scope and fitting_form == "NonBreit":
        raise ValueError("fit_scope values containing 'FH' currently require fitting_form='Breit'")


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
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator over ``[tmin, tmax)`` with an n-state ansatz."""
    fit_t = np.arange(tmin, tmax, dtype=int)
    fit_y = np.asarray(pt2_gv)[fit_t] * rescale
    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(fit_t, fit_y),
        prior=prior if prior is not None else pt2_prior(nstate),
        fcn=lambda t, p: pt2_re_fcn(t, p, Lt, nstate=nstate),
        svdcut=svdcut,
        maxit=10000,
        **kwargs,
    )


def fit_ratio(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Fit real/imag 3pt/2pt ratio data with an n-state ansatz (scale-invariant)."""
    parts = _parts(part)
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)
    x_vecs = [np.array(ts, dtype=float), np.array(taus, dtype=float)]
    all_y = {"re": re_vals, "im": im_vals}
    y_data = {key: all_y[key] for key in parts}

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        return {key: pt3_ratio_fcn(x[0], x[1], p, Lt, nstate=nstate, part=key) for key in parts}

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_vecs, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_nonbreit_ratio(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Fit real/imag non-forward 3pt ratio data with separate i/f spectra."""
    parts = _parts(part)
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)
    x_vecs = [np.array(ts, dtype=float), np.array(taus, dtype=float)]
    all_y = {"re": re_vals, "im": im_vals}
    y_data = {key: all_y[key] for key in parts}

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        return {key: pt3_nonbreit_ratio_fcn(x[0], x[1], p, Lt, nstate=nstate, part=key) for key in parts}

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_vecs, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_joint(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Jointly fit 2pt data and real/imag 3pt/2pt ratios."""
    parts = _parts(part)
    fit_t = np.arange(tmin, tmax, dtype=int)
    fit_pt2 = np.asarray(pt2_gv)[fit_t] * rescale
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)

    x_data = {
        "pt2_t": fit_t,
        "ratio_t": np.array(ts, dtype=float),
        "ratio_tau": np.array(taus, dtype=float),
    }
    y_data: dict[str, Any] = {"pt2": fit_pt2}
    if "re" in parts:
        y_data["ratio_re"] = re_vals
    if "im" in parts:
        y_data["ratio_im"] = im_vals

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values = {"pt2": pt2_re_fcn(x["pt2_t"], p, Lt, nstate=nstate)}
        if "re" in parts:
            values["ratio_re"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="re")
        if "im" in parts:
            values["ratio_im"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="im")
        return values

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_data, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_nonbreit_joint(
    pt2_i_gv: np.ndarray,
    pt2_f_gv: np.ndarray,
    tmin: int,
    tmax: int,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Jointly fit initial/final 2pt data and a non-forward 3pt ratio."""
    parts = _parts(part)
    fit_t = np.arange(tmin, tmax, dtype=int)
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)

    x_data = {
        "pt2_t": fit_t,
        "ratio_t": np.array(ts, dtype=float),
        "ratio_tau": np.array(taus, dtype=float),
    }
    y_data: dict[str, Any] = {
        "pt2_i": np.asarray(pt2_i_gv)[fit_t] * rescale,
        "pt2_f": np.asarray(pt2_f_gv)[fit_t] * rescale,
    }
    if "re" in parts:
        y_data["ratio_re"] = re_vals
    if "im" in parts:
        y_data["ratio_im"] = im_vals

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values = {
            "pt2_i": _pt2_re_fcn_with_suffix(x["pt2_t"], p, Lt, nstate=nstate, suffix="_i"),
            "pt2_f": _pt2_re_fcn_with_suffix(x["pt2_t"], p, Lt, nstate=nstate, suffix="_f"),
        }
        if "re" in parts:
            values["ratio_re"] = pt3_nonbreit_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="re")
        if "im" in parts:
            values["ratio_im"] = pt3_nonbreit_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="im")
        return values

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_data, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_fh(
    fh_re: np.ndarray,
    fh_im: np.ndarray,
    tsep_ls: list[int],
    tau_cut: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Fit real/imag FH data with a finite-difference summed-ratio ansatz."""
    parts = _parts(part)
    fit_t = np.asarray(tsep_ls[:-1], dtype=float)
    re_vals = np.asarray(fh_re, dtype=object)
    im_vals = np.asarray(fh_im, dtype=object)
    if fit_t.size == 0:
        raise ValueError("FH fit window must contain at least one point")
    if re_vals.ndim != 1 or im_vals.ndim != 1:
        raise ValueError("FH fit data must be one-dimensional")
    if len(re_vals) != len(fit_t) or len(im_vals) != len(fit_t):
        raise ValueError("FH data length must match len(tsep_ls) - 1")
    all_y = {"re": re_vals, "im": im_vals}
    y_data = {key: all_y[key] for key in parts}
    dt = _fh_dt(tsep_ls)

    def fcn(t: np.ndarray, p: dict) -> dict[str, np.ndarray]:
        return {key: fh_fcn(t, tau_cut, p, nstate=nstate, part=key, dt=dt) for key in parts}

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(fit_t, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_ratio_fh(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    fh_re: np.ndarray,
    fh_im: np.ndarray,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Jointly fit ratio and FH observables without refitting the 2pt data."""
    parts = _parts(part)
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)
    fh_t = np.asarray(tsep_ls[:-1], dtype=float)
    x_data = {
        "ratio_t": np.array(ts, dtype=float),
        "ratio_tau": np.array(taus, dtype=float),
        "fh_t": fh_t,
    }
    y_data: dict[str, Any] = {}
    if "re" in parts:
        y_data["ratio_re"] = re_vals
        y_data["fh_re"] = np.asarray(fh_re, dtype=object)
    if "im" in parts:
        y_data["ratio_im"] = im_vals
        y_data["fh_im"] = np.asarray(fh_im, dtype=object)
    dt = _fh_dt(tsep_ls)

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values: dict[str, Any] = {}
        if "re" in parts:
            values["ratio_re"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="re")
            values["fh_re"] = fh_re_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        if "im" in parts:
            values["ratio_im"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="im")
            values["fh_im"] = fh_im_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        return values

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_data, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_joint_fh(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    fh_re: np.ndarray,
    fh_im: np.ndarray,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Jointly fit 2pt data and FH data."""
    parts = _parts(part)
    fit_t = np.arange(tmin, tmax, dtype=int)
    fh_t = np.asarray(tsep_ls[:-1], dtype=float)
    x_data = {"pt2_t": fit_t, "fh_t": fh_t}
    y_data: dict[str, Any] = {"pt2": np.asarray(pt2_gv)[fit_t] * rescale}
    if "re" in parts:
        y_data["fh_re"] = np.asarray(fh_re, dtype=object)
    if "im" in parts:
        y_data["fh_im"] = np.asarray(fh_im, dtype=object)
    dt = _fh_dt(tsep_ls)

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values: dict[str, Any] = {"pt2": pt2_re_fcn(x["pt2_t"], p, Lt, nstate=nstate)}
        if "re" in parts:
            values["fh_re"] = fh_re_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        if "im" in parts:
            values["fh_im"] = fh_im_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        return values

    kwargs = {"p0": p0} if p0 is not None else {}
    return lsf.nonlinear_fit(
        data=(x_data, y_data), prior=prior, fcn=fcn, svdcut=svdcut, maxit=10000, **kwargs
    )


def fit_joint_ratio_fh(
    pt2_gv: np.ndarray,
    tmin: int,
    tmax: int,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    fh_re: np.ndarray,
    fh_im: np.ndarray,
    tsep_ls: list[int],
    tau_cut: int,
    Lt: int,
    *,
    nstate: int = 2,
    part: str = "both",
    svdcut: float = 1e-2,
    rescale: float = 1.0,
    prior: gv.BufferDict,
    p0: dict[str, float] | None = None,
) -> lsf.nonlinear_fit:
    """Jointly fit 2pt, ratio, and FH data."""
    parts = _parts(part)
    fit_t = np.arange(tmin, tmax, dtype=int)
    ts, taus, re_vals, im_vals = _ratio_points(ratio_re, ratio_im, tsep_ls, tau_cut)
    fh_t = np.asarray(tsep_ls[:-1], dtype=float)
    x_data = {
        "pt2_t": fit_t,
        "ratio_t": np.array(ts, dtype=float),
        "ratio_tau": np.array(taus, dtype=float),
        "fh_t": fh_t,
    }
    y_data: dict[str, Any] = {"pt2": np.asarray(pt2_gv)[fit_t] * rescale}
    if "re" in parts:
        y_data["ratio_re"] = re_vals
        y_data["fh_re"] = np.asarray(fh_re, dtype=object)
    if "im" in parts:
        y_data["ratio_im"] = im_vals
        y_data["fh_im"] = np.asarray(fh_im, dtype=object)
    dt = _fh_dt(tsep_ls)

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values: dict[str, Any] = {"pt2": pt2_re_fcn(x["pt2_t"], p, Lt, nstate=nstate)}
        if "re" in parts:
            values["ratio_re"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="re")
            values["fh_re"] = fh_re_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        if "im" in parts:
            values["ratio_im"] = pt3_ratio_fcn(x["ratio_t"], x["ratio_tau"], p, Lt, nstate=nstate, part="im")
            values["fh_im"] = fh_im_fcn(x["fh_t"], tau_cut, p, nstate=nstate, dt=dt)
        return values

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
    ratio_points = _ratio_data_count([int(t) for t in spec["tsep_ls"]], int(spec["tau_cut"]))
    fh_points = max(len(spec["tsep_ls"]) - 1, 0)

    total = 0
    if strategy == "joint":
        total += pt2_points * (2 if fitting_form == "NonBreit" else 1)
    if fit_scope in {"ratio", "ratio+FH"}:
        total += ratio_points * components
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
        try:
            p0[key] = float(gv.mean(fit.p[key]))
        except Exception:
            p0[key] = float(gv.mean(prior[key]))
    return p0


def _anchor_pt2_prior(prior: gv.BufferDict, pt2_fit: lsf.nonlinear_fit, suffix: str = "") -> None:
    """Pin E0 and z0 of a ratio prior to widened 2pt posteriors (chained mode)."""
    for key in ("E0", "z0"):
        value = pt2_fit.p[key]
        prior[_state_key(key, suffix=suffix)] = gv.gvar(gv.mean(value), gv.sdev(value) * PT2_PRIOR_ERROR_SCALE)


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


def _bare_samples(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract O00/(2*E0) real/imag per record (NaN for failed fits)."""
    real: list[float] = []
    imag: list[float] = []
    for rec in records:
        fit = rec.get("fit")
        if fit is None:
            real.append(np.nan)
            imag.append(np.nan)
            continue
        real.append(float(gv.mean(fit.p["O00_re"] / (2 * fit.p["E0"]))))
        imag.append(float(gv.mean(fit.p["O00_im"] / (2 * fit.p["E0"]))))
    return np.asarray(real, dtype=float), np.asarray(imag, dtype=float)


def _bare_matrix_element_from_fit(p: dict, *, part: str, fitting_form: str) -> Any:
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
) -> float:
    """Return zero for the component that was intentionally excluded from the fit."""
    if output_part not in _parts(fit_part):
        return 0.0
    return float(gv.mean(_bare_matrix_element_from_fit(p, part=output_part, fitting_form=fitting_form)))


def _ratio_prior_template(fitting_form: str, nstate: int) -> gv.BufferDict:
    if fitting_form == "NonBreit":
        return pt3_nonbreit_ratio_prior(nstate)
    return pt3_ratio_prior(nstate)


def _scope_prior_template(fitting_form: str, nstate: int, fit_scope: str, strategy: str) -> gv.BufferDict:
    _validate_scope_form(fit_scope, fitting_form)
    if fit_scope == "ratio":
        return _ratio_prior_template(fitting_form, nstate)
    if fit_scope == "FH":
        return _joint_fh_prior(nstate) if strategy == "joint" else fh_prior(nstate)
    if fit_scope == "ratio+FH":
        return _ratio_fh_prior(nstate)
    raise ValueError(f"unsupported fit_scope {fit_scope!r}")


def _scope_prior_with_width(
    fitting_form: str, nstate: int, fit_scope: str, strategy: str, prior_width: float
) -> gv.BufferDict:
    return _vary_prior_width(_scope_prior_template(fitting_form, nstate, fit_scope, strategy), prior_width)


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


def _read_2pt(path: str, *, source_sink: str, gamma: str, momentum: str) -> np.ndarray:
    """Read one 2pt dataset as a complex (n_cfg, Lt) array."""
    with h5py.File(path, "r") as h5f:
        return np.swapaxes(np.asarray(h5f[source_sink][gamma][momentum]), 0, 1)


def _read_3pt(
    path: str,
    *,
    source_sink: str,
    gamma: str,
    momentum: str,
    b_dir: str,
    eta: str,
    bt: str,
    bz: str,
    tsep: int,
) -> np.ndarray:
    """Read one 3pt slice as a complex (n_cfg, tsep+1) array."""
    dset = f"{source_sink}/{gamma}/{momentum}/{b_dir}/{eta}/{bt}/{bz}"
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


def _normalise_pt2_windows(windows: list[dict[str, int]] | None, *, Lt: int) -> list[dict[str, int]]:
    if windows is not None:
        return [{"tmin": int(w["tmin"]), "tmax": int(w["tmax"])} for w in windows]
    quarter = max(Lt // 4, 1)
    tmins = list(range(2, quarter - 3))
    return [{"tmin": tmin, "tmax": quarter} for tmin in tmins[:DEFAULT_MAX_PT2_WINDOWS]]


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
    momentum: str,
    z: int,
    fit_label: str,
    fitting_form: str = "Breit",
    part: str = "both",
) -> dict[str, str]:
    stem = log_dir / f"{fit_label}_{momentum}_z{z}_sample0"
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
            "fit_re": fh_re_fcn(fit_t, tau_cut, p, nstate=nstate, dt=dt),
            "fit_im": fh_im_fcn(fit_t, tau_cut, p, nstate=nstate, dt=dt),
            "color": COLOR_CYCLE[0],
        }
    ]


def _plot_sample0_fh(
    *,
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    rec: dict[str, Any],
    log_dir: Path,
    momentum: str,
    z: int,
    fit_label: str,
    part: str = "both",
) -> dict[str, str]:
    stem = log_dir / f"{fit_label}_{momentum}_z{z}_sample0"
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
    momentum: str,
    fit_label: str,
) -> dict[str, str]:
    stem = log_dir / f"{fit_label}_{momentum}_sample0"
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


def _split_log_paths(
    *,
    log_dir: Path,
    log_path: str | Path | None,
    ensemble: str,
    tag: str,
    variant: str,
    direction: str,
    momentum: str,
    b_label: str,
    fit_mode: str,
) -> tuple[Path, Path]:
    if log_path is not None:
        base = Path(log_path)
        suffix = base.suffix or ".log"
        return base.with_name(f"{base.stem}_tuning{suffix}"), base.with_name(f"{base.stem}_samples{suffix}")
    stem = f"{ensemble}_{tag}_{variant}_{direction}_{momentum}_{b_label}_{fit_mode}"
    return log_dir / f"{stem}_tuning.log", log_dir / f"{stem}_samples.log"


def _ensemble_resample_name(mode: str) -> str:
    if mode == "bs":
        return "bootstrap"
    if mode == "jk":
        return "jackknife"
    return mode


def _artifact_ensemble_info(ensemble_id: str) -> EnsembleInfo:
    return EnsembleInfo("", str(ensemble_id), 1.0, 1.0, 1, 1, 0.0)


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
            "bare_re_sys_sdev": json.dumps([float(rec.get("real_sys_sdev", 0.0)) for rec in sorted_records]),
            "bare_im_sys_sdev": json.dumps([float(rec.get("imag_sys_sdev", 0.0)) for rec in sorted_records]),
        }
    )
    return EnsembleData(
        ensemble=_artifact_ensemble_info(str(bare_attrs.get("ensemble", ""))),
        resample=_ensemble_resample_name(resample_mode),
        values=values,
        dims=("z",),
        coords={"z": z_values},
        attrs={key: str(value) for key, value in bare_attrs.items() if value is not None},
        name="bare_matrix_element",
    )


def _write_outputs(
    records: list[dict[str, Any]],
    *,
    artifacts_dir: Path,
    save_path: str | None,
    ensemble: str,
    tag: str,
    variant: str,
    direction: str,
    momentum: str,
    b_label: str,
    resample_mode: str,
    matrix_element_label: str = r"Bare matrix element $O_{00}/(2E_0)$",
    plot_title: str | None = None,
    ylim: tuple[float, float] = (-0.2, 1.2),
    part: str = "both",
) -> dict[str, Any]:
    """Write the bare matrix-element NetCDF plus diagnostic plot."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    resolved_save = resolve_plot_save_path(save_path, artifacts_dir=artifacts_dir, default_stem="bare_matrix_elements")
    plotted_parts = _parts(part)

    z_values: list[int] = []
    real_mean: list[float] = []
    real_err: list[float] = []
    imag_mean: list[float] = []
    imag_err: list[float] = []
    outputs: list[dict[str, Any]] = []

    for rec in sorted(records, key=lambda item: item["z"]):
        z = rec["z"]
        real = np.asarray(rec["real_samples"], dtype=float)
        imag = np.asarray(rec["imag_samples"], dtype=float)
        r_mean, r_err = sample_mean_err(real, mode=resample_mode)
        i_mean, i_err = sample_mean_err(imag, mode=resample_mode)
        rec["real_mean"] = r_mean
        rec["imag_mean"] = i_mean
        rec["real_stat_sdev"] = r_err
        rec["imag_stat_sdev"] = i_err
        z_values.append(z)
        real_mean.append(r_mean)
        real_err.append(r_err)
        imag_mean.append(i_mean)
        imag_err.append(i_err)
        outputs.append(
            {
                "z": z,
                "n_samples": int(real.shape[0]),
                "n_failed_samples": int(np.count_nonzero(~np.isfinite(real) | ~np.isfinite(imag))),
                "real_mean": r_mean,
                "real_sdev": r_err,
                "real_stat_sdev": r_err,
                "real_sys_sdev": float(rec.get("real_sys_sdev", 0.0)),
                "imag_mean": i_mean,
                "imag_sdev": i_err,
                "imag_stat_sdev": i_err,
                "imag_sys_sdev": float(rec.get("imag_sys_sdev", 0.0)),
                "window": rec["window"],
                "sample0_plot_paths": rec.get("sample0_plot_paths", {}),
            }
        )

    fig, ax = default_plot()
    if "re" in plotted_parts:
        ax.errorbar(z_values, real_mean, real_err, label="Re", color=COLOR_CYCLE[0], **ERRORBAR_STYLE)
    if "im" in plotted_parts:
        ax.errorbar(z_values, imag_mean, imag_err, label="Im", color=COLOR_CYCLE[1], marker="s", **ERRORBAR_STYLE)
    ax.set_xlabel(r"$z/a$", **FONT_SIZE)
    ax.set_ylabel(matrix_element_label, **FONT_SIZE)
    ax.set_title(plot_title or f"{ensemble} {momentum} {direction} bare matrix elements", **FONT_SIZE)
    ax.set_ylim(*ylim)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    pdf_path = f"{resolved_save}.pdf"
    svg_path = f"{resolved_save}.svg"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    data = _bare_records_to_ensemble(
        records,
        resample_mode=resample_mode,
        attrs={
            "ensemble": ensemble,
            "tag": tag,
            "variant": variant,
            "direction": direction,
            "momentum": momentum,
            "b_label": b_label,
            "resample_mode": resample_mode,
            "part": part,
        },
    )
    artifact = f"{resolved_save}.nc"
    data.to_netcdf(artifact)
    return {
        "artifact": artifact,
        "netcdf_path": artifact,
        "data": data,
        "plot_pdf": pdf_path,
        "plot_svg": svg_path,
        "n_z": len(records),
        "n_sample": data.n_sample,
        "outputs": outputs,
    }


def _progress(iterable, *, desc: str):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc)


# --- tool 1: inspect the 2pt scale ------------------------------------------


def inspect_correlator_scale(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_windows: list[dict[str, int]] | None = None,
    source_sink: str = "SS",
    gamma: str = "5",
    momentum: str = "PX0PY0PZ0",
    selectors: dict[str, Any] | None = None,
    out: str = "correlator_scale_inspection",
) -> dict[str, Any]:
    """Report 2pt magnitudes so the agent can choose a power-of-ten correlator_rescale."""
    if selectors is not None:
        source_sink = str(selectors.get("source_sink") or source_sink)
        gamma = str(selectors.get("gamma") or selectors.get("pt2_gamma") or gamma)
        momentum = str(selectors.get("momentum") or momentum)
    pt2_real = np.real(_read_2pt(pt2_path, source_sink=source_sink, gamma=gamma, momentum=momentum))
    n_cfg, Lt = pt2_real.shape
    windows = _normalise_pt2_windows(pt2_windows, Lt=Lt)
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
        "source_sink": source_sink,
        "gamma": gamma,
        "momentum": momentum,
        "n_cfg": int(n_cfg),
        "Lt": int(Lt),
        "windows": window_stats,
        "target_typical_abs_range": [0.0001, 0.01],
    }
    store[out] = result
    return result


# --- tool 2: tune the 2pt ground state (2pt-only path) ----------------------


def tune_ground_state(
    store: dict[str, Any],
    *,
    pt2_path: str,
    source_sink: str = "SS",
    gamma: str = "5",
    momentum: str = "PX0PY0PZ0",
    pt2_windows: list[dict[str, int]] | None = None,
    nstate: int = 2,
    svdcut: float = 1e-2,
    correlator_rescale: float = 1.0,
    resample_mode: str = "jk",
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
    mode = _check_mode(resample_mode)
    scale = _check_rescale(correlator_rescale)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"

    pt2_complex = _read_2pt(pt2_path, source_sink=source_sink, gamma=gamma, momentum=momentum)
    n_cfg, Lt = pt2_complex.shape
    re_samples, _ = resample_config_samples(np.real(pt2_complex), mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size)
    pt2_gv = samples_to_gvar(re_samples, mode=mode)
    store["Lt"] = int(Lt)

    windows = _normalise_pt2_windows(pt2_windows, Lt=Lt)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for window in windows:
        try:
            fit = fit_two_point(pt2_gv, window["tmin"], window["tmax"], Lt, nstate=nstate, svdcut=svdcut, rescale=scale)
            records.append(_record(fit, tmin=window["tmin"], tmax=window["tmax"], nstate=nstate, correlator_rescale=scale))
        except Exception as exc:
            rejected.append({**window, "reason": str(exc)})
    if not records:
        raise ValueError("all 2pt windows failed: " + "; ".join(str(item) for item in rejected[:5]))
    store["pt2_scan"] = records

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
    raise ValueError(f"fit_strategy must be 'joint' or 'chained', got {value!r}")


def _fit_mode_label(strategy: str, scope_label: str) -> str:
    return f"{strategy}_2pt_{scope_label}"


def _fit_average(
    spec: dict[str, Any],
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
) -> lsf.nonlinear_fit:
    """Fit one candidate window on sample-average data for the chosen strategy."""
    template = _scope_prior_with_width(fitting_form, nstate, fit_scope, strategy, prior_width)
    fh_re = fh_im = None
    if "FH" in fit_scope:
        fh_re, fh_im = _fh_samples_from_ratios(ratio_re, ratio_im, spec["tsep_ls"], spec["tau_cut"])
    if strategy == "joint":
        if fitting_form == "NonBreit":
            return fit_nonbreit_joint(
                pt2_gv, pt2_f_gv if pt2_f_gv is not None else pt2_gv,
                spec["tmin"], spec["tmax"], ratio_re, ratio_im, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=template,
            )
        if fit_scope == "FH":
            return fit_joint_fh(
                pt2_gv, spec["tmin"], spec["tmax"], fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=template,
            )
        if fit_scope == "ratio+FH":
            return fit_joint_ratio_fh(
                pt2_gv, spec["tmin"], spec["tmax"], ratio_re, ratio_im, fh_re, fh_im,
                spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=template,
            )
        return fit_joint(
            pt2_gv, spec["tmin"], spec["tmax"], ratio_re, ratio_im, spec["tsep_ls"], spec["tau_cut"], Lt,
            nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=template,
        )
    prior = template
    if fitting_form == "NonBreit":
        _anchor_pt2_prior(prior, pt2_best["fit"], suffix="_i")
        _anchor_pt2_prior(prior, (pt2_f_best or pt2_best)["fit"], suffix="_f")
        return fit_nonbreit_ratio(
            ratio_re, ratio_im, spec["tsep_ls"], spec["tau_cut"], Lt,
            nstate=nstate, part=part, svdcut=svdcut, prior=prior,
        )
    if fit_scope == "FH":
        _anchor_fh_energy_prior(prior, pt2_best["fit"], nstate)
        return fit_fh(
            fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"],
            nstate=nstate, part=part, svdcut=svdcut, prior=prior,
        )
    if fit_scope == "ratio+FH":
        _anchor_pt2_prior(prior, pt2_best["fit"])
        _anchor_fh_energy_prior(prior, pt2_best["fit"], nstate)
        return fit_ratio_fh(
            ratio_re, ratio_im, fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"], Lt,
            nstate=nstate, part=part, svdcut=svdcut, prior=prior,
        )
    _anchor_pt2_prior(prior, pt2_best["fit"])
    return fit_ratio(
        ratio_re, ratio_im, spec["tsep_ls"], spec["tau_cut"], Lt,
        nstate=nstate, part=part, svdcut=svdcut, prior=prior,
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
        try:
            fit = _fit_average(
                spec, strategy=strategy, fit_scope=fit_scope, pt2_gv=pt2_gv, pt2_f_gv=pt2_f_gv,
                ratio_re=ratio_re, ratio_im=ratio_im, pt2_best=pt2_best, pt2_f_best=pt2_f_best,
                Lt=Lt, nstate=nstate, part=part, svdcut=svdcut, scale=scale, fitting_form=fitting_form,
                prior_width=prior_width,
            )
            usable, reason = _fit_usable(fit, template)
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
        except Exception as exc:
            rejected.append({**size_metadata, "nstate": nstate, "prior_width": prior_width, "reason": str(exc)})
    return records, rejected


def _candidate_specs(
    *,
    strategy: str,
    pt2_window_specs: list[dict[str, int]],
    pt3_window_specs: list[dict[str, Any]],
    pt2_best: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Cartesian window candidates: joint pairs 2pt windows with ratio windows."""
    if strategy == "joint":
        return [
            {"tmin": w["tmin"], "tmax": w["tmax"], "tsep_ls": p["tsep_ls"], "tau_cut": p["tau_cut"]}
            for w, p in product(pt2_window_specs, pt3_window_specs)
        ]
    tmin = pt2_best["tmin"]
    tmax = pt2_best["tmax"]
    return [{"tmin": tmin, "tmax": tmax, "tsep_ls": p["tsep_ls"], "tau_cut": p["tau_cut"]} for p in pt3_window_specs]


# --- tool 3: tune the bare matrix on sample-average data ---------------------


def tune_bare_matrix(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_out_path: str | None = None,
    pt3_paths: dict[str, str] | list[str],
    tsep_ls: list[int],
    momentum: str,
    momentum_out: str | None = None,
    pt3_momentum: str | None = None,
    fitting_form: str = "Breit",
    tune_z: int = 0,
    source_sink: str = "SS",
    pt2_gamma: str = "5",
    pt3_gamma: str = "T",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bt: str = "bT0",
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
    n_boot: int = 200,
    seed: int | None = 1984,
    bin_size: int = 1,
    part: str = "both",
    q_min: float = 0.05,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "bare_tune",
) -> dict[str, Any]:
    """Scan bare-matrix fit windows on sample-average data for one representative z.

    Returns ranked candidate diagnostics and writes a tuning ratio
    plot so the agent can choose one shared window to pass to ``fit_bare_matrix_grid``.
    """
    form = _normalise_fitting_form(fitting_form)
    scale = _check_rescale(correlator_rescale)
    mode = _check_mode(resample_mode)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    tseps = [int(t) for t in tsep_ls]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    final_momentum = momentum if momentum_out is None else momentum_out
    three_point_momentum = momentum if pt3_momentum is None else pt3_momentum

    pt2_complex = _read_2pt(pt2_path, source_sink=source_sink, gamma=pt2_gamma, momentum=momentum)
    n_cfg, Lt = pt2_complex.shape
    re_samples, pt2_complex_samples, indices = _resample_pt2(pt2_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size)
    pt2_gv = samples_to_gvar(re_samples, mode=mode)
    pt2_f_gv = None
    pt2_f_complex_samples = pt2_complex_samples
    if form == "NonBreit":
        pt2_f_complex = _read_2pt(pt2_out_path or pt2_path, source_sink=source_sink, gamma=pt2_gamma, momentum=final_momentum)
        if pt2_f_complex.shape != pt2_complex.shape:
            raise ValueError(f"initial/final 2pt shape mismatch: {pt2_complex.shape} != {pt2_f_complex.shape}")
        re_f_samples, pt2_f_complex_samples, _ = _resample_pt2(
            pt2_f_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
        )
        pt2_f_gv = samples_to_gvar(re_f_samples, mode=mode)

    ratio_re: dict[int, np.ndarray] = {}
    ratio_im: dict[int, np.ndarray] = {}
    for tsep in tseps:
        pt3 = _read_3pt(
            paths_by_tsep[tsep], source_sink=source_sink, gamma=pt3_gamma, momentum=three_point_momentum,
            b_dir=b_dir, eta=eta, bt=bt, bz=f"bz{int(tune_z)}", tsep=tsep,
        )
        pt3_samples, _ = resample_config_samples(pt3, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices)
        if form == "NonBreit":
            re_s, im_s = _non_forward_ratio_samples(pt2_complex_samples, pt2_f_complex_samples, pt3_samples, tsep)
        else:
            re_s, im_s = _ratio_samples(pt2_complex_samples, pt3_samples, tsep)
        ratio_re[tsep] = samples_to_gvar(re_s, mode=mode)
        ratio_im[tsep] = samples_to_gvar(im_s, mode=mode)

    pt2_window_specs = _normalise_pt2_windows(pt2_windows, Lt=Lt)
    pt3_window_specs = _normalise_pt3_windows(pt3_windows, tsep_ls=tseps, tau_cuts=pt3_tau_cuts)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    strategies = fit_strategies or ([fit_strategy] if fit_strategy is not None else ["joint"])
    scopes = fit_scope_values or ([fit_scope] if fit_scope is not None else ["ratio"])
    states = nstate_values or ([nstate] if nstate is not None else [2])
    prior_widths = _normalise_prior_width(prior_width)
    for strategy_value in strategies:
        strategy, _ = _normalise_strategy(strategy_value)
        for scope_value in scopes:
            scope, _ = _normalise_fit_scope(scope_value)
            _validate_scope_form(scope, form)
            for nstate_value in states:
                nstate_int = int(nstate_value)
                if "FH" in scope and nstate_int > 2:
                    rejected.append({"fit_strategy": strategy, "fit_scope": scope, "nstate": nstate_int, "reason": "FH fits currently support nstate <= 2"})
                    continue
                for prior_width_value in prior_widths:
                    pt2_best = None
                    pt2_f_best = None
                    if strategy == "chained":
                        pt2_records: list[dict[str, Any]] = []
                        pt2_f_records: list[dict[str, Any]] = []
                        pt2_prior_template = _vary_prior_width(pt2_prior(nstate_int), prior_width_value)
                        for window in pt2_window_specs:
                            try:
                                fit = fit_two_point(
                                    pt2_gv, window["tmin"], window["tmax"], Lt,
                                    nstate=nstate_int, svdcut=svdcut, rescale=scale, prior=pt2_prior_template,
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
                            except Exception as exc:
                                rejected.append({
                                    **window,
                                    "fit_strategy": strategy,
                                    "fit_scope": scope,
                                    "nstate": nstate_int,
                                    "prior_width": prior_width_value,
                                    "reason": str(exc),
                                })
                            if form == "NonBreit" and pt2_f_gv is not None:
                                try:
                                    fit_f = fit_two_point(
                                        pt2_f_gv, window["tmin"], window["tmax"], Lt,
                                        nstate=nstate_int, svdcut=svdcut, rescale=scale, prior=pt2_prior_template,
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
                                except Exception as exc:
                                    rejected.append({
                                        **window,
                                        "fit_strategy": strategy,
                                        "fit_scope": scope,
                                        "nstate": nstate_int,
                                        "prior_width": prior_width_value,
                                        "reason": str(exc),
                                    })
                        if not pt2_records:
                            continue
                        pt2_best = pt2_records[select_best(pt2_records, q_min=q_min)[0]]
                        if form == "NonBreit":
                            if not pt2_f_records:
                                continue
                            pt2_f_best = pt2_f_records[select_best(pt2_f_records, q_min=q_min)[0]]

                    specs = _candidate_specs(
                        strategy=strategy,
                        pt2_window_specs=pt2_window_specs,
                        pt3_window_specs=pt3_window_specs,
                        pt2_best=pt2_best,
                    )
                    found, failed = _scan_average(
                        specs, strategy=strategy, fit_scope=scope, pt2_gv=pt2_gv, pt2_f_gv=pt2_f_gv,
                        ratio_re=ratio_re, ratio_im=ratio_im, pt2_best=pt2_best, pt2_f_best=pt2_f_best,
                        Lt=Lt, nstate=nstate_int, part=part, svdcut=svdcut, scale=scale, fitting_form=form,
                        prior_width=prior_width_value,
                    )
                    for rec in found:
                        rec["fit_strategy"] = strategy
                        rec["fit_scope"] = scope
                    records.extend(found)
                    rejected.extend({**item, "fit_strategy": strategy, "fit_scope": scope, "nstate": nstate_int} for item in failed)
    if not records:
        raise ValueError("all bare-matrix tuning windows failed: " + "; ".join(str(item) for item in rejected[:5]))
    store[out] = records

    best_index, fallback = select_data_window(records, q_min=q_min)
    best = records[best_index]

    candidates = []
    for i, rec in enumerate(records):
        p = rec["fit"].p
        candidate = {
            "index": i,
            "fit_strategy": rec["fit_strategy"],
            "fit_scope": rec["fit_scope"],
            "nstate": rec["nstate"],
            "prior_width": rec["prior_width"],
            "tmin": rec["tmin"],
            "tmax": rec["tmax"],
            "tsep_ls": rec["tsep_ls"],
            "tau_cut": rec["tau_cut"],
            "Q": rec["Q"],
            "chi2_dof": rec["chi2_dof"],
            "logGBF": rec["logGBF"],
            "n_data": rec["n_data"],
            "n_params": rec["n_params"],
            "dof_is_positive": rec["dof_is_positive"],
            "bare_re": str(_bare_matrix_element_from_fit(p, part="re", fitting_form=form)),
            "bare_im": str(_bare_matrix_element_from_fit(p, part="im", fitting_form=form)),
        }
        if form == "Breit":
            candidate["O00_re_over_2E0"] = candidate["bare_re"]
            candidate["O00_im_over_2E0"] = candidate["bare_im"]
        candidates.append(candidate)
    return {
        "out": out,
        "fit_strategies": strategies,
        "fit_scopes": scopes,
        "nstate_values": states,
        "prior_width": prior_widths,
        "tune_z": int(tune_z),
        "Lt": int(Lt),
        "n_cfg": int(n_cfg),
        "correlator_rescale": scale,
        "fitting_form": form,
        "candidates": candidates,
        "rejected": rejected,
        "recommended_index": best_index,
        "recommended_fallback_no_q_passing": fallback,
        "recommended_window": _fit_summary(best, fallback=fallback, index=best_index),
        "tuning_diagnostic_pdfs": {},
    }


# --- tool 4: apply one shared setting to all samples and z -------------------


def _fit_one_sample(
    spec: dict[str, Any],
    sample_index: int,
    prior: gv.BufferDict,
    p0: dict[str, float],
    *,
    strategy: str,
    fit_scope: str,
    pt2_samples: np.ndarray,
    pt2_gv: np.ndarray,
    pt2_f_samples: np.ndarray | None,
    pt2_f_gv: np.ndarray | None,
    samples_re: dict[int, np.ndarray],
    samples_im: dict[int, np.ndarray],
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
    tseps: list[int],
    Lt: int,
    nstate: int,
    part: str,
    svdcut: float,
    scale: float,
    fitting_form: str,
) -> tuple[lsf.nonlinear_fit, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Refit one resampled sample with the fixed shared window and tuned prior."""
    rre = {t: _recenter(samples_re[t][sample_index], ratio_re[t]) for t in tseps}
    rim = {t: _recenter(samples_im[t][sample_index], ratio_im[t]) for t in tseps}
    fh_re = fh_im = None
    if "FH" in fit_scope:
        fh_re, fh_im = _fh_samples_from_ratios(rre, rim, spec["tsep_ls"], spec["tau_cut"])
    if strategy == "joint":
        pt2_s = _recenter(pt2_samples[sample_index], pt2_gv)
        if fitting_form == "NonBreit":
            pt2_f_sample_data = pt2_f_samples if pt2_f_samples is not None else pt2_samples
            pt2_f_template = pt2_f_gv if pt2_f_gv is not None else pt2_gv
            pt2_f_s = _recenter(pt2_f_sample_data[sample_index], pt2_f_template)
            fit = fit_nonbreit_joint(
                pt2_s, pt2_f_s, spec["tmin"], spec["tmax"], rre, rim, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=prior, p0=p0,
            )
        else:
            if fit_scope == "FH":
                fit = fit_joint_fh(
                    pt2_s, spec["tmin"], spec["tmax"], fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"], Lt,
                    nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=prior, p0=p0,
                )
            elif fit_scope == "ratio+FH":
                fit = fit_joint_ratio_fh(
                    pt2_s, spec["tmin"], spec["tmax"], rre, rim, fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"], Lt,
                    nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=prior, p0=p0,
                )
            else:
                fit = fit_joint(
                    pt2_s, spec["tmin"], spec["tmax"], rre, rim, spec["tsep_ls"], spec["tau_cut"], Lt,
                    nstate=nstate, part=part, svdcut=svdcut, rescale=scale, prior=prior, p0=p0,
                )
    else:
        if fitting_form == "NonBreit":
            fit = fit_nonbreit_ratio(
                rre, rim, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, prior=prior, p0=p0,
            )
        elif fit_scope == "FH":
            fit = fit_fh(
                fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"],
                nstate=nstate, part=part, svdcut=svdcut, prior=prior, p0=p0,
            )
        elif fit_scope == "ratio+FH":
            fit = fit_ratio_fh(
                rre, rim, fh_re, fh_im, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, prior=prior, p0=p0,
            )
        else:
            fit = fit_ratio(
                rre, rim, spec["tsep_ls"], spec["tau_cut"], Lt,
                nstate=nstate, part=part, svdcut=svdcut, prior=prior, p0=p0,
            )
    return fit, rre, rim


def fit_bare_matrix_grid(
    store: dict[str, Any],
    *,
    pt2_path: str,
    pt2_out_path: str | None = None,
    pt3_paths: dict[str, str] | list[str],
    tsep_ls: list[int],
    z_values: list[int],
    ensemble: str,
    tag: str,
    momentum: str,
    momentum_out: str | None = None,
    pt3_momentum: str | None = None,
    fitting_form: str = "Breit",
    hadron: str | None = None,
    gfix: str | None = None,
    direction: str = "X",
    variant: str = "free",
    source_sink: str = "SS",
    pt2_gamma: str = "5",
    pt3_gamma: str = "T",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bt: str = "bT0",
    b_label: str = "b0",
    pt2_window: dict[str, int] | None = None,
    pt3_window: dict[str, Any] | None = None,
    pt3_tau_cut: int | None = None,
    pt2_windows: list[dict[str, int]] | None = None,
    pt3_windows: list[dict[str, Any]] | None = None,
    pt3_tau_cuts: list[int] | None = None,
    model_average: bool = False,
    tune_z: int | None = None,
    fit_strategy: str = "joint",
    fit_scope: str = "ratio",
    nstate: int | list[int] = 2,
    nstate_values: list[int] | None = None,
    prior_width: float | list[float] | None = None,
    resample_mode: str = "bs",
    n_boot: int = 200,
    seed: int | None = 1984,
    bin_size: int = 1,
    svdcut: float = 1e-2,
    part: str = "both",
    q_min: float = 0.05,
    posterior_prior_error_scale: float = 3.0,
    correlator_rescale: float = 1.0,
    job_id: str | None = None,
    a_fm: float | None = None,
    pz_gev: float | None = None,
    pz_out_gev: float | None = None,
    save_path: str | None = None,
    log_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    out: str = "bare_matrix_grid",
) -> dict[str, Any]:
    """Apply one shared window to all samples and z, then export bare matrix elements.

    Window/tau-cut choices are tuned once on sample-average data (for ``tune_z``)
    and used for every z and every resampled sample. ``model_average=True`` varies
    fit-function choices (nstate and prior_width) within that fixed data window.
    """
    del out
    form = _normalise_fitting_form(fitting_form)
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
    fit_mode = _fit_mode_label(strategy, scope_label)
    scale = _check_rescale(correlator_rescale)
    mode = _check_mode(resample_mode)
    fitted_parts = _parts(part)
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    fit_log_dir = Path(log_dir) if log_dir is not None else out_dir / "fit_logs"
    fit_log_dir.mkdir(parents=True, exist_ok=True)
    tuning_log_path, sample_log_path = _split_log_paths(
        log_dir=fit_log_dir, log_path=log_path, ensemble=ensemble, tag=tag, variant=variant,
        direction=direction, momentum=momentum, b_label=b_label, fit_mode=fit_mode,
    )
    tuning_logger = setup_logger(tuning_log_path, logger_name="correlator_tuning_logger")
    sample_logger = setup_logger(sample_log_path, logger_name="correlator_sample_logger")
    tuning_logger.info("Starting %s bare matrix grid fit (model_average=%s)", fit_mode, model_average)
    tuning_logger.info("ensemble=%s tag=%s momentum=%s direction=%s rescale=%s", ensemble, tag, momentum, direction, scale)

    tseps = [int(t) for t in tsep_ls]
    if pt3_window is None and pt3_tau_cut is not None:
        pt3_window = {"tsep_ls": tseps, "tau_cut": int(pt3_tau_cut)}
    z_list = [int(z) for z in z_values]
    paths_by_tsep = _normalise_pt3_paths(pt3_paths, tsep_ls=tseps)
    missing = [tsep for tsep in tseps if tsep not in paths_by_tsep]
    if missing:
        raise ValueError(f"pt3_paths missing tsep entries: {missing}")
    tune_z_value = z_list[0] if tune_z is None else int(tune_z)
    final_momentum = momentum if momentum_out is None else momentum_out
    three_point_momentum = momentum if pt3_momentum is None else pt3_momentum

    pt2_complex = _read_2pt(pt2_path, source_sink=source_sink, gamma=pt2_gamma, momentum=momentum)
    n_cfg, Lt = pt2_complex.shape
    pt2_samples, pt2_complex_samples, indices = _resample_pt2(pt2_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size)
    pt2_gv = samples_to_gvar(pt2_samples, mode=mode)
    pt2_f_samples = None
    pt2_f_gv = None
    pt2_f_complex_samples = pt2_complex_samples
    if form == "NonBreit":
        pt2_f_complex = _read_2pt(pt2_out_path or pt2_path, source_sink=source_sink, gamma=pt2_gamma, momentum=final_momentum)
        if pt2_f_complex.shape != pt2_complex.shape:
            raise ValueError(f"initial/final 2pt shape mismatch: {pt2_complex.shape} != {pt2_f_complex.shape}")
        pt2_f_samples, pt2_f_complex_samples, _ = _resample_pt2(
            pt2_f_complex, mode=mode, n_boot=n_boot, seed=seed, bin_size=bin_size, indices=indices
        )
        pt2_f_gv = samples_to_gvar(pt2_f_samples, mode=mode)
    n_samples = int(pt2_samples.shape[0])
    pt2_window_specs = _normalise_pt2_windows(pt2_windows, Lt=Lt)
    if pt2_window is not None:
        explicit_pt2_spec = {"tmin": int(pt2_window["tmin"]), "tmax": int(pt2_window["tmax"])}
        if explicit_pt2_spec not in pt2_window_specs:
            pt2_window_specs = [explicit_pt2_spec, *pt2_window_specs]
    pt3_window_specs = _normalise_pt3_windows(pt3_windows, tsep_ls=tseps, tau_cuts=pt3_tau_cuts)
    tuning_logger.info("Lt=%s n_cfg=%s mode=%s n_samples=%s", Lt, n_cfg, mode, n_samples)

    def read_ratios(z: int):
        samples_re: dict[int, np.ndarray] = {}
        samples_im: dict[int, np.ndarray] = {}
        gv_re: dict[int, np.ndarray] = {}
        gv_im: dict[int, np.ndarray] = {}
        for tsep in tseps:
            pt3 = _read_3pt(
                paths_by_tsep[tsep], source_sink=source_sink, gamma=pt3_gamma, momentum=three_point_momentum,
                b_dir=b_dir, eta=eta, bt=bt, bz=f"bz{z}", tsep=tsep,
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
            gv_re[tsep] = samples_to_gvar(samples_re[tsep], mode=mode)
            gv_im[tsep] = samples_to_gvar(samples_im[tsep], mode=mode)
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
            except Exception as exc:
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
                except Exception as exc:
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
                pt2_sample=pt2_sample0, rec=rec0, Lt=Lt, log_dir=fit_log_dir, momentum=momentum, fit_label="chained_fit"
            )
        except Exception as exc:
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

    for z in _progress(z_list, desc=f"fit bare matrix {ensemble} {momentum} {direction}"):
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
            else 0.0
        )
        imag_sys_sdev = (
            _weighted_model_sdev(avg_im_vals, avg_weights, center=avg_im_mean)
            if model_average and "im" in fitted_parts
            else 0.0
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
        for sample_index in range(n_samples):
            try:
                re_vals: list[float] = []
                im_vals: list[float] = []
                sample_records: list[dict[str, Any]] = []
                first_fit = None
                first_rre = first_rim = None
                first_meta = None
                for candidate_index, (rec, template, (prior, p0)) in enumerate(zip(avg_records, templates, priors)):
                    spec = {"tmin": rec["tmin"], "tmax": rec["tmax"], "tsep_ls": rec["tsep_ls"], "tau_cut": rec["tau_cut"]}
                    fit, rre, rim = _fit_one_sample(
                        spec, sample_index, prior, p0, nstate=int(rec["nstate"]), **common
                    )
                    usable, reason = _fit_usable(fit, template)
                    if not usable:
                        if not model_average:
                            raise ValueError(str(reason))
                        sample_logger.info(
                            "Rejected %s z=%s sample=%s nstate=%s prior_width=%s: %s",
                            fit_mode, z, sample_index, rec["nstate"], rec["prior_width"], reason,
                        )
                        continue
                    sample_record = _record(
                        fit,
                        candidate_index=candidate_index,
                        nstate=int(rec["nstate"]),
                        prior_width=float(rec["prior_width"]),
                        part=part,
                        fit_scope=scope,
                        correlator_rescale=scale,
                        **spec,
                    )
                    sample_records.append(sample_record)
                    log_nonlinear_fit_quality(
                        fit, kind=f"sample ground-state {fit_mode}",
                        label=(
                            f"z={z} sample={sample_index} t=[{spec['tmin']},{spec['tmax']}) "
                            f"tseps={spec['tsep_ls']} tau_cut={spec['tau_cut']} "
                            f"nstate={rec['nstate']} prior_width={rec['prior_width']}"
                        ),
                        logger=sample_logger, q_min=q_min,
                    )
                    re_vals.append(
                        _bare_matrix_element_mean_for_part(
                            fit.p,
                            output_part="re",
                            fit_part=part,
                            fitting_form=form,
                        )
                    )
                    im_vals.append(
                        _bare_matrix_element_mean_for_part(
                            fit.p,
                            output_part="im",
                            fit_part=part,
                            fitting_form=form,
                        )
                    )
                    if first_fit is None:
                        first_fit, first_rre, first_rim = fit, rre, rim
                        first_meta = {**spec, "nstate": int(rec["nstate"]), "prior_width": float(rec["prior_width"])}
                if not sample_records:
                    raise ValueError("all fit-function candidates failed")
                sample_weights = _loggbf_weights(sample_records) if model_average else np.ones(1, dtype=float)
                for weight, sample_record in zip(sample_weights, sample_records):
                    weight_sums[int(sample_record["candidate_index"])] += float(weight)
                weight_counts += 1
                real_samples[sample_index] = float(np.sum(sample_weights * np.array(re_vals)))
                imag_samples[sample_index] = float(np.sum(sample_weights * np.array(im_vals)))
                if sample_index == 0:
                    rec0 = _record(first_fit, **first_meta, part=part, fit_scope=scope, correlator_rescale=scale)
                    if scope != "FH":
                        sample0_paths.update(_plot_sample0_ratio(
                            ratio_re=first_rre, ratio_im=first_rim, rec=rec0, Lt=Lt, log_dir=fit_log_dir,
                            momentum=momentum, z=z, fit_label=f"{strategy}_{scope_label}_fit", fitting_form=form,
                            part=part,
                        ))
                    if "FH" in scope:
                        sample0_paths.update(_plot_sample0_fh(
                            ratio_re=first_rre, ratio_im=first_rim, rec=rec0, log_dir=fit_log_dir,
                            momentum=momentum, z=z, fit_label=f"{strategy}_{scope_label}_fit", part=part,
                        ))
            except Exception as exc:
                failures.append({"sample": sample_index, "error": str(exc)})
                sample_logger.info("Bad %s z=%s sample=%s: %s", fit_mode, z, sample_index, exc)

        if not np.any(np.isfinite(real_samples)):
            raise ValueError(f"all resampled fits failed for z={z}")
        real_mean, real_sdev = sample_mean_err(real_samples, mode=mode)
        imag_mean, imag_sdev = sample_mean_err(imag_samples, mode=mode)
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

    if form == "NonBreit":
        q2 = None if pz_gev is None or pz_out_gev is None else (float(pz_out_gev) - float(pz_gev)) ** 2
        xi = None if pz_gev is None or pz_out_gev is None else (float(pz_gev) - float(pz_out_gev)) / (float(pz_gev) + float(pz_out_gev))
        plot_title = rf"{ensemble} $Q^2={q2:g}\,\mathrm{{GeV}}^2$, $\xi={xi:g}$ {direction} bare matrix elements"
    else:
        p_label = "n/a" if pz_gev is None else f"{float(pz_gev):g}"
        plot_title = rf"{ensemble} $p={p_label}\,\mathrm{{GeV}}$ {direction} bare matrix elements"
    output = _write_outputs(
        z_records, artifacts_dir=out_dir, save_path=save_path, ensemble=ensemble, tag=tag, variant=variant,
        direction=direction, momentum=momentum, b_label=b_label, resample_mode=mode,
        matrix_element_label=(
            r"Bare matrix element $O_{00}/(E_{0}^{i}+E_{0}^{f})$"
            if form == "NonBreit"
            else r"Bare matrix element $O_{00}/(2E_0)$"
        ),
        plot_title=plot_title,
        part=part,
    )
    bare_data = output.pop("data")
    bare_data.array.attrs.update(
        {
            key: str(value)
            for key, value in {
                "job_id": job_id,
                "a_fm": a_fm,
                "pz_gev": pz_gev,
                "pz_out_gev": pz_out_gev,
                "fitting_form": form,
                "fit_scope": scope,
                "part": part,
                "nstate_values": json.dumps(fit_nstates),
                "prior_width": json.dumps(prior_widths),
                "momentum_out": final_momentum,
                "pt3_momentum": three_point_momentum,
                "hadron": hadron,
                "gfix": gfix,
            }.items()
            if value is not None
        }
    )
    bare_data.to_netcdf(output["netcdf_path"])
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
        "z_values": z_list,
        "tune_z": tune_z_value,
        "z_fits": z_report,
        "sample0_pt2_plot_paths": sample0_pt2_paths,
    }


STAGE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_correlator_scale": inspect_correlator_scale,
    "tune_ground_state": tune_ground_state,
    "tune_bare_matrix": tune_bare_matrix,
    "fit_bare_matrix_grid": fit_bare_matrix_grid,
}
