"""Two-state form-factor ratio / sum / joint fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from ..utils.logger import log_nonlinear_fit_quality

from .fit_funcs import ff_ratio_fcn, ff_sum_fcn, general_prior


def _ff_prior() -> gv.BufferDict:
    g = general_prior(nstate=2)
    pr = gv.BufferDict()
    pr["log(dE1)"] = g["log(dE1)"]
    pr["log(ff)"] = g["log(ff)"]
    pr["ff_excited_coeff"] = g["ff_excited_coeff"]
    pr["ff_den_exp_coeff"] = g["ff_den_exp_coeff"]
    return pr


def ff_ratio_two_state_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_by_tsep: Mapping[int, np.ndarray],
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
    data_scale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Fit ratio data R(t_sep, tau) with the two-state ansatz."""

    ts: list[int] = []
    taus: list[int] = []
    ys: list = []
    for tsep in tsep_ls:
        row = np.asarray(ratio_by_tsep[tsep], dtype=object)
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            ts.append(tsep)
            taus.append(tau)
            ys.append(row[tau] * data_scale)
    x_vecs = [np.array(ts, dtype=float), np.array(taus, dtype=float)]

    priors = _ff_prior() if prior is None else prior

    def fcn(x: list[np.ndarray], p: dict) -> np.ndarray:
        return ff_ratio_fcn((x[0], x[1]), p)

    fit_res = lsf.nonlinear_fit(
        data=(x_vecs, ys),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )
    log_nonlinear_fit_quality(fit_res, kind="ff ratio", label=label)
    return fit_res


def ff_sum_two_state_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    sum_by_tsep: Mapping[int, gv.GVar | float],
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
    data_scale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Fit tau-averaged sum points S(t_sep) with the two-state ansatz."""

    x = np.array(tsep_ls, dtype=float)
    y = np.array([sum_by_tsep[t] * data_scale for t in tsep_ls])
    priors = _ff_prior() if prior is None else prior

    def fcn(t: np.ndarray, p: dict) -> np.ndarray:
        return ff_sum_fcn(t, tau_cut, p)

    fit_res = lsf.nonlinear_fit(
        data=(x, y),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )
    log_nonlinear_fit_quality(fit_res, kind="ff sum", label=label)
    return fit_res


def ff_joint_two_state_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_by_tsep: Mapping[int, np.ndarray],
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
    data_scale: float = 1.0,
) -> lsf.nonlinear_fit:
    """Joint fit of ratio points and matching tau-averaged sums (same two-state ansatz)."""

    ts: list[int] = []
    taus: list[int] = []
    y_ratio: list = []
    for tsep in tsep_ls:
        row = np.asarray(ratio_by_tsep[tsep], dtype=object)
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            ts.append(tsep)
            taus.append(tau)
            y_ratio.append(row[tau] * data_scale)

    # Sum data = mean of ratio(r,tau) over tau in [tau_cut, tsep - tau_cut]; matches ff_sum_fcn.
    y_sum = []
    for tsep in tsep_ls:
        row = np.asarray(ratio_by_tsep[tsep], dtype=object)
        acc = row[tau_cut] * data_scale
        for tau in range(tau_cut + 1, tsep + 1 - tau_cut):
            acc = acc + row[tau] * data_scale
        y_sum.append(acc / (tsep - 2 * tau_cut + 1))

    x_ratio = [np.array(ts, dtype=float), np.array(taus, dtype=float)]
    priors = _ff_prior() if prior is None else prior
    x_dic = {"ratio": x_ratio, "sum": np.array(tsep_ls, dtype=float)}
    y_dic = {"ratio": y_ratio, "sum": y_sum}

    def fcn(x: dict, p: dict) -> dict:
        return {
            "ratio": ff_ratio_fcn((x["ratio"][0], x["ratio"][1]), p),
            "sum": ff_sum_fcn(x["sum"], tau_cut, p),
        }

    fit_res = lsf.nonlinear_fit(
        data=(x_dic, y_dic),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )
    log_nonlinear_fit_quality(fit_res, kind="ff joint", label=label)
    return fit_res
