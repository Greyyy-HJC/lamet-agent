"""Two-point correlator ground-state fits."""

from __future__ import annotations

import gvar as gv
import lsqfit as lsf
import numpy as np

from ..utils.logger import log_nonlinear_fit_quality

from .fit_funcs import general_prior, pt2_re_fcn


def _pt2_prior(nstate: int) -> gv.BufferDict:
    prior = general_prior(nstate=nstate)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = prior["E0"]
    for state in range(1, nstate):
        pt2_prior[f"log(dE{state})"] = prior[f"log(dE{state})"]
    for state in range(nstate):
        pt2_prior[f"z{state}"] = prior[f"z{state}"]
    return pt2_prior


def pt2_fit(
    pt2_avg: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with an n-state spectral decomposition."""

    pt2_avg = np.asarray(pt2_avg)
    if pt2_avg.ndim != 1:
        raise ValueError(f"pt2_avg must be one-dimensional, got shape {pt2_avg.shape}")

    fit_t = np.arange(tmin, tmax, dtype=int)
    if fit_t.size == 0:
        raise ValueError("fit window must contain at least one slice (require tmax > tmin)")
    if np.any(fit_t < 0) or np.any(fit_t >= len(pt2_avg)):
        raise ValueError(
            f"fit window [{tmin}, {tmax}) must lie within data length {len(pt2_avg)}"
        )

    priors = _pt2_prior(nstate=nstate) if prior is None else prior
    fit_pt2 = pt2_avg[fit_t]

    def fcn(t: np.ndarray, p: dict) -> np.ndarray:
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

    fit_res = lsf.nonlinear_fit(
        data=(fit_t, fit_pt2),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )

    log_nonlinear_fit_quality(fit_res, kind="2pt", label=label)

    return fit_res


def pt2_two_state_fit(
    pt2_avg: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with the default two-state model."""

    return pt2_fit(
        pt2_avg,
        tmin,
        tmax,
        Lt,
        nstate=2,
        prior=prior,
        label=label,
        maxit=maxit,
    )
