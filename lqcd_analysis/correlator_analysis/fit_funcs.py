"""Prior settings and fit functions for ground-state fits."""

import numpy as np
import gvar as gv


def _validate_nstate(nstate: int) -> None:
    if isinstance(nstate, bool) or not isinstance(nstate, int) or nstate < 1:
        raise ValueError("nstate must be a positive integer")


def general_prior(nstate: int = 2) -> gv.BufferDict:
    """Return broad priors for an n-state ground-state fit."""
    _validate_nstate(nstate)

    priors = gv.BufferDict()

    priors["E0"] = gv.gvar(1, 10)
    for state in range(1, nstate):
        priors[f"log(dE{state})"] = gv.gvar(0, 10)

    for row in range(nstate):
        for col in range(row, nstate):
            priors[f"O{row}{col}_re"] = gv.gvar(1, 10)
            priors[f"O{row}{col}_im"] = gv.gvar(1, 10)

    for state in range(nstate):
        priors[f"z{state}"] = gv.gvar(1, 10)

    if nstate == 1:
        priors["sum_re_const"] = gv.gvar(0, 10)
        priors["sum_im_const"] = gv.gvar(0, 10)
    if nstate == 2:
        priors["sum_re_excited_coeff"] = gv.gvar(0, 10)
        priors["sum_re_offset"] = gv.gvar(0, 10)
        priors["sum_re_exp_offset"] = gv.gvar(0, 10)
        priors["sum_im_excited_coeff"] = gv.gvar(0, 10)
        priors["sum_im_offset"] = gv.gvar(0, 10)
        priors["sum_im_exp_offset"] = gv.gvar(0, 10)
        priors["sum_den_exp_coeff"] = gv.gvar(0, 10)
    if nstate >= 2:
        priors["log(ff)"] = gv.gvar(-2, 5)
        priors["ff_excited_coeff"] = gv.gvar(0, 10)
        priors["ff_den_exp_coeff"] = gv.gvar(0, 10)

    return priors


def pt2_re_fcn(
    pt2_t: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Compute the real part of the n-state two-point correlator."""
    _validate_nstate(nstate)

    val = 0.0
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]

        z = p[f"z{state}"]
        val = val + z**2 / (2 * energy) * (
            np.exp(-energy * pt2_t) + np.exp(-energy * (Lt - pt2_t))
        )

    return val


def pt3_ratio_re_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Compute the real part of the n-state 3pt/2pt ratio."""
    return _pt3_ratio_fcn(ra_t, ra_tau, p, Lt, nstate=nstate, part="re")


def pt3_ratio_im_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Compute the imaginary part of the n-state 3pt/2pt ratio."""
    return _pt3_ratio_fcn(ra_t, ra_tau, p, Lt, nstate=nstate, part="im")


def _pt3_ratio_fcn(
    ra_t: float | np.ndarray,
    ra_tau: float | np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int,
    part: str,
) -> float | np.ndarray:
    _validate_nstate(nstate)

    energies = []
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]
        energies.append(energy)

    numerator = 0.0
    for source_state, source_energy in enumerate(energies):
        for sink_state, sink_energy in enumerate(energies):
            row = min(source_state, sink_state)
            col = max(source_state, sink_state)
            matrix_element = p[f"O{row}{col}_{part}"]
            numerator = numerator + (
                matrix_element
                * p[f"z{source_state}"]
                * p[f"z{sink_state}"]
                * np.exp(-source_energy * (ra_t - ra_tau))
                * np.exp(-sink_energy * ra_tau)
                / (2 * source_energy)
                / (2 * sink_energy)
            )

    denominator = pt2_re_fcn(ra_t, p, Lt, nstate=nstate)
    return numerator / denominator


def sum_re_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    nstate: int = 2,
    dE1: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute the real summed-ratio fit function for one or two states."""
    return _sum_fcn(t, tau_cut, p, nstate=nstate, part="re", dE1=dE1)


def sum_im_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    nstate: int = 2,
    dE1: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute the imaginary summed-ratio fit function for one or two states."""
    return _sum_fcn(t, tau_cut, p, nstate=nstate, part="im", dE1=dE1)


def fh_re_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    nstate: int = 2,
    dt: int = 1,
) -> float | np.ndarray:
    """Compute the real FH fit function from neighboring summed ratios."""
    return _fh_fcn(t, tau_cut, p, nstate=nstate, part="re", dt=dt)


def fh_im_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    nstate: int = 2,
    dt: int = 1,
) -> float | np.ndarray:
    """Compute the imaginary FH fit function from neighboring summed ratios."""
    return _fh_fcn(t, tau_cut, p, nstate=nstate, part="im", dt=dt)


def _sum_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    *,
    nstate: int,
    part: str,
    dE1: float | np.ndarray | None,
) -> float | np.ndarray:
    _validate_nstate(nstate)
    if nstate > 2:
        raise ValueError("sum fit functions currently support nstate <= 2")

    e0 = p["E0"]
    if nstate == 1:
        return (
            p[f"O00_{part}"] * (t - 2 * tau_cut + 1) / (2 * e0)
            + p[f"sum_{part}_const"]
        )

    if dE1 is None:
        dE1 = p["dE1"]

    exp_term = np.exp(-dE1 * t)
    numerator = (
        p[f"O00_{part}"]
        * (t - 2 * tau_cut + 1)
        * (1 + p[f"sum_{part}_excited_coeff"] * exp_term)
        + p[f"sum_{part}_offset"]
        + p[f"sum_{part}_exp_offset"] * exp_term
    )
    denominator = 2 * e0 * (1 + p["sum_den_exp_coeff"] * exp_term)
    return numerator / denominator


def _fh_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
    *,
    nstate: int,
    part: str,
    dt: int,
) -> float | np.ndarray:
    _validate_nstate(nstate)
    if nstate > 2:
        raise ValueError("FH fit functions currently support nstate <= 2")

    if nstate == 1:
        return p[f"O00_{part}"] / (2 * p["E0"]) + t * 0

    term1 = _sum_fcn(t + dt, tau_cut, p, nstate=nstate, part=part, dE1=None)
    term2 = _sum_fcn(t, tau_cut, p, nstate=nstate, part=part, dE1=None)
    return (term1 - term2) / dt


def qda_re_fcn(
    qda_t: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Compute the real qDA/TMDWF fit function."""
    return _qda_fcn(qda_t, p, Lt, nstate=nstate, part="re")


def qda_im_fcn(
    qda_t: float | np.ndarray,
    p: dict,
    Lt: int,
    nstate: int = 2,
) -> float | np.ndarray:
    """Compute the imaginary qDA/TMDWF fit function."""
    return _qda_fcn(qda_t, p, Lt, nstate=nstate, part="im")


def _qda_fcn(
    qda_t: float | np.ndarray,
    p: dict,
    Lt: int,
    *,
    nstate: int,
    part: str,
) -> float | np.ndarray:
    _validate_nstate(nstate)

    val = 0.0
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]

        val = val + p[f"z{state}"] / (2 * energy) * p[f"O0{state}_{part}"] * (
            np.exp(-energy * qda_t) + np.exp(-energy * (Lt - qda_t))
        )

    return val


def ff_ratio_fcn(
    x: tuple[float | np.ndarray, float | np.ndarray],
    p: dict,
) -> float | np.ndarray:
    """Compute the two-state form-factor ratio fit function."""
    ra_t, ra_tau = x
    dE = p["dE1"] if "dE1" in p else np.exp(p["log(dE1)"])
    ff = p["ff"] if "ff" in p else np.exp(p["log(ff)"])

    return (
        -ff
        * (
            1
            + p["ff_excited_coeff"]
            * (np.exp(-dE * ra_tau) + np.exp(-dE * (ra_t - ra_tau)))
        )
        / (1 + p["ff_den_exp_coeff"] * np.exp(-dE * ra_t / 2))
    )


def _ff_sum_avg_at_t(t_val: float, tau_cut: int, p: dict) -> float | np.ndarray:
    width = t_val - 2 * tau_cut + 1
    if width <= 0:
        raise ValueError(
            "ff_sum_fcn requires t >= 2 * tau_cut - 1 for a non-empty tau sum"
        )
    taus = np.arange(tau_cut, int(t_val) + 1 - tau_cut, dtype=float)
    return np.sum(ff_ratio_fcn((t_val, taus), p)) / width


def ff_sum_fcn(
    t: float | np.ndarray,
    tau_cut: int,
    p: dict,
) -> float | np.ndarray:
    """Compute the tau-averaged two-state form-factor ratio fit function."""
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        return _ff_sum_avg_at_t(float(t_arr.item()), tau_cut, p)
    stacked = [_ff_sum_avg_at_t(float(tv), tau_cut, p) for tv in t_arr.ravel()]
    return np.asarray(stacked).reshape(t_arr.shape)

