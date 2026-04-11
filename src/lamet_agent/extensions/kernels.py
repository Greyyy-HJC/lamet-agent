"""Perturbative matching kernels for LaMET quasi distributions.

Implements the Collins-Soper kernel matching functions used in CG (coulomb-gauge)
qTMDWF analyses.  The formulae follow:

  * CG_tmd_kernel_RGR  — Eq.(D.2) of arXiv:1002.2213
  * CG_tmdwf_kernel_RGR — product of two CG_tmd_kernel_RGR factors (for the
    wave-function renormalisation structure of the quasi-TMDWF)

These functions depend only on NumPy and SciPy, so they are available
regardless of whether the optional ``gvar``/``lsqfit`` stack is installed.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# QCD constants
# ---------------------------------------------------------------------------
_CF: float = 4.0 / 3.0
_CA: float = 3.0
_TF: float = 0.5

# alphas(mu=2 GeV) / (4*pi) initial condition used in lametlat
_AS_REF: float = 0.293 / (4.0 * np.pi)


def _beta(order: int, Nf: int = 3) -> float:
    """QCD beta function coefficient b_order."""
    if order == 0:
        return 11.0 / 3.0 * _CA - 4.0 / 3.0 * _TF * Nf
    if order == 1:
        return 34.0 / 3.0 * _CA**2 - (20.0 / 3.0 * _CA + 4.0 * _CF) * _TF * Nf
    if order == 2:
        return (
            2857.0 / 54.0 * _CA**3
            + (2.0 * _CF**2 - 205.0 / 9.0 * _CF * _CA - 1415.0 / 27.0 * _CA**2) * _TF * Nf
            + (44.0 / 9.0 * _CF + 158.0 / 27.0 * _CA) * _TF**2 * Nf**2
        )
    raise ValueError(f"beta not implemented for order={order}")


def _cusp0() -> float:
    return 2.0 * _CF


def _cusp1(Nf: int = 3) -> float:
    return 2.0 * _CF * (
        (67.0 / 9.0 - np.pi**2 / 3.0) * _CA - 20.0 / 9.0 * _TF * Nf
    )


def _CG_gamma_c() -> float:
    """Non-cusp anomalous dimension for CG quark bilinear (Eq.A4 of 2504.04625)."""
    return -6.0 * _CF


def alphas_nloop(mu: float, order: int = 2, Nf: int = 3) -> float:
    """Running coupling alpha_s(mu) at n-loop order.

    Parameters
    ----------
    mu:
        Renormalisation scale in GeV.
    order:
        Loop order (0=1-loop, 1=2-loop, 2=3-loop).
    Nf:
        Number of active quark flavours.
    """
    b0 = _beta(0, Nf)
    b1 = _beta(1, Nf)
    aS = _AS_REF
    temp = 1.0 + aS * b0 * np.log((mu / 2.0) ** 2)

    if order == 0:
        return aS * 4.0 * np.pi / temp
    if order == 1:
        return aS * 4.0 * np.pi / (temp + aS * b1 / b0 * np.log(temp))
    if order == 2:
        b2 = _beta(2, Nf)
        return aS * 4.0 * np.pi / (
            temp
            + aS * b1 / b0 * np.log(temp)
            + aS**2 * (
                b2 / b0 * (1.0 - 1.0 / temp)
                + b1**2 / b0**2 * (np.log(temp) / temp + 1.0 / temp - 1.0)
            )
        )
    raise ValueError(f"alphas_nloop not implemented for order={order}")


def CG_tmd_kernel_RGR(
    x: float | np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    vary_eps: float = 1.0,
) -> float | np.ndarray:
    """CG TMD hard-matching kernel with RG resummation (NLL).

    Implements Appendix D.2 of arXiv:1002.2213 for the Coulomb-gauge
    quasi-TMDWF matching.

    Parameters
    ----------
    x:
        Parton momentum fraction.
    pz_gev:
        Hadron momentum P^z in GeV.
    mu:
        Matching scale in GeV (default 2 GeV).
    vary_eps:
        Multiplicative variation of the initial scale for scale-uncertainty
        estimates (default 1).
    """
    x = np.asarray(x, dtype=float)
    zeta = (2.0 * x * pz_gev * vary_eps) ** 2

    b0 = _beta(0)
    b1 = _beta(1)

    a0 = alphas_nloop(np.sqrt(zeta), order=1)  # 2-loop
    amu = alphas_nloop(mu, order=1)
    r = amu / a0

    # Cusp contribution (NLL)
    term1 = 4.0 * np.pi / a0 * (1.0 - 1.0 / r - np.log(r))
    term2 = (_cusp1() / _cusp0() - b1 / b0) * (1.0 - r + np.log(r))
    term3 = b1 / (2.0 * b0) * np.log(r) ** 2
    k_cusp = -_cusp0() / (4.0 * b0**2) * (term1 + term2 + term3)

    # Non-cusp gamma_C contribution (LL)
    a0_ll = alphas_nloop(np.sqrt(zeta), order=0)
    amu_ll = alphas_nloop(mu, order=0)
    r_ll = amu_ll / a0_ll
    k_gammac = -_CG_gamma_c() / (2.0 * b0) * np.log(r_ll)

    integral = -2.0 * k_cusp + k_gammac
    return np.exp(integral)


def CG_tmdwf_kernel_RGR(
    x: float | np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    vary_eps: float = 1.0,
) -> float | np.ndarray:
    """CG quasi-TMDWF hard-matching kernel (product of two CG_tmd_kernel_RGR).

    The quasi-TMDWF involves two quark-bilinear factors (one per quark leg),
    so the full hard kernel is the product::

        H(x, P^z) * H(1-x, P^z)

    Parameters
    ----------
    x, pz_gev, mu, vary_eps:
        See :func:`CG_tmd_kernel_RGR`.
    """
    return CG_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * CG_tmd_kernel_RGR(
        1.0 - x, pz_gev, mu, vary_eps
    )
