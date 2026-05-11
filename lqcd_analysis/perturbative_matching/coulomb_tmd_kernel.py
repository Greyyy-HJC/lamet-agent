from typing import Union

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# QCD constants
# ---------------------------------------------------------------------------
C_F: float = 4.0 / 3.0
C_A: float = 3.0
T_F: float = 0.5

# alphas(mu=2 GeV) / (4*pi) initial condition used in lametlat
alpha_s_2gev: float = 0.293 / (4.0 * np.pi)


def _beta_0(n_f: int) -> float:
    """arXiv:1002.2213 Eq.(D.15)"""
    return 11.0 / 3.0 * C_A - 4.0 / 3.0 * T_F * n_f


def _beta_1(n_f: int) -> float:
    """arXiv:1002.2213 Eq.(D.15)"""
    return 34.0 / 3.0 * C_A**2 - (20.0 / 3.0 * C_A + 4.0 * C_F) * T_F * n_f


def _beta_2(n_f: int) -> float:
    """arXiv:1002.2213 Eq.(D.15)"""
    return (
        2857.0 / 54.0 * C_A**3
        + (2.0 * C_F**2 - 205.0 / 9.0 * C_F * C_A - 1415.0 / 27.0 * C_A**2) * T_F * n_f
        + (44.0 / 9.0 * C_F + 158.0 / 27.0 * C_A) * T_F**2 * n_f**2
    )


def _Gamma_0() -> float:
    return 2.0 * C_F


def _Gamma_1(n_f: int = 3) -> float:
    return 2.0 * C_F * ((67.0 / 9.0 - np.pi**2 / 3.0) * C_A - 20.0 / 9.0 * T_F * n_f)


def _gamma_0() -> float:
    """Non-cusp anomalous dimension for CG quark bilinear (Eq.A4 of 2504.04625)."""
    return -6.0 * C_F


def _alpha_s_0(mu: NDArray[np.float64], n_f: int = 3) -> NDArray[np.float64]:
    """1-loop running coupling for the LL contribution."""
    beta_0 = _beta_0(n_f)
    X = 1.0 + alpha_s_2gev * beta_0 * 2 * np.log(mu / 2.0)
    return alpha_s_2gev / X * (4.0 * np.pi)


def _alpha_s_1(mu: NDArray[np.float64], n_f: int = 3) -> NDArray[np.float64]:
    """2-loop running coupling for the NLL contribution."""
    beta_0 = _beta_0(n_f)
    beta_1 = _beta_1(n_f)
    X = 1.0 + alpha_s_2gev * beta_0 * 2 * np.log(mu / 2.0)
    return alpha_s_2gev / (X + alpha_s_2gev * beta_1 / beta_0 * np.log(X)) * (4.0 * np.pi)


def _alpha_s_2(mu: NDArray[np.float64], n_f: int = 3) -> NDArray[np.float64]:
    """3-loop running coupling for the NNLL contribution."""
    beta_0 = _beta_0(n_f)
    beta_1 = _beta_1(n_f)
    beta_2 = _beta_2(n_f)
    X = 1.0 + alpha_s_2gev * beta_0 * 2 * np.log(mu / 2.0)
    return (
        alpha_s_2gev
        / (
            X
            + alpha_s_2gev * beta_1 / beta_0 * np.log(X)
            + alpha_s_2gev**2
            * (beta_2 / beta_0 * (1.0 - 1.0 / X) + beta_1**2 / beta_0**2 * (np.log(X) / X + 1.0 / X - 1.0))
        )
        * (4.0 * np.pi)
    )


def coulomb_tmd_kernel_nlo(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0
) -> Union[float, np.ndarray]:
    """CG TMD hard-matching kernel (NLO expansion in alpha_s).

    arXiv:2311.01391; ``pz_gev`` is longitudinal momentum in GeV.

    Parameters
    ----------
    x:
        Parton momentum fraction.
    pz_gev:
        Hadron momentum P^z in GeV.
    mu:
        Matching scale in GeV (default 2 GeV).
    """
    x = np.asarray(x, dtype=float)
    zeta_scale = (2.0 * x * pz_gev) ** 2

    log_mu2_over_zeta = np.log((mu**2) / zeta_scale)
    temp = (
        0.5 * log_mu2_over_zeta**2
        + 3.0 * log_mu2_over_zeta
        + 12.0
        - np.pi**2 * 7.0 / 12.0
    )

    alpha_s_mu = _alpha_s_1(np.asarray(mu))
    h = -C_F * alpha_s_mu / (4.0 * np.pi) * temp
    return 1.0 + h


def coulomb_tmdwf_kernel_nlo(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0
) -> Union[float, np.ndarray]:
    """CG quasi-TMDWF NLO kernel (product of two single-leg kernels).

    Parameters
    ----------
    x, pz_gev, mu:
        See :func:`coulomb_tmd_kernel_nlo`.
    """
    return coulomb_tmd_kernel_nlo(x, pz_gev, mu) * coulomb_tmd_kernel_nlo(1.0 - x, pz_gev, mu)


def coulomb_tmdpdf_kernel_nlo(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0
) -> Union[float, np.ndarray]:
    """CG quasi-TMDPDF hard-matching kernel at NLO (PDF normalization).

    Same Sudakov-like polynomial as :func:`coulomb_tmd_kernel_nlo` with the PDF
    coupling prefactor (alpha_s / (2 pi) instead of alpha_s / (4 pi)).

    arXiv:2311.01391; ``pz_gev`` in GeV.

    Parameters
    ----------
    x:
        Parton momentum fraction.
    pz_gev:
        Hadron momentum P^z in GeV.
    mu:
        Matching scale in GeV (default 2 GeV).
    """
    x = np.asarray(x, dtype=float)
    zeta_scale = (2.0 * x * pz_gev) ** 2

    log_mu2_over_zeta = np.log((mu**2) / zeta_scale)
    temp = (
        0.5 * log_mu2_over_zeta**2
        + 3.0 * log_mu2_over_zeta
        + 12.0
        - np.pi**2 * 7.0 / 12.0
    )

    alpha_s_mu = _alpha_s_1(np.asarray(mu))
    h = -C_F * alpha_s_mu / (2.0 * np.pi) * temp
    return 1.0 + h


def coulomb_tmd_kernel_rg_nll(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0
) -> Union[float, np.ndarray]:
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

    beta_0 = _beta_0(3)
    beta_1 = _beta_1(3)

    alpha_s_mu_0 = _alpha_s_1(np.sqrt(zeta))  # 2-loop
    alpha_s_mu = _alpha_s_1(np.asarray(mu))
    r = alpha_s_mu / alpha_s_mu_0

    # Cusp contribution (NLL)
    term1 = 4.0 * np.pi / alpha_s_mu_0 * (1.0 - 1.0 / r - np.log(r))
    term2 = (_Gamma_1() / _Gamma_0() - beta_1 / beta_0) * (1.0 - r + np.log(r))
    term3 = beta_1 / (2.0 * beta_0) * np.log(r) ** 2
    K_Gamma = -_Gamma_0() / (4.0 * beta_0**2) * (term1 + term2 + term3)

    # Non-cusp gamma_C contribution (LL)
    alpha_s_mu_0_ll = _alpha_s_0(np.sqrt(zeta))  # 1-loop
    alpha_s_mu_ll = _alpha_s_0(np.asarray(mu))
    r_ll = alpha_s_mu_ll / alpha_s_mu_0_ll
    K_gamma = -_gamma_0() / (2.0 * beta_0) * np.log(r_ll)

    integral = -2.0 * K_Gamma + K_gamma
    return np.exp(integral)


def coulomb_tmdwf_kernel_rg_nll(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0
) -> Union[float, np.ndarray]:
    """CG quasi-TMDWF hard-matching kernel (product of two single-leg kernels).

    The quasi-TMDWF involves two quark-bilinear factors (one per quark leg),
    so the full hard kernel is the product:

        H(x, P^z) * H(1-x, P^z)

    Parameters
    ----------
    x, pz_gev, mu, vary_eps:
        See :func:`coulomb_tmd_kernel_rg_nll`.
    """
    return coulomb_tmd_kernel_rg_nll(x, pz_gev, mu, vary_eps) * coulomb_tmd_kernel_rg_nll(
        1.0 - x, pz_gev, mu, vary_eps
    )


def coulomb_tmdpdf_kernel_rg_nll(
    x: Union[float, np.ndarray], pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0
) -> Union[float, np.ndarray]:
    """CG quasi-TMDPDF hard-matching kernel (product of two single-leg kernels).

    The quasi-TMDPDF involves two quark-bilinear factors (one per quark leg),
    so the full hard kernel is the product:

        H(x, P^z) * H(x, P^z)

    Parameters
    ----------
    x, pz_gev, mu, vary_eps:
        See :func:`coulomb_tmd_kernel_rg_nll`.
    """
    return coulomb_tmd_kernel_rg_nll(x, pz_gev, mu, vary_eps) * coulomb_tmd_kernel_rg_nll(
        x, pz_gev, mu, vary_eps
    )
