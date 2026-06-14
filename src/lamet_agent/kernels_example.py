"""Reference perturbative kernels for smoke tests and workflow examples."""

from __future__ import annotations

from typing import Final

import numpy as np


def identity_kernel(x: np.ndarray) -> np.ndarray:
    """Return identity mapping for kernel smoke tests."""
    return x


"""
Below is an example of matching kernel for unpolarized PDF in Coulomb guage, reference https://arxiv.org/pdf/2602.11283
Usage: kernel_matrix = unpolarized_matching_kernel_nlo_gT(), lightcone = np.dot(kernel_matrix, quasi)
"""

# Conversion factor: 1 fm^{-1} = 0.1973269631 GeV
GEV_FM: Final[float] = 0.1973269631
CF: Final[float] = 4.0 / 3.0
NF: Final[int] = 3
CA: Final[float] = 3.0
TF: Final[float] = 1.0 / 2.0

def beta(order: int = 0, Nf: int = 3) -> float:
    """Return QCD beta-function coefficient at LO/NLO/NNLO."""
    if order == 0:
        return 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * Nf
    if order == 1:
        return 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * Nf
    if order == 2:
        return (
            2857.0 / 54.0 * CA**3
            + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * Nf
            + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * Nf**2
        )

    raise NotImplementedError(f"beta coefficient at order={order} is not implemented.")


def alphas_nloop(mu: float, order: int = 0, Nf: int = 3) -> float:
    """Compute running coupling :math:`alpha_s(mu)` up to NNLO."""
    a_s_ref = 0.293 / (4.0 * np.pi)
    b0 = beta(0, Nf)
    temp = 1.0 + a_s_ref * b0 * np.log((mu / 2.0) ** 2)

    if order == 0:
        return a_s_ref * 4.0 * np.pi / temp
    if order == 1:
        b1 = beta(1, Nf)
        return a_s_ref * 4.0 * np.pi / (temp + a_s_ref * b1 / b0 * np.log(temp))
    if order == 2:
        b1 = beta(1, Nf)
        b2 = beta(2, Nf)
        correction = (
            temp
            + a_s_ref * b1 / b0 * np.log(temp)
            + a_s_ref**2
            * (b2 / b0 * (1.0 - 1.0 / temp) + b1**2 / b0**2 * (np.log(temp) / temp + 1.0 / temp - 1.0))
        )
        return a_s_ref * 4.0 * np.pi / correction

    raise NotImplementedError(f"alpha_s at order={order} is not implemented.")


def unpolarized_matching_kernel_nlo_gT(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO unpolarized kernel for ``gamma^t`` in MSbar.

    Eq. (2.14) of arXiv:2602.11283.
    """
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep xi=x/y finite.")

    y_step = np.diff(y_grid) #! step[i] = y_grid[i+1] - y_grid[i]
    dy = float(np.abs(y_step[0])) #! here we assume the step is the same for all i
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")
    
    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    nx = len(x_grid)
    ny = len(y_grid)
    identity = np.zeros((nx, ny))
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)
    nlo_matrix = np.zeros((nx, ny))

    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            if np.isclose(x_val, y_val, atol=eps, rtol=0.0):
                identity[idx, idy] = 1.0

            xi = x_val / y_val
            one_minus_xi = 1.0 - xi
            if np.abs(one_minus_xi) <= eps:
                continue

            y_norm = np.abs(y_val)
            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)
            entry = 0.0

            if eps < xi < 1.0 - eps:
                splitting = (1.0 + xi**2) / (1.0 - xi)
                entry += splitting * log_scale + xi - 1.0

            if xi < 0.5 - eps:
                sqrt_term = np.sqrt(1.0 - 2.0 * xi)
                atan_piece = (3.0 * xi - 1.0) / (xi - 1.0 + eps)
                atan_piece *= np.arctan(sqrt_term / (np.abs(xi) + eps)) / (sqrt_term + eps)
            elif xi > 0.5 + eps:
                sqrt_term = np.sqrt(2.0 * xi - 1.0)
                atan_piece = (3.0 * xi - 1.0) / (xi - 1.0 + eps)
                atan_piece *= np.arctanh(sqrt_term / (np.abs(xi) + eps)) / (sqrt_term + eps)
            else:
                atan_piece = (3.0 * xi - 1.0) / (xi - 1.0)

            sign_safe_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
            signed_logs = (
                np.sign(xi) * np.log(np.abs(xi) + eps)
                + np.sign(one_minus_xi) * np.log(np.abs(one_minus_xi) + eps)
            )
            entry += (1.0 + xi**2) / sign_safe_denominator * signed_logs
            entry += np.sign(xi) + atan_piece - 1.0 / (np.abs(one_minus_xi) + eps)

            nlo_matrix[idx, idy] = entry / y_norm

    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy]) #! plus function: sum over row gives zero
        nlo_matrix[int(diag_row), idy] += 0.5 * (
            1.0 + np.log(4.0 * y_grid[idy] ** 2 * pz_gev**2 / mu**2)
        ) / dy

    return identity - alpha_s * CF / (2.0 * np.pi) * nlo_matrix * dy


def helicity_matching_kernel_nlo_gTg5(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO helicity kernel for ``gamma^t gamma5`` in MSbar."""
    return unpolarized_matching_kernel_nlo_gT(x_ls=x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps)
