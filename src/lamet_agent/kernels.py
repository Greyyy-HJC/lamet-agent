"""Reference perturbative kernels for smoke tests and workflow examples."""

from __future__ import annotations

from typing import Final

import numpy as np


def identity_kernel(x: np.ndarray) -> np.ndarray:
    """Return identity mapping for kernel smoke tests."""
    # no-op kernel used in smoke tests: returns its input unchanged.
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
    # Return the beta-function coefficient for the given perturbative order.
    if order == 0:
        # LO: beta_0.
        return 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * Nf
    if order == 1:
        # NLO: beta_1.
        return 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * Nf
    if order == 2:
        # NNLO: beta_2.
        return (
            2857.0 / 54.0 * CA**3
            + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * Nf
            + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * Nf**2
        )

    raise NotImplementedError(f"beta coefficient at order={order} is not implemented.")


def alphas_nloop(mu: float, order: int = 0, Nf: int = 3) -> float:
    """Compute running coupling :math:`alpha_s(mu)` up to NNLO."""
    # Reference alpha_s at mu=2 GeV, converted to the alpha_s/(4*pi) form.
    a_s_ref = 0.293 / (4.0 * np.pi)
    # Build the common factor that recurs in the LO running.
    b0 = beta(0, Nf)
    temp = 1.0 + a_s_ref * b0 * np.log((mu / 2.0) ** 2)

    if order == 0:
        # LO running coupling.
        return a_s_ref * 4.0 * np.pi / temp
    if order == 1:
        # NLO running coupling: add the beta_1 correction on top of the LO factor.
        b1 = beta(1, Nf)
        return a_s_ref * 4.0 * np.pi / (temp + a_s_ref * b1 / b0 * np.log(temp))
    if order == 2:
        # NNLO running coupling: further add the beta_2 and beta_1^2 corrections.
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
    # 1. Coerce the input x/y grids to 1D float arrays; default y grid = x grid when y_ls is None.
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    # 2. Check grid dimensions and the y=0 singularity so xi=x/y stays finite.
    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep xi=x/y finite.")

    # 3. Compute the y grid spacing and require a uniformly spaced integration grid.
    y_step = np.diff(y_grid) #! step[i] = y_grid[i+1] - y_grid[i]
    dy = float(np.abs(y_step[0])) #! here we assume the step is the same for all i
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")

    # 4. Compute the NLO running coupling alpha_s at the renormalization scale mu.
    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    # 5. Initialize the identity matrix, the diagonal-row map, and the NLO correction matrix.
    nx = len(x_grid)
    ny = len(y_grid)
    identity = np.zeros((nx, ny))
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)
    nlo_matrix = np.zeros((nx, ny))

    # 6. Double loop over every matrix element C(x_i, y_j).
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            # 6.1 If x_i and y_j are the same grid point, put 1 in the LO identity kernel.
            if np.isclose(x_val, y_val, atol=eps, rtol=0.0):
                identity[idx, idy] = 1.0

            # 6.2 Compute the convolution variable xi=x/y; the xi=1 singularity is handled later by the plus prescription.
            xi = x_val / y_val
            one_minus_xi = 1.0 - xi
            if np.abs(one_minus_xi) <= eps:
                continue

            # 6.3 Prepare the |y| normalization and scale logarithm for this matrix element.
            y_norm = np.abs(y_val)
            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)
            entry = 0.0

            # 6.4 In the physical region 0<xi<1, add the splitting function times the scale logarithm.
            if eps < xi < 1.0 - eps:
                splitting = (1.0 + xi**2) / (1.0 - xi)
                entry += splitting * log_scale + xi - 1.0

            # 6.5 The closed form contains sqrt(1-2*xi). For xi>1/2 that root is imaginary,
            #     so write the equivalent real analytic continuation using arctanh.
            if xi < 0.5 - eps:
                sqrt_term = np.sqrt(1.0 - 2.0 * xi)
                atan_piece = (3.0 * xi - 1.0) / (xi - 1.0 + eps)
                atan_piece *= np.arctan(sqrt_term / (np.abs(xi) + eps)) / (sqrt_term + eps)
            elif xi > 0.5 + eps:
                sqrt_term = np.sqrt(2.0 * xi - 1.0)
                atan_piece = (3.0 * xi - 1.0) / (xi - 1.0 + eps)
                atan_piece *= np.arctanh(sqrt_term / (np.abs(xi) + eps)) / (sqrt_term + eps)
            else:
                atan_piece = (3.0 * xi - 1.0) / ((xi - 1.0) * np.abs(xi))

            # 6.6 Use signed logs and an eps-regulated denominator to handle the signs of xi and 1-xi stably.
            sign_safe_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
            signed_logs = (
                np.sign(xi) * np.log(np.abs(xi) + eps)
                + np.sign(one_minus_xi) * np.log(np.abs(one_minus_xi) + eps)
            )
            entry += (1.0 + xi**2) / sign_safe_denominator * signed_logs
            entry += np.sign(xi) + atan_piece - 1.0 / (np.abs(one_minus_xi) + eps)

            # 6.7 Divide by |y| to get this NLO element of the discrete convolution matrix.
            nlo_matrix[idx, idy] = entry / y_norm

    # 7. Apply the plus prescription per y_j column and add the finite delta(1-xi) term.
    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy]) #! plus function: sum over row gives zero
        nlo_matrix[int(diag_row), idy] += 0.5 * (
            1.0 + np.log(4.0 * y_grid[idy] ** 2 * pz_gev**2 / mu**2)
        ) / dy

    # 8. Assemble the final matching matrix: LO identity minus the NLO correction, times the discrete step dy.
    return identity - alpha_s * CF / (2.0 * np.pi) * nlo_matrix * dy


def helicity_matching_kernel_nlo_gTg5(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO helicity kernel for ``gamma^t gamma5`` in MSbar."""
    # The helicity kernel currently shares the same implementation as the unpolarized gamma^t kernel.
    return unpolarized_matching_kernel_nlo_gT(x_ls=x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps)


"""
Below is the NLO unpolarized GLUON matching kernel in MSbar, written in exactly
the same form as ``unpolarized_matching_kernel_nlo_gT`` above (same signature,
same plus-prescription, same direct LO - NLO return).
Usage: kernel_matrix = unpolarized_gluon_matching_kernel_nlo(...)
       lightcone = np.dot(kernel_matrix, quasi)
"""


def unpolarized_gluon_matching_kernel_nlo(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO unpolarized **gluon** kernel in MSbar.

    Gluon analogue of :func:`unpolarized_matching_kernel_nlo_gT`: same signature
    and same structure, but the matching is ``C_A``-proportional and uses the
    gluon splitting/coefficient functions. No Wilson-line (z_s) term -- this is
    the plain MSbar gluon kernel, not the hybrid/ratio scheme.
    """
    # 1. Coerce the input x/y grids to 1D float arrays; default y grid = x grid when y_ls is None.
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    # 2. Check grid dimensions and the y=0 singularity so xi=x/y stays finite.
    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep xi=x/y finite.")

    # 3. Compute the y grid spacing and require a uniformly spaced integration grid.
    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")

    # 4. Compute the NLO running coupling alpha_s at the renormalization scale mu.
    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    # 5. Initialize the identity matrix, the diagonal-row map, and the NLO correction matrix.
    nx = len(x_grid)
    ny = len(y_grid)
    identity = np.zeros((nx, ny))
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)
    nlo_matrix = np.zeros((nx, ny))

    # 6. Double loop over every matrix element C(x_i, y_j).
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            # 6.1 If x_i and y_j are the same grid point, put 1 in the LO identity kernel.
            if np.isclose(x_val, y_val, atol=eps, rtol=0.0):
                identity[idx, idy] = 1.0

            # 6.2 Compute the convolution variable xi=x/y; the xi=1 singularity is handled later by the plus prescription.
            xi = x_val / y_val
            one_minus_xi = 1.0 - xi
            if np.abs(one_minus_xi) <= eps:
                continue

            # 6.3 Prepare the |y| normalization and scale logarithm for this matrix element.
            y_norm = np.abs(y_val)
            # log(4 y^2 pz^2 / mu^2) == -log(r^2/4) with r = mu/(|y| pz)
            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)

            # 6.4 Precompute the polynomial parts that recur in the gluon kernel's piecewise formula.
            poly = 2.0 * (1.0 - xi + xi**2) ** 2 / (1.0 - xi)
            cubic = (11.0 - 28.0 * xi + 18.0 * xi**2 - 12.0 * xi**3) / (6.0 * (1.0 - xi))

            # 6.5 Fill in the analytic expression for the region xi falls in.
            entry = 0.0
            if xi > 1.0 + eps:
                entry = poly * np.log(xi / (xi - 1.0)) + cubic
            elif eps < xi < 1.0 - eps:
                entry = poly * (log_scale + np.log(xi * (1.0 - xi))) - (
                    15.0 - 56.0 * xi + 102.0 * xi**2 - 96.0 * xi**3 + 48.0 * xi**4
                ) / (6.0 * (1.0 - xi))
            elif xi < -eps:
                entry = -poly * np.log(xi / (xi - 1.0)) - cubic

            # 6.6 Divide by |y| to get this NLO element of the discrete convolution matrix.
            nlo_matrix[idx, idy] = entry / y_norm

    # 7. Apply the plus prescription per y_j column so each column's correction integrates to zero.
    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy]) #! plus function: column sum gives zero

    # 8. Assemble the final matching matrix: LO identity minus the NLO correction, times dy (same form as the PDF kernel).
    return identity - alpha_s * CA / (2.0 * np.pi) * nlo_matrix * dy
