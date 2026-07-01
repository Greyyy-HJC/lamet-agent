"""NLO Coulomb-gauge matching kernels (arXiv:2602.11283).

Written in the same spirit as a hand script: a few explicit *coefficient
functions* ``C_<scheme>(ksi, ...)`` carry the closed-form physics, and a single
*discretization* ``build_matching_matrix`` turns any of them into the matching
matrix. ``x`` and ``y`` stay independent open grids (lists/arrays):

    lightcone(x) = sum_y  K(x, y) * quasi(y)          # K = kernel @ quasi

with ``ksi = x / y`` and the lamet log scale ``L = log(4 y^2 P_z^2 / mu^2)``.

Three operator classes are implemented (arXiv:2602.11283, Eqs. 2.14-2.21), each in
the ratio / msbar / hybrid scheme via ``CG_<operator>_PDF_<scheme>``:

    gt    / gtg5    gamma^t / gamma^t gamma5  (unpolarized / helicity, time comp.)
    gz    / gzg5    gamma^z / gamma^z gamma5  (unpolarized / helicity, z comp.)
    gtgpg5          gamma^t gamma_perp gamma5 (transversity)

Scheme structure straight from the paper:
  * gt/gtg5: ratio = C_ratio (2.16); msbar = +1/(2|1-ksi|) + diagonal (2.14);
    hybrid = +Wilson-line Si term (2.19-2.20).
  * gz/gzg5: ratio and hybrid are identical to gt (2.16, 2.20); only msbar differs,
    by +2(1-ksi)_+ + delta(1-ksi) (2.15).
  * gtgpg5: ratio = msbar = hybrid = C_ratio_perp (2.17, 2.18, 2.21) -- no scheme
    dependence at NLO.

Each public kernel is a one-line wrapper picking a coefficient function; the
discretization (loop + plus prescription + LO delta) is shared.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np


# --- physical constants & running coupling ----------------------------------

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


def _sine_integral(value: float) -> float:
    """Return Si(value), using scipy when available and a local fallback otherwise."""
    try:
        from scipy.special import sici

        return float(sici(value)[0])
    except ModuleNotFoundError:
        pass

    if np.isclose(value, 0.0, atol=1e-14, rtol=0.0):
        return 0.0

    sign = 1.0 if value > 0.0 else -1.0
    upper = abs(value)
    n_steps = max(256, int(128 * upper))
    if n_steps % 2:
        n_steps += 1

    grid = np.linspace(0.0, upper, n_steps + 1)
    integrand = np.ones_like(grid)
    integrand[1:] = np.sin(grid[1:]) / grid[1:]
    h = upper / n_steps
    integral = h / 3.0 * (
        integrand[0]
        + integrand[-1]
        + 4.0 * np.sum(integrand[1:-1:2])
        + 2.0 * np.sum(integrand[2:-2:2])
    )
    return sign * float(integral)


# --- coefficient functions C_<scheme>(ksi, ...) -----------------------------
# Each returns the bare regular coefficient C^(1) for one (x, y) pair. They know
# nothing about grids, the plus prescription or alpha_s -- that is the job of
# build_matching_matrix below. ``ksi = x / y`` and ``log_scale = log(4 y^2 P_z^2 / mu^2)``.


def _atan_piece(ksi: float, eps: float) -> float:
    """The (3ksi-1)/(ksi-1) * arctan/arctanh term shared by C_ratio and C_ratio_perp.

    Identical in Eq. (2.16) and Eq. (2.18); the branch is chosen by where ksi sits
    relative to 1/2 (analytic across ksi = 1/2 despite the apparent square roots).
    """
    if ksi < 0.5 - eps:
        sqrt_term = np.sqrt(1.0 - 2.0 * ksi)
        piece = (3.0 * ksi - 1.0) / (ksi - 1.0 + eps)
        return piece * np.arctan(sqrt_term / (np.abs(ksi) + eps)) / (sqrt_term + eps)
    if ksi > 0.5 + eps:
        sqrt_term = np.sqrt(2.0 * ksi - 1.0)
        piece = (3.0 * ksi - 1.0) / (ksi - 1.0 + eps)
        return piece * np.arctanh(sqrt_term / (np.abs(ksi) + eps)) / (sqrt_term + eps)
    # ksi = 1/2: analytic limit, since arctan(sqrt(1-2ksi)/|ksi|)/sqrt(1-2ksi) -> 1/|ksi|.
    return (3.0 * ksi - 1.0) / (ksi - 1.0) / (np.abs(ksi) + eps)


def C_ratio(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Ratio-scheme regular coefficient C_r^(1)(ksi), Eq. (2.16).

    Backbone of the unpolarized/helicity (gamma^t, gamma^z) schemes; MSbar and
    hybrid add a finite correction on top of it.
    """
    one_minus_ksi = 1.0 - ksi
    entry = 0.0

    # Splitting-function piece, only inside the physical 0 < ksi < 1 window.
    if eps < ksi < 1.0 - eps:
        entry += (1.0 + ksi**2) / one_minus_ksi * log_scale + ksi - 1.0

    # Logarithmic + remaining regular terms. The trailing -1.5/|1-ksi| is the bare
    # ratio coefficient; MSbar/hybrid shift it via their own corrections.
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    signed_logs = (
        np.sign(ksi) * np.log(np.abs(ksi) + eps)
        + np.sign(one_minus_ksi) * np.log(np.abs(one_minus_ksi) + eps)
    )
    entry += (1.0 + ksi**2) / sign_safe_denominator * signed_logs
    entry += np.sign(ksi) + _atan_piece(ksi, eps) - 1.5 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def C_ratio_perp(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Transversity ratio coefficient C_r^perp(1)(ksi), Eq. (2.18).

    Same shape as C_ratio but with the transversity splitting 2 ksi/(1-ksi), no
    ``+ksi-1`` / ``+sgn(ksi)`` terms, and a ``-1/|1-ksi|`` tail (vs ``-1.5/|1-ksi|``).
    For the transversity operator MSbar = ratio = hybrid all equal this (Eqs 2.17, 2.21).
    """
    one_minus_ksi = 1.0 - ksi
    entry = 0.0

    if eps < ksi < 1.0 - eps:
        entry += 2.0 * ksi / one_minus_ksi * log_scale

    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    signed_logs = (
        np.sign(ksi) * np.log(np.abs(ksi) + eps)
        + np.sign(one_minus_ksi) * np.log(np.abs(one_minus_ksi) + eps)
    )
    entry += 2.0 * ksi / sign_safe_denominator * signed_logs
    entry += _atan_piece(ksi, eps) - 1.0 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def C_msbar(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """MSbar off-diagonal coefficient: C_ratio + 0.5/|1-ksi|, Eq. (2.14).

    The finite *diagonal* conversion term (``0.5(1 + log_scale)``) is not part of
    the per-element coefficient; it is added on the plus-prescription row inside
    build_matching_matrix.
    """
    return C_ratio(ksi, log_scale, eps) + 0.5 / (np.abs(1.0 - ksi) + eps)


def C_msbar_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """gamma^z MSbar off-diagonal coefficient, Eq. (2.15).

    ``C_msbar^{gamma^z} = C_msbar^{gamma^t} + 2(1-ksi)`` on 0 < ksi < 1 (plus a
    ``delta(1-ksi)`` term carried on the diagonal -- see CG_gz_PDF_msbar). The
    ``2(1-ksi)`` is plus-prescribed at ksi = 1 by the shared discretization.
    """
    entry = C_msbar(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_hybrid(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    """Hybrid coefficient: C_ratio + Wilson-line Si correction, Eq. (2.19)-(2.20).

    ``zspz = z_s * P_z`` is the dimensionless Wilson-line length (constant in y).
    The parton momentum is ``y P_z``, so the per-y Wilson-line scale that enters
    the sine integral is ``z_s y P_z = |y| * zspz`` -- the ``|y|`` factor is what
    makes the correction y-dependent. The term replaces the MSbar ``0.5/|1-ksi|``.
    """
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    wilson_scale = np.abs(y) * zspz  # z_s * |y| * P_z
    delta = 0.5 * (
        1.0 / (np.abs(one_minus_ksi) + eps)
        - 2.0 * _sine_integral(one_minus_ksi * wilson_scale) / (np.pi * sign_safe_denominator)
    )
    return C_ratio(ksi, log_scale, eps) + delta


# --- the unified discretization ---------------------------------------------
# A coefficient function has the signature ``coeff(ksi, log_scale, y) -> float``.
# ``y`` is passed so y-dependent scales (the hybrid Wilson-line scale z_s y P_z)
# can be formed; schemes that do not need it simply ignore the argument.
CoeffFn = Callable[[float, float, float], float]


def _lo_interp_matrix(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """LO delta(x - y) as the matrix form of the examples' ``np.interp`` grid move.

    Built column by column straight from ``np.interp`` (each y basis vector), so
    ``(matrix @ q)[i] == np.interp(x_grid[i], y_grid, q, left=0, right=0)`` by
    construction. Equals the identity when the grids coincide, and keeps the LO term
    alive (instead of dropping to all-NLO) when ``x_ls`` and ``y_ls`` are staggered.
    """
    order = np.argsort(y_grid)  # np.interp needs an increasing y grid
    ys = y_grid[order]
    lo_sorted = np.column_stack(
        [np.interp(x_grid, ys, unit, left=0.0, right=0.0) for unit in np.eye(len(y_grid))]
    )
    lo = np.empty_like(lo_sorted)
    lo[:, order] = lo_sorted  # undo the sort so columns line up with y_grid
    return lo


def build_matching_matrix(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float,
    y_ls: np.ndarray | None,
    eps: float,
    *,
    coeff: CoeffFn,
    color_factor: float = CF,
    diagonal_extra: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Discretize a coefficient function ``coeff(ksi, log_scale)`` into an (nx, ny) matrix.

    ``x_ls`` and ``y_ls`` are independent open grids (``y_ls`` defaults to ``x_ls``).
    The loop fills the off-diagonal (ksi != 1) entries from ``coeff``; the LO
    delta(x - y) is a linear-interpolation stencil from the y grid onto each x (the
    same ``np.interp`` the examples use to move a curve between grids), so it survives
    when the grids are staggered and collapses to the identity when they coincide; the
    plus-prescription makes every y column integrate to zero and restores the
    ksi = 1 singularity; ``diagonal_extra(log_scale)`` (MSbar only) adds the finite
    diagonal conversion term. Returns ``identity - alpha_s C_x/(2 pi) * matrix * dy``.
    """
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep ksi=x/y finite.")

    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))  # uniform integration measure
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")

    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    nx, ny = len(x_grid), len(y_grid)
    nlo_matrix = np.zeros((nx, ny))
    # LO delta(x - y): a linear-interpolation stencil from the y grid onto each x,
    # the same np.interp(..., left=0, right=0) trick the examples use to move a curve
    # between grids. Collapses to the identity when the grids coincide.
    identity = _lo_interp_matrix(x_grid, y_grid)
    # For each y column, the x row closest to that y point carries the plus-function.
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)

    # 1) Off-diagonal (ksi != 1) regular coefficients from the coeff function.
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            ksi = x_val / y_val
            if np.abs(1.0 - ksi) <= eps:
                continue  # the ksi = 1 singularity is restored by the plus prescription

            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)
            nlo_matrix[idx, idy] = coeff(ksi, log_scale, y_val) / np.abs(y_val)

    # 2) Plus-prescription: make every y column integrate to zero, then add the
    #    optional finite scheme-conversion term on that column's nearest x row.
    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy])
        if diagonal_extra is not None:
            log_scale = np.log(4.0 * y_grid[idy] ** 2 * pz_gev**2 / mu**2)
            nlo_matrix[int(diag_row), idy] += diagonal_extra(log_scale) / dy

    # 3) Assemble: LO identity minus the NLO correction (times the dy measure).
    return identity - alpha_s * color_factor / (2.0 * np.pi) * nlo_matrix * dy


# --- public quark kernels: CG_<operator>_PDF_<scheme> ------------------------
# Each is one line: pick a coefficient function (and, for MSbar, the diagonal
# conversion term) and hand it to build_matching_matrix.


def CG_gt_PDF_ratio(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO ratio-scheme kernel ``C_r`` for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.16)."""
    del zspz  # ratio scheme has no Wilson-line scale; kept for a uniform signature.
    return build_matching_matrix(
        x_ls, pz_gev, mu, y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_ratio(ksi, log_scale, eps),
    )


def CG_gt_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar kernel for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.14)."""
    del zspz  # MSbar has no Wilson-line scale; kept for a uniform signature.
    return build_matching_matrix(
        x_ls, pz_gev, mu, y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_msbar(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale),
    )


def CG_gt_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme kernel for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.19-2.20).

    ``zspz = z_s * P_z`` (the dimensionless Wilson-line length) is required.
    """
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix(
        x_ls, pz_gev, mu, y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_hybrid(ksi, log_scale, y, z, eps),
    )


# --- helicity gamma^t gamma5 PDF --------------------------------------------
# The helicity kernels share the unpolarized gamma^t structure, so each scheme
# simply delegates to the corresponding CG_gt_PDF_<scheme> builder above.


def CG_gtg5_PDF_ratio(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_PDF_ratio(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gtg5_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_PDF_msbar(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gtg5_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_PDF_hybrid(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


# --- gamma^z / gamma^z gamma5 PDF -------------------------------------------
# Eq. (2.15): only the MSbar scheme differs from gamma^t (by 2(1-ksi)_+ + delta).
# In the ratio and hybrid schemes gamma^z shares gamma^t's coefficient
# (C_r in Eq. 2.16; delta C_hyb in Eq. 2.20 is identical for gamma^t and gamma^z),
# so those two delegate to the gamma^t builders.


def CG_gz_PDF_ratio(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.16; = gamma^t)."""
    return CG_gt_PDF_ratio(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gz_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.15).

    ``= gamma^t MSbar + 2(1-ksi)_+ + delta(1-ksi)``: the off-diagonal carries the
    extra ``2(1-ksi)`` and the diagonal carries the extra ``delta(1-ksi)`` (coefficient 1).
    """
    del zspz  # MSbar has no Wilson-line scale; kept for a uniform signature.
    return build_matching_matrix(
        x_ls, pz_gev, mu, y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_msbar_gz(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale) + 1.0,
    )


def CG_gz_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.19-2.20; = gamma^t)."""
    return CG_gt_PDF_hybrid(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gzg5_PDF_ratio(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_PDF_ratio(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gzg5_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_PDF_msbar(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gzg5_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_PDF_hybrid(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


# --- transversity gamma^t gamma_perp gamma5 PDF -----------------------------
# Eqs. (2.17), (2.18), (2.21): the transversity coefficient is C_r^perp in *every*
# scheme -- MSbar = ratio (no extra finite term, Eq. 2.17) and the hybrid Wilson-line
# correction vanishes (delta C_hyb = 0, Eq. 2.21). So all three schemes coincide.


def CG_gtgpg5_PDF_ratio(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the Coulomb-gauge transversity ``gamma^t gamma_perp gamma5`` PDF (Eq. 2.18)."""
    del zspz  # transversity has no Wilson-line scale at NLO (Eq. 2.21).
    return build_matching_matrix(
        x_ls, pz_gev, mu, y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_ratio_perp(ksi, log_scale, eps),
    )


def CG_gtgpg5_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar transversity kernel (Eq. 2.17: equals the ratio coefficient C_r^perp)."""
    return CG_gtgpg5_PDF_ratio(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)


def CG_gtgpg5_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid transversity kernel (Eq. 2.21: delta C_hyb = 0, so equals C_r^perp)."""
    return CG_gtgpg5_PDF_ratio(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps, zspz=zspz)
