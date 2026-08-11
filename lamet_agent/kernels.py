"""NLO Coulomb-gauge matching kernels (arXiv:2602.11283).

Written in the same spirit as a hand script: a few explicit *coefficient
functions* ``C_<scheme>(ksi, ...)`` carry the closed-form physics, and a single
*discretization* ``build_matching_matrix`` turns any of them into the matching
matrix. ``x`` and ``y`` stay independent open grids (lists/arrays):

    lightcone(x) = sum_y  K(x, y) * quasi(y)          # K = kernel @ quasi

with ``ksi = x / y`` and the lamet log scale ``L = log(4 y^2 P_z^2 / mu^2)``.

Three operator classes are implemented (arXiv:2602.11283, Eqs. 2.14-2.21), each in
the ratio / msbar / hybrid scheme via ``CG_<operator>_quark_PDF_<scheme>_NLO``
(``quark_PDF`` marks the quark kernels; a gluon kernel would read ``gluon_PDF``):

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

The gauge-invariant (straight Wilson line) kernels are provided as
``GI_<operator>_quark_PDF_<scheme>_NLO`` in the ratio / hybrid schemes, with gt/gtg5 from
arXiv:2412.20461 Eqs. (23)-(24), gz/gzg5 from arXiv:2604.00143 Eqs. (C6)-(C8) and
transversity from arXiv:2208.08008 Eqs. (22)-(23). As
in the Coulomb-gauge case, unpolarized and helicity share one coefficient at NLO, and
gamma^z differs from gamma^t only by +2(1-ksi) on 0 < ksi < 1. Each kernel is tagged
with its paper via ``@kernel_reference``; the matching report cites that tag.

Each public kernel is a one-line wrapper picking a coefficient function; the
discretization (loop + plus prescription + LO delta) is shared.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Final

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


# --- coordinate-space MSbar conversion for self-renormalization -------------
# Used by the renormalization stage (stages/renorm/functions.py), not by the matching
# kernels below.


def ZMSbar(z_fm: np.ndarray | float, *, mu: float = 2.0, offset: float, order: int = 0, Nf: int = 3) -> np.ndarray:
    """1-loop coordinate-space conversion to MSbar at scale ``mu`` (GeV).

    ``offset`` is the finite constant: ``5/2`` for PDF and ``7/2`` for DA.
    """
    z_arr = np.asarray(z_fm, dtype=float)
    alphas = alphas_nloop(mu, order=order, Nf=Nf)
    log_term = np.log(mu**2 * (z_arr / GEV_FM) ** 2 * np.exp(2.0 * np.euler_gamma) / 4.0)
    return 1.0 + alphas * CF / (2.0 * np.pi) * (1.5 * log_term + offset)


def ZMSbar_pdf(z_fm: np.ndarray | float, mu: float = 2.0, order: int = 0, Nf: int = 3) -> np.ndarray:
    """1-loop MSbar conversion factor for PDF self-renormalization."""
    return ZMSbar(z_fm, mu=mu, offset=2.5, order=order, Nf=Nf)


def ZMSbar_da(z_fm: np.ndarray | float, mu: float = 2.0, order: int = 0, Nf: int = 3) -> np.ndarray:
    """1-loop MSbar conversion factor for DA self-renormalization."""
    return ZMSbar(z_fm, mu=mu, offset=3.5, order=order, Nf=Nf)


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


def C_msbar_plus(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Integrand of the MSbar delta subtraction, Eq. (2.14).

    Identical to ``C_msbar`` except that the ``0.5/|1-ksi|`` piece only enters for
    ksi in [0, 2]: its delta partner in Eq. (2.14) is ``-1/2 int_0^2 d ksi'``, so
    outside [0, 2] the term is a plain regular contribution with no subtraction.
    This is the ``plus_coeff`` handed to ``_build_pdf_matrix`` for the MSbar scheme.
    """
    entry = C_ratio(ksi, log_scale, eps)
    if 0.0 <= ksi <= 2.0:
        entry += 0.5 / (np.abs(1.0 - ksi) + eps)
    return entry


def C_msbar_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """gamma^z MSbar off-diagonal coefficient, Eq. (2.15).

    ``C_msbar^{gamma^z} = C_msbar^{gamma^t} + 2(1-ksi)`` on 0 < ksi < 1 (plus a
    ``delta(1-ksi)`` term carried on the diagonal -- see CG_gz_quark_PDF_msbar_NLO). The
    ``2(1-ksi)`` is plus-prescribed at ksi = 1 by the shared discretization.
    """
    entry = C_msbar(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_msbar_gz_plus(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Integrand of the gamma^z MSbar delta subtraction (Eqs. 2.14-2.15).

    Like ``C_msbar_plus`` (0.5/|1-ksi| only on [0, 2]) plus the ``2(1-ksi)`` piece,
    which is nonzero only on 0 < ksi < 1 so it is subtracted on its whole support.
    """
    entry = C_msbar_plus(ksi, log_scale, eps)
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
# A kernel density has the signature ``density(x, y) -> float``: the regular (x != y)
# NLO integrand for one grid pair, *including its own measure factors*, with alpha_s
# C_F / (2 pi) factored out. Anything else the formula needs (P_z, mu, the Wilson-line
# scale, ...) is closed over by the public kernel that builds it.
#
# Keeping it a raw function of (x, y) -- rather than of ksi = x/y -- is what lets one
# discretization serve both PDFs and DAs. A PDF coefficient depends only on ksi and is
# integrated with dy/|y|, so its density is ``C(x/y, L(y)) / |y|``. A meson DA kernel
# V(x, y) (arXiv:2212.14415 Eq. 4.15) is a genuine two-variable function -- terms like
# |x-y|/(y(y-1)) are not functions of x/y -- and is integrated with a plain dy, so its
# density is just V(x, y). Everything downstream (skip the x = y singular line, restore
# it by the plus prescription, LO delta(x - y), the dy measure) is common to both.
DensityFn = Callable[[float, float], float]


def _lo_interp_matrix(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """LO delta(x - y) as the matrix form of the examples' ``np.interp`` grid move.

    Built column by column straight from ``np.interp`` (each y basis vector), so
    ``(matrix @ q)[i] == np.interp(x_grid[i], y_grid, q, left=0, right=0)`` by
    construction. Equals the identity when the grids coincide, and keeps the LO term
    alive (instead of dropping to all-NLO) when the x and y grids are staggered.
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
    lc_x_ls: np.ndarray,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    *,
    density: DensityFn,
    color_factor: float = CF,
    diagonal_extra: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Discretize a kernel density ``density(x, y)`` into an (nx, ny) matching matrix.

    This is the column-sum plus prescription: it knows no physics beyond "a regular NLO
    integrand singular on x = y, made a plus function by forcing each y column to sum to
    zero". It serves the meson-DA kernel, a genuine two-variable V(x, y) integrated with a
    plain dy (the ``0.5/|1-ksi|``-over-[0, 2] structure of the CG PDFs needs the ksi
    integral of ``_build_pdf_matrix`` instead, so PDF kernels use that, not this).

    The matrix rows live on ``lc_x_ls`` (the output light-cone x grid) and its
    columns on ``quasi_y_ls`` (the input quasi x grid), so the caller forms the matched
    distribution as ``lightcone = matrix @ quasi``. The two are independent open grids;
    ``quasi_y_ls`` defaults to ``lc_x_ls``. The loop fills the off-diagonal
    (x != y) entries from ``density``; the LO delta(x - y) is a linear-interpolation
    stencil from the y grid onto each x, so it survives when the grids are staggered and
    collapses to the identity when they coincide; the plus prescription makes every y
    column integrate to zero and thereby restores the x = y singularity;
    ``diagonal_extra(y)`` adds an optional finite diagonal conversion term.
    Returns ``identity - alpha_s C_x/(2 pi) * matrix * dy`` -- the matched distribution
    is one matrix product, with no matrix inverse anywhere.
    """
    x_grid = np.asarray(lc_x_ls, dtype=float)
    y_grid = np.asarray(x_grid if quasi_y_ls is None else quasi_y_ls, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("`lc_x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`quasi_y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        # Every density here carries a 1/y (PDF: the dy/|y| measure; DA: V's own 1/y),
        # so a y = 0 column is singular for either.
        raise ValueError("`quasi_y_ls` must avoid values too close to 0.")

    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))  # uniform integration measure
    if dy <= eps:
        raise ValueError("`quasi_y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`quasi_y_ls` must be uniformly spaced.")

    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    nx, ny = len(x_grid), len(y_grid)
    nlo_matrix = np.zeros((nx, ny))
    # LO delta(x - y): a linear-interpolation stencil from the y grid onto each x,
    # the same np.interp(..., left=0, right=0) trick the examples use to move a curve
    # between grids. Collapses to the identity when the grids coincide.
    identity = _lo_interp_matrix(x_grid, y_grid)
    # For each y column, the x row closest to that y point carries the plus-function.
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)

    # 1) Off-diagonal (x != y) regular entries from the density. The tolerance is
    #    relative to |y| -- for a PDF density that is exactly the old |1 - x/y| <= eps.
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            if np.abs(x_val - y_val) <= eps * np.abs(y_val):
                continue  # the x = y singularity is restored by the plus prescription
            nlo_matrix[idx, idy] = density(x_val, y_val)

    # 2) Plus-prescription: make every y column integrate to zero, then add the
    #    optional finite scheme-conversion term on that column's nearest x row.
    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy])
        if diagonal_extra is not None:
            nlo_matrix[int(diag_row), idy] += diagonal_extra(float(y_grid[idy])) / dy

    # 3) Assemble: LO identity minus the NLO correction (times the dy measure).
    return identity - alpha_s * color_factor / (2.0 * np.pi) * nlo_matrix * dy


# --- PDF coefficient signature ----------------------------------------------
# A PDF coefficient knows only ksi = x/y, the lamet log L(y) and y; the dy/|y| measure
# and the frozen-delta ksi integral are supplied by ``_build_pdf_matrix`` below, so every
# PDF kernel stays a one-line wrapper handing it a ``coeff`` (and, for MSbar, a
# ``plus_coeff`` and a ``diagonal_extra``).
CoeffFn = Callable[[float, float, float], float]


def _pdf_log_scale(y: float, momentum_gev: float, mu: float) -> float:
    """The lamet log ``L = ln(4 y^2 P_z^2 / mu^2)`` that the PDF coefficients expect."""
    return float(np.log(4.0 * y**2 * momentum_gev**2 / mu**2))


def _pdf_density(coeff: CoeffFn, momentum_gev: float, mu: float) -> DensityFn:
    """Wrap a PDF coefficient ``coeff(ksi, L, y)`` into the ``density(x, y)`` the
    discretization consumes: evaluate it at ksi = x/y and divide by |y| (the dy/|y|
    measure of the PDF factorization formula).

    Used by the ratio / hybrid PDF kernels, which go through the column-sum
    ``build_matching_matrix`` (2nd-order accurate, smooth on the run grid). Only the
    MSbar kernels take the ``_build_pdf_matrix`` ksi-integral path below, whose
    ``int_0^2`` plus counterterm they genuinely need.
    """

    def density(x: float, y: float) -> float:
        return coeff(x / y, _pdf_log_scale(y, momentum_gev, mu), y) / np.abs(y)

    return density


# --- PDF MSbar discretization: unpol_matching's ksi-integral plus prescription ---
# Only the MSbar kernels use this path (the ratio / hybrid kernels use the column-sum
# build_matching_matrix, which is 2nd-order accurate and stays smooth -- the ksi-integral
# below is only 1st-order and adds grid-scale noise to the Wilson-line hybrid term).
# The PDF kernels use a frozen-at-delta-point ksi-integral for the plus subtraction
# (not build_matching_matrix's column-sum): each x row's delta carries
# ``diagonal_extra(L(x)) - int d(ksi) plus_coeff(ksi, L(x), x)``, the integral summed
# on the grid ksi lattice ``x / y_j`` (measure d(ksi) = |x| dy / y^2) and completed by
# ``_integrate`` over the ksi regions the y grid cannot reach. This reproduces the
# collaborator reference kernels (msbar_kernel/*.npy, hybrid_nlo/*.npy); the plain
# column-sum in build_matching_matrix does not, because the MSbar 1/(2|1-ksi|) term is
# plus-prescribed only over ksi in [0, 2] and its far tail leaves the finite grid.


def _integrate(func: Callable[[float], float], lo: float, hi: float) -> float:
    """Integrate a smooth coefficient tail over (lo, hi); +-inf limits allowed.

    Uses scipy when available; the fallback maps infinite tails onto u = 1/ksi
    (the coefficients decay like 1/ksi^2, so the transformed integrand is finite)
    and applies a fixed Simpson rule.
    """
    if hi <= lo:
        return 0.0
    try:
        import warnings

        from scipy import integrate as _si

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", _si.IntegrationWarning)
            value, _ = _si.quad(func, lo, hi, limit=400)
        return float(value)
    except ModuleNotFoundError:
        pass

    def simpson(f: Callable[[float], float], a: float, b: float, n: int = 2000) -> float:
        grid = np.linspace(a, b, n + 1)
        vals = np.array([f(t) for t in grid])
        h = (b - a) / n
        return float(h / 3.0 * (vals[0] + vals[-1] + 4.0 * vals[1:-1:2].sum() + 2.0 * vals[2:-2:2].sum()))

    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    if finite_lo and finite_hi:
        return simpson(func, lo, hi)
    # Map an infinite tail via u = 1/ksi (ksi = 1/u, d ksi = -du / u^2).
    if finite_lo and not finite_hi:  # (lo, +inf), lo > 0
        return simpson(lambda u: func(1.0 / u) / u**2, 0.0 + 1e-9, 1.0 / lo)
    if not finite_lo and finite_hi:  # (-inf, hi), hi < 0
        return simpson(lambda u: func(1.0 / u) / u**2, 1.0 / hi, 0.0 - 1e-9)
    return 0.0


def _uncovered_ksi_intervals(
    x_val: float, y_grid: np.ndarray, dy: float, eps: float
) -> list[tuple[float, float]]:
    """Return the ksi intervals NOT reached by ``ksi = x / y`` on the y grid.

    For fixed x > 0 the y grid maps onto one ksi interval per sign of y; the
    complement (the gap around ksi = 0 from |y| beyond the grid ends, plus the far
    |ksi| tails from |y| below the smallest grid value) is added to the delta
    subtraction by direct integration. Intervals are clipped half a grid step away
    from ksi = 1, where the discrete sum takes over the (regulated) plus singularity.
    """
    covered: list[tuple[float, float]] = []
    for sign in (1.0, -1.0):
        ys = y_grid[y_grid * sign > eps]
        if ys.size:
            lo, hi = x_val / (sign * np.max(sign * ys)), x_val / (sign * np.min(sign * ys))
            covered.append((min(lo, hi), max(lo, hi)))
    covered.sort()

    gaps: list[tuple[float, float]] = []
    edge = -np.inf
    for lo, hi in covered:
        gaps.append((edge, lo))
        edge = hi
    gaps.append((edge, np.inf))

    guard = 0.5 * dy / np.abs(x_val)
    clipped = []
    for lo, hi in gaps:
        for piece_lo, piece_hi in ((lo, min(hi, 1.0 - guard)), (max(lo, 1.0 + guard), hi)):
            if piece_hi > piece_lo:
                clipped.append((piece_lo, piece_hi))
    return clipped


def _build_pdf_matrix(
    x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    y_ls: np.ndarray | None,
    eps: float,
    *,
    coeff: CoeffFn,
    plus_coeff: CoeffFn | None = None,
    color_factor: float = CF,
    diagonal_extra: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Discretize a PDF coefficient ``coeff(ksi, log_scale, y)`` into the full (nx, ny)
    matching matrix (LO identity + NLO correction), using the ksi-integral plus
    prescription of ``unpol_matching`` -- see the module note above.

    ``coeff`` fills every regular (ksi != 1) entry with ``coeff(x/y, L(y), y)/|y|``.
    Each x row then receives one delta term spread over the two columns bracketing
    y = x, carrying ``diagonal_extra(L(x)) - int d(ksi) plus_coeff(ksi, L(x), x)``.
    ``plus_coeff`` defaults to ``coeff`` (MSbar passes a [0, 2]-restricted variant).
    """
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`quasi_y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`quasi_y_ls` must avoid values too close to 0 to keep ksi=x/y finite.")

    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))
    if dy <= eps:
        raise ValueError("`quasi_y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`quasi_y_ls` must be uniformly spaced.")
    if np.any(x_grid < np.min(y_grid) - 0.5 * dy) or np.any(x_grid > np.max(y_grid) + 0.5 * dy):
        raise ValueError("every x must lie inside the y grid to place its delta term.")

    if plus_coeff is None:
        plus_coeff = coeff

    alpha_s = alphas_nloop(mu, order=1, Nf=3)
    nx, ny = len(x_grid), len(y_grid)
    identity = _lo_interp_matrix(x_grid, y_grid)
    nlo_matrix = np.zeros((nx, ny))

    # 1) Regular (ksi != 1) coefficients.
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            ksi = x_val / y_val
            if np.abs(1.0 - ksi) <= eps:
                continue
            log_scale = _pdf_log_scale(y_val, momentum_gev, mu)
            nlo_matrix[idx, idy] = coeff(ksi, log_scale, y_val) / np.abs(y_val)

    # 2) Delta terms: one per x row, spread over the two bracketing columns, carrying
    #    diagonal_extra(L(x)) - int d(ksi) plus_coeff frozen at the delta point.
    ascending = y_grid[1] > y_grid[0]
    y_sorted = y_grid if ascending else y_grid[::-1]
    for idx, x_val in enumerate(x_grid):
        if np.abs(x_val) <= eps:
            # x = 0: ksi = x/y is 0 for every column, so the row never crosses the
            # ksi = 1 plus singularity and needs no delta subtraction. (L(x) and the
            # d(ksi) = |x| dy / y^2 measure are both singular here; skipping is the
            # finite, well-defined limit -- the row is pure off-diagonal + LO.)
            continue
        pos = int(np.searchsorted(y_sorted, x_val))
        pos = min(max(pos, 1), ny - 1)
        w_hi = np.clip((x_val - y_sorted[pos - 1]) / (y_sorted[pos] - y_sorted[pos - 1]), 0.0, 1.0)
        cols_weights = (
            ((pos - 1 if ascending else ny - pos), 1.0 - w_hi),
            ((pos if ascending else ny - 1 - pos), w_hi),
        )

        log_scale = _pdf_log_scale(x_val, momentum_gev, mu)

        subtraction = 0.0
        for y_val in y_grid:
            ksi = x_val / y_val
            if np.abs(1.0 - ksi) <= eps:
                continue
            subtraction += plus_coeff(ksi, log_scale, x_val) * np.abs(x_val) * dy / y_val**2
        for lo, hi in _uncovered_ksi_intervals(x_val, y_grid, dy, eps):
            subtraction += _integrate(lambda ksi: plus_coeff(ksi, log_scale, x_val), lo, hi)

        delta_entry = -subtraction
        if diagonal_extra is not None:
            delta_entry += diagonal_extra(log_scale)
        for col, weight in cols_weights:
            nlo_matrix[idx, int(col)] += weight * delta_entry / dy

    return identity - alpha_s * color_factor / (2.0 * np.pi) * nlo_matrix * dy


# --- provenance: which paper each kernel is transcribed from -----------------


def kernel_reference(arxiv_id: str, equations: str) -> Callable[[Any], Any]:
    """Tag a kernel with the paper and equations it is transcribed from.

    The tag is the single source of truth for the kernel's provenance: the matching
    report reads ``fn.arxiv_id`` / ``fn.equations`` off the kernel it actually ran
    (see ``stages/matching/reporting.py``) to cite the right paper and to fetch the
    right LaTeX source for the formula section. Kernels from different papers can
    therefore coexist in this file without any table to keep in sync.

    The decorator only attaches attributes -- it does not wrap the function, so the
    kernel stays directly callable and ``inspect.getsource`` still shows its body.
    """

    def tag(fn: Any) -> Any:
        fn.arxiv_id = arxiv_id
        fn.equations = equations
        return fn

    return tag


# --- public quark kernels: CG_<operator>_quark_PDF_<scheme>_NLO --------------------
# Each is one line: pick a coefficient function (and, for MSbar, the diagonal
# conversion term) and hand it to build_matching_matrix.


@kernel_reference("2602.11283", "Eq. (2.16)")
def CG_gt_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel ``C_r`` for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.16)."""
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_ratio(ksi, log_scale, eps), momentum_gev, mu),
    )


@kernel_reference("2602.11283", "Eq. (2.14)")
def CG_gt_quark_PDF_msbar_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar kernel for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.14)."""
    del zspz  # MSbar has no Wilson-line scale; kept for a uniform signature.
    # Off-diagonal carries the full C_msbar (C_ratio + 0.5/|1-ksi|); the delta
    # subtraction uses C_msbar_plus, which restricts 0.5/|1-ksi| to the paper's
    # int_0^2 counterterm, and _build_pdf_matrix completes the grid-uncovered tail.
    return _build_pdf_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_msbar(ksi, log_scale, eps),
        plus_coeff=lambda ksi, log_scale, y: C_msbar_plus(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale),
    )


@kernel_reference("2602.11283", "Eqs. (2.19)-(2.20)")
def CG_gt_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
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
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )


# --- helicity gamma^t gamma5 PDF --------------------------------------------
# The helicity kernels share the unpolarized gamma^t structure, so each scheme
# simply delegates to the corresponding CG_gt_quark_PDF_<scheme>_NLO builder above.


@kernel_reference("2602.11283", "Eq. (2.16)")
def CG_gtg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2602.11283", "Eq. (2.14)")
def CG_gtg5_quark_PDF_msbar_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_quark_PDF_msbar_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


@kernel_reference("2602.11283", "Eqs. (2.19)-(2.20)")
def CG_gtg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme helicity kernel for the Coulomb-gauge ``gamma^t gamma5`` PDF."""
    return CG_gt_quark_PDF_hybrid_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


# --- gamma^z / gamma^z gamma5 PDF -------------------------------------------
# Eq. (2.15): only the MSbar scheme differs from gamma^t (by 2(1-ksi)_+ + delta).
# In the ratio and hybrid schemes gamma^z shares gamma^t's coefficient
# (C_r in Eq. 2.16; delta C_hyb in Eq. 2.20 is identical for gamma^t and gamma^z),
# so those two delegate to the gamma^t builders.


@kernel_reference("2602.11283", "Eq. (2.16)")
def CG_gz_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.16; = gamma^t)."""
    return CG_gt_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2602.11283", "Eq. (2.15)")
def CG_gz_quark_PDF_msbar_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.15).

    ``= gamma^t MSbar + 2(1-ksi)_+ + delta(1-ksi)``: the off-diagonal carries the
    extra ``2(1-ksi)`` and the diagonal carries the extra ``delta(1-ksi)`` (coefficient 1).
    """
    del zspz  # MSbar has no Wilson-line scale; kept for a uniform signature.
    # Off-diagonal carries the full C_msbar_gz; the delta subtraction uses
    # C_msbar_gz_plus (0.5/|1-ksi| restricted to [0, 2], the 2(1-ksi) piece on its
    # whole (0,1) support). The extra delta(1-ksi) of Eq. 2.15 rides on diagonal_extra.
    return _build_pdf_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coeff=lambda ksi, log_scale, y: C_msbar_gz(ksi, log_scale, eps),
        plus_coeff=lambda ksi, log_scale, y: C_msbar_gz_plus(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale) + 1.0,
    )


@kernel_reference("2602.11283", "Eqs. (2.19)-(2.20)")
def CG_gz_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme kernel for the Coulomb-gauge ``gamma^z`` PDF (Eq. 2.19-2.20; = gamma^t)."""
    return CG_gt_quark_PDF_hybrid_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


@kernel_reference("2602.11283", "Eq. (2.16)")
def CG_gzg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2602.11283", "Eq. (2.15)")
def CG_gzg5_quark_PDF_msbar_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_quark_PDF_msbar_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


@kernel_reference("2602.11283", "Eqs. (2.19)-(2.20)")
def CG_gzg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme helicity kernel for the Coulomb-gauge ``gamma^z gamma5`` PDF."""
    return CG_gz_quark_PDF_hybrid_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


# --- transversity gamma^t gamma_perp gamma5 PDF -----------------------------
# Eqs. (2.17), (2.18), (2.21): the transversity coefficient is C_r^perp in *every*
# scheme -- MSbar = ratio (no extra finite term, Eq. 2.17) and the hybrid Wilson-line
# correction vanishes (delta C_hyb = 0, Eq. 2.21). So all three schemes coincide.


@kernel_reference("2602.11283", "Eq. (2.18)")
def CG_gtgpg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the Coulomb-gauge transversity ``gamma^t gamma_perp gamma5`` PDF (Eq. 2.18)."""
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_ratio_perp(ksi, log_scale, eps), momentum_gev, mu),
    )


@kernel_reference("2602.11283", "Eq. (2.17)")
def CG_gtgpg5_quark_PDF_msbar_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO MSbar transversity kernel (Eq. 2.17: equals the ratio coefficient C_r^perp)."""
    del zspz  # CG transversity has no Wilson-line scale at NLO (Eq. 2.21).
    return CG_gtgpg5_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2602.11283", "Eq. (2.21)")
def CG_gtgpg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid transversity kernel (Eq. 2.21: delta C_hyb = 0, so equals C_r^perp)."""
    del zspz  # CG transversity has no Wilson-line scale at NLO (Eq. 2.21).
    return CG_gtgpg5_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


# --- gauge-invariant (straight Wilson line) coefficients, Eqs. (23)-(24) -----
# The GI counterpart of the CG coefficients above: same discretization, different
# closed form. Both the gamma^t (unpolarized) and gamma^t gamma5 (helicity) PDFs
# share one coefficient at NLO, so the four GI kernels below are two wrappers each.


def C_ratio_gi(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Gauge-invariant ratio-scheme coefficient, Eq. (23), written region by region.

    The paper's three branches (ksi > 1, 0 < ksi < 1, ksi < 0) share two terms once
    the region signs are worked out: the ``3/(2(1-ksi))`` tail is ``+1.5/|1-ksi|``
    everywhere, and the constant ``+-1`` is ``sgn(ksi)``. Only the log piece differs:
    outside [0, 1] it is ``sgn(ksi) * S * ln|ksi/(ksi-1)|``, inside it carries the
    lamet log. ``S = (1+ksi^2)/(1-ksi)`` is the splitting function.

    ``log_scale = ln(4 y^2 P_z^2 / mu^2)`` is the discretization's convention, while
    Eq. (23) uses ``-ln(mu^2/(y^2 P_z^2)) = ln(y^2 P_z^2 / mu^2)``; the two differ by
    the constant ``ln 4``, removed here. The whole expression sits under the paper's
    plus prescription, which build_matching_matrix restores (each y column sums to zero).
    """
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    splitting = (1.0 + ksi**2) / sign_safe_denominator

    if eps < ksi < 1.0 - eps:
        # 0 < ksi < 1: S * (ln(y^2 Pz^2/mu^2) + ln(4 ksi (1-ksi)) - 1) + 1
        lamet_log = log_scale - np.log(4.0)  # ln(4 y^2 Pz^2/mu^2) -> ln(y^2 Pz^2/mu^2)
        entry = splitting * (
            lamet_log + np.log(4.0 * ksi * one_minus_ksi + eps) - 1.0
        ) + 1.0
    else:
        # ksi > 1 and ksi < 0: +-[S * ln(ksi/(ksi-1)) + 1], the sign following sgn(ksi).
        # ksi/(ksi-1) is positive in both regions, so take it in absolute value.
        log_ratio = np.log((np.abs(ksi) + eps) / (np.abs(ksi - 1.0) + eps))
        entry = np.sign(ksi) * (splitting * log_ratio + 1.0)

    entry += 1.5 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def _hybrid_gi_delta(ksi: float, y: float, zspz: float, eps: float, strength: float) -> float:
    """Ratio -> hybrid switch for the GI kernels: the same ``R``, a per-operator strength.

    Every GI paper writes the same Wilson-line correction
    ``R = -1/|1-ksi| + 2 Si((1-ksi) |y| z_s P_z) / (pi (1-ksi))``, differing only in the
    prefactor. The discretization factors out ``alpha_s C_F / (2 pi)``, so ``strength`` is
    what remains of each paper's prefactor:

      * 3/2 -- gamma^t/gamma^z: 2412.20461 Eq. (24) writes ``3 alpha_s C_F / (4 pi)``, and
        2604.00143 Eq. (C8) writes ``(alpha_s C_F / 2 pi)(3/2)`` -- the same number;
      * 2   -- transversity: 2208.08008 Eq. (23) writes ``alpha_s C_F / pi``.

    ``zspz = z_s * P_z``, so ``|y| * zspz`` is the papers' lambda_s built on the parton
    momentum ``y P_z`` (2208.08008 folds the ``|y|`` into its own lambda_s).
    """
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    wilson_scale = np.abs(y) * zspz  # |y| * z_s * P_z
    return strength * (
        -1.0 / (np.abs(one_minus_ksi) + eps)
        + 2.0 * _sine_integral(one_minus_ksi * wilson_scale) / (np.pi * sign_safe_denominator)
    )


def C_hybrid_gi(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    """Gauge-invariant gamma^t hybrid coefficient: C_ratio_gi + the Si term, Eq. (24)."""
    return C_ratio_gi(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=1.5)


def C_ratio_gi_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """Gauge-invariant gamma^z ratio coefficient, arXiv:2604.00143 Eq. (C7).

    Equal to the gamma^t coefficient C_ratio_gi (Eq. 23 of arXiv:2412.20461) plus
    ``2(1-ksi)`` on 0 < ksi < 1 -- the same gamma^z vs gamma^t shift the Coulomb-gauge
    pair C_msbar / C_msbar_gz shows. The term is plus-prescribed at ksi = 1 by the
    shared discretization.

    The two papers write the log differently but mean the same thing: Eq. (C7) uses
    ``ksi = w/y`` and ``-ln(mu0^2 / (4 w^2 P_z^2)) + ln((1-ksi)/ksi)``, which with
    ``w = ksi y`` is ``ln(y^2 P_z^2 / mu0^2) + ln(4 ksi (1-ksi))`` -- exactly Eq. (23)'s
    combination, so C_ratio_gi already carries it.
    """
    entry = C_ratio_gi(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_hybrid_gi_gz(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    """Gauge-invariant gamma^z hybrid coefficient: Eq. (C6) = LO + Eq. (C7) + Eq. (C8).

    The LO delta(1-ksi) is the discretization's identity, so only the two NLO pieces
    are assembled here. Eq. (C6) also carries a ``delta C_M`` (leading-renormalon /
    mass) term and an NNLO piece; neither is implemented -- this kernel is NLO only.
    """
    return C_ratio_gi_gz(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=1.5)


def C_ratio_gi_perp(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    """GI transversity ratio coefficient, arXiv:2208.08008 Eq. (22), branch by branch.

    The paper's three branches are transcribed as written, without merging them: the
    transversity splitting is ``2 ksi / (1 - ksi)``, the ``2/(1-ksi)`` tail appears only
    outside [0, 1] and the ``+2`` constant only inside it -- unlike the gamma^t/gamma^z
    coefficient, where the tail is common to all three regions.

    Eq. (22) writes the log as ``ln(4 p_z^2 / mu^2) + ln(ksi (1-ksi))`` with ``p_z`` the
    *parton* momentum (its coefficient is ``C_r(x, mu/p_z)``), i.e. ``p_z = y P_z``. That
    first term is exactly this discretization's ``log_scale = ln(4 y^2 P_z^2 / mu^2)``.
    """
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    splitting = 2.0 * ksi / sign_safe_denominator  # 2 ksi / (1 - ksi)
    # ksi/(ksi-1) is positive on both outer branches, so form it in absolute value.
    log_ratio = np.log((np.abs(ksi) + eps) / (np.abs(ksi - 1.0) + eps))

    if ksi > 1.0 + eps:
        # 2 ksi/(1-ksi) * ln(ksi/(ksi-1)) - 2/(1-ksi)
        entry = splitting * log_ratio - 2.0 / sign_safe_denominator
    elif eps < ksi < 1.0 - eps:
        # 2 ksi/(1-ksi) * (ln(4 p_z^2/mu^2) + ln(ksi(1-ksi))) + 2
        entry = splitting * (log_scale + np.log(ksi * one_minus_ksi + eps)) + 2.0
    elif ksi < -eps:
        # -2 ksi/(1-ksi) * ln(ksi/(ksi-1)) + 2/(1-ksi)
        entry = -splitting * log_ratio + 2.0 / sign_safe_denominator
    else:
        # ksi = 0 (and the ksi = 1 point, which the discretization never asks for):
        # every term above carries a factor ksi or is finite, so the coefficient is 0.
        entry = 0.0
    return float(entry)


def C_hybrid_gi_perp(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    """GI transversity hybrid coefficient, arXiv:2208.08008 Eq. (23): C_r + the Si term.

    Eq. (23)'s correction carries the prefactor ``alpha_s C_F / pi``, twice the
    ``alpha_s C_F / (2 pi)`` the discretization factors out -- hence ``strength=2``, where
    the gamma^t/gamma^z papers give 3/2 for the same ``R``.
    """
    return C_ratio_gi_perp(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=2.0)


# --- public quark kernels: GI_<operator>_quark_PDF_<scheme>_NLO --------------------


@kernel_reference("2412.20461", "Eq. (23)")
def GI_gt_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the gauge-invariant ``gamma^t`` PDF (Eq. 23)."""
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_ratio_gi(ksi, log_scale, eps), momentum_gev, mu),
    )


@kernel_reference("2412.20461", "Eq. (24)")
def GI_gt_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid kernel for the gauge-invariant ``gamma^t`` PDF (Eq. 24).

    ``zspz = z_s * P_z`` (the paper's lambda_s) is required.
    """
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid_gi(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )


@kernel_reference("2412.20461", "Eq. (23)")
def GI_gtg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel: same coefficient as GI_gt_quark_PDF_ratio_NLO (Eq. 23)."""
    return GI_gt_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2412.20461", "Eq. (24)")
def GI_gtg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid helicity kernel: same coefficient as GI_gt_quark_PDF_hybrid_NLO (Eq. 24)."""
    return GI_gt_quark_PDF_hybrid_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


# --- gauge-invariant gamma^z / gamma^z gamma5, arXiv:2604.00143 Eqs. (C6)-(C8) ---
# Same structure as the gamma^t pair above (the hybrid switch Eq. (C8) is literally
# Eq. (24)); the coefficient differs only by the +2(1-ksi) of Eq. (C7). As for gamma^t,
# unpolarized and helicity share one NLO coefficient.


@kernel_reference("2604.00143", "Eq. (C7)")
def GI_gz_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the gauge-invariant ``gamma^z`` PDF (Eq. C7)."""
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_ratio_gi_gz(ksi, log_scale, eps), momentum_gev, mu),
    )


@kernel_reference("2604.00143", "Eqs. (C6)-(C8)")
def GI_gz_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid kernel for the gauge-invariant ``gamma^z`` PDF (Eqs. C6-C8).

    ``zspz = z_s * P_z`` is required. NLO only: Eq. (C6)'s ``delta C_M`` and its NNLO
    term are not implemented.
    """
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid_gi_gz(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )


@kernel_reference("2604.00143", "Eq. (C7)")
def GI_gzg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme helicity kernel: same coefficient as GI_gz_quark_PDF_ratio_NLO (Eq. C7)."""
    return GI_gz_quark_PDF_ratio_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps)


@kernel_reference("2604.00143", "Eqs. (C6)-(C8)")
def GI_gzg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid helicity kernel: same coefficient as GI_gz_quark_PDF_hybrid_NLO (Eqs. C6-C8)."""
    return GI_gz_quark_PDF_hybrid_NLO(lc_x_ls, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=quasi_y_ls, eps=eps, zspz=zspz)


# --- gauge-invariant transversity, arXiv:2208.08008 Eqs. (22)-(23) -----------
# The transversity operator gamma^t gamma_perp gamma5 with a straight Wilson line. Unlike
# the Coulomb-gauge transversity (where ratio = msbar = hybrid), the hybrid scheme here
# does carry a Wilson-line correction, so ratio and hybrid are genuinely different.


@kernel_reference("2208.08008", "Eq. (22)")
def GI_gtgpg5_quark_PDF_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the GI transversity ``gamma^t gamma_perp gamma5`` PDF (Eq. 22)."""
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_ratio_gi_perp(ksi, log_scale, eps), momentum_gev, mu),
    )


@kernel_reference("2208.08008", "Eq. (23)")
def GI_gtgpg5_quark_PDF_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid kernel for the GI transversity PDF (Eq. 23).

    ``zspz = z_s * P_z`` (the paper's lambda_s) is required.
    """
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix(
        lc_x_ls, mu, quasi_y_ls, eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid_gi_perp(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )


# --- meson distribution amplitude, arXiv:2212.14415 Eqs. (4.15)-(4.16) -------
# The DA kernel V(x, y) is a genuine two-variable function -- terms like |x-y|/(y(y-1))
# are not functions of ksi = x/y -- so it cannot go through the PDF adapter. It is a
# kernel density in its own right: it already carries its own 1/y and 1/(1-y) poles and
# is integrated with a plain dy (no 1/|y| measure). Everything else -- skipping x = y,
# the plus prescription, the LO delta(x - y), the dy measure -- is the shared machinery.
#
# Normalization: the paper writes ``a_s C_F {...}`` with a_s = alpha_s/(4 pi), while this
# discretization factors out alpha_s C_F/(2 pi). The braces below are transcribed exactly
# as printed, and the factor 1/2 is applied where the density is built.


def _da_log(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    """The DA logs of Eq. (4.16): ``l = ln(4 P_z^2 v^2 / mu^2)``.

    Eq. (4.16) spells out ``l_xbar = ln(4 P_z^2 (1-x)^2 / mu^2)``; ``l_x`` and ``l_xy``
    are the same form with ``v = x`` and ``v = x - y``. The square makes each log
    well defined for either sign of ``v``.
    """
    return float(np.log(4.0 * momentum_gev**2 * value**2 / mu**2 + eps))


def V_qq_t(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    """DA matching kernel ``V_qq,t^(1)``, arXiv:2212.14415 Eq. (4.15), third line.

    Returns the brace as printed (the ``a_s C_F`` prefactor is applied by the caller).
    """
    l_x = _da_log(x, momentum_gev, mu, eps)
    l_xbar = _da_log(1.0 - x, momentum_gev, mu, eps)
    l_xy = _da_log(x - y, momentum_gev, mu, eps)
    return (
        np.abs(x) / (y * (y - x)) * (l_x - 1.0)
        + np.abs(1.0 - x) / ((1.0 - y) * (x - y)) * (l_xbar - 1.0)
        + (x + y - 2.0 * x * y) / (np.abs(x - y) * y * (1.0 - y)) * (l_xy - 1.0)
    )


def V_qq_h(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    """DA matching kernel ``V_qq,h^(1)``, arXiv:2212.14415 Eq. (4.15), first line.

    ``V_qq,h = a_s C_F {...} + V_qq,t``; the brace is transcribed as printed and the
    ``V_qq,t`` of the third line is added on. Returns the value with ``a_s C_F`` factored
    out, so the caller supplies the prefactor.

    Note the sibling ``V_qq,p = V_qq,h + 2 a_s C_F {|x|/y + |1-x|/(1-y) + |x-y|/((y-1)y)}``
    of the second line: that brace vanishes identically outside 0 < x < 1 (put over the
    common denominator y(1-y) the numerator cancels), so h and p differ only inside the
    physical window.
    """
    l_x = _da_log(x, momentum_gev, mu, eps)
    l_xbar = _da_log(1.0 - x, momentum_gev, mu, eps)
    l_xy = _da_log(x - y, momentum_gev, mu, eps)
    brace = (
        np.abs(x) / y * (l_x - 1.0)
        + np.abs(1.0 - x) / (1.0 - y) * (l_xbar - 1.0)
        + np.abs(x - y) / (y * (y - 1.0)) * (l_xy - 1.0)
    )
    return float(brace + V_qq_t(x, y, momentum_gev, mu, eps))


def V_qq_p(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    """DA matching kernel ``V_qq,p^(1)``, arXiv:2212.14415 Eq. (4.15), second line.

    ``V_qq,p = V_qq,h + 2 a_s C_F {|x|/y + |1-x|/(1-y) + |x-y|/((y-1)y)}``, transcribed
    as printed and returned with ``a_s C_F`` factored out.

    The added brace vanishes identically outside 0 < x < 1: over the common denominator
    y(1-y) its numerator is ``-x(1-y) + y(1-x) - (y-x) = 0`` there. So V_qq,p and V_qq,h
    coincide outside the physical window and differ only inside it.
    """
    extra = (
        np.abs(x) / y
        + np.abs(1.0 - x) / (1.0 - y)
        + np.abs(x - y) / ((y - 1.0) * y)
    )
    return float(V_qq_h(x, y, momentum_gev, mu, eps) + 2.0 * extra)


def V_qq_rto(x: float, y: float) -> float:
    """The Wilson-line term ``3/(2 |x - y|)`` that rides along with V_qq in the ratio scheme.

    Not an optional scheme flourish -- it is what makes the DA kernel integrable. The
    density ``V_qq,p / 2`` falls off as ``-3/(2 |x|)``, an *even* tail, so ``int dx V`` --
    exactly the integral the plus prescription subtracts on each y column -- diverges
    logarithmically on its own and the subtraction would depend on how wide an x grid one
    happened to pick. This term carries the equal and opposite ``+3/(2 |x|)`` tail, so the
    sum is integrable and the plus prescription is well defined and grid independent.

    It is the ``z_s -> infinity`` limit of the hybrid scheme's Si term: as its argument
    grows, ``Si -> (pi/2) sgn(y - x)`` and ``3 Si(z_s P_z (y-x)) / (pi (y-x))`` collapses
    to exactly this. Ratio and hybrid differ only in that one term.
    """
    return float(1.5 / np.abs(x - y))


def _da_matrix(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    *,
    coefficient: Callable[[float, float, float, float, float], float],
    wilson_line: Callable[[float, float], float],
) -> np.ndarray:
    """Discretize a meson-DA kernel: ``coefficient / 2`` plus the scheme's Wilson-line term.

    ``coefficient`` selects the operator, and the two DA operators differ by exactly the
    ``Delta C`` the papers quote for gamma^z gamma_5:

        V_qq,h / 2 = C^{gt g5},   V_qq,p / 2 = C^{gz g5} = C^{gt g5} + Delta C.

    ``0.5 (V_qq_p - V_qq_h) == Delta C`` holds identically (a test pins it), so gamma^z gamma_5
    needs no separate correction term -- adding one would count it twice. The two agree outside
    0 < x < 1, where Delta C vanishes, so both carry the same ``-3/(2|x|)`` tail and the same
    ``wilson_line`` cancels it. Ratio and hybrid differ only in ``wilson_line``.
    """
    def density(x: float, y: float) -> float:
        if not (eps < y < 1.0 - eps):
            return 0.0
        return 0.5 * coefficient(x, y, momentum_gev, mu, eps) + wilson_line(x, y)

    return build_matching_matrix(lc_x_ls, mu, quasi_y_ls, eps, density=density)


def _da_wilson_line(scheme: str, zspz: float | None, eps: float) -> Callable[[float, float], float]:
    """The Wilson-line term that turns the bare coefficient into a ratio or hybrid kernel.

    Ratio takes ``3/(2|x-y|)``; hybrid takes ``3 Si(z_s P_z (y-x)) / (pi (y-x))``, of which
    the ratio term is the ``z_s -> infinity`` limit. ``_hybrid_gi_delta`` supplies the Si term
    already carrying ``-3/(2|x-y|)`` -- it is written to convert a density that *has* the ratio
    term into the hybrid one -- so hybrid adds both and the ratio term cancels out of it.
    """
    if scheme == "ratio":
        return V_qq_rto
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)

    def hybrid(x: float, y: float) -> float:
        return V_qq_rto(x, y) + _hybrid_gi_delta(x / y, y, z, eps, strength=1.5) / np.abs(y)

    return hybrid


# --- public quark kernel: GI_<operator>_DA_<scheme>_NLO ----------------------


# The two DA operators share every piece of machinery and differ only in which coefficient
# of Eq. (4.15) they use: gamma^t gamma_5 takes V_qq,h, gamma^z gamma_5 takes V_qq,p (which
# is V_qq,h plus that operator's Delta C). The scheme picks the Wilson-line term on top.
#
# Unlike the PDF kernels the density is the two-variable V(x, y) itself: it carries its own
# 1/y and 1/(1-y) poles, so it is integrated with a plain dy rather than the PDF's dy/|y|.
# The 1/2 converts the paper's ``a_s = alpha_s/(4 pi)`` to the ``alpha_s C_F/(2 pi)`` this
# discretization factors out. The density is zero outside 0 < y < 1: in the factorization
# formula the y integral runs over the DA's support, so V is only ever defined there.


@kernel_reference(
    "2212.14415",
    "Eq. (4.15) V_qq,h (the gamma^t gamma_5 coefficient), with the ratio-scheme Wilson-line "
    "term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto)",
)
def GI_gtg5_DA_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the meson DA measured with the gamma^t gamma_5 operator."""
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coefficient=V_qq_h,
        wilson_line=_da_wilson_line("ratio", None, eps),
    )


@kernel_reference(
    "2405.20120",
    "Eq. (4.2), the gamma^t gamma_5 coefficient (V_qq_h below), with the hybrid-scheme "
    "Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x))",
)
def GI_gtg5_DA_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid kernel for the meson DA measured with the gamma^t gamma_5 operator.

    ``zspz = z_s * P_z``.
    """
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coefficient=V_qq_h,
        wilson_line=_da_wilson_line("hybrid", zspz, eps),
    )


@kernel_reference(
    "2212.14415",
    "Eq. (4.15) V_qq,p (the gamma^z gamma_5 coefficient), with the ratio-scheme Wilson-line "
    "term 3/(2|x-y|) (the z_s -> infinity limit of the hybrid Si term; see V_qq_rto)",
)
def GI_gzg5_DA_ratio_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO ratio-scheme kernel for the meson DA measured with the gamma^z gamma_5 operator."""
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coefficient=V_qq_p,
        wilson_line=_da_wilson_line("ratio", None, eps),
    )


@kernel_reference(
    "2405.20120",
    "Eq. (4.5), the gamma^z gamma_5 coefficient (V_qq_p below), with the hybrid-scheme "
    "Wilson-line term 3 Si(z_s P_z (y-x))/(pi (y-x))",
)
def GI_gzg5_DA_hybrid_NLO(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid kernel for the meson DA measured with the gamma^z gamma_5 operator.

    ``zspz = z_s * P_z``.
    """
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps,
        coefficient=V_qq_p,
        wilson_line=_da_wilson_line("hybrid", zspz, eps),
    )


# --- leading-renormalon resummation (LRR), arXiv:2305.05212 -------------------
# The gauge-invariant hybrid kernel above is a fixed-order (NLO) truncation of an
# asymptotic series: the straight Wilson line carries a linear UV (self-energy)
# divergence whose renormalon makes the matching coefficients c_n grow like
# n! (beta0/2 pi)^n -- the *same* growth as the heavy-quark pole mass. Leading
# renormalon resummation (LRR) sums that divergence to all orders in the
# principal-value (PV) Borel scheme and subtracts the low orders already in the
# fixed-order kernel, so the O(Lambda_QCD z) power correction is removed without
# double counting (arXiv:2305.05212 Eqs. 12-17, transcribed from LRR.nb).
#
# The construction is *matrix valued* -- it is not one more per-element coefficient
# fed to build_matching_matrix, but a product of discretized matrices with a matrix
# exponential:
#
#     M_LRR = (M_fix + r0 MCz) . exp(-MCz rsumPV)                     (LRR.nb MfixGen)
#
# where M_fix = I - C1 is exactly the fixed-order GI hybrid kernel already built
# above (its C1 = alpha_s C_F/(2 pi)(the GI ratio + Wilson-line Si terms) is the
# notebook's -Log[(mu/Pz)^2] P + Cratio1pluswomu + Chybrid1plus, which sum to
# C_hybrid_gi -- a test pins this), MCz is the *bare* plus-prescribed discretization
# of the renormalon shape C_z (no alpha_s, no C_F), and rsumPV, r0 are the two scalar
# renormalon numbers below. Both grids must coincide (the matrix exponential needs a
# square matrix), so the LRR kernels take lc_x_ls == quasi_y_ls.


# Renormalon normalization N_m (the pole-mass residue), arXiv:2305.05212 around Eq. (12).
# nf-indexed constants copied verbatim from LRR.nb; only nf = 3 is used for the pion.
_NM_RENORMALON: Final[dict[int, float]] = {
    3: 0.5749687262865643,
    4: 0.5522713118193284,
    5: 0.5235323457364502,
}


def _renormalon_params(nf: int) -> tuple[float, float, float, float]:
    """Return ``(beta0, b, c1, c2)`` for the renormalon series (LRR.nb rnasym/dPVasym).

    ``beta0..beta3`` are the standard MS-bar coefficients (beta0 = 11 - 2 nf/3 = 9 at
    nf = 3, beta1 = 102 - 38 nf/3, ...); the notebook writes them divided by (4 pi)^n and
    then multiplied back, which cancels, so the plain integer-coefficient forms are used.
    ``b``, ``c1``, ``c2`` are the sub-asymptotic corrections of arXiv:hep-ph/0105008.
    """
    beta0 = 11.0 - 2.0 * nf / 3.0
    beta1 = 102.0 - 38.0 * nf / 3.0
    beta2 = 2857.0 / 2.0 - 5033.0 * nf / 18.0 + 325.0 * nf**2 / 54.0
    beta3 = 29243.0 - 6946.3 * nf + 405.089 * nf**2 + 1.49931 * nf**3
    b = beta1 / (2.0 * beta0**2)
    c1 = 1.0 / (4.0 * b * beta0**3) * (beta1**2 / beta0 - beta2)
    c2 = (
        beta1**4
        + 4.0 * beta0**3 * beta1 * beta2
        - 2.0 * beta0 * beta1**2 * beta2
        + beta0**2 * (-2.0 * beta1**3 + beta2**2)
        - 2.0 * beta0**4 * beta3
    ) / (32.0 * (b - 1.0) * b * beta0**8)
    return beta0, b, c1, c2


def rnasym(n: int, z: float, mu: float, nf: int = 3) -> float:
    """Asymptotic leading-renormalon coefficient ``r_n``, arXiv:2305.05212 Eq. (12).

    ``r_n = N_m |z mu| (beta0/2 pi)^n Gamma(n+1+b)/Gamma(1+b) (1 + b c1/(n+b) + ...)``.
    In the matching only ``r_0`` is used (``rnasym(0, 1, mu, nf) * alpha_s`` is the
    single-power subtraction that stops the exponential double counting the O(alpha_s)
    renormalon already in the fixed-order kernel).
    """
    from math import gamma  # local: gamma of a non-integer argument, no numpy equivalent

    beta0, b, c1, c2 = _renormalon_params(nf)
    tail = 1.0 + b * c1 / (n + b) + b * (b - 1.0) * c2 / ((n + b) * (n + b - 1.0))
    return float(
        _NM_RENORMALON[nf]
        * abs(z * mu)
        * (beta0 / (2.0 * np.pi)) ** n
        * gamma(n + 1.0 + b)
        / gamma(1.0 + b)
        * tail
    )


def dPVasym(z: float, mu: float, nf: int, alphas: float) -> float:
    """Principal-value Borel sum of the renormalon series, arXiv:2305.05212 Eq. (13).

    The closed form (LRR.nb) evaluates the PV integral as a sum of exponential-integral
    functions ``E_nu(w)`` at ``w = -2 pi/(alpha_s beta0)``::

        dPVasym = N_m |z mu| e^w (-2 pi/beta0) Re[E_{1+b}(w) + c1 E_b(w) + c2 E_{-1+b}(w)]

    ``E_nu`` at non-integer order and negative real argument is complex (Borel cut), so
    the PV prescription takes the real part; ``mpmath.expint`` supplies ``E_nu``.
    Validated against LRR.nb's stored outputs (dPVasym = 4.9371 at mu = 100 GeV,
    2.8577 at mu = 50 GeV, with the paper's threshold-crossing alpha_s).
    """
    try:
        import mpmath  # local: the only consumer of the non-integer exponential integral
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without mpmath
        raise ModuleNotFoundError(
            "The LRR matching kernels need mpmath for the exponential-integral E_nu; "
            "install the 'analysis' extra (pip install -e '.[analysis]')."
        ) from exc

    beta0, b, c1, c2 = _renormalon_params(nf)
    w = -2.0 * np.pi / (alphas * beta0)
    borel = (
        mpmath.expint(1.0 + b, w)
        + c1 * mpmath.expint(b, w)
        + c2 * mpmath.expint(-1.0 + b, w)
    )
    value = _NM_RENORMALON[nf] * abs(z * mu) * mpmath.e**w * (-2.0 * np.pi / beta0) * mpmath.re(borel)
    return float(value)


def C_z_lrr(
    ksi: float,
    y: float,
    momentum_gev: float,
    zspz: float,
    eps_m: float = 0.005,
    eps: float = 1e-12,
) -> float:
    """Renormalon shape ``C_z(ksi)`` of the LRR matching kernel, arXiv:2305.05212 Eq. (17).

    This is the Fourier transform of the regularized linear-``z`` Wilson-line tail (LRR.nb
    ``Cz``), *without* the ``N_m mu`` normalization -- that scalar rides on ``rsumPV`` and
    ``r0`` instead. ``eps_m`` (GeV) is the paper's ``epsilon_m`` long-distance regulator
    (``z -> z e^{-eps_m|z|}``), 0.005 GeV in the notebook. The partonic momentum is
    ``pz = |y| P_z`` (``y`` is the quasi/integration momentum fraction) and the Wilson-line
    length is ``zs = zspz / P_z`` in GeV^{-1}, so ``pz zs = |y| zspz`` and ``eps_m zs`` are
    the notebook's dimensionless combinations.

    The ``ksi = 1`` limit is finite (``e^{-eps_m zs} pz (1 + eps_m zs + eps_m^2 zs^2)/(eps_m^2 pi)``)
    but the discretization skips ``x = y`` and restores it by the plus prescription, so this
    branch only guards grid points that fall numerically on ``ksi = 1``.
    """
    M = eps_m
    pz = abs(y) * momentum_gev
    zs = zspz / momentum_gev  # z_s in GeV^{-1}: pz*zs = |y|*zspz, M*zs = eps_m*zspz/Pz
    emz = np.exp(-M * zs)

    om = ksi - 1.0  # the notebook's (-1 + ksi)
    if abs(om) <= eps:
        return float(emz * pz * (1.0 + M * zs + M**2 * zs**2) / (M**2 * np.pi))

    phi = pz * zs * (1.0 - ksi)  # = pz zs - pz ksi zs; note pz(-1+ksi)zs = -phi (cos is even)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    denom = (M**2 + pz**2 * om**2) ** 2

    term1 = -(emz * zs * sin_phi) / (np.pi * om)
    term2 = (emz * pz / (np.pi * denom)) * (
        (M**2 - pz**2 * om**2 + M**3 * zs + M * pz**2 * om**2 * zs) * cos_phi
        + pz * om * (2.0 * M + M**2 * zs + pz**2 * om**2 * zs) * sin_phi
    )
    return float(term1 + term2)


def _plus_prescription_matrix(
    x_grid: np.ndarray, y_grid: np.ndarray, density: DensityFn, eps: float
) -> np.ndarray:
    """The bare plus-prescribed integral of a density -- LRR.nb ``SubMGen``, no alpha_s/C_F/LO.

    Same off-diagonal fill and column-sum-to-zero plus prescription as build_matching_matrix
    (so MCz lines up row-for-row with the fixed-order kernel), but it returns only the NLO
    integral ``sum_y density(x, y) dy`` -- there is no LO identity, no ``alpha_s C_F/(2 pi)``
    prefactor -- because the LRR assembly supplies those (or not) itself.
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    dy = float(np.abs(np.diff(y)[0]))
    matrix = np.zeros((len(x), len(y)))
    diag_rows = np.abs(x[:, None] - y[None, :]).argmin(axis=0)
    for idx, x_val in enumerate(x):
        for idy, y_val in enumerate(y):
            if np.abs(x_val - y_val) <= eps * np.abs(y_val):
                continue
            matrix[idx, idy] = density(x_val, y_val)
    for idy, diag_row in enumerate(diag_rows):
        matrix[int(diag_row), idy] -= np.sum(matrix[:, idy])
    return matrix * dy


def _lrr_improve(
    fixed_order_matrix: np.ndarray,
    x_ls: np.ndarray,
    y_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    zspz: float,
    eps: float,
    *,
    nf: int = 3,
    eps_m: float = 0.005,
    restrict_unit_interval: bool = False,
) -> np.ndarray:
    """Wrap a fixed-order GI hybrid kernel with the LRR resummation (LRR.nb ``MfixGen``).

    ``M_LRR = (M_fix + r0 MCz) . exp(-MCz rsumPV)``. The renormalon shape ``C_z`` is
    discretized with the ``dy/|y|`` measure (bare, no alpha_s/C_F -- the same measure the GI
    hybrid Wilson-line Si term already uses, see ``_da_wilson_line``), and the two scalar
    renormalon numbers are ``rsumPV = dPVasym(1, mu, nf, alpha_s)`` and
    ``r0 = rnasym(0, 1, mu, nf) alpha_s`` at the NLO running coupling. ``restrict_unit_interval``
    zeroes ``C_z`` outside 0 < y < 1, matching a DA kernel's support (see ``_da_matrix``).
    """
    from scipy.linalg import expm  # local: the only matrix-exponential in the package

    def density_cz(x: float, y: float) -> float:
        if restrict_unit_interval and not (eps < y < 1.0 - eps):
            return 0.0
        return C_z_lrr(x / y, y, momentum_gev, zspz, eps_m, eps) / np.abs(y)

    m_cz = _plus_prescription_matrix(x_ls, y_ls, density_cz, eps)
    alpha_s = alphas_nloop(mu, order=1, Nf=nf)
    rsum_pv = dPVasym(1.0, mu, nf, alpha_s)
    r0 = rnasym(0, 1.0, mu, nf) * alpha_s
    return (fixed_order_matrix + r0 * m_cz) @ expm(-m_cz * rsum_pv)


def _require_square_grid(lc_x_ls: np.ndarray, quasi_y_ls: np.ndarray | None) -> np.ndarray:
    """Return the shared grid, rejecting distinct lc/quasi grids the matrix exponential can't take."""
    x = np.asarray(lc_x_ls, dtype=float)
    if quasi_y_ls is None:
        return x
    y = np.asarray(quasi_y_ls, dtype=float)
    if x.shape != y.shape or not np.allclose(x, y, rtol=0.0, atol=1e-12):
        raise ValueError(
            "LRR kernels resum via a matrix exponential and so need lc_x_ls == quasi_y_ls "
            "(a single square grid); pass matching grids or leave lc_x_ls unset."
        )
    return x


def _lrr_from_fixed_order(
    fixed_order_builder: Callable[..., np.ndarray],
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    zspz: float | None,
    *,
    restrict_unit_interval: bool = False,
) -> np.ndarray:
    """Shared body of every GI+LRR kernel below: fixed-order matrix, then the universal resummation.

    Only ``fixed_order_builder`` (the operator's own ``GI_*_hybrid_NLO``) changes between
    kernels -- the leading renormalon is a property of the straight Wilson line, so its ``C_z``
    shape and the scalar numbers ``r0``/``rsumPV`` are operator independent. This is exactly why
    one implementation covers gamma^t, gamma^z, transversity and the meson DA at once.
    """
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid (LRR) matching kernel.")
    x = _require_square_grid(lc_x_ls, quasi_y_ls)
    m_fix = fixed_order_builder(x, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=x, eps=eps, zspz=zspz)
    return _lrr_improve(m_fix, x, x, momentum_gev, mu, float(zspz), eps, restrict_unit_interval=restrict_unit_interval)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gt_quark_PDF_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR GI ``gamma^t`` hybrid kernel: GI_gt fixed order + the Wilson-line renormalon (arXiv:2305.05212)."""
    return _lrr_from_fixed_order(GI_gt_quark_PDF_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gtg5_quark_PDF_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR helicity kernel: same renormalon and (via the shared gamma^t fixed order) coefficient."""
    return _lrr_from_fixed_order(GI_gtg5_quark_PDF_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gz_quark_PDF_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR GI ``gamma^z`` hybrid kernel: GI_gz fixed order (the +2(1-ksi) of Eq. C7) + same renormalon."""
    return _lrr_from_fixed_order(GI_gz_quark_PDF_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gzg5_quark_PDF_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR helicity kernel: same renormalon and (via the shared gamma^z fixed order) coefficient."""
    return _lrr_from_fixed_order(GI_gzg5_quark_PDF_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gtgpg5_quark_PDF_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR GI transversity kernel: GI_gtgpg5 fixed order + the same Wilson-line renormalon.

    The linear divergence is spin independent, so transversity carries the identical C_z / r0 /
    rsumPV as the unpolarized kernels; only the fixed order (arXiv:2208.08008 Eq. 22-23) differs.
    """
    return _lrr_from_fixed_order(GI_gtgpg5_quark_PDF_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz)


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gtg5_DA_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR meson-DA (gamma^t gamma5) kernel: GI_gtg5_DA fixed order + the Wilson-line renormalon.

    The quasi-DA carries the same straight Wilson line and hence the same linear renormalon, so by
    universality the C_z / r0 / rsumPV are reused, restricted to the DA support 0 < y < 1 (as the
    DA fixed-order kernel is). The C_z closed form itself is transcribed for the PDF in arXiv:2305.05212;
    applying it to the DA is the universality assumption, pending a DA-specific reference.
    """
    return _lrr_from_fixed_order(
        GI_gtg5_DA_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz, restrict_unit_interval=True
    )


@kernel_reference("2305.05212", "Eqs. (12)-(17)")
def GI_gzg5_DA_hybrid_LRR_NLO(
    lc_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None
) -> np.ndarray:
    """NLO+LRR meson-DA (gamma^z gamma5) kernel: GI_gzg5_DA fixed order + the same Wilson-line renormalon.

    See GI_gtg5_DA_hybrid_LRR_NLO for the DA universality assumption; only the fixed order differs.
    """
    return _lrr_from_fixed_order(
        GI_gzg5_DA_hybrid_NLO, lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, zspz, restrict_unit_interval=True
    )


# --- self-describing structure for the matching report -----------------------
# A kernel declares HOW its factorization should be rendered, exactly as it declares its
# paper via ``@kernel_reference``. The report reads ``fn.matching_structure`` and renders
# whatever it finds -- it never enumerates kernel families (no ``if is_lrr``/``if is_da``
# in the report). So a new kernel needs no report change: it either inherits a default
# below (matched by the id convention) or overrides by setting ``.matching_structure``
# on its function.
#
# A structure dict has:
#   factorization : the continuum factorization, as a display-equation LaTeX body.
#   result_noun   : English name of the matched distribution ("light-cone PDF").
#   source_noun   : English name of the lattice input ("quasi-PDF").
#   notation      : the notation guidance handed to the formula LLM (bullet lines).
#   extra_structure : optional extra display equation (e.g. an all-orders resummation).
#   extra_note      : optional extra LLM instruction documenting that structure.
# The content lives here (with the kernels) rather than in the report so that adding a
# kernel -- or a whole new structure -- touches only this file.

_PDF_STRUCTURE: dict[str, Any] = {
    "factorization": (
        r"f(x,\mu)=\int\frac{dy}{|y|}\,C^{-1}\!\left(\frac{x}{y},\frac{\mu}{yP_z}\right)"
        r"\tilde f\!\left(y,P_z\right),"
    ),
    "result_noun": "light-cone PDF",
    "source_noun": "quasi-PDF",
    "notation": (
        "- Define notation once: $\\xi=x/y$ and $L=\\ln(4y^2P_z^2/\\mu^2)$.\n"
        "- The coefficient has a plus-prescription at $\\xi=1$ (the code restores it by making "
        "each $y$-column integrate to zero). Reproduce it using the paper's EXACT "
        "plus-prescription notation, copying the bracket structure verbatim from the LaTeX "
        "above: the paper writes $[\\,g(\\xi)\\,]^{D}_{+(x_0)}$ where the subscript $+(x_0)$ marks "
        "the subtraction point ($x_0=1$, i.e. $+(1)$) and the superscript $D$ marks the domain. "
        "Keep that subscript/superscript placement precisely -- do NOT move the $(1)$ into the "
        "superscript or drop the domain. The paper splits the coefficient into more than one "
        "plus-bracket over different domains (e.g. $[0,1]$ and $(-\\infty,\\infty)$): reproduce "
        "exactly that split, and include the paper's definition of $[g]^{D}_{+(x_0)}$ plus any "
        "$\\delta(1-\\xi)$ term.\n"
    ),
    "extra_structure": None,
    "extra_note": None,
}

_DA_STRUCTURE: dict[str, Any] = {
    "factorization": (
        r"\phi(x,\mu)=\int_0^1 dy\,\Big[\delta(x-y)-\frac{\alpha_s C_F}{2\pi}"
        r"\big[V(x,y)\big]_+\Big]\,\tilde\phi\!\left(y,P_z\right)+O(\alpha_s^2),"
    ),
    "result_noun": "light-cone DA",
    "source_noun": "quasi-DA",
    "notation": (
        "- This is a distribution-amplitude kernel. Its density is a genuine two-variable "
        "$V(x,y)$ carrying its own $1/y$ and $1/(1-y)$ poles, integrated with a plain $dy$ "
        "over the DA's support $y\\in[0,1]$. Do NOT introduce $\\xi=x/y$ and do NOT write the "
        "coefficient as a function of $x/y$ alone -- it is not one.\n"
        "- Define the notation the paper itself uses for $V(x,y)$ and its logarithm scale.\n"
        "- The coefficient carries a plus-prescription at $x=y$ (the code restores it by "
        "making each $y$-column integrate to zero over the $x$ grid). Reproduce the paper's "
        "EXACT bracket notation for it, copying the structure verbatim from the LaTeX above, "
        "including the paper's definition of the bracket and its subtraction domain.\n"
    ),
    "extra_structure": None,
    "extra_note": None,
}

# Leading-renormalon resummation: an add-on any fixed-order kernel can carry. It does not
# replace the base factorization -- it wraps the fixed-order matrix in an all-orders
# matrix exponential -- so it is merged onto whichever base (PDF or DA) applies.
_LRR_EXTRA: dict[str, Any] = {
    "extra_structure": (
        r"M_{\mathrm{LRR}}=\left(M_{\mathrm{fix}}+r_0\,M_{C_z}\right)"
        r"\exp\!\left(-\,M_{C_z}\,r_{\mathrm{sumPV}}\right),"
    ),
    "extra_note": (
        "- This kernel does NOT stop at fixed order: on top of the fixed-order matrix "
        "$M_{\\mathrm{fix}}$ it resums the leading Wilson-line renormalon to ALL orders via a "
        "matrix exponential, $M_{\\mathrm{LRR}}=(M_{\\mathrm{fix}}+r_0 M_{C_z})"
        "\\exp(-M_{C_z}\\,r_{\\mathrm{sumPV}})$, where $M_{C_z}$ is built from the renormalon "
        "shape $C_z(\\xi)$ and $r_{\\mathrm{sumPV}}$ is its principal-value Borel sum. Document "
        "this resummation explicitly -- the shape $C_z$, the coefficient $r_0$, the PV sum "
        "$r_{\\mathrm{sumPV}}$, and the matrix exponential -- reading it from the code. Do NOT "
        "present this kernel as a plain fixed-order coefficient.\n"
    ),
}


def _default_matching_structure(kernel_id: str) -> dict[str, Any]:
    """Pick a kernel's structure from its id convention (``..._DA_...`` / ``..._LRR_...``).

    The id already encodes the distribution and the resummation, so the default reads them
    off it: a DA base or a PDF base, plus the renormalon add-on when the id carries ``LRR``.
    A kernel that does not fit the convention can set ``.matching_structure`` directly.
    """
    parts = str(kernel_id).split("_")
    base = dict(_DA_STRUCTURE if any(p in {"DA", "qDA", "gDA"} for p in parts) else _PDF_STRUCTURE)
    if "LRR" in parts:
        base.update(_LRR_EXTRA)
    return base


def _attach_default_structures() -> None:
    """Give every kernel a ``matching_structure`` unless it already declares one.

    Runs once at import over this module's kernel functions (named ``..._NLO``). Keeping
    the family logic here -- next to the kernels -- is what lets the report stay a generic
    reader with no per-kernel branches of its own.
    """
    for _name, _obj in list(globals().items()):
        if not isinstance(_obj, types.FunctionType) or not _name.endswith("_NLO"):
            continue
        if getattr(_obj, "matching_structure", None) is None:
            _obj.matching_structure = _default_matching_structure(_name)


_attach_default_structures()
