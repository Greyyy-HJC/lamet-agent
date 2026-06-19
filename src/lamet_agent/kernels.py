"""Compact (DRY) NLO unpolarized ``gamma^t`` matching kernels.

This is the deliberately *simple* counterpart of the fully-inlined
``kernels_example.py`` style: instead of copying the whole discretization into
every scheme, a single builder ``_quark_matching_kernel`` does the work once and
the three public schemes differ only by a couple of clearly-marked lines chosen
via the ``scheme`` argument.

    CG_gt_PDF_ratio   C_r   -- Eq. (2.16) of arXiv:2602.11283 (the bare regular part)
    CG_gt_PDF_msbar   C_MS = C_r + finite MSbar conversion        -- Eq. (2.14)
    CG_gt_PDF_hybrid  C_hy = C_r + Wilson-line Si correction       -- Eq. (2.19)-(2.20)

The three differ by exactly:

    off-diagonal       diagonal (plus-prescription row)
    -----------------  ------------------------------------
    ratio    + 0                + 0
    MSbar    + 0.5/|1-xi|       + 0.5 (1 + log) / dy
    hybrid   + delta_hybrid     + 0

Numerically ``CG_gt_PDF_msbar`` reproduces
``kernels_example.unpolarized_matching_kernel_nlo_gT`` to floating-point
rounding (~1e-16).
"""

from __future__ import annotations

from typing import Final

import numpy as np


# --- shared numeric plumbing (alpha_s, Si, colour factors) ------------------
# Kept inline so this module is self-contained (no dependency on other kernel
# files). The kernel *structure* below is what the schemes actually differ in.

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


# --- grid validation --------------------------------------------------------


def _validate_grids(
    x_ls: np.ndarray,
    y_ls: np.ndarray | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return validated (x_grid, y_grid, dy) for a uniform y integration grid.

    ``y_ls`` defaults to ``x_ls`` (quasi- and light-cone PDFs share the grid).
    ``dy`` is the uniform spacing used as the integration measure.
    """
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)

    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep xi=x/y finite.")

    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")

    return x_grid, y_grid, dy


# --- per-element coefficients ----------------------------------------------


def _ratio_regular_entry(xi: float, log_scale: float, eps: float) -> float:
    """Bare ratio-scheme regular coefficient C_r^(1)(xi), Eq. (2.16).

    This is the common backbone of all three schemes. MSbar and hybrid only add
    a small correction on top of it (see ``_quark_matching_kernel``).
    """
    one_minus_xi = 1.0 - xi
    entry = 0.0

    # Splitting-function piece, only inside the physical 0 < xi < 1 window.
    if eps < xi < 1.0 - eps:
        splitting = (1.0 + xi**2) / one_minus_xi
        entry += splitting * log_scale + xi - 1.0

    # arctan/arctanh piece, with the branch chosen by where xi sits relative to 1/2.
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

    # Logarithmic + remaining regular terms. The trailing -1.5/|1-xi| is the
    # bare ratio coefficient; MSbar/hybrid shift it via their own corrections.
    sign_safe_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
    signed_logs = (
        np.sign(xi) * np.log(np.abs(xi) + eps)
        + np.sign(one_minus_xi) * np.log(np.abs(one_minus_xi) + eps)
    )
    entry += (1.0 + xi**2) / sign_safe_denominator * signed_logs
    entry += np.sign(xi) + atan_piece - 1.5 / (np.abs(one_minus_xi) + eps)
    return float(entry)


def _hybrid_delta_entry(xi: float, zspz: float, eps: float) -> float:
    """Hybrid-minus-ratio off-diagonal correction, Eq. (2.20).

    ``zspz`` is the dimensionless Wilson-line length ``z_s * P_z``. This replaces
    the MSbar ``+0.5/|1-xi|`` term by the ``Si(z_s P_z (1-xi))`` sine-integral.
    """
    one_minus_xi = 1.0 - xi
    sign_safe_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
    return 0.5 * (
        1.0 / (np.abs(one_minus_xi) + eps)
        - 2.0 * _sine_integral(one_minus_xi * zspz) / (np.pi * sign_safe_denominator)
    )


# --- shared builder ---------------------------------------------------------


def _quark_matching_kernel(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    *,
    scheme: str,
    zspz: float | None = None,
) -> np.ndarray:
    """Build the NLO unpolarized ``gamma^t`` matching matrix for one scheme.

    The result maps a quasi-PDF on ``y_ls`` to a light-cone PDF on ``x_ls`` via
    ``lightcone = kernel @ quasi``. The discretization (loop + plus prescription)
    is identical for every scheme; only the marked ``scheme``-specific lines differ.
    """
    if scheme not in ("ratio", "msbar", "hybrid"):
        raise ValueError(f"Unknown matching scheme: {scheme!r}.")
    if scheme == "hybrid" and zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")

    x_grid, y_grid, dy = _validate_grids(x_ls, y_ls, eps)
    alpha_s = alphas_nloop(mu, order=1, Nf=3)

    nx, ny = len(x_grid), len(y_grid)
    identity = np.zeros((nx, ny))
    nlo_matrix = np.zeros((nx, ny))
    # For each y column, the x row closest to that y point carries the plus-function.
    diag_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)

    # 1) Off-diagonal (xi != 1) regular coefficients.
    for idx, x_val in enumerate(x_grid):
        for idy, y_val in enumerate(y_grid):
            if np.isclose(x_val, y_val, atol=eps, rtol=0.0):
                identity[idx, idy] = 1.0  # leading-order delta(x - y)

            xi = x_val / y_val
            if np.abs(1.0 - xi) <= eps:
                continue  # the xi = 1 singularity is restored by the plus prescription

            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)
            entry = _ratio_regular_entry(xi, log_scale, eps)
            if scheme == "msbar":
                entry += 0.5 / (np.abs(1.0 - xi) + eps)          # <-- MSbar off-diagonal
            elif scheme == "hybrid":
                entry += _hybrid_delta_entry(xi, float(zspz), eps)  # <-- hybrid off-diagonal

            nlo_matrix[idx, idy] = entry / np.abs(y_val)

    # 2) Diagonal plus-prescription: make every y column integrate to zero, then
    #    add the MSbar-only finite conversion term.
    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy])
        if scheme == "msbar":
            nlo_matrix[int(diag_row), idy] += 0.5 * (
                1.0 + np.log(4.0 * y_grid[idy] ** 2 * pz_gev**2 / mu**2)
            ) / dy

    # 3) Assemble: LO identity minus the NLO correction (times the dy measure).
    return identity - alpha_s * CF / (2.0 * np.pi) * nlo_matrix * dy


# --- public scheme wrappers -------------------------------------------------


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
    return _quark_matching_kernel(x_ls, pz_gev, mu, y_ls, eps, scheme="ratio")


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
    return _quark_matching_kernel(x_ls, pz_gev, mu, y_ls, eps, scheme="msbar")


def CG_gt_PDF_hybrid(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    """NLO hybrid-scheme kernel for the Coulomb-gauge ``gamma^t`` PDF (Eq. 2.19-2.20).

    ``zspz`` is the dimensionless Wilson-line length ``z_s * P_z`` and is required.
    """
    return _quark_matching_kernel(x_ls, pz_gev, mu, y_ls, eps, scheme="hybrid", zspz=zspz)


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


# --- unpolarized gluon PDF --------------------------------------------------


def CG_gluon_PDF_msbar(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO unpolarized **gluon** kernel for the Coulomb-gauge PDF in MSbar.

    Gluon analogue of :func:`CG_gt_PDF_msbar`: same signature and structure, but the
    matching is ``C_A``-proportional and uses the gluon splitting/coefficient
    functions. No Wilson-line (z_s) term -- plain MSbar, no ratio/hybrid scheme.
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

            poly = 2.0 * (1.0 - xi + xi**2) ** 2 / (1.0 - xi)
            cubic = (11.0 - 28.0 * xi + 18.0 * xi**2 - 12.0 * xi**3) / (6.0 * (1.0 - xi))

            entry = 0.0
            if xi > 1.0 + eps:
                entry = poly * np.log(xi / (xi - 1.0)) + cubic
            elif eps < xi < 1.0 - eps:
                entry = poly * (log_scale + np.log(xi * (1.0 - xi))) - (
                    15.0 - 56.0 * xi + 102.0 * xi**2 - 96.0 * xi**3 + 48.0 * xi**4
                ) / (6.0 * (1.0 - xi))
            elif xi < -eps:
                entry = -poly * np.log(xi / (xi - 1.0)) - cubic

            nlo_matrix[idx, idy] = entry / y_norm

    for idy, diag_row in enumerate(diag_rows):
        nlo_matrix[int(diag_row), idy] -= np.sum(nlo_matrix[:, idy]) #! plus function: column sum gives zero

    return identity - alpha_s * CA / (2.0 * np.pi) * nlo_matrix * dy
