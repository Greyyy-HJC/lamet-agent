from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    C_msbar_gz,
    C_msbar_gz_plus,
    _build_pdf_matrix,
    _p_nlo_valence,
    _rgr_from_fixed_order,
)


def _fixed_order(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    return _build_pdf_matrix(
        lc_x_ls,
        momentum_gev,
        mu,
        quasi_y_ls,
        eps,
        coeff=lambda ksi, log_scale, y: C_msbar_gz(ksi, log_scale, eps),
        plus_coeff=lambda ksi, log_scale, y: C_msbar_gz_plus(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale) + 1.0,
    )


def kernel(
    x_out: np.ndarray,
    x_in: np.ndarray,
    *,
    momentum_gev: float,
    scale_gev: float,
    kappa: float = 1.0,
    mu_min_gev: float = 0.6,
    eps: float = 1e-12,
) -> np.ndarray:
    return _rgr_from_fixed_order(
        _fixed_order,
        _p_nlo_valence,
        np.asarray(x_out, dtype=float),
        momentum_gev,
        scale_gev,
        np.asarray(x_in, dtype=float),
        eps,
        None,
        needs_zspz=False,
        takes_zspz=True,
        needs_zpsi=True,
        kappa=kappa,
        mu_min=mu_min_gev,
    )
