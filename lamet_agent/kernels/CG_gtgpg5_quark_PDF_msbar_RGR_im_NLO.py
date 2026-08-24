from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    C_ratio_perp,
    _p_nlo_transversity,
    _pdf_density,
    _rgr_from_fixed_order,
    build_matching_matrix,
)

def _fixed_order(lc_x_ls: np.ndarray, momentum_gev: float, mu: float=2.0, quasi_y_ls: np.ndarray | None=None, eps: float=1e-12) -> np.ndarray:
    return build_matching_matrix(lc_x_ls, mu, quasi_y_ls, eps, density=_pdf_density(lambda ksi, log_scale, y: C_ratio_perp(ksi, log_scale, eps), momentum_gev, mu))

def kernel(x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, kappa: float=1.0, mu_min_gev: float=0.6, eps: float=1e-12) -> np.ndarray:
    return _rgr_from_fixed_order(_fixed_order, _p_nlo_transversity, np.asarray(x_out, dtype=float), momentum_gev, scale_gev, np.asarray(x_in, dtype=float), eps, None, needs_zspz=False, takes_zspz=False, needs_zpsi=False, kappa=kappa, mu_min=mu_min_gev)
