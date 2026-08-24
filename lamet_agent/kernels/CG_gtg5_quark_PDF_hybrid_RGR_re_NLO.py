from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    C_hybrid,
    GEV_FM,
    _p_nlo_full_helicity,
    _pdf_density,
    _rgr_from_fixed_order,
    build_matching_matrix,
)

def _fixed_order(lc_x_ls: np.ndarray, momentum_gev: float, mu: float=2.0, quasi_y_ls: np.ndarray | None=None, eps: float=1e-12, zspz: float | None=None) -> np.ndarray:
    if zspz is None:
        raise ValueError('`zspz` is required for the hybrid matching kernel.')
    z = float(zspz)
    return build_matching_matrix(lc_x_ls, mu, quasi_y_ls, eps, density=_pdf_density(lambda ksi, log_scale, y: C_hybrid(ksi, log_scale, y, z, eps), momentum_gev, mu))

def kernel(x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float, kappa: float=1.0, mu_min_gev: float=0.6, eps: float=1e-12) -> np.ndarray:
    return _rgr_from_fixed_order(_fixed_order, _p_nlo_full_helicity, np.asarray(x_out, dtype=float), momentum_gev, scale_gev, np.asarray(x_in, dtype=float), eps, zs_fm * momentum_gev / GEV_FM, needs_zspz=True, takes_zspz=True, needs_zpsi=False, kappa=kappa, mu_min=mu_min_gev)
