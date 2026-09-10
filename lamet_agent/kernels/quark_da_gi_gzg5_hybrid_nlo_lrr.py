from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    GEV_FM,
    V_qq_p,
    _da_matrix,
    _da_wilson_line,
    _lrr_from_fixed_order,
)


def _fixed_order(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float = 2.0,
    quasi_y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
    zspz: float | None = None,
) -> np.ndarray:
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, coefficient=V_qq_p, wilson_line=_da_wilson_line("hybrid", zspz, eps)
    )


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float, eps: float = 1e-12
) -> np.ndarray:
    return _lrr_from_fixed_order(
        _fixed_order,
        np.asarray(x_out, dtype=float),
        momentum_gev,
        scale_gev,
        np.asarray(x_in, dtype=float),
        eps,
        zs_fm * momentum_gev / GEV_FM,
        restrict_unit_interval=True,
    )
