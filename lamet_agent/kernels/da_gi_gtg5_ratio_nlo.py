from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    V_qq_h,
    _da_matrix,
    _da_wilson_line,
)


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, eps: float = 1e-12
) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    return _da_matrix(
        lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, coefficient=V_qq_h, wilson_line=_da_wilson_line("ratio", None, eps)
    )
