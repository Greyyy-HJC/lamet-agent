from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    C_msbar,
    C_msbar_plus,
    _build_pdf_matrix,
)

def kernel(x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, eps: float=1e-12) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    return _build_pdf_matrix(lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, coeff=lambda ksi, log_scale, y: C_msbar(ksi, log_scale, eps), plus_coeff=lambda ksi, log_scale, y: C_msbar_plus(ksi, log_scale, eps), diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale))
