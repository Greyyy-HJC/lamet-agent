from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    C_hybrid_gi,
    GEV_FM,
    _pdf_density,
    build_matching_matrix,
)


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float, eps: float = 1e-12
) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    zspz = zs_fm * momentum_gev / GEV_FM
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix(
        lc_x_ls,
        mu,
        quasi_y_ls,
        eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid_gi(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )
