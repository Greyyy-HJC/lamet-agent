import numpy as np

from .implementation import ZMSbar_pdf


def kernel(z_fm: np.ndarray | float, mu: float = 2.0) -> np.ndarray:
    return ZMSbar_pdf(z_fm, mu)
