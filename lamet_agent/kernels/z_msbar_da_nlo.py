import numpy as np

from .implementation import ZMSbar_da


def kernel(z_fm: np.ndarray | float, mu: float = 2.0) -> np.ndarray:
    return ZMSbar_da(z_fm, mu)
