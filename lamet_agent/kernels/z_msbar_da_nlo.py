import numpy as np

from .implementation import ZMSbar


def kernel(z_fm: np.ndarray | float, mu: float = 2.0) -> np.ndarray:
    return ZMSbar(z_fm, mu=mu, offset=3.5)
