import numpy as np

from .implementation import ZMSbar


def kernel(z_fm: np.ndarray | float, mu: float = 2.0, order: int = 0, Nf: int = 3) -> np.ndarray:
    return ZMSbar(z_fm, mu=mu, offset=3.5, order=order, Nf=Nf)
