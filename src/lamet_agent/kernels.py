"""Reference perturbative kernels for smoke tests."""

from __future__ import annotations

import numpy as np


def identity_kernel(x: np.ndarray) -> np.ndarray:
    """Return identity mapping for kernel smoke tests."""
    return x


def damped_identity_kernel(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return a simple damped kernel transform for fake-data checks."""
    return x * np.exp(-alpha * np.abs(x))

# lightcone = np.dot(matrix, quasi)
def qpdf_kernel(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(x_array), len(y_array)))
    return matrix
