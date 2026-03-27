"""Reference hard-kernel module for users who want to author kernels outside JSON first.

This file is not imported by the default demo manifest. It exists to show the expected
callable shape before the function is inlined into manifest JSON.

Example usage:
    from examples.demo_kernel import demo_hard_kernel
"""

from __future__ import annotations

import numpy as np


def demo_hard_kernel(momentum_axis: np.ndarray, values: np.ndarray, metadata: dict) -> np.ndarray:
    """Apply a simple damping factor to demo Fourier-space values."""
    scale = float(metadata.get("matching_scale", 1.0))
    width = float(metadata.get("kernel_width", 0.25))
    return scale * values / (1.0 + width * np.square(momentum_axis))
