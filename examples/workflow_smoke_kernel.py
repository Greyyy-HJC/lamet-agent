"""Reference hard-kernel module for the workflow smoke example.

Purpose:
    Show the expected callable shape before the function is inlined into
    manifest JSON.

Inputs and outputs:
    - input: `momentum_axis`, transformed values, and manifest metadata
    - output: one NumPy array with the same shape as `values`

Example usage:
    from examples.workflow_smoke_kernel import workflow_smoke_kernel
"""

from __future__ import annotations

import numpy as np


def workflow_smoke_kernel(momentum_axis: np.ndarray, values: np.ndarray, metadata: dict) -> np.ndarray:
    """Apply a simple damping factor to the workflow-smoke Fourier-space values."""
    scale = float(metadata.get("matching_scale", 1.0))
    width = float(metadata.get("kernel_width", 0.25))
    return scale * values / (1.0 + width * np.square(momentum_axis))
