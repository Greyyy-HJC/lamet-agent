"""Kernel loading and orientation-preserving matrix application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_kernel


def load_data(value: Any) -> EnsembleData:
    """Load one quasi-distribution source."""
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, Path):
        if value.suffix.lower() != ".nc":
            raise ValueError("matching input must be a .nc artifact")
        return EnsembleData.from_netcdf(value)
    raise TypeError("matching input is neither EnsembleData nor a NetCDF Path")


def inspect_callable(kernel, *, parameter_values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate explicit kernel keyword parameters against its signature."""
    import inspect

    signature = inspect.signature(kernel)
    parameters = list(signature.parameters.values())
    if len(parameters) < 4 or parameters[0].name != "x_out" or parameters[1].name != "x_in" or parameters[2].name != "momentum_gev" or parameters[3].name != "scale_gev":
        raise TypeError("kernel signature must be kernel(x_out, x_in, *, momentum_gev, scale_gev, ...)")
    if any(parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in parameters[2:]):
        raise TypeError("kernel momentum, scale, and specific parameters must be keyword-only")
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        raise TypeError("kernel-specific parameters must be explicit, not **kwargs")
    required = [parameter.name for parameter in parameters[2:] if parameter.default is inspect.Parameter.empty and parameter.name not in {"momentum_gev", "scale_gev"}]
    accepted = [parameter.name for parameter in parameters[2:] if parameter.name not in {"momentum_gev", "scale_gev"}]
    missing = [name for name in required if name not in parameter_values]
    unexpected = [name for name in parameter_values if name not in accepted]
    if missing or unexpected:
        raise ValueError(f"kernel parameters missing={missing}, unexpected={unexpected}")
    return accepted, required


def apply_matrix(data: EnsembleData, matrix: np.ndarray, x_out: list[float]) -> EnsembleData:
    """Apply ``matched[...,i]=sum_j matrix[i,j]*quasi[...,j]`` sample-wise."""
    if "x" not in data.dims:
        raise ValueError("matching input must have x dimension")
    x_axis = data.array.dims.index("x") - 1
    moved = np.moveaxis(np.asarray(data.values), x_axis + 1, -1)
    transformed = np.einsum("oi,...i->...o", matrix, moved)
    transformed = np.moveaxis(transformed, -1, x_axis + 1)
    coords = dict(data.coords)
    coords["x"] = list(x_out)
    attrs = data.attrs
    return EnsembleData(data.ensemble, data.resample, [sample for sample in transformed], data.dims, coords, attrs=attrs, name="matched_distribution")
