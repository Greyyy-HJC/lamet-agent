"""Helpers for compiling and validating inline hard-kernel callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import FunctionType
from typing import Any

import numpy as np
import scipy

from lamet_agent.errors import KernelLoadError
from lamet_agent.schemas import KernelSpec


def load_kernel(spec: KernelSpec) -> Callable[..., Any]:
    """Compile an inline kernel definition and validate the declared callable."""
    namespace: dict[str, Any] = {"np": np, "scipy": scipy}
    try:
        compiled = compile(spec.source, "<manifest-kernel>", "exec")
        exec(compiled, namespace, namespace)
    except Exception as exc:  # pragma: no cover - exact exception type is input dependent.
        raise KernelLoadError(f"Failed to compile inline kernel source: {exc}") from exc
    candidate = namespace.get(spec.callable_name)
    if not isinstance(candidate, FunctionType):
        raise KernelLoadError(
            f"Kernel callable {spec.callable_name!r} was not defined by the provided source."
        )
    validate_kernel_signature(candidate)
    return candidate


def validate_kernel_signature(kernel: Callable[..., Any]) -> None:
    """Validate the expected hard-kernel interface."""
    signature = inspect.signature(kernel)
    parameters = list(signature.parameters.values())
    positional_like = [
        parameter
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_variadic = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    if len(positional_like) < 2 and not accepts_variadic:
        raise KernelLoadError(
            "Kernel callable must accept at least two positional arguments: coordinate_axis and values."
        )
