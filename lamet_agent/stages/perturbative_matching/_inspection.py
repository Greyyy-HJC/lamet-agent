"""Load the selected kernel and inspect its explicit callable contract."""

from __future__ import annotations

import inspect
import math

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.kernels import load_kernel, load_kernel_document, matching_kernel_id
from lamet_agent.stages.perturbative_matching.physics import load_data, inspect_callable


_COMPONENT_ALIASES = {"re": "re", "real": "re", "im": "im", "imag": "im", "imaginary": "im"}


def _matching_component(resummation_part: str, attrs: dict) -> str:
    """Return the quasi component matched by one resummation choice."""
    if resummation_part == "both":
        return "both"
    required = resummation_part or None
    declared = _COMPONENT_ALIASES.get(str(attrs.get("component", "")).lower())
    if required is not None and declared is not None and declared != required:
        raise ValueError(
            f"resummation_part '{resummation_part}' matches the {required} component "
            f"but the quasi input declares component '{declared}'"
        )
    return required or declared or "re"


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("matching accepts one quasi source")
        return value[0]
    return value


def run(context: ToolContext) -> dict[str, object]:
    """Load one kernel module and store its input/output grid summary."""
    data = load_data(_one(context.inputs["quasi"]))
    component = _matching_component(str(context.params.get("resummation_part", "")), data.attrs)
    if np.iscomplexobj(data.values) and component != "both":
        data = data.imag if component == "im" else data.real
    data.array.attrs["matching_component"] = component
    momentum = data.attrs.get("momentum_gev")
    if (
        not isinstance(momentum, (int, float))
        or isinstance(momentum, bool)
        or not math.isfinite(float(momentum))
        or not float(momentum) > 0
    ):
        raise ValueError("quasi input requires a finite positive momentum_gev")
    if "x" not in data.dims or len(data.coords["x"]) < 1:
        raise ValueError("quasi input requires a nonempty x coordinate")
    root = context.state.get("kernel_root")
    kernel_id = matching_kernel_id(
        data.attrs,
        scheme=str(context.params["scheme"]),
        order=str(context.params.get("order", "nlo")),
        resummation=str(context.params.get("resummation", "")),
        resummation_part=str(context.params.get("resummation_part", "")),
    )
    context.params["kernel_id"] = kernel_id
    kernel = load_kernel(kernel_id, root=root)
    parameter_values = dict(context.params["kernel_parameters"])
    kernel_uses_zs = "zs_fm" in inspect.signature(kernel).parameters
    scheme_uses_zs = context.params["scheme"] == "hybrid"
    if kernel_uses_zs != scheme_uses_zs:
        expected = "include" if scheme_uses_zs else "omit"
        raise ValueError(f"kernel '{kernel_id}' must {expected} zs_fm for scheme {context.params['scheme']!r}")
    if scheme_uses_zs:
        parameter_values.setdefault("zs_fm", context.params["zs_fm"])
    parameter_names, required = inspect_callable(kernel, parameter_values=parameter_values)
    document = load_kernel_document(kernel_id, root=root)
    context.state["kernel"] = kernel
    context.state["quasi"] = data
    context.state["kernel_inspection"] = {
        "kernel_id": kernel_id,
        "parameters": parameter_names,
        "required": required,
        "x_count": len(data.coords.get("x", [])),
        "dims": data.dims,
        "momentum_gev": float(momentum),
        "matching_component": component,
        "document": document,
    }
    return {
        "summary": f"loaded kernel {kernel_id}",
        "metrics": {key: value for key, value in context.state["kernel_inspection"].items() if key != "document"},
        "state_keys": ["kernel", "quasi", "kernel_inspection"],
        "artifacts": [],
    }
