"""Load the selected kernel and inspect its explicit callable contract."""

from __future__ import annotations

import math
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.kernels import load_kernel, load_kernel_document
from lamet_agent.stages.perturbative_matching.physics import load_data, inspect_callable


_COMPONENT_ALIASES = {"re": "re", "real": "re", "im": "im", "imag": "im", "imaginary": "im"}
_ORDER_COMPONENTS = {"rgr_nlo_re": "re", "rgr_nlo_im": "im"}


def _matching_component(order: str, attrs: dict) -> str:
    """Return the quasi component matched by one kernel order."""
    required = _ORDER_COMPONENTS.get(order)
    declared = _COMPONENT_ALIASES.get(str(attrs.get("component", "")).lower())
    if required is not None and declared is not None and declared != required:
        raise ValueError(
            f"kernel order '{order}' matches the {required} component "
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
    component = _matching_component(str(context.params["order"]), data.attrs)
    if np.iscomplexobj(data.values):
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
    kernel = load_kernel(context.params["kernel_id"], root=root)
    parameter_values = dict(context.params["kernel_parameters"])
    if context.params["scheme"] == "hybrid":
        parameter_values.setdefault("zs_fm", context.params["zs_fm"])
    parameter_names, required = inspect_callable(kernel, parameter_values=parameter_values)
    document = load_kernel_document(context.params["kernel_id"], root=root)
    attrs = data.attrs
    required_tokens = {
        "gfix": attrs.get("gfix"),
        "kernel_operator": attrs.get("kernel_operator"),
        "target_observable": attrs.get("target_observable"),
        "renormalization_scheme": attrs.get("renormalization_scheme"),
    }
    tokens = set(context.params["kernel_id"].split("_"))
    missing_tokens = [
        key
        for key, value in required_tokens.items()
        if not isinstance(value, str) or not value or value.lower() not in tokens
    ]
    if missing_tokens:
        raise ValueError(f"kernel id does not match quasi provenance fields: {missing_tokens}")
    context.state["kernel"] = kernel
    context.state["quasi"] = data
    context.state["kernel_inspection"] = {
        "kernel_id": context.params["kernel_id"],
        "parameters": parameter_names,
        "required": required,
        "x_count": len(data.coords.get("x", [])),
        "dims": data.dims,
        "momentum_gev": float(momentum),
        "matching_component": component,
        "document": document,
    }
    return {
        "summary": f"loaded kernel {context.params['kernel_id']}",
        "metrics": {key: value for key, value in context.state["kernel_inspection"].items() if key != "document"},
        "state_keys": ["kernel", "quasi", "kernel_inspection"],
        "artifacts": [],
    }
