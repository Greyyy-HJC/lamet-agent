"""Load the selected kernel and inspect its explicit callable contract."""

from __future__ import annotations

import math
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.kernels import load_kernel, load_kernel_document
from lamet_agent.stages.perturbative_matching.physics import load_data, inspect_callable


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("matching accepts one quasi source")
        return value[0]
    return value


def run(context: ToolContext) -> dict[str, object]:
    """Load one kernel module and store its input/output grid summary."""
    data = load_data(_one(context.inputs["quasi"]))
    if np.iscomplexobj(data.values):
        component = str(data.attrs.get("component", "re")).lower()
        data = data.imag if component in {"im", "imag", "imaginary"} else data.real
        attrs = data.attrs
        attrs["matching_component"] = "im" if component in {"im", "imag", "imaginary"} else "re"
    momentum = data.attrs.get("momentum_gev")
    if not isinstance(momentum, (int, float)) or isinstance(momentum, bool) or not math.isfinite(float(momentum)) or not float(momentum) > 0:
        raise ValueError("quasi input requires a finite positive momentum_gev")
    if "x" not in data.dims or len(data.coords["x"]) < 1:
        raise ValueError("quasi input requires a nonempty x coordinate")
    root = context.state.get("kernel_root")
    kernel = load_kernel(context.params["kernel_id"], root=root)
    parameter_values = dict(context.params["kernel_parameters"])
    if "rgr_kappa" in parameter_values:
        parameter_values["kappa"] = parameter_values.pop("rgr_kappa")
    if "rgr_mu_min_gev" in parameter_values:
        parameter_values["mu_min_gev"] = parameter_values.pop("rgr_mu_min_gev")
    if "hybrid" in context.params:
        parameter_values["zs_fm"] = context.params["hybrid"]["zs_fm"]
    parameter_names, required = inspect_callable(kernel, parameter_values=parameter_values)
    document = load_kernel_document(context.params["kernel_id"], root=root)
    attrs = data.attrs
    required_tokens = {
        "gfix": attrs.get("gfix"),
        "kernel_operator": attrs.get("kernel_operator"),
        "target_observable": str(attrs.get("target_observable", "")).upper(),
        "renormalization_scheme": attrs.get("renormalization_scheme"),
    }
    tokens = set(context.params["kernel_id"].split("_"))
    missing_tokens = [key for key, value in required_tokens.items() if not isinstance(value, str) or not value or value not in tokens]
    if missing_tokens:
        raise ValueError(f"kernel id does not match quasi provenance fields: {missing_tokens}")
    context.state["kernel"] = kernel
    context.state["quasi"] = data
    context.state["kernel_inspection"] = {"kernel_id": context.params["kernel_id"], "parameters": parameter_names, "required": required, "x_count": len(data.coords.get("x", [])), "dims": data.dims, "momentum_gev": float(momentum), "document": document}
    return {"summary": f"loaded kernel {context.params['kernel_id']}", "metrics": {key: value for key, value in context.state["kernel_inspection"].items() if key != "document"}, "state_keys": ["kernel", "quasi", "kernel_inspection"], "artifacts": []}
