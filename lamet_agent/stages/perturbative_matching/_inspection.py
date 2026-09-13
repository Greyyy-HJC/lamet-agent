"""Load the selected kernel and inspect its explicit callable contract."""

from __future__ import annotations

import inspect
import math

from lamet_agent.agent import ToolContext
from lamet_agent.kernels import load_kernel, load_kernel_document, matching_kernel_id
from lamet_agent.stages.perturbative_matching.contract import _kernel_parameter_issues
from lamet_agent.stages.perturbative_matching.physics import inspect_callable, load_data, select_output_component


def _one(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("matching accepts one quasi source")
        return value[0]
    return value


def run(context: ToolContext) -> dict[str, object]:
    """Load one kernel module and store its input/output grid summary."""
    data = load_data(_one(context.inputs["quasi"]))
    source_component = str(data.attrs.get("source_component", "")).strip().lower()
    if source_component not in {"re", "im", "both"}:
        raise ValueError("quasi input requires source_component='re', 'im', or 'both'")
    data = select_output_component(data)
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
        order=str(context.params.get("order", "nlo")),
        resummation=str(context.params.get("resummation", "")),
    )
    context.params["kernel_id"] = kernel_id
    scheme = str(data.attrs["renormalization_scheme"]).strip().lower()
    kernel = load_kernel(kernel_id, root=root)
    parameter_values = dict(context.params["kernel_parameters"])
    if "zs_fm" in parameter_values:
        raise ValueError("kernel_parameters.zs_fm is supplied by upstream renormalization attrs")
    kernel_uses_zs = "zs_fm" in inspect.signature(kernel).parameters
    scheme_uses_zs = scheme == "hybrid"
    if kernel_uses_zs != scheme_uses_zs:
        expected = "include" if scheme_uses_zs else "omit"
        raise ValueError(f"kernel '{kernel_id}' must {expected} zs_fm for scheme {scheme!r}")
    if scheme_uses_zs:
        switch = data.attrs.get("zs_fm")
        if (
            not isinstance(switch, (int, float))
            or isinstance(switch, bool)
            or not math.isfinite(float(switch))
            or switch <= 0
        ):
            raise ValueError("hybrid quasi input requires a finite positive zs_fm attr")
        parameter_values["zs_fm"] = float(switch)
    issues = _kernel_parameter_issues(kernel, parameter_values)
    if issues:
        raise ValueError("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
    parameter_names, required = inspect_callable(kernel, parameter_values=parameter_values)
    document = load_kernel_document(kernel_id, root=root)
    context.state["kernel"] = kernel
    context.state["quasi"] = data
    context.state["kernel_inspection"] = {
        "kernel_id": kernel_id,
        "scheme": scheme,
        "parameters": parameter_names,
        "required": required,
        "x_count": len(data.coords.get("x", [])),
        "dims": data.dims,
        "momentum_gev": float(momentum),
        "source_component": source_component,
        "output_component": data.attrs["output_component"],
        "document": document,
    }
    return {
        "summary": f"loaded kernel {kernel_id}",
        "metrics": {key: value for key, value in context.state["kernel_inspection"].items() if key != "document"},
        "state_keys": ["kernel", "quasi", "kernel_inspection"],
        "artifacts": [],
    }
