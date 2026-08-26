"""Align renormalization inputs and report coordinate coverage."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.stages.renormalization.physics import load_data, physical_z_coordinates


def run(context: ToolContext) -> dict[str, object]:
    """Load and align target, denominator, zR, or reference inputs."""
    aligned = {}
    for role, value in context.inputs.items():
        if isinstance(value, list):
            aligned[role] = [physical_z_coordinates(load_data(item)) for item in value]
        elif value.__class__.__name__ in {"EnsembleData", "WindowsPath", "PosixPath"}:
            data = load_data(value)
            if role == "reference" and data.attrs.get("coord_unit") is None:
                attrs = data.attrs
                if attrs.get("z_unit") != "fm":
                    raise ValueError("self-renormalization reference must declare z_unit='fm'")
                attrs["coord_unit"] = "fm"
                data = EnsembleData(data.ensemble, data.resample, [sample for sample in data.values], data.dims, data.coords, attrs=attrs, name=data.name)
            aligned[role] = physical_z_coordinates(data)
    if not aligned:
        raise ValueError("no numerical renormalization input was supplied")
    context.state["aligned_inputs"] = aligned
    coverage = {}
    for role, data in aligned.items():
        values = data if isinstance(data, list) else [data]
        coverage[role] = [{"dims": item.dims, "coords": {dim: len(coords) for dim, coords in item.coords.items()}, "n_sample": item.n_sample} for item in values]
    return {"summary": f"aligned {len(aligned)} renormalization inputs", "metrics": coverage, "state_keys": ["aligned_inputs"], "artifacts": []}
