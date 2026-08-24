"""Build sample-bearing matrix-element candidates from inspected data."""

from __future__ import annotations

from typing import Literal

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.stages.correlator_analysis.physics import matrix_element_samples


def run(context: ToolContext, *, method: Literal["ratio", "summation", "qda"], t_min: int, t_max: int, tau_min: int | None = None) -> dict[str, object]:
    """Append a matrix-element candidate with one value per physical coordinate."""
    correlators = context.state.get("correlators")
    lsqfit = context.params["lsqfit"]
    required_scope = {"ratio": "3pt_ratio", "summation": "FH", "qda": "qda_ratio"}[method]
    if required_scope not in lsqfit["fit_scope"] or "independent" not in lsqfit["fit_strategy"]:
        raise ValueError(f"matrix-element method '{method}' is not allowed by fit_scope/fit_strategy")
    if t_min < lsqfit["time_range"]["min"] or t_max > lsqfit["time_range"]["max"] or t_min > t_max:
        raise ValueError("matrix-element fit window is outside the authored time range")
    if not correlators:
        raise RuntimeError("inspect_correlators must run before fit_matrix_element")
    qda_settings = lsqfit if method == "qda" else None
    if qda_settings is not None and {"tmin": t_min, "tmax": t_max} not in lsqfit["pt2_windows"]:
        raise ValueError("qDA fit window must be selected from params.lsqfit.pt2_windows")
    values, z_coordinates, diagnostics = matrix_element_samples(
        correlators,
        method=method,
        t_min=t_min,
        t_max=t_max,
        tau_min=tau_min,
        lsqfit=qda_settings,
        sample_error_mode=str(context.manifest["metadata"]["sample_error_mode"]),
        workers=context.workers,
        _parallel=context._parallel,
    )
    component = context.params["component"]
    if component == "re":
        values = values.real
    elif component == "im":
        values = values.imag
    dims = ["z"]
    coords = {"z": z_coordinates}
    samples = [sample for sample in values]
    source = next(value for value in correlators.values() if value.attrs.get("correlator_type") in {"three_point", "qda"})
    attrs = dict(source.attrs)
    attrs.update({"observable": "matrix_element", "method": method, "n_states": context.params["nstate"][0], "prior_width": lsqfit["prior_width"][0], "sample_error_mode": context.manifest["metadata"]["sample_error_mode"], "units": '{"values":"dimensionless","z":"lattice"}'})
    candidate = EnsembleData(source.ensemble, source.resample, samples, dims, coords, attrs=attrs, name="bare_matrix_element")
    candidates = context.state.setdefault("matrix_element_candidates", [])
    candidate_id = f"matrix_{len(candidates) + 1:03d}"
    candidates.append({"id": candidate_id, "method": method, "observable": "matrix_element", "window": {"t_min": t_min, "t_max": t_max, "tau_min": tau_min}, "component": component, "data": candidate, **diagnostics})
    return {"summary": f"stored matrix-element candidate {candidate_id}", "metrics": {"candidate_id": candidate_id, "coordinate_count": len(candidate.coords[dims[0]]), **{key: diagnostics[key] for key in ("Q", "chi2_dof", "quality_passed") if key in diagnostics}}, "state_keys": ["matrix_element_candidates"], "artifacts": []}
