"""Stage-local guidance and validation for Fourier-transform jobs."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params


STAGE_SKILL = """
Fourier transformation extends finite coordinate-space matrix elements with the
configured asymptotic model, transforms every resampled sample, and preserves
the sample axis in an EnsembleData(x) output. Fit ranges are selected once from
sample-average tail-fit diagnostics over the configured zmin/zmax grid. After
that range is fixed, scheme_scan.model_average controls per-sample averaging
over fit-model candidates defined by order and posterior_prior_error_scale;
the method argument is a fixed theory choice and is not scanned.
""".strip()

TOOL_CATALOG = {
    "load_renormalized_matrix_element_samples": "Load the external NetCDF source for a partial run; skip this for an in-memory upstream input.",
    "run_fourier_transform": "Run tail fits, Fourier transform, plots, and write the job NetCDF to store['output']; the runner writes one stage report after all Fourier jobs.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    if set(job.inputs) != {"input"}:
        return ["A fourier_transform job requires exactly one input role."]
    params = merge_stage_params(manifest.stages["fourier_transform"].defaults, job.params)
    params = {**derive_job_kinematics(manifest, job), **params}
    missing = [key for key in ("order", "coord_unit", "y_grid", "momentum_gev") if key not in params]
    if "sector" not in params and "part" not in params:
        missing.append("sector")
    if missing:
        return [f"Fourier job {job.id!r} is missing parameters: {missing}"]
    orders = params["order"] if isinstance(params["order"], list) else [params["order"]]
    if any(order not in {"LA", "NLA"} for order in orders):
        return ["Fourier order must be 'LA' or 'NLA'."]
    sectors = {"pdf": {"valence", "total", "full", "sea"}, "da": {"full"}, "gpd": {"valence", "total", "full", "sea"}}
    if "sector" in params and str(params["sector"]).lower() not in sectors[manifest.metadata.target_observable]:
        return [f"Fourier sector must be one of {sorted(sectors[manifest.metadata.target_observable])}."]
    if "sector" not in params and params.get("part") not in {"re", "im", "both"}:
        return ["Fourier part must be 're', 'im', or 'both'."]
    return []
