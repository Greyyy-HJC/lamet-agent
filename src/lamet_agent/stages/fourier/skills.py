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
For DA only, symmetry_guarantee defaults to true: rotate by exp(+i*z*Pz/2),
discard the rotated imaginary part, rotate the retained real part back by
exp(-i*z*Pz/2), then run the ordinary extension and Fourier transform.
Set it false to use the DA input unchanged. It has no effect for PDF or GPD.
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
    missing = [key for key in ("momentum_gev",) if key not in params]
    if missing:
        return [f"Fourier job {job.id!r} is missing parameters: {missing}"]
    orders = params["order"] if isinstance(params.get("order"), list) else [params.get("order")] if "order" in params else []
    if orders and any(order not in {"LA", "NLA"} for order in orders):
        return ["Fourier order must be 'LA' or 'NLA'."]
    sectors = {"pdf": {"sea", "valence", "singlet", "full"}, "da": {"full"}, "gpd": {"sea", "valence", "singlet", "full"}}
    if "sector" in params and str(params["sector"]).lower() not in sectors[manifest.metadata.target_observable]:
        return [f"Fourier sector must be one of {sorted(sectors[manifest.metadata.target_observable])}."]
    if "sector" not in params and "part" in params and params.get("part") not in {"re", "im", "both"}:
        return ["Fourier part must be 're', 'im', or 'both'."]
    if "symmetry_guarantee" in params and not isinstance(params["symmetry_guarantee"], bool):
        return ["Fourier symmetry_guarantee must be a boolean."]
    return []
