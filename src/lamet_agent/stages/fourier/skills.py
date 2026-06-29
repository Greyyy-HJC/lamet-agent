"""Stage-local guidance and validation for Fourier-transform jobs."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


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
    params = {**manifest.stages["fourier_transform"].defaults, **job.params}
    missing = [key for key in ("order", "part", "coord_unit", "y_grid", "pz_gev") if key not in params]
    if missing:
        return [f"Fourier job {job.id!r} is missing parameters: {missing}"]
    orders = params["order"] if isinstance(params["order"], list) else [params["order"]]
    if any(order not in {"LA", "NLA"} for order in orders):
        return ["Fourier order must be 'LA' or 'NLA'."]
    if params["part"] not in {"re", "im", "both"}:
        return ["Fourier part must be 're', 'im', or 'both'."]
    return []
