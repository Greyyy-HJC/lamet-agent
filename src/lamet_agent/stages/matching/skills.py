"""Stage-local guidance and validation for perturbative-matching jobs."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params
from lamet_agent.stages.matching.functions import is_hybrid_kernel, resolve_kernel_id


STAGE_SKILL = """
Perturbative matching applies the selected NLO kernel matrix independently to
every quasi-PDF sample. The stage-owned scheme is ratio, hybrid, or msbar and
must match the token in the exact declared kernel_id. Hybrid kernels use zs_fm
and momentum_gev to form z_s P_z.

Two grids, both optional and both taken from the manifest. quasi_y_ls is the grid
the kernel integrates over (its columns); it defaults to the grid the Fourier stage
produced, must stay inside that grid's range, must not contain zero (the kernels
carry a 1/y measure), and must be uniformly spaced. Setting it to anything else
linearly interpolates every quasi sample onto it. lc_x_ls is the grid the matched
PDF comes out on (the kernel's rows); it defaults to the quasi grid and is
otherwise unconstrained -- it may contain zero and need not be uniform.
""".strip()

TOOL_CATALOG = {
    "load_quasi_pdf": "Select the requested real/imaginary component from the job's in-memory or external Fourier input.",
    "build_matching_kernel": "Build the manifest-selected NLO matching matrix.",
    "apply_matching": "Apply the kernel sample by sample and write the job NetCDF to store['output'].",
    "plot_matched_pdf": "Plot quasi and matched PDFs.",
    "report_matching_result": "Regenerate an optional per-job English/Chinese report; the runner writes one stage report after all matching jobs.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def effective_matching_params(manifest: AnalysisManifest, job: StageJob) -> dict[str, Any]:
    """Merge matching defaults and job params, inferring kernel_id from inputs.kernels."""
    params = merge_stage_params(manifest.stages["perturbative_matching"].defaults, job.params)
    if "kernel_id" in params:
        return params
    matching_kernels = [item for item in manifest.kernels if item.stage == "perturbative_matching"]
    if len(matching_kernels) == 1:
        params["kernel_id"] = matching_kernels[0].kernel_id
    return params


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    if set(job.inputs) != {"quasi"}:
        return ["A perturbative_matching job requires exactly one quasi input role."]
    params = effective_matching_params(manifest, job)
    params = {**derive_job_kinematics(manifest, job), **params}
    missing = [key for key in ("kernel_id", "momentum_gev", "scheme") if key not in params]
    if missing:
        return [f"Matching job {job.id!r} is missing parameters: {missing}"]
    declaration = next((item for item in manifest.kernels if item.kernel_id == params["kernel_id"]), None)
    if declaration is None:
        return [f"Matching kernel {params['kernel_id']!r} is not declared in inputs.kernels."]
    try:
        resolved = resolve_kernel_id(declaration.kernel_id, str(params["scheme"]))
    except ValueError as exc:
        return [str(exc)]
    if is_hybrid_kernel(resolved) and "zs_fm" not in params:
        return ["A hybrid matching job requires flat parameter zs_fm in stage defaults or job params."]
    return []
