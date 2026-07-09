"""Stage-local guidance and validation for perturbative-matching jobs."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.stages.matching.functions import resolve_kernel_id


STAGE_SKILL = """
Perturbative matching applies the selected NLO kernel matrix independently to
every quasi-PDF sample. The job's logical kernel_id resolves through the matching
kernel declaration and its scheme; hybrid kernels use zs_fm and pz_gev to form
z_s P_z. The x grid must not contain zero.
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
    params = {**manifest.stages["perturbative_matching"].defaults, **job.params}
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
    missing = [key for key in ("kernel_id", "pz_gev", "mu", "component") if key not in params]
    if missing:
        return [f"Matching job {job.id!r} is missing parameters: {missing}"]
    declaration = next((item for item in manifest.kernels if item.kernel_id == params["kernel_id"]), None)
    if declaration is None:
        return [f"Matching kernel {params['kernel_id']!r} is not declared in inputs.kernels."]
    try:
        resolved = resolve_kernel_id(declaration.kernel_id, declaration.scheme)
    except ValueError as exc:
        return [str(exc)]
    if resolved.endswith("_hybrid") and "zs_fm" not in declaration.kernel_parameters:
        return ["A hybrid matching kernel requires kernel_parameters.zs_fm."]
    return []
