"""Stage-local validation for perturbative-matching jobs."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params
from lamet_agent.stages.matching.functions import is_hybrid_kernel, resolve_kernel_id


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
