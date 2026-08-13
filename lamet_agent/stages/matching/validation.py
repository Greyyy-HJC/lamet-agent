"""Stage-local validation for perturbative-matching jobs."""

from __future__ import annotations

from typing import Any

import numpy as np

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params
from lamet_agent.stages.matching.functions import (
    is_hybrid_kernel,
    lc_finer_than_quasi_message,
    resolve_grid_spec,
    resolve_kernel_id,
)


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


def matching_grid_warnings(manifest: AnalysisManifest) -> list[str]:
    """Return matching-grid density issues when ``lc_x_ls`` is denser than the quasi grid.

    ``validate`` prints these as a boxed warning and then fails. ``run`` prints the
    same warning without failing; kernel construction still rejects the density.

    Resolves grids from matching ``quasi_y_ls`` / ``lc_x_ls`` or, if quasi is omitted,
    the upstream Fourier job's ``y_grid``. Jobs whose grids cannot be read from the
    manifest (artifact-only quasi with no ``quasi_y_ls``) are skipped.
    """
    if "perturbative_matching" not in manifest.metadata.stages:
        return []
    matching = manifest.stages.get("perturbative_matching")
    if matching is None:
        return []
    fourier = manifest.stages.get("fourier_transform")
    fourier_jobs = {job.id: job for job in (fourier.jobs if fourier is not None else [])}
    warnings: list[str] = []
    for job in matching.jobs:
        params = merge_stage_params(matching.defaults, job.params)
        quasi_spec = params.get("quasi_y_ls")
        if quasi_spec is None:
            quasi_ref = job.inputs.get("quasi")
            if isinstance(quasi_ref, str) and quasi_ref in fourier_jobs and fourier is not None:
                ft_params = merge_stage_params(fourier.defaults, fourier_jobs[quasi_ref].params)
                quasi_spec = ft_params.get("y_grid")
        if quasi_spec is None:
            continue
        lc_spec = params.get("lc_x_ls")
        if lc_spec is None:
            continue
        try:
            y_ls = np.asarray(resolve_grid_spec(quasi_spec, name="quasi_y_ls"), dtype=float)
            x_ls = np.asarray(resolve_grid_spec(lc_spec, name="lc_x_ls"), dtype=float)
        except (TypeError, ValueError, KeyError):
            continue
        message = lc_finer_than_quasi_message(x_ls, y_ls)
        if message:
            warnings.append(f"Matching job {job.id!r}: {message}")
    return warnings
