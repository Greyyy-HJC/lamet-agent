"""Executable manifest contract for perturbative matching."""

from __future__ import annotations

from typing import Any

import numpy as np

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import ConstraintSpec, ParameterSpec, RuleViolation, StageParamContract, StageValidationContext, merge_stage_params, resolve_stage_params


def _parameter(summary: str, physics: str, **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _kernel_field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _scheme_message(value: Any) -> str | None:
    if value in {"ratio", "hybrid", "msbar"}:
        return None
    return "perturbative_matching scheme must be 'ratio', 'hybrid', or 'msbar'."


def _grid_message(name: str):
    def validate(value: Any) -> str | None:
        from lamet_agent.stages.matching.functions import resolve_grid_spec

        if isinstance(value, dict):
            if not {"start", "stop"}.issubset(value):
                return f"{name} object requires start and stop."
            selectors = {key for key in ("num", "step") if key in value}
            if len(selectors) != 1:
                return f"{name} object requires exactly one of num or step."
        try:
            grid = np.asarray(resolve_grid_spec(value, name=name), dtype=float)
        except (TypeError, ValueError, KeyError) as exc:
            return str(exc)
        if not np.all(np.isfinite(grid)):
            return f"{name} must contain only finite values."
        if name == "quasi_y_ls":
            if grid.size < 2:
                return "quasi_y_ls must resolve to at least 2 points."
            if np.any(np.isclose(grid, 0.0)):
                return "quasi_y_ls must not contain zero because matching kernels carry a 1/y measure."
            spacing = np.diff(grid)
            if not np.allclose(spacing, spacing[0]):
                return "quasi_y_ls must be uniformly spaced."
        elif grid.size < 1:
            return "lc_x_ls must resolve to at least 1 point."
        return None

    return validate


def _violation(context: StageValidationContext, message: str, *, parameter: str, cause: str, path: str | None = None) -> RuleViolation:
    return RuleViolation(message, path or context.parameter_path(parameter), cause, (parameter,))


def _effective_kernel_id(context: StageValidationContext) -> Any:
    if context.params.get("kernel_id") is not None:
        return context.params["kernel_id"]
    kernels = [
        item
        for item in context.resources.get("kernels", [])
        if _kernel_field(item, "stage") in {"matching", "perturbative_matching"}
        and _kernel_field(item, "kernel_id")
    ]
    return _kernel_field(kernels[0], "kernel_id") if len(kernels) == 1 else None


def _check_input(context: StageValidationContext) -> RuleViolation | None:
    if set(context.inputs) == {"quasi"}:
        return None
    return _violation(
        context,
        "A perturbative_matching job requires exactly one quasi input role.",
        parameter="inputs.quasi",
        path=f"{context.job_path}.inputs",
        cause=f"The effective input roles are {sorted(context.inputs)}.",
    )


def _check_momentum(context: StageValidationContext) -> RuleViolation | None:
    if context.params.get("momentum_gev") is not None:
        return None
    return _violation(
        context,
        f"Matching job {context.job_id!r} has no derivable physical momentum.",
        parameter="derived.momentum_gev",
        path=f"{context.job_path}.inputs",
        cause="The upstream source does not provide momentum, volume, and lattice_spacing_fm.",
    )


def _check_kernel(context: StageValidationContext) -> RuleViolation | None:
    from lamet_agent.stages.matching.functions import resolve_kernel_id

    kernel_id = _effective_kernel_id(context)
    scheme = context.params.get("scheme")
    if kernel_id is None:
        return _violation(
            context,
            f"perturbative_matching job {context.job_id!r} is missing kernel_id.",
            parameter="kernel_id",
            cause="The job does not select a kernel and the manifest does not declare exactly one matching kernel to infer.",
        )
    if scheme not in {"ratio", "hybrid", "msbar"}:
        return None
    declaration = next(
        (item for item in context.resources.get("kernels", []) if _kernel_field(item, "kernel_id") == kernel_id),
        None,
    )
    if declaration is None:
        return _violation(
            context,
            f"Matching kernel {kernel_id!r} is not declared in inputs.kernels.",
            parameter="kernel_id",
            cause="The selected identifier has no manifest kernel declaration.",
        )
    if _kernel_field(declaration, "stage") not in {"matching", "perturbative_matching"}:
        return _violation(
            context,
            f"Matching kernel {kernel_id!r} is not declared for perturbative_matching.",
            parameter="kernel_id",
            cause=f"The declaration stage is {_kernel_field(declaration, 'stage')!r}.",
        )
    try:
        resolve_kernel_id(str(kernel_id), str(scheme))
    except ValueError as exc:
        return _violation(context, str(exc), parameter="kernel_id", cause="The kernel registry or encoded scheme rejects this selection.")
    return None


def _check_hybrid(context: StageValidationContext) -> RuleViolation | None:
    from lamet_agent.stages.matching.functions import is_hybrid_kernel

    kernel_id = _effective_kernel_id(context)
    if kernel_id is None or not is_hybrid_kernel(str(kernel_id)) or "zs_fm" in context.params:
        return None
    return _violation(
        context,
        "A hybrid matching job requires flat parameter zs_fm in stage defaults or job params.",
        parameter="zs_fm",
        cause="The hybrid kernel needs the dimensionless transition scale zs_fm * momentum_gev.",
    )


_GRID_FIELDS = {
    "num": _parameter("Number of uniformly spaced grid points.", "Grid density controls numerical discretization of the convolution.", expected=int),
    "start": _parameter("First grid coordinate.", "The endpoints define the represented momentum-fraction domain.", expected=float),
    "step": _parameter("Positive grid spacing.", "This is an alternative to num for a uniform grid.", expected=float),
    "stop": _parameter("Last grid coordinate.", "The endpoints define the represented momentum-fraction domain.", expected=float),
}

_PLOT_FIELDS = {
    "xlim": _parameter("Horizontal plot limits.", "This changes presentation only.", expected=list, items=float),
    "ylim": _parameter("Vertical plot limits.", "This changes presentation only.", expected=list, items=float),
}


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="matching",
    summary="Convert a finite-momentum quasi-distribution to a light-cone distribution with a declared perturbative kernel.",
    physics="The operator, renormalization scheme, momentum, and hybrid transition scale jointly define the perturbative convolution.",
    planning_notes=("kernel_id is inferred when exactly one perturbative_matching kernel is declared.", "momentum_gev is runner-derived from upstream discrete kinematics."),
    input_roles=("quasi",),
    input_role_descriptions={
        "quasi": "One Fourier-stage quasi-distribution to be convolved with the perturbative matching kernel.",
    },
    schema={
        "component": _parameter(
            "Single complex component extracted from the quasi-distribution.",
            "Matching acts on one real-valued quasi channel at a time; the chosen channel must follow the Fourier observable and sector convention.",
            expected=str,
            choices=("re", "im"),
            choice_descriptions={
                "re": "Match the real quasi-distribution channel.",
                "im": "Match the imaginary quasi-distribution channel.",
            },
            required=True,
        ),
        "endpoint_cut": _parameter("Numerical endpoint exclusion.", "The cut regulates finite-grid evaluation near singular convolution endpoints.", expected=float),
        "kernel_id": _parameter("Exact declared matching-kernel identifier.", "The identifier fixes gauge treatment, operator, observable, scheme, and perturbative order; it is inferred when exactly one matching kernel is declared.", expected=str),
        "lc_x_ls": _parameter(
            "Output light-cone momentum-fraction grid.",
            "This explicit grid supplies the kernel rows and may use a different domain or include zero, but it must not be denser than the quasi grid: otherwise some rows miss the plus-prescription subtraction and acquire oscillatory discretization artifacts. Use a numeric list, or an object with start, stop, and exactly one of num or positive step.",
            expected=(list, dict),
            items=float,
            required=True,
            schema=_GRID_FIELDS,
            validator=_grid_message("lc_x_ls"),
        ),
        "mu": _parameter("Perturbative matching scale.", "The truncated kernel retains residual dependence on this factorization scale.", expected=float, unit="GeV", required=True),
        "plot": _parameter("Matching plot settings.", "These settings affect presentation only.", expected=dict, schema=_PLOT_FIELDS),
        "quasi_y_ls": _parameter(
            "Input quasi-distribution integration-grid override.",
            "This explicit grid supplies the kernel columns and its 1/y integration measure. It must be uniform, exclude zero, and stay within the Fourier grid; setting it interpolates every quasi sample. Use a numeric list, or an object with start, stop, and exactly one of num or positive step.",
            expected=(list, dict),
            items=float,
            required=True,
            schema=_GRID_FIELDS,
            validator=_grid_message("quasi_y_ls"),
        ),
        "r": _parameter(
            "Symmetric factorization-scale variation ratio used to generate systematics branches.",
            "When r differs from 1, manifest expansion keeps the central scale mu and clones matching jobs at mu/r and mu*r so residual perturbative-scale dependence can enter the systematic budget.",
            expected=float,
            default=1.0,
        ),
        "scheme": _parameter(
            "Renormalization scheme encoded by kernel_id.",
            "The stage scheme must equal the ratio, hybrid, or msbar token in the exact kernel name because these kernels contain different perturbative coefficients.",
            expected=str,
            required=True,
            choices=("ratio", "hybrid", "msbar"),
            choice_descriptions={
                "ratio": "Use the ratio-scheme matching coefficient.",
                "hybrid": "Use the hybrid coefficient and the dimensionless transition scale zs_fm * Pz/(hbar*c).",
                "msbar": "Use the MS-bar-scheme matching coefficient.",
            },
            validator=_scheme_message,
        ),
        "sector": _parameter(
            "Partonic sector projected after matching.",
            "Sea, valence, singlet, and full choices retain different quark/antiquark combinations on the signed-x domain.",
            expected=str,
            choices=("sea", "valence", "singlet", "full"),
            choice_descriptions={
                "sea": "Retain the sea/antiquark combination.",
                "valence": "Retain quark minus antiquark.",
                "singlet": "Retain quark plus antiquark.",
                "full": "Keep the complete signed-x matched distribution.",
            },
        ),
        "xlim": _parameter("Legacy horizontal plot limits.", "This changes presentation only.", expected=list, items=float),
        "ylim": _parameter("Legacy vertical plot limits.", "This changes presentation only.", expected=list, items=float),
        "zs_fm": _parameter("Hybrid transition distance or uncertainty-bearing systematics value.", "Together with hadron momentum it sets the dimensionless Wilson-line scale in a hybrid kernel. Uncertainty strings are expanded into numerical branches before execution.", expected=(float, str), unit="fm"),
    },
    removed={},
    constraints=(
        ConstraintSpec("matching.inputs.exactly_one", ("inputs.quasi",), "Each job consumes exactly one quasi input.", "One convolution maps one quasi-distribution artifact to one light-cone result.", 'Set inputs to {"quasi": "<fourier job or artifact>"}.', _check_input),
        ConstraintSpec("matching.kinematics.momentum_required", ("inputs.quasi", "derived.momentum_gev"), "Physical momentum must be derivable from the upstream source.", "The matching kernel depends explicitly on the finite hadron momentum in GeV.", "Declare discrete momentum, volume, and lattice_spacing_fm on the upstream source or artifact.", _check_momentum),
        ConstraintSpec("matching.kernel.compatibility", ("kernel_id", "scheme", "inputs.kernels"), "kernel_id must be declared, registered, and encode the selected scheme.", "Different operator and scheme kernels are not interchangeable perturbative coefficients.", "Select one declared kernel and set scheme to its encoded token.", _check_kernel),
        ConstraintSpec("matching.hybrid.zs_required", ("kernel_id", "zs_fm"), "Hybrid kernels require flat zs_fm.", "The kernel depends on zs * Pz as a dimensionless transition scale.", "Declare zs_fm in matching defaults or job params.", _check_hybrid),
    ),
)


def effective_matching_params(manifest: AnalysisManifest, job: StageJob) -> dict[str, Any]:
    """Merge authored params and infer kernel_id when exactly one matching kernel exists."""
    stage = manifest.stages["perturbative_matching"]
    params = resolve_stage_params("perturbative_matching", stage.defaults, job.params)
    if "kernel_id" not in params:
        kernels = [item for item in manifest.kernels if item.stage == "perturbative_matching"]
        if len(kernels) == 1:
            params["kernel_id"] = kernels[0].kernel_id
    return params


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved matching context consumed by the shared evaluator."""
    authored = merge_stage_params(manifest.stages["perturbative_matching"].defaults, job.params)
    params = {**derive_job_kinematics(manifest, job), **effective_matching_params(manifest, job)}
    return StageValidationContext(
        stage="perturbative_matching",
        job_id=job.id,
        job_path=f"stages.perturbative_matching.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        resources={"kernels": list(manifest.kernels)},
        authored_params=authored,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise diagnostics."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]


def matching_grid_warnings(manifest: AnalysisManifest) -> list[str]:
    """Return failures where the light-cone grid is denser than the quasi grid."""
    from lamet_agent.stages.matching.functions import lc_finer_than_quasi_message, resolve_grid_spec

    if "perturbative_matching" not in manifest.metadata.stages:
        return []
    matching = manifest.stages.get("perturbative_matching")
    if matching is None:
        return []
    warnings: list[str] = []
    for job in matching.jobs:
        params = resolve_stage_params("perturbative_matching", matching.defaults, job.params)
        quasi_spec = params.get("quasi_y_ls")
        if quasi_spec is None or params.get("lc_x_ls") is None:
            continue
        try:
            y_ls = np.asarray(resolve_grid_spec(quasi_spec, name="quasi_y_ls"), dtype=float)
            x_ls = np.asarray(resolve_grid_spec(params["lc_x_ls"], name="lc_x_ls"), dtype=float)
        except (TypeError, ValueError, KeyError):
            continue
        message = lc_finer_than_quasi_message(x_ls, y_ls)
        if message:
            warnings.append(f"Matching job {job.id!r}: {message}")
    return warnings
