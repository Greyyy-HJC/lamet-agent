"""Executable manifest contract for continuum and infinite-momentum extrapolation."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import ConstraintSpec, ParameterSpec, RuleViolation, StageParamContract, StageValidationContext, merge_stage_params


def _parameter(summary: str, physics: str, **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _positive_workers(value: Any) -> str | None:
    return None if type(value) is int and value >= 1 else "extrapolation workers must be a positive integer."


def _check_inputs(context: StageValidationContext) -> RuleViolation | None:
    if context.params.get("operation") == "systematics_budget":
        if "main" in context.inputs:
            return None
        return RuleViolation(
            "A systematics_budget job requires an input role named main.",
            f"{context.job_path}.inputs",
            "The generated budget job has no central extrapolation result to combine with systematic branches.",
            ("inputs.main",),
        )
    lightcone = context.inputs.get("lightcone")
    if not isinstance(lightcone, list):
        return RuleViolation(
            "An extrapolation job requires a list input role named lightcone.",
            f"{context.job_path}.inputs",
            f"The effective lightcone input is {lightcone!r}.",
            ("inputs.lightcone",),
        )
    if not lightcone:
        return RuleViolation(
            "An extrapolation job requires at least one perturbative_matching input.",
            f"{context.job_path}.inputs.lightcone",
            "The lightcone input list is empty.",
            ("inputs.lightcone",),
        )
    return None


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="extrapolation",
    summary="Jointly remove lattice-spacing and finite-momentum effects from matched distributions.",
    physics="The continuum and infinite-momentum limits are inferred by fitting controlled powers of a, aPz, and 1/Pz across compatible ensembles and boosts.",
    planning_notes=("The lightcone role is a non-empty list because the extrapolation needs several lattice spacings and/or momenta.",),
    input_roles=("lightcone", "main"),
    schema={
        "allow_order_a": _parameter("Candidate powers of lattice spacing.", "These terms parameterize continuum discretization effects.", expected=list, items=int, examples=([2],), coerce_scalar_to_list=True),
        "allow_order_1overp": _parameter("Candidate inverse-momentum powers.", "These terms parameterize higher-twist finite-momentum corrections.", expected=list, items=int, examples=([2, 4],), coerce_scalar_to_list=True),
        "allow_order_ap": _parameter("Candidate powers of a times momentum.", "These terms capture boost-enhanced cutoff artifacts.", expected=list, items=int, coerce_scalar_to_list=True),
        "allow_order_a_sym": _parameter("Symmetric-sector lattice-spacing powers.", "The symmetric x component may have a distinct allowed discretization expansion.", expected=list, items=int),
        "allow_order_1overp_sym": _parameter("Symmetric-sector inverse-momentum powers.", "The symmetric x component may have a distinct higher-twist expansion.", expected=list, items=int),
        "allow_order_ap_sym": _parameter("Symmetric-sector aPz powers.", "The symmetric x component may have distinct boost-enhanced cutoff artifacts.", expected=list, items=int),
        "fitting_param_xdep": _parameter("x dependence assigned to fit coefficients.", "This controls how smoothly systematic coefficients vary across momentum fraction.", expected=list, coerce_scalar_to_list=True),
        "posterior_prior_error_scale": _parameter("Prior-width scale for extrapolation coefficients.", "The width controls regularization of weakly constrained continuum and momentum corrections.", expected=float),
        "pdep_gev": _parameter("Reference momentum scales used by the fit ansatz.", "Explicit GeV scales make inverse-momentum coefficients dimensionally consistent.", expected=list, items=float, coerce_scalar_to_list=True),
        "sample_error_mode": _parameter("Sample-error propagation mode.", "The mode determines how resampled uncertainty is summarized after the joint fit.", expected=str),
        "workers": _parameter("Parallel fit workers.", "Parallelism changes runtime only.", expected=int, validator=_positive_workers),
    },
    removed={
        "lattice_spacing_allow_order": "was replaced by allow_order_a, for example [2].",
        "momentum_allow_order": "was replaced by allow_order_1overp, for example [2] or [2, 4].",
    },
    constraints=(
        ConstraintSpec(
            "extrapolation.inputs.required",
            ("inputs.lightcone", "inputs.main"),
            "Analysis jobs consume a non-empty lightcone list; generated budget jobs consume main.",
            "A limit fit requires matched results at varying lattice spacings or momenta, while a budget needs its central result.",
            'Set inputs.lightcone to matching job/artifact ids, or inputs.main on a generated systematics budget job.',
            _check_inputs,
        ),
    ),
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved extrapolation context consumed by the shared evaluator."""
    params = merge_stage_params(manifest.stages["extrapolation"].defaults, job.params)
    authored = {key: value for key, value in params.items() if key != "operation"}
    return StageValidationContext(
        stage="extrapolation",
        job_id=job.id,
        job_path=f"stages.extrapolation.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        authored_params=authored,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise diagnostics."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
