"""Executable manifest contract for continuum and infinite-momentum extrapolation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import ConstraintSpec, ParameterSpec, RuleViolation, StageParamContract, StageValidationContext, merge_stage_params


def _parameter(summary: str, physics: str, **kwargs: object) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _x_dependence_message(value: object) -> str | None:
    if isinstance(value, list) and len(value) == 3 and all(type(item) is bool for item in value):
        return None
    return "extrapolation fitting_param_xdep must contain exactly three booleans: [a_xdep, p_xdep, include_ap]."


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
    input_role_descriptions={
        "lightcone": "A non-empty list of matched light-cone distributions spanning the lattice spacings and/or momenta needed by the limit fit.",
        "main": "The central extrapolated result consumed by an automatically generated systematics-budget job.",
    },
    schema={
        "allow_order_a": _parameter(
            "Powers of lattice spacing included in the central fit.",
            "Terms a^n parameterize continuum discretization effects; they contribute only when the inputs span more than one lattice spacing.",
            expected=list,
            items=int,
            examples=([2],),
            coerce_scalar_to_list=True,
        ),
        "allow_order_1overp": _parameter(
            "Inverse-momentum powers included in the central fit.",
            "Terms 1/Pz^n parameterize higher-twist and finite-boost corrections; they contribute only when momenta vary within an ensemble.",
            expected=list,
            items=int,
            examples=([2, 4],),
            coerce_scalar_to_list=True,
        ),
        "allow_order_ap": _parameter(
            "Powers of the dimensionless aPz combination included in the central fit.",
            "Terms (a Pz)^n capture boost-enhanced cutoff artifacts and require both lattice-spacing and momentum variation.",
            expected=list,
            items=int,
            coerce_scalar_to_list=True,
        ),
        "allow_order_a_sym": _parameter(
            "Alternative lattice-spacing powers used to generate a systematics branch.",
            "These values do not enter the central fit directly; manifest expansion clones the job with this a^n model and adds the result to the systematic budget.",
            expected=list,
            items=int,
        ),
        "allow_order_1overp_sym": _parameter(
            "Alternative inverse-momentum powers used to generate a systematics branch.",
            "Manifest expansion clones the central job with this 1/Pz^n model so model-form sensitivity can enter the systematic budget.",
            expected=list,
            items=int,
        ),
        "allow_order_ap_sym": _parameter(
            "Alternative aPz powers used to generate a systematics branch.",
            "Manifest expansion clones the central job with this boost-enhanced cutoff model and includes the difference in the systematic budget.",
            expected=list,
            items=int,
        ),
        "fitting_param_xdep": _parameter(
            "Three booleans controlling coefficient dependence on momentum fraction x.",
            "The entries respectively make the a^n coefficients x-dependent, make the 1/Pz^n coefficients x-dependent, and enable the declared (aPz)^n terms. The default is [false, true, false].",
            expected=list,
            items=bool,
            examples=([False, True, False],),
            validator=_x_dependence_message,
        ),
        "posterior_prior_error_scale": _parameter(
            "Scale used to build per-sample coefficient priors from the sample-average fit.",
            "Larger values loosen the resampled fits around the sample-average posterior; smaller values regularize weakly constrained continuum and momentum corrections more strongly.",
            expected=float,
            default="3.0",
        ),
        "pdep_gev": _parameter(
            "Physical momenta shown in the momentum-dependence diagnostic plot.",
            "These values evaluate the fitted finite-momentum curve for presentation only and do not change the extrapolation ansatz or fitted limit.",
            expected=list,
            items=float,
            unit="GeV",
            coerce_scalar_to_list=True,
        ),
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
