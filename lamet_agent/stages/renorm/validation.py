"""Executable manifest contract for renormalization."""

from __future__ import annotations

import math
from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import ConstraintSpec, ParameterSpec, RuleViolation, StageParamContract, StageValidationContext, merge_stage_params


def _parameter(summary: str, physics: str, **kwargs: Any) -> ParameterSpec:
    return ParameterSpec(summary=summary, physics=physics, **kwargs)


def _scheme_message(value: Any) -> str | None:
    legacy = {
        "hybrid_ratio": "use scheme='hybrid' with strategy='external_denominator'",
        "hybrid_self_renormalization": "use scheme='ratio' with strategy='self_renormalization'",
        "self_renormalization": "use scheme='ratio' with strategy='self_renormalization'",
    }
    if value in legacy:
        return f"renormalization scheme {value!r} is no longer supported; {legacy[value]}."
    if value not in {"ratio", "hybrid", "msbar"}:
        return f"Unsupported renormalization scheme: {value!r}; use 'ratio', 'hybrid', or 'msbar'."
    return None


def _strategy_message(value: Any) -> str | None:
    if value == "ratio":
        return "renormalization strategy 'ratio' is no longer supported; use strategy='external_denominator'."
    if value not in {"external_denominator", "self_renormalization"}:
        return f"Unsupported renormalization strategy: {value!r}; use 'external_denominator' or 'self_renormalization'."
    return None


def _normalization_message(value: Any) -> str | None:
    return None if isinstance(value, bool) else "renormalization.defaults.normalization must be a boolean when provided."


def _lambda_message(value: Any) -> str | None:
    valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    return None if valid else "self_renormalization LambdaQCD_gev must be a finite positive value."


def _coverage_message(value: Any) -> str | None:
    if value in {"strict", "intersection", "extrapolate"}:
        return None
    return "self_renormalization z_coverage_policy must be 'strict', 'intersection', or 'extrapolate'."


def _kernel_field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _violation(context: StageValidationContext, message: str, *, parameter: str, cause: str, path: str | None = None) -> RuleViolation:
    return RuleViolation(message, path or context.parameter_path(parameter), cause, (parameter,))


def _valid_selection(context: StageValidationContext) -> tuple[str, str] | None:
    scheme = context.params.get("scheme")
    strategy = context.params.get("strategy")
    if scheme not in {"ratio", "hybrid", "msbar"} or strategy not in {"external_denominator", "self_renormalization"}:
        return None
    return str(scheme), str(strategy)


def _check_external(context: StageValidationContext) -> RuleViolation | list[RuleViolation] | None:
    selection = _valid_selection(context)
    if selection is None or selection[1] != "external_denominator":
        return None
    scheme, _ = selection
    issues: list[RuleViolation] = []
    if scheme == "msbar":
        issues.append(_violation(
            context,
            "renormalization strategy 'external_denominator' does not implement scheme 'msbar'.",
            parameter="strategy",
            cause="Direct MS-bar conversion requires the fitted self-renormalization factor rather than an external ratio denominator.",
        ))
    scheme_parameters = context.params.get("scheme_parameters", {})
    if isinstance(scheme_parameters, dict):
        self_only = sorted({"LambdaQCD_gev", "d", "svdcut", "z_coverage_policy"}.intersection(scheme_parameters))
        if self_only:
            issues.append(_violation(
                context,
                "strategy 'external_denominator' does not accept self-renormalization scheme_parameters: " + ", ".join(self_only) + ".",
                parameter="scheme_parameters",
                cause=f"These fit-only parameters are present: {self_only!r}.",
            ))
    if set(context.inputs) != {"target", "denominator"}:
        issues.append(_violation(
            context,
            f"A {scheme}+external_denominator renormalization job requires target and denominator inputs.",
            parameter="inputs",
            path=f"{context.job_path}.inputs",
            cause=f"The effective input roles are {sorted(context.inputs)}.",
        ))
    if scheme == "hybrid" and "zs_fm" not in context.params:
        issues.append(_violation(
            context,
            "hybrid scheme requires flat parameter zs_fm in stage defaults or job params.",
            parameter="zs_fm",
            cause="The ratio-to-mass-counterterm transition distance is absent.",
        ))
    return issues or None


def _check_self_parameters(context: StageValidationContext) -> RuleViolation | list[RuleViolation] | None:
    selection = _valid_selection(context)
    if selection is None or selection[1] != "self_renormalization":
        return None
    scheme, _ = selection
    scheme_parameters = context.params.get("scheme_parameters", {})
    if not isinstance(scheme_parameters, dict):
        return _violation(context, "self_renormalization scheme_parameters must be an object.", parameter="scheme_parameters", cause=f"The effective value is {scheme_parameters!r}.")
    issues: list[RuleViolation] = []
    if "LambdaQCD_gev" not in scheme_parameters:
        issues.append(_violation(
            context,
            "self_renormalization requires scheme_parameters.LambdaQCD_gev on every fit and apply job.",
            parameter="scheme_parameters.LambdaQCD_gev",
            cause="The perturbative running scale is absent from the effective scheme parameters.",
        ))
    coverage = scheme_parameters.get("z_coverage_policy", "extrapolate")
    coverage_message = _coverage_message(coverage)
    if coverage_message is not None:
        issues.append(_violation(
            context,
            coverage_message,
            parameter="scheme_parameters.z_coverage_policy",
            cause=f"The effective coverage policy is {coverage!r}.",
        ))
    roles = set(context.inputs)
    if roles == {"reference"}:
        if "d" not in scheme_parameters:
            issues.append(_violation(context, "self_renormalization fit job requires scheme_parameters.d.", parameter="scheme_parameters.d", cause="The fixed discretization coefficient is absent from the reference fit."))
        if "m0_gev" in scheme_parameters:
            issues.append(_violation(
                context,
                "self_renormalization fit jobs determine the reference m0; remove scheme_parameters.m0_gev here (apply jobs may override target m0_gev).",
                parameter="scheme_parameters.m0_gev",
                cause="A fixed m0_gev would duplicate the parameter determined by the reference fit.",
            ))
    else:
        expected = {"target", "denominator", "zR"} if scheme == "hybrid" else {"target", "zR"}
        if roles != expected:
            issues.append(_violation(
                context,
                "A self_renormalization job requires either {reference} (fit) or " + f"{sorted(expected)} (apply) inputs for scheme {scheme!r}.",
                parameter="inputs",
                path=f"{context.job_path}.inputs",
                cause=f"The effective input roles are {sorted(roles)}.",
            ))
        if scheme == "hybrid" and "zs_fm" not in context.params:
            issues.append(_violation(context, "hybrid scheme requires flat parameter zs_fm in stage defaults or job params.", parameter="zs_fm", cause="The hybrid transition distance is absent from this apply job."))
    return issues or None


def _check_self_kernel(context: StageValidationContext) -> RuleViolation | None:
    selection = _valid_selection(context)
    if selection is None or selection[1] != "self_renormalization":
        return None
    kernels = [item for item in context.resources.get("kernels", []) if _kernel_field(item, "stage") == "renormalization"]
    if not kernels:
        return _violation(context, "self_renormalization requires a kernel with stage='renormalization' in inputs.kernels.", parameter="inputs.kernels", path="inputs.kernels", cause="No renormalization kernel declaration is available.")
    kernel_id = context.params.get("kernel_id") or (_kernel_field(kernels[0], "kernel_id") if len(kernels) == 1 else None)
    if kernel_id is None:
        return _violation(context, "self_renormalization requires kernel_id when multiple renormalization kernels are declared.", parameter="kernel_id", cause="More than one renormalization kernel is declared.")
    declaration = next((item for item in kernels if _kernel_field(item, "kernel_id") == kernel_id), None)
    if declaration is None:
        return _violation(context, f"Renormalization kernel {kernel_id!r} is not declared in inputs.kernels.", parameter="kernel_id", cause="The selected identifier does not match a renormalization kernel declaration.")
    if kernel_id not in {"ZMSbar_pdf", "ZMSbar_da"}:
        return _violation(context, f"Unsupported renormalization kernel_id {kernel_id!r}; use ZMSbar_pdf or ZMSbar_da.", parameter="kernel_id", cause="The runtime has no conversion kernel registered for this identifier.")
    return None


_SCHEME_PARAMETER_FIELDS = {
    "LambdaQCD_gev": _parameter("QCD scale used in perturbative running.", "It fixes the coupling entering the self-renormalization short-distance ansatz. Declare it in defaults so fit and apply share one value; an apply-job override is used as-is for remap and long-distance reconstruction.", expected=float, unit="GeV", validator=_lambda_message),
    "d": _parameter("Fixed discretization coefficient for a self-renormalization operator.", "A fit job requires the reference-operator value; an apply-job override remaps the upstream zR to a target operator with different lattice artifacts.", expected=float),
    "delta_m_gev": _parameter("External-hybrid target/denominator mass-gap offset.", "Together with m0_gev it controls the long-distance exponential branch beyond zs_fm; it is not a self-renormalization fit control.", expected=float, unit="GeV"),
    "m0_gev": _parameter("Target-specific residual mass offset for apply jobs.", "Reference fit jobs determine m0 and therefore must not fix it.", expected=float, unit="GeV"),
    "svdcut": _parameter("Covariance singular-value cut for the reference fit.", "Regularization stabilizes the correlated fit of the renormalization factor.", expected=float, default="1e-12"),
    "z_coverage_policy": _parameter(
        "String policy for target coordinates beyond the fitted zR range.",
        "For self-renormalization, strict requires zR at every nonzero target coordinate; intersection keeps only the target/zR overlap; extrapolate fits the long-distance f1 tail and rebuilds zR only at missing target coordinates.",
        default="extrapolate",
        examples=("strict",),
    ),
}


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="renorm",
    summary="Remove Wilson-line and ultraviolet divergences from bare matrix elements.",
    physics="External-denominator and self-renormalization strategies define distinct estimators; their inputs and nuisance parameters cannot be interchanged.",
    planning_notes=("scheme and strategy are independent required choices.", "Keep self-renormalization controls inside scheme_parameters and keep hybrid zs_fm flat."),
    input_roles=("target", "denominator", "reference", "zR"),
    input_role_descriptions={
        "target": "Bare matrix element to renormalize.",
        "denominator": "Bare zero-momentum or scheme denominator used by external-ratio and hybrid jobs.",
        "reference": "Zero-momentum reference used to fit the self-renormalization factor.",
        "zR": "Previously fitted self-renormalization factor consumed by an apply job.",
    },
    schema={
        "ensemble": _parameter("Optional ensemble selector.", "It identifies the lattice ensemble associated with a self-renormalization reference.", expected=str),
        "kernel_id": _parameter("Declared conversion kernel identifier.", "Self-renormalization uses a PDF- or DA-specific MS-bar conversion factor.", expected=str),
        "mu": _parameter("Renormalization scale.", "The perturbative conversion factor and short-distance logarithms are evaluated at this scale; the value may also come from the selected kernel declaration.", expected=float, unit="GeV", default="2.0"),
        "normalization": _parameter(
            "Normalize every bare job input by its lattice z=0 value before applying the scheme.",
            "This fixes the local-current normalization convention upstream of either estimator; false preserves the raw bare normalization.",
            expected=bool,
            choices=(False, True),
            choice_descriptions={
                False: "Pass raw bare matrix elements directly to the renormalization estimator.",
                True: "Divide each bare input by its own z=0 sample before any renormalization tool runs.",
            },
            default="true",
            validator=_normalization_message,
        ),
        "scheme": _parameter(
            "Physical renormalization scheme.",
            "The scheme chooses the short-/long-distance definition of the renormalized matrix element; it is independent of the estimator strategy used to construct the factors.",
            expected=str,
            required=True,
            choices=("ratio", "hybrid", "msbar"),
            choice_descriptions={
                "ratio": "Use a ratio prescription over the full nonzero coordinate range.",
                "hybrid": "Use a ratio below zs_fm and a continuous mass-counterterm branch above it.",
                "msbar": "Apply the fitted self-renormalization factor directly in the MS-bar definition; external_denominator does not implement this choice.",
            },
            validator=_scheme_message,
        ),
        "strategy": _parameter(
            "Estimator used to construct the renormalization factor.",
            "An external denominator forms a direct sample-by-sample ratio, while self-renormalization fits the Wilson-line divergence on a reference and transfers it to target jobs.",
            expected=str,
            required=True,
            choices=("external_denominator", "self_renormalization"),
            choice_descriptions={
                "external_denominator": "Consume target and denominator in one apply job; supported for ratio and hybrid schemes.",
                "self_renormalization": "Fit zR from a reference job, then consume target and zR in separate apply jobs.",
            },
            validator=_strategy_message,
        ),
        "scheme_parameters": _parameter("Strategy-specific numerical controls.", "These parameters belong to the self-renormalization ansatz and coverage prescription.", expected=dict, schema=_SCHEME_PARAMETER_FIELDS),
        "zs_fm": _parameter("Hybrid transition distance or uncertainty-bearing systematics value.", "Below zs the ratio prescription is used; above it the mass counterterm controls the Wilson line. A string such as '0.17(2)' requests low/high systematics branches before execution.", expected=(float, str), unit="fm"),
    },
    removed={
        "LambdaQCD": "was renamed; use scheme_parameters.LambdaQCD_gev and specify the value explicitly.",
        "d": "belongs to strategy='self_renormalization'; use scheme_parameters.d.",
        "m0_gev": "is scheme/strategy-specific; use scheme_parameters.m0_gev.",
        "svdcut": "belongs to strategy='self_renormalization'; use scheme_parameters.svdcut.",
        "z_coverage_policy": "belongs to strategy='self_renormalization'; use scheme_parameters.z_coverage_policy.",
        "alpha_s": "is derived from mu by alphas_nloop and cannot be specified.",
        "Nf": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu).",
        "order": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu).",
        "b0": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
        "cf": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
        "f1_extension_zmin_fm": "is no longer supported; apply-time extension is automatic with z_coverage_policy='extrapolate'.",
        "k": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
        "lqcd": "was renamed; use scheme_parameters.LambdaQCD_gev and specify the value explicitly.",
        "scheme_parameters.zs_fm": "is no longer supported; use flat stages.renormalization.defaults.zs_fm or the corresponding jobs[].params.zs_fm.",
        "zms_kind": "is no longer supported; select a declared ZMSbar_pdf or ZMSbar_da kernel_id.",
        "zr_zmax_fm": "is no longer supported; the target grid determines automatic apply-time extension with z_coverage_policy='extrapolate'.",
    },
    constraints=(
        ConstraintSpec("renorm.external.contract", ("scheme", "strategy", "scheme_parameters", "inputs", "zs_fm"), "external_denominator supports ratio/hybrid, accepts target+denominator, and needs zs_fm for hybrid.", "A direct ratio and a fitted self-renormalization factor are different estimators with different nuisance parameters.", "Choose compatible scheme/strategy values and provide exactly the required input roles.", _check_external),
        ConstraintSpec("renorm.self.contract", ("scheme_parameters", "inputs", "zs_fm"), "self_renormalization requires LambdaQCD_gev; fit and apply jobs have distinct roles and parameters.", "The reference fit determines the divergence and residual mass, while apply jobs transfer that fitted factor to target data.", "Provide reference+d for a fit, or the scheme-specific target/zR roles for apply.", _check_self_parameters),
        ConstraintSpec("renorm.self.kernel", ("kernel_id", "inputs.kernels"), "self_renormalization selects one declared ZMSbar_pdf or ZMSbar_da kernel.", "The conversion factor depends on whether the target is a PDF or DA operator.", "Declare and select the matching renormalization kernel.", _check_self_kernel),
    ),
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved renormalization context consumed by the shared evaluator."""
    stage_config = manifest.stages["renormalization"]
    authored = merge_stage_params(stage_config.defaults, job.params)
    return StageValidationContext(
        stage="renormalization",
        job_id=job.id,
        job_path=f"stages.renormalization.jobs.{job.id}",
        params=authored,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        resources={"kernels": list(manifest.kernels)},
        authored_params=authored,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise diagnostics."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
