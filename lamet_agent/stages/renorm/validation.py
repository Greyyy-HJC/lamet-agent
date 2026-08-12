"""Stage-local validation for renormalization."""

from __future__ import annotations

import math

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import merge_stage_params


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    params = merge_stage_params(manifest.stages["renormalization"].defaults, job.params)
    scheme = params.get("scheme")
    strategy = params.get("strategy")
    normalization = params.get("normalization", True)
    if not isinstance(normalization, bool):
        return ["renormalization.defaults.normalization must be a boolean when provided."]

    legacy_schemes = {
        "hybrid_ratio": "use scheme='hybrid' with strategy='external_denominator'",
        "hybrid_self_renormalization": "use scheme='ratio' with strategy='self_renormalization'",
        "self_renormalization": "use scheme='ratio' with strategy='self_renormalization'",
    }
    if scheme in legacy_schemes:
        return [f"renormalization scheme {scheme!r} is no longer supported; {legacy_schemes[scheme]}."]
    if scheme not in {"ratio", "hybrid", "msbar"}:
        return [f"Unsupported renormalization scheme: {scheme!r}; use 'ratio', 'hybrid', or 'msbar'."]
    if strategy == "ratio":
        return [
            "renormalization strategy 'ratio' is no longer supported; "
            "use strategy='external_denominator'."
        ]
    if strategy not in {"external_denominator", "self_renormalization"}:
        return [
            f"Unsupported renormalization strategy: {strategy!r}; "
            "use 'external_denominator' or 'self_renormalization'."
        ]

    if strategy == "external_denominator":
        if scheme == "msbar":
            return [
                "renormalization strategy 'external_denominator' does not implement scheme 'msbar'."
            ]
        scheme_parameters = params.get("scheme_parameters", {})
        if isinstance(scheme_parameters, dict):
            self_only = sorted(
                {"LambdaQCD_gev", "d", "svdcut", "z_coverage_policy"}.intersection(scheme_parameters)
            )
            if self_only:
                return [
                    "strategy 'external_denominator' does not accept "
                    "self-renormalization scheme_parameters: "
                    + ", ".join(self_only)
                    + "."
                ]
        if set(job.inputs) != {"target", "denominator"}:
            return [
                f"A {scheme}+external_denominator renormalization job requires "
                "target and denominator inputs."
            ]
        if scheme == "hybrid" and "zs_fm" not in params:
            return ["hybrid scheme requires flat parameter zs_fm in stage defaults or job params."]
        return []

    if strategy == "self_renormalization":
        scheme_parameters = params.get("scheme_parameters", {})
        if not isinstance(scheme_parameters, dict):
            return ["self_renormalization scheme_parameters must be an object."]
        if "LambdaQCD_gev" not in scheme_parameters:
            return [
                "self_renormalization requires scheme_parameters.LambdaQCD_gev "
                "on every fit and apply job."
            ]
        try:
            lambdaqcd_gev = float(scheme_parameters["LambdaQCD_gev"])
        except (TypeError, ValueError):
            return ["self_renormalization LambdaQCD_gev must be a finite positive value."]
        if not math.isfinite(lambdaqcd_gev) or lambdaqcd_gev <= 0.0:
            return ["self_renormalization LambdaQCD_gev must be a finite positive value."]
        coverage_policy = scheme_parameters.get("z_coverage_policy", "extrapolate")
        if coverage_policy not in {"strict", "intersection", "extrapolate"}:
            return [
                "self_renormalization z_coverage_policy must be "
                "'strict', 'intersection', or 'extrapolate'."
            ]
        roles = set(job.inputs)
        if roles == {"reference"}:
            if "d" not in scheme_parameters:
                return ["self_renormalization fit job requires scheme_parameters.d."]
            if "m0_gev" in scheme_parameters:
                return [
                    "self_renormalization fit jobs determine the reference m0; "
                    "remove scheme_parameters.m0_gev here (apply jobs may override target m0_gev)."
                ]
        else:
            expected = {"target", "denominator", "zR"} if scheme == "hybrid" else {"target", "zR"}
            if roles != expected:
                return [
                    "A self_renormalization job requires either {reference} (fit) or "
                    f"{sorted(expected)} (apply) inputs for scheme {scheme!r}."
                ]
            if scheme == "hybrid" and "zs_fm" not in params:
                return ["hybrid scheme requires flat parameter zs_fm in stage defaults or job params."]
        renorm_kernels = [item for item in manifest.kernels if item.stage == "renormalization"]
        if not renorm_kernels:
            return ["self_renormalization requires a kernel with stage='renormalization' in inputs.kernels."]
        kernel_id = params.get("kernel_id") or (renorm_kernels[0].kernel_id if len(renorm_kernels) == 1 else None)
        if kernel_id is None:
            return ["self_renormalization requires kernel_id when multiple renormalization kernels are declared."]
        declaration = next((item for item in renorm_kernels if item.kernel_id == kernel_id), None)
        if declaration is None:
            return [f"Renormalization kernel {kernel_id!r} is not declared in inputs.kernels."]
        if declaration.kernel_id not in {"ZMSbar_pdf", "ZMSbar_da"}:
            return [f"Unsupported renormalization kernel_id {declaration.kernel_id!r}; use ZMSbar_pdf or ZMSbar_da."]
        return []

    return []
