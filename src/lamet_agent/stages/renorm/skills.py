"""Stage-local skill guidance and validation for renormalization."""

from __future__ import annotations

import math

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import merge_stage_params


STAGE_SKILL = """
Renormalization consumes bare EnsembleData from the current job store.
When normalization=true, the runner already divides each bare matrix element by
its lattice z=0 value before any tool calls.

ratio and hybrid_ratio use roles target and denominator. ratio divides them
pointwise on the complete z grid. hybrid_ratio additionally requires flat
job/defaults parameter zs_fm; m0_gev and delta_m_gev remain in scheme_parameters.
hybrid_self_renormalization combines a full-z self-renormalization fit with
short-distance MSbar matching to fix the finite renormalization. It splits into:
- scheme_parameters.LambdaQCD_gev is required and is used by the
  self-renormalization ansatz; use one value throughout the fit/apply chain.
- fit job inputs {reference}: require scheme_parameters.d; fit the reference-operator m0 from
  short-distance g(z), use one discretization family, and never extrapolate
  beyond the reference grid. Writes store['output']/store['zR'].
- apply job inputs {target, zR}: apply H/(zR*ZMSbar). Optional scheme_parameters.d / m0_gev
  remap upstream zR (e.g. PDF-fit factor → DA d/m0).
  scheme_parameters.z_coverage_policy extrapolates the single-family
  long-distance f1 by default when the target
  extends past zR; strict and intersection remain explicit alternatives.
""".strip()

TOOL_CATALOG = {
    "apply_ratio_scheme_renormalization": "ratio/hybrid_ratio: consume target/denominator and write store['output'] plus the job NetCDF.",
    "fit_self_renormalization_factor": "hybrid_self_renormalization fit job: fit zR using scheme_parameters (including required LambdaQCD_gev and d); short-distance MSbar matching fixes m0.",
    "apply_self_renormalization": "hybrid_self_renormalization apply job: apply H/(zR*ZMSbar) with the declared LambdaQCD_gev; optional scheme_parameters d/m0_gev remap zR.",
    "plot_self_renormalization_diagnostics": "hybrid_self_renormalization: fit-job panels, or apply-job zmsbar_compare (+ stage-level discrete_effect once).",
    "plot_renormalized_matrix_element": "Plot store['output'] to PDF (apply jobs).",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    params = merge_stage_params(manifest.stages["renormalization"].defaults, job.params)
    scheme = params.get("scheme")
    normalization = params.get("normalization", True)
    if not isinstance(normalization, bool):
        return ["renormalization.defaults.normalization must be a boolean when provided."]

    if scheme in {"ratio", "hybrid_ratio"}:
        scheme_parameters = params.get("scheme_parameters", {})
        if isinstance(scheme_parameters, dict):
            self_only = sorted(
                {"LambdaQCD_gev", "d", "svdcut", "z_coverage_policy"}.intersection(scheme_parameters)
            )
            if self_only:
                return [
                    f"{scheme} does not accept hybrid-self-only scheme_parameters: "
                    + ", ".join(self_only)
                    + "."
                ]
        if set(job.inputs) != {"target", "denominator"}:
            return [f"A {scheme} renormalization job requires target and denominator inputs."]
        if scheme == "hybrid_ratio" and "zs_fm" not in params:
            return ["hybrid_ratio requires flat parameter zs_fm in stage defaults or job params."]
        return []

    if scheme == "self_renormalization":
        return [
            "renormalization scheme 'self_renormalization' was renamed; "
            "use 'hybrid_self_renormalization'."
        ]

    if scheme == "hybrid_self_renormalization":
        scheme_parameters = params.get("scheme_parameters", {})
        if not isinstance(scheme_parameters, dict):
            return ["hybrid_self_renormalization scheme_parameters must be an object."]
        if "LambdaQCD_gev" not in scheme_parameters:
            return [
                "hybrid_self_renormalization requires scheme_parameters.LambdaQCD_gev "
                "on every fit and apply job."
            ]
        try:
            lambdaqcd_gev = float(scheme_parameters["LambdaQCD_gev"])
        except (TypeError, ValueError):
            return ["hybrid_self_renormalization LambdaQCD_gev must be a finite positive value."]
        if not math.isfinite(lambdaqcd_gev) or lambdaqcd_gev <= 0.0:
            return ["hybrid_self_renormalization LambdaQCD_gev must be a finite positive value."]
        coverage_policy = scheme_parameters.get("z_coverage_policy", "extrapolate")
        if coverage_policy not in {"strict", "intersection", "extrapolate"}:
            return [
                "hybrid_self_renormalization z_coverage_policy must be "
                "'strict', 'intersection', or 'extrapolate'."
            ]
        roles = set(job.inputs)
        if roles == {"reference"}:
            if "d" not in scheme_parameters:
                return ["hybrid_self_renormalization fit job requires scheme_parameters.d."]
            if "m0_gev" in scheme_parameters:
                return [
                    "hybrid_self_renormalization fit jobs determine the reference m0; "
                    "remove scheme_parameters.m0_gev here (apply jobs may override target m0_gev)."
                ]
        elif roles == {"target", "zR"}:
            pass
        else:
            return [
                "A hybrid_self_renormalization job requires either {reference} (fit) "
                "or {target, zR} (apply) inputs."
            ]
        renorm_kernels = [item for item in manifest.kernels if item.stage == "renormalization"]
        if not renorm_kernels:
            return ["hybrid_self_renormalization requires a kernel with stage='renormalization' in inputs.kernels."]
        kernel_id = params.get("kernel_id") or (renorm_kernels[0].kernel_id if len(renorm_kernels) == 1 else None)
        if kernel_id is None:
            return ["hybrid_self_renormalization requires kernel_id when multiple renormalization kernels are declared."]
        declaration = next((item for item in renorm_kernels if item.kernel_id == kernel_id), None)
        if declaration is None:
            return [f"Renormalization kernel {kernel_id!r} is not declared in inputs.kernels."]
        if declaration.kernel_id not in {"ZMSbar_pdf", "ZMSbar_da"}:
            return [f"Unsupported renormalization kernel_id {declaration.kernel_id!r}; use ZMSbar_pdf or ZMSbar_da."]
        return []

    return [f"Unsupported renormalization scheme: {scheme!r}."]
