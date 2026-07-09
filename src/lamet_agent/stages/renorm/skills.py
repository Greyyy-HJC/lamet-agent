"""Stage-local skill guidance and validation for renormalization."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
Renormalization consumes bare EnsembleData from the current job store.
When normalization=true, the runner already divides each bare matrix element by
its lattice z=0 value before any tool calls.

hybrid_ratio uses roles target and denominator with scheme_parameters.zs_fm.
self_renormalization splits into:
- fit job inputs {reference}: require job params.d; optional params.m0_gev
  (omit to fit m0 from short-distance g(z); set to freeze). Writes store['output']/store['zR'].
- apply job inputs {target, zR}: apply H/(zR*ZMSbar). Optional params.d / params.m0_gev
  remap upstream zR (e.g. PDF-fit factor → DA d/m0).
""".strip()

TOOL_CATALOG = {
    "apply_ratio_scheme_renormalization": "hybrid_ratio: consume target/denominator and write store['output'] plus the job NetCDF.",
    "fit_self_renormalization_factor": "self_renormalization fit job: fit zR from store['reference'] with required d and optional m0_gev; write store['zR']/store['output'].",
    "apply_self_renormalization": "self_renormalization apply job: apply H/(zR*ZMSbar); optional d/m0_gev remap upstream zR.",
    "plot_self_renormalization_diagnostics": "self_renormalization: fit-job panels, or apply-job zmsbar_compare (+ stage-level discrete_effect once).",
    "plot_renormalized_matrix_element": "Plot store['output'] to PDF (apply jobs).",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    params = {**manifest.stages["renormalization"].defaults, **job.params}
    scheme = params.get("scheme")
    normalization = params.get("normalization", True)
    if not isinstance(normalization, bool):
        return ["renormalization.defaults.normalization must be a boolean when provided."]

    if scheme == "hybrid_ratio":
        if set(job.inputs) != {"target", "denominator"}:
            return ["A hybrid_ratio renormalization job requires target and denominator inputs."]
        scheme_parameters = params.get("scheme_parameters")
        if not isinstance(scheme_parameters, dict) or "zs_fm" not in scheme_parameters:
            return ["hybrid_ratio requires scheme_parameters.zs_fm."]
        return []

    if scheme == "self_renormalization":
        roles = set(job.inputs)
        if roles == {"reference"}:
            if "d" not in params:
                return ["self_renormalization fit job requires params.d."]
        elif roles == {"target", "zR"}:
            pass
        else:
            return [
                "A self_renormalization job requires either {reference} (fit) "
                "or {target, zR} (apply) inputs."
            ]
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

    return [f"Unsupported renormalization scheme: {scheme!r}."]
