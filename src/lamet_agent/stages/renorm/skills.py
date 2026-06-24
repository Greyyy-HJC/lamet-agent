"""Stage-local skill guidance and validation for renormalization."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
Hybrid-ratio renormalization consumes target and denominator EnsembleData from
the current job store. The physical switch length zs_fm is converted with the
target artifact's a_fm, and the nearest available denominator z is used.
""".strip()

TOOL_CATALOG = {
    "apply_ratio_scheme_renormalization": "Consume target/denominator roles and write the renormalized NetCDF plus store['output']; the runner writes one stage report.",
    "plot_renormalized_matrix_element": "Plot store['output'] to PDF.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    if set(job.inputs) != {"target", "denominator"}:
        return ["A renormalization job requires target and denominator inputs."]
    params = {**manifest.stages["renormalization"].defaults, **job.params}
    if params.get("scheme") != "hybrid_ratio":
        return ["The current renormalization stage supports scheme='hybrid_ratio'."]
    scheme_parameters = params.get("scheme_parameters")
    if not isinstance(scheme_parameters, dict) or "zs_fm" not in scheme_parameters:
        return ["hybrid_ratio requires scheme_parameters.zs_fm."]
    return []
