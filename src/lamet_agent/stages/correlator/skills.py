"""Stage-local skill guidance and validation for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
Correlator-analysis physics:
- Fit the symmetric 2pt correlator only in the first half of the lattice.
- Form 3pt/2pt ratios after resampling both correlators with shared indices.
- The optional fitting_form selects the default Breit ratio or a NonBreit ratio
  with separate initial/final 2pt correlators matched by pz_gev/pz_out_gev.
- Tune nstate, fit strategy, and windows on sample-average data, then pass one
  selected scalar nstate and fit_strategy to fit_bare_matrix_grid.
- The bare matrix element is O00/(2*E0) and is invariant under 2pt rescaling.
""".strip()

TOOL_CATALOG = {
    "inspect_correlator_scale": "Inspect the selected job's 2pt magnitude.",
    "tune_bare_matrix": "Scan every configured nstate, fit strategy, and fit window.",
    "fit_bare_matrix_grid": "Apply the selected scalar configuration to every z/sample and write store['output'].",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
    params = {**manifest.stages["correlator_analysis"].defaults, **job.params}
    fitting_form = str(params.get("fitting_form", "Breit"))
    if fitting_form not in {"Breit", "NonBreit"}:
        return ["fitting_form must be 'Breit' or 'NonBreit'."]
    n_2pt = len([item for item in selected if item.kind == "2pt"])
    if fitting_form == "Breit" and n_2pt != 1:
        return ["A Breit correlator_analysis job requires exactly one 2pt correlator."]
    if fitting_form == "NonBreit" and n_2pt != 2:
        return ["A NonBreit correlator_analysis job requires exactly two 2pt correlators."]
    pt3 = [item for item in selected if item.kind == "3pt"]
    if not pt3:
        return ["A correlator_analysis job requires at least one 3pt correlator."]
    if any(item.bt is None or len(item.bt) != 1 for item in pt3):
        return ["The current correlator stage requires exactly one bt value per 3pt correlator."]
    return []
