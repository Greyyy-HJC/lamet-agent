"""Stage-local skill guidance and validation for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
Correlator-analysis physics:
- Fit the symmetric 2pt correlator only in the first half of the lattice.
- Form 3pt/2pt ratios after resampling both correlators with shared indices.
- fit_scope selects the 3pt observable to fit: ratio, FH, or ratio+FH. FH is
  constructed by summing ratio data over tau after pt3_tau_cuts and finite
  differencing neighboring tsep values.
- The optional fitting_form selects the default Breit ratio or a NonBreit ratio
  with separate initial/final 2pt correlators matched by pz_gev/pz_out_gev.
- Tune data windows on sample-average data at multiple representative z values
  chosen by the agent. fit_bare_matrix_grid then keeps one shared window and
  either selects one fit function on sample-average data or, when model_average
  is enabled, averages nstate/prior_width fit functions sample by sample.
- A shared data window must pass sample-average joint fits at every tune z the
  agent selects; a good chi2/dof at only the smallest tune z is not sufficient.
- Data-window candidates with different pt2/pt3 points should not be ranked by
  raw logGBF. Choose windows after the Q and n_data > n_params gates, favoring
  cross-z feasibility, good chi2/dof, and more data points when chi2/dof values
  are comparable.
- The bare matrix element is O00/(2*E0) and is invariant under 2pt rescaling.
""".strip()

TOOL_CATALOG = {
    "inspect_correlator_scale": "Inspect the selected job's 2pt magnitude.",
    "tune_bare_matrix": (
        "Scan every configured nstate, prior_width, fit strategy, and fit window "
        "at LLM-supplied tune_z_values; return cross-z feasibility and "
        "recommended_robust_index."
    ),
    "fit_bare_matrix_grid": "Apply one shared data window, optionally model-average fit functions per sample, and write store['output']; the runner writes one stage report with fit_logs links.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
    params = {**manifest.stages["correlator_analysis"].defaults, **job.params}
    fitting_form = str(params.get("fitting_form", "Breit"))
    if fitting_form not in {"Breit", "NonBreit"}:
        return ["fitting_form must be 'Breit' or 'NonBreit'."]
    raw_scopes = params.get("fit_scope", ["ratio"])
    scopes = raw_scopes if isinstance(raw_scopes, list) else [raw_scopes]
    normalised_scopes = {str(scope).strip().lower().replace(" ", "") for scope in scopes}
    allowed_scopes = {"ratio", "fh", "ratio+fh"}
    if not normalised_scopes.issubset(allowed_scopes):
        return ["fit_scope must contain only 'ratio', 'FH', or 'ratio+FH'."]
    if fitting_form == "NonBreit" and any("fh" in scope for scope in normalised_scopes):
        return ["fit_scope values containing 'FH' currently require fitting_form 'Breit'."]
    n_2pt = len([item for item in selected if item.kind == "2pt"])
    if fitting_form == "Breit" and n_2pt != 1:
        return ["A Breit correlator_analysis job requires exactly one 2pt correlator."]
    if fitting_form == "NonBreit" and n_2pt != 2:
        return ["A NonBreit correlator_analysis job requires exactly two 2pt correlators."]
    pt3 = [item for item in selected if item.kind == "3pt"]
    if not pt3:
        return ["A correlator_analysis job requires at least one 3pt correlator."]
    if any("fh" in scope for scope in normalised_scopes) and len(pt3) < 2:
        return ["FH correlator_analysis jobs require at least two 3pt tsep correlators."]
    if any(item.bt is None or len(item.bt) != 1 for item in pt3):
        return ["The current correlator stage requires exactly one bt value per 3pt correlator."]
    return []
