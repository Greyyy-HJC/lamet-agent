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
  with initial/final 2pt slices selected by their discrete momentum labels.
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
    "tune_ground_state": (
        "Optionally scan 2pt-only windows and model-average the selected "
        "ground-state fits."
    ),
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
    if "variant" in manifest.stages["correlator_analysis"].defaults or "variant" in job.params:
        return ["variant is not a supported correlator_analysis parameter."]
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
    pt2 = [item for item in selected if item.correlator_type == "2pt"]
    pt3 = [item for item in selected if item.correlator_type == "3pt"]
    if fitting_form == "Breit":
        momentum = params.get("momentum")
        if not isinstance(momentum, str):
            return ["A Breit correlator_analysis job requires scalar params.momentum."]
        if not any(momentum in item.momentum for item in pt2):
            return [f"No selected 2pt correlator declares momentum {momentum!r}."]
        selected_pt2 = [item for item in pt2 if momentum in item.momentum]
        selected_pt3 = [item for item in pt3 if momentum in item.momentum]
    else:
        initial = params.get("initial_momentum")
        final = params.get("final_momentum")
        if not isinstance(initial, str) or not isinstance(final, str):
            return ["A NonBreit correlator_analysis job requires params.initial_momentum and params.final_momentum."]
        if not any(initial in item.momentum for item in pt2):
            return [f"No selected 2pt correlator declares initial_momentum {initial!r}."]
        if not any(final in item.momentum for item in pt2):
            return [f"No selected 2pt correlator declares final_momentum {final!r}."]
        selected_pt2 = [item for item in pt2 if initial in item.momentum or final in item.momentum]
        selected_pt3 = [item for item in pt3 if final in item.momentum]
    if not selected_pt3:
        return ["A correlator_analysis job requires at least one 3pt correlator."]
    tseps = {tsep for item in selected_pt3 for tsep in (item.tsep or [])}
    if any("fh" in scope for scope in normalised_scopes) and len(tseps) < 2:
        return ["FH correlator_analysis jobs require at least two 3pt tsep values."]
    if any(item.bT is None or len(item.bT) != 1 for item in selected_pt3):
        return ["The current correlator stage requires exactly one bT value per 3pt correlator."]
    reference = selected_pt3[0]
    if any(
        (item.source_operator, item.sink_operator, item.current_operator, item.bz_direction, item.bT, item.bz)
        != (
            reference.source_operator,
            reference.sink_operator,
            reference.current_operator,
            reference.bz_direction,
            reference.bT,
            reference.bz,
        )
        for item in selected_pt3[1:]
    ):
        return ["Selected 3pt correlators must use the same operators, bz_direction, bT, and bz grid."]
    if any(
        (item.source_operator, item.sink_operator) != (reference.source_operator, reference.sink_operator)
        for item in selected_pt2
    ):
        return ["Selected 2pt and 3pt correlators must use the same source and sink operators."]
    provenance = (reference.ensemble, reference.hadron, reference.gfix, reference.volume, reference.lattice_spacing_fm)
    if any(
        (item.ensemble, item.hadron, item.gfix, item.volume, item.lattice_spacing_fm) != provenance
        for item in [*selected_pt2, *selected_pt3]
    ):
        return ["Selected correlators must use the same ensemble, hadron, gfix, volume, and lattice spacing."]
    return []
