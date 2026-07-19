"""Stage-local skill guidance and validation for correlator analysis."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import merge_stage_params


STAGE_SKILL = """
Correlator-analysis physics:
- Fit the symmetric 2pt correlator only in the first half of the lattice.
- Form 3pt/2pt ratios after resampling both correlators with shared indices.
- fit_scope selects 3pt_ratio, FH, 3pt_ratio+FH, or qda_ratio. FH is
  constructed by summing ratio data over tau after pt3_tau_cuts and finite
  differencing neighboring tsep values.
- The optional fitting_form selects the default Breit ratio or a NonBreit ratio
  with initial/final 2pt slices selected by their discrete momentum labels.
- Tune data windows on sample-average data at multiple representative z values
  chosen by the agent. fit_bare_matrix_grid then keeps one shared window and
  either selects one fit function on sample-average data or, when model_average
  is enabled, averages nstate/prior_width fit functions sample by sample.
- When manifest windows are omitted, generate bounded 2pt candidates from the
  first-half resampled signal and 3pt candidates from the available tsep grid.
  Explicit pt2_windows, pt3_windows, and pt3_tau_cuts remain exact overrides.
- A shared data window must pass sample-average joint fits at every tune z the
  agent selects; a good chi2/dof at only the smallest tune z is not sufficient.
- Data-window candidates with different pt2/pt3 points should not be ranked by
  raw logGBF. Choose windows after the Q and n_data > n_params gates, favoring
  cross-z feasibility, good chi2/dof, and more data points when chi2/dof values
  are comparable.
- 3pt/FH bare matrix elements use O00/(2*E0). qda_ratio uses O00/z0 with an
  ordinary local-local denominator, or O00/zprime0 when the qDA operator's
  bz=0 correlator supplies the denominator. Both outputs are invariant under
  2pt rescaling.
- qda_ratio uses a nonlocal qDA 2pt numerator and optionally an ordinary
  local-source/local-sink 2pt denominator. Without the ordinary input, bz=0
  of the qDA input supplies the denominator and is fit with distinct source
  and sink overlaps z_n*zprime_n. bT and bz are data selectors, never operator
  name patterns. qda_ratio has no 3pt, tsep, tau_cut, or current insertion.
""".strip()

TOOL_CATALOG = {
    "inspect_correlator_scale": "Inspect the selected job's 2pt magnitude.",
    "tune_ground_state": (
        "Optionally scan 2pt-only windows and model-average the selected "
        "ground-state fits."
    ),
    "tune_bare_matrix": (
        "Scan every configured nstate, prior_width, fit strategy, and explicit or automatic fit window "
        "at LLM-supplied tune_z_values; return cross-z feasibility and "
        "recommended_robust_index."
    ),
    "fit_bare_matrix_grid": "Apply one shared data window, optionally model-average fit functions per sample, and write store['output']; the runner writes one stage report with fit_logs links.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    selected = [item for item in manifest.correlators if item.correlator_id in job.correlator_ids]
    params = merge_stage_params(manifest.stages["correlator_analysis"].defaults, job.params)
    if "variant" in manifest.stages["correlator_analysis"].defaults or "variant" in job.params:
        return ["variant is not a supported correlator_analysis parameter."]
    fitting_form = str(params.get("fitting_form", "Breit"))
    raw_scopes = params.get("fit_scope", ["3pt_ratio"])
    scopes = raw_scopes if isinstance(raw_scopes, list) else [raw_scopes]
    allowed_scopes = {"3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"}
    if not scopes or any(not isinstance(scope, str) or scope not in allowed_scopes for scope in scopes):
        return ["fit_scope must contain only '3pt_ratio', 'FH', '3pt_ratio+FH', or 'qda_ratio'."]
    if "qda_ratio" in scopes and len(scopes) != 1:
        return ["fit_scope='qda_ratio' cannot be mixed with 3pt/FH scopes in one job."]
    pt2 = [item for item in selected if item.correlator_type == "2pt"]
    pt3 = [item for item in selected if item.correlator_type == "3pt"]
    if scopes == ["qda_ratio"]:
        if fitting_form != "Breit":
            return ["fit_scope='qda_ratio' requires fitting_form 'Breit'."]
        momentum = params.get("momentum")
        if not isinstance(momentum, str):
            return ["A qda_ratio correlator_analysis job requires scalar params.momentum."]
        matching_pt2 = [item for item in pt2 if momentum in item.momentum]
        qda_pt2 = [
            item
            for item in matching_pt2
            if item.bz is not None
            and ("_nonlocal" in item.source_operator or "_nonlocal" in item.sink_operator)
        ]
        local_pt2 = [
            item
            for item in matching_pt2
            if "_nonlocal" not in item.source_operator and "_nonlocal" not in item.sink_operator
        ]
        if len(qda_pt2) != 1 or len(local_pt2) > 1:
            return [
                "A qda_ratio job requires exactly one nonlocal qDA 2pt correlator "
                "with a bz grid and at most one ordinary local-source/local-sink 2pt correlator."
            ]
        qda_input = qda_pt2[0]
        if qda_input.bT is None or len(qda_input.bT) != 1:
            return ["A qda_ratio qDA 2pt correlator must declare exactly one bT value."]
        if not local_pt2 and 0 not in (qda_input.bz or []):
            return [
                "A qda_ratio job without an ordinary local 2pt correlator requires "
                "bz=0 in the nonlocal qDA 2pt grid."
            ]
        if any(token in qda_input.source_operator or token in qda_input.sink_operator for token in ("<bz>", "{bz}")):
            return ["qDA source_operator and sink_operator must not encode bz placeholders."]
        provenance = lambda item: (
            item.ensemble,
            item.hadron,
            item.gfix,
            item.volume,
            item.lattice_spacing_fm,
            item.temporal_extent,
        )
        if local_pt2 and provenance(qda_input) != provenance(local_pt2[0]):
            return ["The qDA and ordinary 2pt correlators must have matching ensemble provenance."]
        return []
    if fitting_form not in {"Breit", "NonBreit"}:
        return ["fitting_form must be 'Breit' or 'NonBreit'."]
    if fitting_form == "NonBreit" and any("FH" in scope for scope in scopes):
        return ["fit_scope values containing 'FH' currently require fitting_form 'Breit'."]
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
    if any("FH" in scope for scope in scopes) and len(tseps) < 2:
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
