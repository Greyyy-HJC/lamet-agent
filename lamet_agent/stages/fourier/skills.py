"""Stage-local validation for Fourier-transform jobs."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params


INFERRED_OBSERVABLES = {
    (target, "quark", hadron, distribution_type): f"{hadron}_quark_{distribution_type}_quasi_{target}"
    for target in ("pdf", "gpd")
    for hadron in ("pion", "nucleon")
    for distribution_type in ("unpolarized", "helicity", "transversity")
}
INFERRED_OBSERVABLES.update(
    {
        ("pdf", "gluon", "pion", "unpolarized"): "pion_gluon_unpolarized_quasi_pdf",
        ("pdf", "gluon", "nucleon", "unpolarized"): "nucleon_gluon_unpolarized_quasi_pdf",
    }
)


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    if set(job.inputs) != {"input"}:
        return ["A fourier_transform job requires exactly one input role."]
    params = merge_stage_params(manifest.stages["fourier_transform"].defaults, job.params)
    params = {**derive_job_kinematics(manifest, job), **params}
    missing = [key for key in ("momentum_gev",) if key not in params]
    target = manifest.metadata.target_observable
    parton = manifest.metadata.parton
    hadron = str(params.get("hadron", "")).lower()
    hadron = "nucleon" if hadron == "proton" else hadron
    distribution_type = str(params.get("distribution_type", "unpolarized")).lower()
    inferred_observable = INFERRED_OBSERVABLES.get((target, parton, hadron, distribution_type))
    if "observable" not in params and parton == "gluon" and hadron in {"pion", "nucleon"} and inferred_observable is None:
        return ["The Fourier backend currently supports only unpolarized gluon PDF observables."]
    if (
        "observable" not in params
        and target in {"pdf", "gpd"}
        and inferred_observable is None
    ):
        missing.append("observable")
    if missing:
        return [f"Fourier job {job.id!r} is missing parameters: {missing}"]
    orders = params["order"] if isinstance(params.get("order"), list) else [params.get("order")] if "order" in params else []
    if orders and any(order not in {"LA", "NLA"} for order in orders):
        return ["Fourier order must be 'LA' or 'NLA'."]
    sectors = (
        {"pdf": {"full"}, "da": {"full"}, "gpd": {"full"}}
        if parton == "gluon"
        else {"pdf": {"sea", "valence", "singlet", "full"}, "da": {"full"}, "gpd": {"sea", "valence", "singlet", "full"}}
    )
    if "sector" in params and str(params["sector"]).lower() not in sectors[target]:
        return [f"Fourier sector must be one of {sorted(sectors[target])}."]
    if "sector" not in params and "part" in params and params.get("part") not in {"re", "im", "both"}:
        return ["Fourier part must be 're', 'im', or 'both'."]
    if "symmetry_guarantee" in params and not isinstance(params["symmetry_guarantee"], bool):
        return ["Fourier symmetry_guarantee must be a boolean."]
    return []
