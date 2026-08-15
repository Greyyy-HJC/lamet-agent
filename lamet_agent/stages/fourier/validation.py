"""Stage-local validation for Fourier-transform jobs."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob, derive_job_kinematics
from lamet_agent.manifest_params import merge_stage_params


INFERRED_OBSERVABLES = {
    (target, "quark", hadron): f"{hadron}_quark_quasi_{target}"
    for target in ("pdf", "gpd")
    for hadron in ("pion", "nucleon")
}
INFERRED_OBSERVABLES.update(
    {
        ("pdf", "gluon", "pion"): "pion_gluon_quasi_pdf",
        ("pdf", "gluon", "nucleon"): "nucleon_gluon_quasi_pdf",
    }
)
PUBLIC_OBSERVABLES = frozenset({*INFERRED_OBSERVABLES.values(), "meson_quasi_da"})


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
    polarization = str(params.get("polarization", "")).lower()
    inferred_observable = INFERRED_OBSERVABLES.get((target, parton, hadron))
    if "observable" in params:
        observable = str(params["observable"]).lower().replace("-", "_").replace(" ", "_")
        if observable not in PUBLIC_OBSERVABLES:
            return [f"Fourier observable must be one of {sorted(PUBLIC_OBSERVABLES)}."]
    if target in {"pdf", "gpd"} and not polarization:
        missing.append("polarization")
    if parton == "gluon" and (target != "pdf" or polarization not in {"", "unpolarized"}):
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
