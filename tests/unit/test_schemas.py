import pytest
from pydantic import ValidationError

from lamet_agent.manifest import AnalysisManifest


def _payload() -> dict:
    return {
        "metadata": {
            "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
            "parton": "quark", "resample_mode": "jk", "random_seed": 1984, "stages": ["correlator_analysis"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]}},
    }


def test_manifest_schema_uses_metadata_inputs_and_stage_jobs() -> None:
    manifest = AnalysisManifest.model_validate(_payload())
    assert manifest.run_id == "demo"
    assert manifest.metadata.workers == 1
    assert manifest.stages["correlator_analysis"].jobs[0].id == "ca"


def test_manifest_accepts_positive_workers() -> None:
    payload = _payload()
    payload["metadata"]["workers"] = 4
    assert AnalysisManifest.model_validate(payload).metadata.workers == 4


@pytest.mark.parametrize("workers", [0, -1, 1.5, "2"])
def test_manifest_rejects_invalid_workers(workers) -> None:
    payload = _payload()
    payload["metadata"]["workers"] = workers
    with pytest.raises(ValidationError, match="workers"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_forward_job_reference() -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["jobs"][0]["inputs"] = {"input": "later"}
    with pytest.raises(ValidationError, match="unavailable upstream"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_bs_mode_without_bs_samples() -> None:
    payload = _payload()
    payload["metadata"]["resample_mode"] = "bs"
    with pytest.raises(ValidationError, match="bs_samples"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_bs_mode_with_bs_samples() -> None:
    payload = _payload()
    payload["metadata"]["resample_mode"] = "bs"
    payload["metadata"]["bs_samples"] = 500
    manifest = AnalysisManifest.model_validate(payload)
    assert manifest.metadata.bs_samples == 500


def test_manifest_rejects_zs_fm_in_kernel_parameters() -> None:
    payload = _payload()
    payload["inputs"]["kernels"] = [
        {
            "stage": "perturbative_matching",
            "kernel_id": "CG_gt_qPDF_hybrid_NLO",
            "kernel_path": "kernels.py",
            "scheme": "hybrid_ratio",
            "kernel_parameters": {"zs_fm": 0.2},
        }
    ]
    with pytest.raises(ValidationError, match=r"inputs\.kernels\[0\]\.kernel_parameters\.zs_fm"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_zs_fm_in_renormalization_scheme_parameters() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization"]
    payload["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid_ratio", "scheme_parameters": {"zs_fm": 0.2}},
            "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
        }
    }
    with pytest.raises(ValidationError, match=r"renormalization\.defaults\.scheme_parameters\.zs_fm"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_flat_zs_fm_defaults_and_job_overrides() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization", "perturbative_matching"]
    payload["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid_ratio", "zs_fm": 0.2},
            "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
        },
        "perturbative_matching": {
            "defaults": {"zs_fm": 0.2},
            "jobs": [{"id": "mt", "inputs": {"quasi": "rn"}, "params": {"zs_fm": 0.3}}],
        },
    }

    manifest = AnalysisManifest.model_validate(payload)

    assert manifest.stages["renormalization"].defaults["zs_fm"] == 0.2
    assert manifest.stages["perturbative_matching"].jobs[0].params["zs_fm"] == 0.3
