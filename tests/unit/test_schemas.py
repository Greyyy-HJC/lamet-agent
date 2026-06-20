import pytest
from pydantic import ValidationError

from lamet_agent.manifest import AnalysisManifest


def _payload() -> dict:
    return {
        "metadata": {
            "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
            "parton": "quark", "resample_mode": "jk", "stages": ["correlator_analysis"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]}},
    }


def test_manifest_schema_uses_metadata_inputs_and_stage_jobs() -> None:
    manifest = AnalysisManifest.model_validate(_payload())
    assert manifest.run_id == "demo"
    assert manifest.stages["correlator_analysis"].jobs[0].id == "ca"


def test_manifest_rejects_forward_job_reference() -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["jobs"][0]["inputs"] = {"input": "later"}
    with pytest.raises(ValidationError, match="unavailable upstream"):
        AnalysisManifest.model_validate(payload)
