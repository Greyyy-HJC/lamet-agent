from lamet_agent.agent import run_agent
from lamet_agent.manifest import AnalysisManifest


def test_run_agent_executes_default_stages_with_mock_model() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "goal": "full_lamet_pipeline",
            "correlators": [
                {
                    "dataset_id": "c2",
                    "kind": "2pt",
                    "path": "fake/c2.h5",
                    "format": "hdf5",
                }
            ],
            "kernels": [
                {
                    "kernel_id": "k1",
                    "function": "lamet_agent.kernels:identity_kernel",
                }
            ],
        }
    )

    result = run_agent(manifest, model="mock")
    assert result["status"] == "completed"
    assert len(result["completed_stages"]) == 5
    assert result["actions"][0]["action"]["action"] == "call_tool"
