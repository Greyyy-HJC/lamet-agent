from lamet_agent.manifest import AnalysisManifest


def test_manifest_schema_accepts_correlators_and_kernels() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "mock_run",
            "correlators": [
                {
                    "dataset_id": "c2",
                    "kind": "2pt",
                    "path": "fake/c2.txt",
                    "format": "txt",
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
    assert manifest.correlators[0].dataset_id == "c2"
