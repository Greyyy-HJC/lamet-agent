import json
from pathlib import Path

from lamet_agent.manifest import validate_manifest_file


def test_validate_manifest_file_checks_kernel_reference(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "demo",
                "correlators": [
                    {
                        "dataset_id": "two_point",
                        "kind": "2pt",
                        "path": "fake/two_point.txt",
                        "format": "txt",
                    }
                ],
                "kernels": [
                    {
                        "kernel_id": "identity",
                        "function": "lamet_agent.kernels:identity_kernel",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = validate_manifest_file(manifest_path)
    assert parsed.run_id == "demo"
    assert parsed.kernels[0].kernel_id == "identity"
