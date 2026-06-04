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

    data_file = tmp_path / "fake" / "two_point.txt"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("0\n", encoding="utf-8")

    parsed = validate_manifest_file(manifest_path)
    assert parsed.run_id == "demo"
    assert parsed.kernels[0].kernel_id == "identity"
    assert parsed.manifest_dir == manifest_path.parent.resolve()
    assert parsed.correlators[0].path == str(data_file.resolve())


def test_validate_manifest_file_resolves_repo_root_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    examples_dir = repo_root / "examples"
    data_dir = examples_dir / "fake_data" / "data"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "fake_2pt.h5"
    data_file.write_bytes(b"")

    (repo_root / "pyproject.toml").write_text('name = "lamet-agent"\n', encoding="utf-8")
    manifest_path = examples_dir / "workflow_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "demo",
                "correlators": [
                    {
                        "dataset_id": "proton_2pt_fake",
                        "kind": "2pt",
                        "path": "examples/fake_data/data/fake_2pt.h5",
                        "format": "hdf5",
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
    assert parsed.project_root == repo_root.resolve()
    assert parsed.correlators[0].path == str(data_file.resolve())
