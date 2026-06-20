import json
from pathlib import Path

from lamet_agent.manifest import validate_manifest_file


def test_validate_manifest_resolves_root_relative_source_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    examples = root / "examples"
    examples.mkdir(parents=True)
    payload = {
        "metadata": {
            "run_id": "demo", "root_directory": "..", "artifacts_directory": "runs/artifacts",
            "target_observable": "pdf", "parton": "quark", "resample_mode": "jk",
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [{
                "correlator_id": "c2", "kind": "2pt", "data_path": "data/c2.h5",
                "ensemble": "E", "hadron": "pion", "gfix": "CG", "source_sink": "SS",
                "momentum": "PX0PY0PZ0", "a_fm": 0.1, "pz_gev": 0.0,
                "src_gamma": "5", "sink_gamma": "5",
            }],
            "artifacts": [], "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2"]}]}},
    }
    path = examples / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = validate_manifest_file(path)
    assert manifest.root_directory == root.resolve()
    assert manifest.artifacts_directory == (root / "runs" / "artifacts").resolve()
    assert manifest.correlators[0].data_path == str((root / "data" / "c2.h5").resolve())
