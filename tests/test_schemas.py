"""Schema validation tests for workflow manifests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.errors import ManifestValidationError
from lamet_agent.schemas import Manifest, load_manifest


def build_manifest_dict(data_path: str = "data.csv", file_format: str = "csv") -> dict:
    """Return a minimal valid manifest dictionary."""
    return {
        "goal": "parton_distribution_function",
        "correlators": [
            {
                "kind": "two_point",
                "path": data_path,
                "file_format": file_format,
                "label": "demo",
            }
        ],
        "metadata": {
            "ensemble": "e1",
            "conventions": "demo",
        },
        "kernel": {
            "source": "def demo_kernel(axis, values, metadata):\n    return values\n",
            "callable_name": "demo_kernel",
        },
    }


class ManifestTests(unittest.TestCase):
    """Cover manifest validation and path checking behavior."""

    def test_manifest_from_dict_accepts_valid_payload(self) -> None:
        manifest = Manifest.from_dict(build_manifest_dict())
        self.assertEqual(manifest.goal, "parton_distribution_function")
        self.assertEqual(manifest.outputs.plot_formats, ["pdf"])

    def test_manifest_from_dict_accepts_txt_correlators(self) -> None:
        manifest = Manifest.from_dict(build_manifest_dict(data_path="data.txt", file_format="txt"))
        self.assertEqual(manifest.correlators[0].file_format, "txt")

    def test_manifest_expands_correlator_families(self) -> None:
        payload = build_manifest_dict(data_path="data_{z:02d}.txt", file_format="txt")
        payload["correlators"][0]["label"] = "toy_z{z:02d}"
        payload["correlators"][0]["metadata"] = {"channel": "gamma_t"}
        payload["correlators"][0]["expand"] = {"z": {"start": 0, "stop": 2}}
        manifest = Manifest.from_dict(payload)
        self.assertEqual([item.label for item in manifest.correlators], ["toy_z00", "toy_z01", "toy_z02"])
        self.assertEqual([item.path for item in manifest.correlators], ["data_00.txt", "data_01.txt", "data_02.txt"])
        self.assertEqual([item.metadata["z"] for item in manifest.correlators], [0, 1, 2])

    def test_custom_goal_requires_explicit_stages(self) -> None:
        payload = build_manifest_dict()
        payload["goal"] = "custom"
        with self.assertRaises(ManifestValidationError):
            Manifest.from_dict(payload)

    def test_metadata_requires_ensemble_and_conventions(self) -> None:
        payload = build_manifest_dict()
        payload["metadata"] = {"ensemble": "e1"}
        with self.assertRaises(ManifestValidationError):
            Manifest.from_dict(payload)

    def test_load_manifest_checks_file_existence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "data.csv").write_text("0.0,1.0\n", encoding="utf-8")
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(json.dumps(build_manifest_dict()), encoding="utf-8")
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.manifest_path, manifest_path.resolve())


if __name__ == "__main__":
    unittest.main()
