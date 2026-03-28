"""Schema validation tests for the v1 manifest contract."""

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
    return {
        "goal": "parton_distribution_function",
        "correlators": [
            {
                "kind": "two_point",
                "path": data_path,
                "file_format": file_format,
                "label": "demo",
                "metadata": {
                    "setup_id": "demo_setup",
                    "momentum": [0, 0, 0],
                    "smearing": "SS",
                },
            }
        ],
        "metadata": {
            "purpose": "physics",
            "analysis": {
                "gauge": "cg",
                "hadron": "pion",
                "channel": "qpdf",
            },
            "conventions": "demo",
            "setups": {
                "demo_setup": {
                    "lattice_action": "demo",
                    "n_f": 2,
                    "lattice_spacing_fm": 0.09,
                    "spatial_extent": 32,
                    "temporal_extent": 64,
                    "pion_mass_valence_gev": 0.3,
                    "pion_mass_sea_gev": 0.3,
                }
            },
        },
        "kernel": {
            "source": "def demo_kernel(axis, values, metadata):\n    return values\n",
            "callable_name": "demo_kernel",
        },
    }


class ManifestTests(unittest.TestCase):
    def test_manifest_from_dict_accepts_valid_payload(self) -> None:
        manifest = Manifest.from_dict(build_manifest_dict())
        self.assertEqual(manifest.analysis_metadata["channel"], "qpdf")
        self.assertEqual(manifest.setup_metadata("demo_setup")["spatial_extent"], 32)

    def test_manifest_expands_nested_three_point_metadata(self) -> None:
        payload = build_manifest_dict(data_path="three_point_z{z:02d}.txt", file_format="txt")
        payload["correlators"][0] = {
            "kind": "three_point",
            "path": "three_point_z{z:02d}.txt",
            "file_format": "txt",
            "label": "toy_z{z:02d}",
            "expand": {"z": {"start": 0, "stop": 2}},
            "metadata": {
                "setup_id": "demo_setup",
                "momentum": [0, 0, 2],
                "smearing": "SS",
                "displacement": {"b": 0, "z": "{z}"},
                "operator": {"gamma": "gt", "flavor": "u-d"},
            },
        }
        manifest = Manifest.from_dict(payload)
        self.assertEqual([item.metadata["displacement"]["z"] for item in manifest.correlators], [0, 1, 2])

    def test_goal_and_channel_must_match(self) -> None:
        payload = build_manifest_dict()
        payload["metadata"]["analysis"]["channel"] = "qda"
        with self.assertRaises(ManifestValidationError):
            Manifest.from_dict(payload)

    def test_unknown_setup_id_fails(self) -> None:
        payload = build_manifest_dict()
        payload["correlators"][0]["metadata"]["setup_id"] = "missing"
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

    def test_curated_pion_cg_qtmdpdf_manifest_loads(self) -> None:
        manifest = load_manifest(ROOT / "examples" / "pion_cg_qtmdpdf_manifest.json")
        self.assertEqual(manifest.goal, "parton_distribution_function")
        self.assertEqual(manifest.analysis_metadata["hadron"], "pion")
        self.assertEqual(manifest.analysis_metadata["channel"], "qpdf")
        self.assertEqual(manifest.observable_name_for_b(4), "qtmdpdf")
        self.assertEqual(len(manifest.correlators), 106)


if __name__ == "__main__":
    unittest.main()
