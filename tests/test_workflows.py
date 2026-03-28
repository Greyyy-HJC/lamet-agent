"""Workflow planner tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.errors import WorkflowResolutionError
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import Manifest
from lamet_agent.workflows import execute_manifest


def base_manifest_payload() -> dict:
    return {
        "goal": "parton_distribution_function",
        "correlators": [
            {
                "kind": "two_point",
                "path": "data.csv",
                "file_format": "csv",
                "label": "demo",
                "metadata": {
                    "setup_id": "demo_setup",
                    "momentum": [0, 0, 0],
                    "smearing": "SS",
                },
            }
        ],
        "metadata": {
            "purpose": "smoke",
            "analysis": {"gauge": "cg", "hadron": "pion", "channel": "qpdf"},
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


class WorkflowPlannerTests(unittest.TestCase):
    def test_pdf_goal_resolves_full_pipeline(self) -> None:
        planner = RuleBasedPlanner()
        manifest = Manifest.from_dict(base_manifest_payload())
        plan = planner.resolve(manifest)
        self.assertEqual(plan.final_observable, "qpdf")
        self.assertEqual(
            plan.stage_names,
            [
                "correlator_analysis",
                "renormalization",
                "fourier_transform",
                "perturbative_matching",
                "physical_limit",
            ],
        )

    def test_missing_two_point_input_fails_default_workflow(self) -> None:
        planner = RuleBasedPlanner()
        payload = base_manifest_payload()
        payload["correlators"][0]["kind"] = "three_point"
        payload["correlators"][0]["metadata"]["displacement"] = {"b": 0, "z": 0}
        payload["correlators"][0]["metadata"]["operator"] = {"gamma": "gt", "flavor": "u-d"}
        manifest = Manifest.from_dict(payload)
        with self.assertRaises(WorkflowResolutionError):
            planner.resolve(manifest)

    def test_execute_manifest_emits_stage_progress_events(self) -> None:
        planner = RuleBasedPlanner()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "data.csv"
            data_path.write_text("0,1.0\n1,0.5\n2,0.25\n", encoding="utf-8")
            payload = base_manifest_payload()
            payload["goal"] = "custom"
            payload["correlators"][0]["path"] = str(data_path)
            payload["workflow"] = {"stages": ["correlator_analysis"]}
            payload["outputs"] = {"directory": str(tmp_path / "outputs"), "plot_formats": ["pdf"], "data_formats": ["json"]}
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            events: list[dict] = []
            run = execute_manifest(manifest_path, planner=planner, progress_callback=events.append)
            self.assertEqual(run.stage_results[0].stage_name, "correlator_analysis")
            self.assertEqual([event["event"] for event in events], ["stage_started", "stage_completed"])


if __name__ == "__main__":
    unittest.main()
