"""Workflow planner tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.errors import WorkflowResolutionError
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import Manifest


def base_manifest_payload() -> dict:
    """Return a minimal manifest payload for planner tests."""
    return {
        "goal": "parton_distribution_function",
        "correlators": [
            {
                "kind": "two_point",
                "path": "data.csv",
                "file_format": "csv",
                "label": "demo",
            }
        ],
        "metadata": {"ensemble": "e1", "conventions": "demo"},
        "kernel": {
            "source": "def demo_kernel(axis, values, metadata):\n    return values\n",
            "callable_name": "demo_kernel",
        },
    }


class WorkflowPlannerTests(unittest.TestCase):
    """Verify rule-based workflow resolution."""

    def test_pdf_goal_resolves_full_pipeline(self) -> None:
        planner = RuleBasedPlanner()
        manifest = Manifest.from_dict(base_manifest_payload())
        plan = planner.resolve(manifest)
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

    def test_distribution_amplitude_changes_final_observable(self) -> None:
        planner = RuleBasedPlanner()
        payload = base_manifest_payload()
        payload["goal"] = "distribution_amplitude"
        manifest = Manifest.from_dict(payload)
        plan = planner.resolve(manifest)
        self.assertEqual(plan.final_observable, "distribution_amplitude")

    def test_missing_two_point_input_fails_default_workflow(self) -> None:
        planner = RuleBasedPlanner()
        payload = base_manifest_payload()
        payload["correlators"][0]["kind"] = "three_point"
        manifest = Manifest.from_dict(payload)
        with self.assertRaises(WorkflowResolutionError):
            planner.resolve(manifest)


if __name__ == "__main__":
    unittest.main()
