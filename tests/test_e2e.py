"""End-to-end smoke test for the demo workflow."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.workflows import execute_manifest


class EndToEndTests(unittest.TestCase):
    """Run the full demo workflow and verify key artifacts."""

    def test_demo_manifest_runs_and_writes_reports(self) -> None:
        example_manifest = ROOT / "examples" / "demo_manifest.json"
        payload = json.loads(example_manifest.read_text(encoding="utf-8"))
        payload["correlators"][0]["path"] = str(ROOT / "examples" / "data" / "qpdf_ft_demo" / "two_point_qpdf_ft_demo.txt")
        payload["correlators"][1]["path"] = str(
            ROOT / "examples" / "data" / "qpdf_ft_demo" / "three_point_qpdf_ft_demo_z{z:02d}.txt"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            payload["outputs"]["directory"] = tmpdir
            temp_manifest = Path(tmpdir) / "manifest.json"
            temp_manifest.write_text(json.dumps(payload), encoding="utf-8")
            run = execute_manifest(temp_manifest, planner=RuleBasedPlanner())
            self.assertTrue((run.run_directory / "report.md").exists())
            self.assertTrue((run.run_directory / "report.json").exists())
            self.assertTrue((run.run_directory / "stages" / "physical_limit" / "physical_limit.pdf").exists())


if __name__ == "__main__":
    unittest.main()
