"""Tests for the integrated two-point correlator analysis workflow."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.extensions.statistics import gv
from lamet_agent.extensions.two_point import fit_two_point_correlator, two_point_fit_function
from lamet_agent.loaders import load_correlator_dataset
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import CorrelatorSpec
from lamet_agent.workflows import execute_manifest


class TwoPointAnalysisTests(unittest.TestCase):
    """Verify raw two-point data is loaded and analyzed end-to-end."""

    def test_txt_two_point_loader_supports_complex_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw = np.array(
                [
                    [0.0, 1.0, 2.0, 0.1, 0.2],
                    [1.0, 3.0, 4.0, 0.3, 0.4],
                ],
                dtype=float,
            )
            path = tmp_path / "two_point_complex.txt"
            np.savetxt(path, raw)
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            spec = CorrelatorSpec.from_dict(
                {
                    "kind": "two_point",
                    "path": str(path),
                    "file_format": "txt",
                    "label": "complex_two_point",
                    "metadata": {"complex_samples": True},
                }
            )
            dataset = load_correlator_dataset(spec, manifest_path)
            self.assertEqual(dataset.samples.shape, (2, 2))
            self.assertIn("imag_samples", dataset.extra_axes)
            np.testing.assert_allclose(dataset.extra_axes["imag_samples"], np.array([[0.1, 0.2], [0.3, 0.4]]))
            np.testing.assert_allclose(dataset.values, np.array([1.5, 3.5]))

    def test_loader_keeps_all_raw_samples(self) -> None:
        manifest_path = ROOT / "examples" / "two_point_analysis_manifest.json"
        spec = CorrelatorSpec.from_dict(
            {
                "kind": "two_point",
                "path": "data/two_point_raw_demo.csv",
                "file_format": "csv",
                "label": "demo_two_point_raw",
            }
        )
        dataset = load_correlator_dataset(spec, manifest_path)
        self.assertEqual(dataset.samples.shape, (64, 314))
        self.assertEqual(dataset.axis.shape, (64,))
        np.testing.assert_allclose(dataset.values, np.mean(dataset.samples, axis=1))

    def test_two_point_manifest_runs_correlator_analysis(self) -> None:
        example_manifest = ROOT / "examples" / "two_point_analysis_manifest.json"
        payload = json.loads(example_manifest.read_text(encoding="utf-8"))
        payload["correlators"][0]["path"] = str(ROOT / "examples" / "data" / "two_point_raw_demo.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            payload["outputs"]["directory"] = tmpdir
            temp_manifest = Path(tmpdir) / "manifest.json"
            temp_manifest.write_text(json.dumps(payload), encoding="utf-8")
            run = execute_manifest(temp_manifest, planner=RuleBasedPlanner())
            stage_dir = run.run_directory / "stages" / "correlator_analysis"
            self.assertTrue((stage_dir / "correlator_analysis.pdf").exists())
            self.assertTrue((stage_dir / "effective_mass.pdf").exists())
            self.assertTrue((stage_dir / "effective_mass_comparison.pdf").exists())
            self.assertTrue((stage_dir / "two_point_fit_summary.json").exists())
            self.assertTrue((stage_dir / "two_point_fit_result.txt").exists())
            self.assertTrue((stage_dir / "correlator_analysis_settings.json").exists())
            payload = run.stage_results[0].payload
            self.assertEqual(payload["resampling"]["configuration_count"], 314)
            self.assertEqual(payload["resampling"]["method"], "bootstrap")
            self.assertEqual(len(payload["effective_mass"]["axis"]), 63)

    def test_fit_handles_small_magnitude_normalized_correlator(self) -> None:
        if gv is None:
            self.skipTest("gvar is not installed")

        times = np.arange(64, dtype=float)
        truth = {"E0": 0.82, "log(dE1)": np.log(0.48), "z0": 0.18, "z1": 0.46}
        scale = 1.0e-33
        mean = scale * np.asarray(
            two_point_fit_function(times, truth, temporal_extent=64, state_count=2, boundary="periodic"),
            dtype=float,
        )
        averaged = gv.gvar(mean, 0.01 * np.abs(mean))
        fit = fit_two_point_correlator(
            averaged,
            temporal_extent=64,
            tmin=4,
            tmax=13,
            state_count=2,
            boundary="periodic",
            normalize=True,
        )
        self.assertGreater(fit.Q, 0.05)
        self.assertAlmostEqual(float(gv.mean(fit.p["E0"])), truth["E0"], delta=0.1)


if __name__ == "__main__":
    unittest.main()
