"""Tests for three-point correlator loading, filtering, and analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.extensions.three_point import filter_bad_points, ratio_imag_function, ratio_real_function
from lamet_agent.extensions.two_point import two_point_fit_function
from lamet_agent.loaders import load_correlator_dataset
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import CorrelatorSpec
from lamet_agent.workflows import execute_manifest


class ThreePointAnalysisTests(unittest.TestCase):
    """Verify the integrated three-point correlator analysis helpers."""

    def test_mad_filter_preserves_uniform_large_scale_and_flags_spike(self) -> None:
        uniform = np.full((3, 10), 1.0e9, dtype=float)
        filtered_uniform, uniform_info = filter_bad_points(uniform, axis=-1, mode="mad")
        np.testing.assert_allclose(filtered_uniform, uniform)
        self.assertEqual(uniform_info.flagged_count, 0)

        spiky = np.ones((2, 8), dtype=float)
        spiky[1, 3] = 1.0e6
        filtered_spiky, spiky_info = filter_bad_points(spiky, axis=-1, mode="mad")
        self.assertEqual(spiky_info.flagged_count, 1)
        self.assertAlmostEqual(filtered_spiky[1, 3], 1.0)

    def test_txt_three_point_loader_reconstructs_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw = np.array(
                [
                    [8, 0, 1.0, 2.0, 0.1, 0.2],
                    [8, 1, 3.0, 4.0, 0.3, 0.4],
                    [10, 0, 5.0, 6.0, 0.5, 0.6],
                    [10, 1, 7.0, 8.0, 0.7, 0.8],
                ],
                dtype=float,
            )
            path = tmp_path / "three_point.txt"
            np.savetxt(path, raw)
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            spec = CorrelatorSpec.from_dict(
                {
                    "kind": "three_point",
                    "path": str(path),
                    "file_format": "txt",
                    "label": "toy_three_point",
                }
            )
            dataset = load_correlator_dataset(spec, manifest_path)
            self.assertEqual(dataset.samples.shape, (2, 2, 2))
            np.testing.assert_array_equal(dataset.axis, np.array([8.0, 10.0]))
            np.testing.assert_array_equal(dataset.extra_axes["tau"], np.array([0.0, 1.0]))
            self.assertTrue(np.iscomplexobj(dataset.samples))

    def test_three_point_manifest_runs_correlator_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = self._write_toy_manifest(tmp_path)
            run = execute_manifest(manifest_path, planner=RuleBasedPlanner())
            stage_dir = run.run_directory / "stages" / "correlator_analysis"
            self.assertTrue((stage_dir / "bare_qpdf_vs_z.pdf").exists())
            self.assertTrue((stage_dir / "bare_qpdf_summary.json").exists())
            self.assertTrue((stage_dir / "toy_z0" / "ratio_real.pdf").exists())
            self.assertTrue((stage_dir / "toy_z0" / "fh_real.pdf").exists())
            self.assertTrue((stage_dir / "toy_z0" / "joint_ratio_fh_fit_result.txt").exists())
            payload = run.stage_results[0].payload
            self.assertIn("three_point", payload)
            self.assertIn("bare_qpdf", payload)
            self.assertIn("qpdf_families", payload)
            self.assertIn("joint_ratio_fh", payload["three_point"][0]["fits"])
            self.assertEqual(
                payload["three_point"][0]["fits"]["joint_ratio_fh"]["fit_windows"]["ratio"]["imag"]["tau_cut"],
                3,
            )
            self.assertEqual(payload["qpdf_families"][0]["sample_count"], 40)
            sample_artifacts = list(stage_dir.glob("bare_qpdf_samples_*.npz"))
            self.assertEqual(len(sample_artifacts), 1)
            with np.load(sample_artifacts[0]) as sample_dump:
                self.assertEqual(sample_dump["real_samples"].shape, (40, 1))
                self.assertEqual(sample_dump["imag_samples"].shape, (40, 1))

    def test_toy_qpdf_ft_manifest_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = self._write_toy_manifest(tmp_path, include_fourier_transform=True)
            run = execute_manifest(manifest_path, planner=RuleBasedPlanner())
            stage_dir = run.run_directory / "stages" / "fourier_transform"
            self.assertTrue((stage_dir / "qpdf_ft_summary.json").exists())
            self.assertTrue((stage_dir / "qpdf_ft_real.pdf").exists())
            payload = run.stage_results[-1].payload
            self.assertIn("momentum_space", payload)
            self.assertEqual(payload["_qpdf_ft_samples"]["real_samples"].shape[0], 40)

    def _write_toy_manifest(self, tmp_path: Path, *, include_fourier_transform: bool = False) -> Path:
        rng = np.random.default_rng(123)
        lt = 32
        configuration_count = 40
        t_axis = np.arange(lt, dtype=float)
        parameters = {
            "E0": 0.45,
            "log(dE1)": np.log(0.35),
            "z0": 1.2,
            "z1": 0.5,
            "O00_re": 0.30,
            "O00_im": 0.04,
            "O01_re": 0.08,
            "O01_im": 0.02,
            "O11_re": 0.05,
            "O11_im": 0.01,
        }
        two_point = np.asarray(
            two_point_fit_function(t_axis, parameters, temporal_extent=lt, state_count=2, boundary="periodic"),
            dtype=float,
        )
        two_point_rows = np.column_stack(
            [t_axis] + [two_point + rng.normal(scale=two_point * 0.01 + 1.0e-6, size=lt) for _ in range(configuration_count)]
        )
        two_point_path = tmp_path / "two_point.txt"
        np.savetxt(two_point_path, two_point_rows)

        z_values = [0, 1, 2, 3, 4, 5] if include_fourier_transform else [0]
        three_point_specs = []
        for z_value in z_values:
            rows = []
            z_scale = np.exp(-0.15 * z_value)
            for tsep in (8, 10, 12):
                denominator = np.asarray(
                    two_point_fit_function(
                        np.array([float(tsep)]),
                        parameters,
                        temporal_extent=lt,
                        state_count=2,
                        boundary="periodic",
                    ),
                    dtype=float,
                )[0]
                for tau in range(12):
                    ratio = (
                        ratio_real_function(float(tsep), float(tau), parameters, lt)
                        + 1j * ratio_imag_function(float(tsep), float(tau), parameters, lt)
                    ) * z_scale
                    real_samples = []
                    imag_samples = []
                    for _ in range(configuration_count):
                        noisy_denominator = denominator + rng.normal(scale=max(abs(denominator) * 0.01, 1.0e-8))
                        noisy_ratio = ratio + (rng.normal(scale=0.002) + 1j * rng.normal(scale=0.002))
                        three_point_value = noisy_ratio * noisy_denominator
                        real_samples.append(three_point_value.real)
                        imag_samples.append(three_point_value.imag)
                    rows.append([tsep, tau, *real_samples, *imag_samples])
            three_point_path = tmp_path / f"three_point_z{z_value}.txt"
            np.savetxt(three_point_path, np.asarray(rows, dtype=float))
            three_point_specs.append(
                {
                    "kind": "three_point",
                    "path": str(three_point_path),
                    "file_format": "txt",
                    "label": f"toy_z{z_value}",
                    "metadata": {"z": z_value, "b": 0, "gamma": "gt", "flavor": "u-d"},
                }
            )

        manifest = {
            "goal": "custom",
            "correlators": [
                {
                    "kind": "two_point",
                    "path": str(two_point_path),
                    "file_format": "txt",
                    "label": "toy_two_point",
                    "metadata": {"Lt": lt},
                },
                *three_point_specs,
            ],
            "metadata": {"ensemble": "toy", "conventions": "toy"},
            "kernel": {
                "callable_name": "identity_kernel",
                "source": "def identity_kernel(axis, values, metadata):\n    return values\n",
            },
            "workflow": {
                "stages": ["correlator_analysis", "renormalization", "fourier_transform"]
                if include_fourier_transform
                else ["correlator_analysis"],
                "stage_parameters": {
                    "correlator_analysis": {
                        "two_point": {
                            "temporal_extent": lt,
                            "fit": {"tmin": 4, "tmax": 13},
                        },
                        "three_point": {
                            "fit_modes": ["ratio", "fh", "joint_ratio_fh"],
                            "primary_fit_mode": "joint_ratio_fh",
                            "ratio": {
                                "fit_tsep": [8, 10, 12],
                                "tau_cut": 2,
                                "imag": {
                                    "fit_tsep": [10, 12],
                                    "tau_cut": 3,
                                },
                            },
                            "fh": {
                                "fit_tsep": [8, 10, 12],
                                "tau_cut": 2,
                                "real": {
                                    "fit_tsep": [10, 12],
                                },
                            },
                        },
                    },
                    "fourier_transform": {
                        "family_selector": {"fit_mode": "joint_ratio_fh", "b": 0, "gamma": "gt", "flavor": "u-d"},
                        "gauge_type": "cg",
                        "physics": {
                            "lattice_spacing_fm": 0.09,
                            "spatial_extent": 64,
                            "momentum_vector": [4, 4, 0],
                            "coordinate_direction": [1, 1, 0],
                        },
                        "x_grid": {"values": [-1.0, 0.0, 1.0]},
                        "extrapolation": {
                            "fit_idx_range": [2, 6],
                            "extrapolated_length": 8.0,
                            "weight_ini": 0.0,
                            "m0": 0.0,
                        },
                    },
                },
            },
            "outputs": {
                "directory": str(tmp_path / "outputs"),
                "plot_formats": ["pdf"],
                "data_formats": ["json"],
                "keep_intermediates": True,
            },
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path


if __name__ == "__main__":
    unittest.main()
