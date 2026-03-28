"""Stage contract tests for the demo pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.kernel import load_kernel
from lamet_agent.loaders import CorrelatorDataset
from lamet_agent.schemas import Manifest
from lamet_agent.stages import CorrelatorAnalysisStage, FourierTransformStage, RenormalizationStage
from lamet_agent.stages.base import StageContext


def manifest_payload() -> dict:
    """Return a manifest payload with PDF and CSV outputs for test runs."""
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
        "outputs": {
            "directory": "outputs",
            "plot_formats": ["pdf"],
            "data_formats": ["csv"],
            "keep_intermediates": True
        }
    }


class StageTests(unittest.TestCase):
    """Ensure stage results follow the normalized contract."""

    def test_fourier_transform_requires_qpdf_payload(self) -> None:
        manifest = Manifest.from_dict(manifest_payload())
        kernel = load_kernel(manifest.kernel)
        dataset = CorrelatorDataset(
            kind="two_point",
            label="demo",
            path=Path("data.csv"),
            axis=[0.0, 1.0, 2.0],
            values=[1.0, 0.5, 0.2],
            samples=None,
            metadata={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            context = StageContext(
                manifest=manifest,
                run_directory=Path(tmpdir),
                datasets={"demo": dataset},
                kernel=kernel,
            )
            correlator_result = CorrelatorAnalysisStage().run(context)
            context.stage_payloads[correlator_result.stage_name] = correlator_result.payload
            renorm_result = RenormalizationStage().run(context)
            self.assertTrue(correlator_result.artifacts)
            self.assertIn("axis", renorm_result.payload)
            context.stage_payloads[renorm_result.stage_name] = renorm_result.payload
            with self.assertRaises(ValueError):
                FourierTransformStage().run(context)

    def test_qpdf_passthrough_and_sample_wise_fourier_transform(self) -> None:
        manifest_data = manifest_payload()
        manifest_data["workflow"] = {
            "stages": ["correlator_analysis", "renormalization", "fourier_transform"],
            "stage_parameters": {
                "fourier_transform": {
                    "family_selector": {"fit_mode": "joint_ratio_fh", "b": 0, "gamma": "gt", "flavor": "u-d"},
                    "gauge_type": "cg",
                    "sample_transform_workers": 2,
                    "physics": {
                        "lattice_spacing_fm": 0.09,
                        "spatial_extent": 64,
                        "momentum_vector": [4, 4, 0],
                        "coordinate_direction": [1, 1, 0],
                    },
                    "x_grid": {"values": [-1.0, 0.0, 1.0]},
                    "extrapolation": {
                        "fit_idx_range": [2, 5],
                        "extrapolated_length": 8.0,
                        "weight_ini": 0.0,
                        "m0": 0.0,
                    },
                }
            },
        }
        manifest = Manifest.from_dict(manifest_data)
        kernel = load_kernel(manifest.kernel)
        dataset = CorrelatorDataset(
            kind="two_point",
            label="demo",
            path=Path("data.csv"),
            axis=[0.0, 1.0, 2.0],
            values=[1.0, 0.5, 0.2],
            samples=None,
            metadata={},
        )
        z_axis = np.arange(6, dtype=float)
        lambda_step = 8.0 * np.pi / 64.0
        lambda_axis = z_axis * lambda_step
        base_real = np.exp(-0.6 * lambda_axis)
        base_imag = 0.2 * np.exp(-0.6 * lambda_axis)
        real_samples = np.vstack([base_real + shift for shift in (-0.03, -0.01, 0.01, 0.03)])
        imag_samples = np.vstack([base_imag + shift for shift in (-0.015, -0.005, 0.005, 0.015)])
        correlator_payload = {
            "axis": np.asarray([0.0, 1.0, 2.0]),
            "values": np.asarray([1.0, 0.5, 0.2]),
            "qpdf_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "ss_sp": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "resampling_method": "jackknife",
                    },
                    "z_axis": z_axis.tolist(),
                    "sample_count": 4,
                    "real": {"mean": np.mean(real_samples, axis=0).tolist(), "error": np.std(real_samples, axis=0, ddof=1).tolist()},
                    "imag": {"mean": np.mean(imag_samples, axis=0).tolist(), "error": np.std(imag_samples, axis=0, ddof=1).tolist()},
                    "sample_artifact": None,
                }
            ],
            "_qpdf_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "ss_sp": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "resampling_method": "jackknife",
                    },
                    "z_axis": z_axis,
                    "sample_count": 4,
                    "real_mean": np.mean(real_samples, axis=0),
                    "real_error": np.std(real_samples, axis=0, ddof=1),
                    "imag_mean": np.mean(imag_samples, axis=0),
                    "imag_error": np.std(imag_samples, axis=0, ddof=1),
                    "real_samples": real_samples,
                    "imag_samples": imag_samples,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            context = StageContext(
                manifest=manifest,
                run_directory=Path(tmpdir),
                datasets={"demo": dataset},
                kernel=kernel,
                stage_payloads={"correlator_analysis": correlator_payload},
            )
            renorm_result = RenormalizationStage().run(context)
            self.assertEqual(renorm_result.payload["mode"], "identity")
            self.assertTrue(renorm_result.payload["_qpdf_families"])
            context.stage_payloads["renormalization"] = renorm_result.payload
            fourier_result = FourierTransformStage().run(context)
            self.assertIn("momentum_space", fourier_result.payload)
            self.assertIn("_qpdf_ft_samples", fourier_result.payload)
            self.assertEqual(fourier_result.payload["_qpdf_ft_samples"]["real_samples"].shape, (4, 3))
            self.assertEqual(fourier_result.payload["extrapolation"]["sample_transform_workers"], 2)
            self.assertTrue((Path(tmpdir) / "stages" / "fourier_transform" / "qpdf_ft_summary.json").exists())

    def test_qpdf_fourier_transform_rejects_unsupported_gauge_type(self) -> None:
        manifest_data = manifest_payload()
        manifest_data["workflow"] = {
            "stages": ["correlator_analysis", "renormalization", "fourier_transform"],
            "stage_parameters": {
                "fourier_transform": {
                    "gauge_type": "gi",
                    "physics": {
                        "lattice_spacing_fm": 0.09,
                        "spatial_extent": 64,
                        "momentum_vector": [4, 4, 0],
                        "coordinate_direction": [1, 1, 0],
                    },
                }
            },
        }
        manifest = Manifest.from_dict(manifest_data)
        kernel = load_kernel(manifest.kernel)
        dataset = CorrelatorDataset(
            kind="two_point",
            label="demo",
            path=Path("data.csv"),
            axis=[0.0, 1.0],
            values=[1.0, 0.5],
            samples=None,
            metadata={},
        )
        correlator_payload = {
            "_qpdf_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "ss_sp": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "resampling_method": "jackknife",
                    },
                    "z_axis": np.arange(6, dtype=float),
                    "sample_count": 3,
                    "real_mean": np.ones(6),
                    "real_error": np.full(6, 0.01),
                    "imag_mean": np.zeros(6),
                    "imag_error": np.full(6, 0.01),
                    "real_samples": np.ones((3, 6)),
                    "imag_samples": np.zeros((3, 6)),
                }
            ],
            "qpdf_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "ss_sp": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "resampling_method": "jackknife",
                    },
                    "z_axis": list(range(6)),
                    "sample_count": 3,
                    "real": {"mean": [1.0] * 6, "error": [0.01] * 6},
                    "imag": {"mean": [0.0] * 6, "error": [0.01] * 6},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            context = StageContext(
                manifest=manifest,
                run_directory=Path(tmpdir),
                datasets={"demo": dataset},
                kernel=kernel,
                stage_payloads={"renormalization": correlator_payload},
            )
            with self.assertRaises(ValueError):
                FourierTransformStage().run(context)


if __name__ == "__main__":
    unittest.main()
