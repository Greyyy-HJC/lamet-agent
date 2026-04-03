"""Focused stage tests for grouped family payloads."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.constants import lattice_unit_to_physical
from lamet_agent.kernel import load_kernel
from lamet_agent.loaders import CorrelatorDataset, load_correlator_dataset
from lamet_agent.extensions.qpdf_fourier import asymptotic_imag_function, asymptotic_real_function
from lamet_agent.schemas import CorrelatorSpec, Manifest
from lamet_agent.stages import CorrelatorAnalysisStage, FourierTransformStage, RenormalizationStage
from lamet_agent.stages.base import StageContext


def manifest_payload() -> dict:
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
                    "momentum": [4, 4, 0],
                    "smearing": "SS",
                },
            }
        ],
        "metadata": {
            "purpose": "physics",
            "analysis": {"gauge": "cg", "hadron": "proton", "channel": "qpdf"},
            "conventions": "demo",
            "setups": {
                "demo_setup": {
                    "lattice_action": "demo",
                    "n_f": 3,
                    "lattice_spacing_fm": 0.09,
                    "spatial_extent": 64,
                    "temporal_extent": 32,
                    "pion_mass_valence_gev": 0.3,
                    "pion_mass_sea_gev": 0.3,
                }
            },
        },
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
    def test_asymptotic_forms_distinguish_gi_and_cg_for_proton_and_pion(self) -> None:
        lam = np.asarray([1.0, 2.0], dtype=float)

        proton_params = {"b": 1.2, "c": 0.3, "d": 0.4, "e": -0.2, "m": 0.5, "n": 0.7}
        proton_gi = asymptotic_real_function(hadron="proton", gauge_type="gi")(lam, proton_params)
        proton_cg = asymptotic_real_function(hadron="proton", gauge_type="cg")(lam, proton_params)
        np.testing.assert_allclose(proton_cg, proton_gi / (lam**proton_params["n"]))

        pion_full_params = {
            "b1": 0.6,
            "b2": 1.1,
            "b3": -0.2,
            "d1": 0.1,
            "d2": 0.05,
            "d3": -0.08,
            "c1": 0.2,
            "c2": -0.1,
            "c3": 0.4,
            "e1": -0.3,
            "e2": 0.25,
            "e3": 0.1,
            "m": 0.35,
            "n": 0.9,
        }
        pion_gi_real = asymptotic_real_function(hadron="pion", gauge_type="gi", quark_sector="full")(lam, pion_full_params)
        pion_cg_real = asymptotic_real_function(hadron="pion", gauge_type="cg", quark_sector="full")(lam, pion_full_params)
        np.testing.assert_allclose(pion_cg_real, pion_gi_real / (lam**pion_full_params["n"]))

        pion_gi_imag = asymptotic_imag_function(hadron="pion", gauge_type="gi", quark_sector="full")(lam, pion_full_params)
        pion_cg_imag = asymptotic_imag_function(hadron="pion", gauge_type="cg", quark_sector="full")(lam, pion_full_params)
        np.testing.assert_allclose(pion_cg_imag, pion_gi_imag / (lam**pion_full_params["n"]))

    def test_pion_valence_and_sea_asymptotic_forms_use_distinct_constraints(self) -> None:
        lam = np.asarray([1.0, 2.0], dtype=float)
        valence_params = {"b1": 0.6, "b2": 1.1, "d1": 0.1, "d2": 0.05, "c1": 0.2, "e1": -0.3, "m": 0.35, "n": 0.9}
        sea_params = {"b2": 1.1, "d2": 0.05, "c2": 0.4, "e2": -0.2, "m": 0.35, "n": 0.9}

        pion_valence_imag = asymptotic_imag_function(hadron="pion", gauge_type="cg", quark_sector="valence")(lam, valence_params)
        np.testing.assert_allclose(pion_valence_imag, np.zeros_like(lam))

        pion_sea_imag = asymptotic_imag_function(hadron="pion", gauge_type="cg", quark_sector="sea")(lam, sea_params)
        self.assertGreater(float(np.max(np.abs(pion_sea_imag))), 0.0)

    def test_txt_two_point_loader_supports_complex_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw = np.array([[0.0, 1.0, 2.0, 0.1, 0.2], [1.0, 3.0, 4.0, 0.3, 0.4]], dtype=float)
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
                    "metadata": {
                        "setup_id": "demo_setup",
                        "momentum": [0, 0, 0],
                        "smearing": "SS",
                        "complex_samples": True,
                    },
                }
            )
            dataset = load_correlator_dataset(spec, manifest_path)
            self.assertEqual(dataset.samples.shape, (2, 2))
            self.assertIn("imag_samples", dataset.extra_axes)

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
                    "metadata": {
                        "setup_id": "demo_setup",
                        "momentum": [0, 0, 2],
                        "smearing": "SS",
                        "displacement": {"b": 0, "z": 1},
                        "operator": {"gamma": "gt", "flavor": "u-d"},
                    },
                }
            )
            dataset = load_correlator_dataset(spec, manifest_path)
            self.assertEqual(dataset.samples.shape, (2, 2, 2))
            self.assertTrue(np.iscomplexobj(dataset.samples))

    def test_grouped_matrix_element_families_flow_through_renormalization_and_fourier(self) -> None:
        manifest_data = manifest_payload()
        manifest_data["workflow"] = {
            "stages": ["correlator_analysis", "renormalization", "fourier_transform"],
            "stage_parameters": {
                "fourier_transform": {
                    "family_selector": {
                        "setup_id": "demo_setup",
                        "fit_mode": "joint_ratio_fh",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "smearing": "SS",
                    },
                    "gauge_type": "cg",
                    "sample_transform_workers": 2,
                    "physics": {"coordinate_direction": [1, 1, 0]},
                    "x_grid": {"values": [-1.0, 0.0, 1.0]},
                    "extrapolation": {
                        "fit_idx_range": [2, 5],
                        "extrapolated_length": 8.0,
                        "weight_ini": 0.0,
                        "m0_gev": 0.1,
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
            metadata={"setup_id": "demo_setup", "momentum": [4, 4, 0], "smearing": "SS"},
        )
        z_axis = np.arange(6, dtype=float)
        lambda_step = 8.0 * np.pi / 64.0
        lambda_axis = z_axis * lambda_step
        base_real = np.exp(-0.6 * lambda_axis)
        base_imag = 0.2 * np.exp(-0.6 * lambda_axis)
        real_samples = np.vstack([base_real + shift for shift in (-0.03, -0.01, 0.01, 0.03)])
        imag_samples = np.vstack([base_imag + shift for shift in (-0.015, -0.005, 0.005, 0.015)])
        correlator_payload = {
            "_matrix_element_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "setup_id": "demo_setup",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "smearing": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "momentum": [4, 4, 0],
                        "observable": "qpdf",
                        "analysis_channel": "qpdf",
                        "gauge": "cg",
                        "hadron": "proton",
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
            "matrix_element_families": [
                {
                    "metadata": {
                        "fit_mode": "joint_ratio_fh",
                        "setup_id": "demo_setup",
                        "b": 0,
                        "gamma": "gt",
                        "flavor": "u-d",
                        "smearing": "SS",
                        "px": 4,
                        "py": 4,
                        "pz": 0,
                        "momentum": [4, 4, 0],
                        "observable": "qpdf",
                        "analysis_channel": "qpdf",
                        "gauge": "cg",
                        "hadron": "proton",
                        "resampling_method": "jackknife",
                    },
                    "z_axis": z_axis.tolist(),
                    "sample_count": 4,
                    "real": {"mean": np.mean(real_samples, axis=0).tolist(), "error": np.std(real_samples, axis=0, ddof=1).tolist()},
                    "imag": {"mean": np.mean(imag_samples, axis=0).tolist(), "error": np.std(imag_samples, axis=0, ddof=1).tolist()},
                    "sample_artifact": None
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_events: list[dict[str, object]] = []
            context = StageContext(
                manifest=manifest,
                run_directory=Path(tmpdir),
                datasets={"demo": dataset},
                kernel=kernel,
                stage_payloads={"correlator_analysis": correlator_payload},
                progress_callback=progress_events.append,
            )
            renorm_result = RenormalizationStage().run(context)
            self.assertIn("_renormalized_families", renorm_result.payload)
            context.stage_payloads["renormalization"] = renorm_result.payload
            fourier_result = FourierTransformStage().run(context)
            self.assertIn("transformed_families", fourier_result.payload)
            self.assertEqual(len(fourier_result.payload["transformed_families"]), 1)
            family = fourier_result.payload["transformed_families"][0]
            components_gev = [
                float(lattice_unit_to_physical(component, a_fm=0.09, spatial_extent=64, dimension="P"))
                for component in (4, 4, 0)
            ]
            total_momentum_gev = float(np.linalg.norm(np.asarray(components_gev, dtype=float)))
            self.assertAlmostEqual(float(family["momentum"]["total_gev"]), total_momentum_gev)
            self.assertAlmostEqual(float(family["extrapolation"]["m0_gev"]), 0.1)
            self.assertAlmostEqual(float(family["extrapolation"]["m0_dimensionless"]), 0.1 / total_momentum_gev)
            self.assertTrue(
                any(
                    event.get("event") == "stage_message"
                    and "|P|=" in str(event.get("message", ""))
                    and "m0=0.100000 GeV" in str(event.get("message", ""))
                    for event in progress_events
                )
            )


if __name__ == "__main__":
    unittest.main()
