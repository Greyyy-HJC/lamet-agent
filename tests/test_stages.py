"""Stage contract tests for the demo pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.kernel import load_kernel
from lamet_agent.loaders import CorrelatorDataset
from lamet_agent.schemas import Manifest
from lamet_agent.stages import CorrelatorAnalysisStage, FourierTransformStage, RenormalizationStage
from lamet_agent.stages.base import StageContext


def manifest_payload() -> dict:
    """Return a manifest payload with SVG and CSV outputs for test runs."""
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
            "plot_formats": ["svg"],
            "data_formats": ["csv"],
            "keep_intermediates": True
        }
    }


class StageTests(unittest.TestCase):
    """Ensure stage results follow the normalized contract."""

    def test_demo_stages_return_payloads_and_artifacts(self) -> None:
        manifest = Manifest.from_dict(manifest_payload())
        kernel = load_kernel(manifest.kernel)
        dataset = CorrelatorDataset(
            kind="two_point",
            label="demo",
            path=Path("data.csv"),
            axis=[0.0, 1.0, 2.0],
            values=[1.0, 0.5, 0.2],
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
            context.stage_payloads[renorm_result.stage_name] = renorm_result.payload
            fourier_result = FourierTransformStage().run(context)
            self.assertTrue(correlator_result.artifacts)
            self.assertIn("axis", renorm_result.payload)
            self.assertIn("magnitude", fourier_result.payload)


if __name__ == "__main__":
    unittest.main()
