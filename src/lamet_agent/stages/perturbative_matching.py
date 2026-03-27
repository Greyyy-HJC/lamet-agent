"""Perturbative matching stage for applying the user-provided hard kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data


@register_stage
class PerturbativeMatchingStage:
    """Apply the hard-kernel callable to the Fourier-space signal."""

    name = "perturbative_matching"
    description = "Apply the user-defined perturbative hard kernel."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["fourier_transform"]
        axis = np.asarray(previous["axis"], dtype=float)
        magnitude = np.asarray(previous["magnitude"], dtype=float)
        matched = np.asarray(context.kernel(axis, magnitude, context.manifest.metadata))
        if matched.shape != magnitude.shape:
            raise ValueError(
                "Kernel output must have the same shape as the Fourier-space magnitude input."
            )
        payload = {"axis": axis, "values": matched}
        artifacts = self._write_artifacts(stage_dir, context, axis, matched)
        summary = "Applied the inline hard kernel to the Fourier-space magnitude to produce matched values."
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "perturbative_matching",
            {"momentum": axis, "matched_value": values},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"perturbative_matching_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Kernel-matched momentum-space data.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"perturbative_matching.{plot_format}"
            save_line_plot(axis, values, plot_path, "Perturbative Matching", "Momentum", "Matched value")
            artifacts.append(
                ArtifactRecord(
                    name=f"perturbative_matching_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description="Kernel-matched momentum-space plot.",
                    format=plot_format,
                )
            )
        return artifacts
