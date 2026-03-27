"""Renormalization stage for simple signal rescaling and stabilization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data


@register_stage
class RenormalizationStage:
    """Apply a simple multiplicative renormalization to the analyzed signal."""

    name = "renormalization"
    description = "Rescale the correlator signal using a simple normalization convention."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["correlator_analysis"]
        axis = np.asarray(previous["axis"], dtype=float)
        values = np.asarray(previous["values"], dtype=float)
        scale = np.max(np.abs(values)) or 1.0
        renormalized = values / scale
        payload = {"axis": axis, "values": renormalized, "scale_factor": scale}
        artifacts = self._write_artifacts(stage_dir, context, axis, renormalized)
        summary = (
            f"Renormalized the combined correlator signal by the maximum absolute amplitude ({scale:.6g})."
        )
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
            stage_dir / "renormalization",
            {"axis": axis, "renormalized_value": values},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"renormalization_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Renormalized correlator signal.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"renormalization.{plot_format}"
            save_line_plot(axis, values, plot_path, "Renormalization", "Coordinate", "Renormalized signal")
            artifacts.append(
                ArtifactRecord(
                    name=f"renormalization_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description="Renormalized correlator signal plot.",
                    format=plot_format,
                )
            )
        return artifacts
