"""Correlator analysis stage for ingesting and normalizing correlator data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data


@register_stage
class CorrelatorAnalysisStage:
    """Build a simple combined correlator signal from the provided inputs."""

    name = "correlator_analysis"
    description = "Combine and precondition input correlators for downstream stages."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        datasets = list(context.datasets.values())
        base_axis = datasets[0].axis
        stacked = np.vstack([np.interp(base_axis, dataset.axis, dataset.values) for dataset in datasets])
        combined = np.mean(stacked, axis=0)
        spread = np.std(stacked, axis=0)
        payload = {
            "axis": base_axis,
            "values": combined,
            "spread": spread,
            "input_labels": [dataset.label for dataset in datasets],
        }
        artifacts = self._write_artifacts(stage_dir, context, base_axis, combined, spread)
        summary = (
            f"Combined {len(datasets)} correlator input(s) onto a common axis and computed a mean signal "
            f"with pointwise spread estimates."
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        spread: np.ndarray,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "correlator_analysis",
            {"axis": axis, "value": values, "spread": spread},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"correlator_analysis_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Combined correlator signal and spread estimates.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"correlator_analysis.{plot_format}"
            save_line_plot(axis, values, plot_path, "Correlator Analysis", "Coordinate", "Signal")
            artifacts.append(
                ArtifactRecord(
                    name=f"correlator_analysis_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description="Combined correlator signal plot.",
                    format=plot_format,
                )
            )
        return artifacts
