"""Fourier transform stage for mapping coordinate-space data to momentum space."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data


@register_stage
class FourierTransformStage:
    """Compute a discrete Fourier transform of the renormalized correlator."""

    name = "fourier_transform"
    description = "Transform the renormalized signal into momentum space."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["renormalization"]
        axis = np.asarray(previous["axis"], dtype=float)
        values = np.asarray(previous["values"], dtype=float)
        spacing = float(np.mean(np.diff(axis))) if len(axis) > 1 else 1.0
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(axis), d=spacing))
        transformed = np.fft.fftshift(np.fft.fft(values))
        magnitude = np.abs(transformed)
        payload = {
            "axis": frequencies,
            "values": transformed,
            "magnitude": magnitude,
        }
        artifacts = self._write_artifacts(stage_dir, context, frequencies, transformed, magnitude)
        summary = "Computed a discrete Fourier transform and exposed both complex values and magnitudes."
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        magnitude: np.ndarray,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "fourier_transform",
            {
                "frequency": axis,
                "real": np.real(values),
                "imag": np.imag(values),
                "magnitude": magnitude,
            },
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"fourier_transform_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Momentum-space correlator values.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"fourier_transform.{plot_format}"
            save_line_plot(axis, magnitude, plot_path, "Fourier Transform", "Momentum", "Magnitude")
            artifacts.append(
                ArtifactRecord(
                    name=f"fourier_transform_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description="Fourier-space magnitude plot.",
                    format=plot_format,
                )
            )
        return artifacts
