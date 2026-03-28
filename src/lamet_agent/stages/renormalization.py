"""Renormalization stage for simple signal rescaling and stabilization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json


@register_stage
class RenormalizationStage:
    """Apply a simple multiplicative renormalization to the analyzed signal."""

    name = "renormalization"
    description = "Rescale the correlator signal using a simple normalization convention."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["correlator_analysis"]
        qpdf_families = previous.get("_qpdf_families")
        if qpdf_families:
            payload = {
                "mode": "identity",
                "renormalization_applied": False,
                "qpdf_families": previous.get("qpdf_families", []),
                "_qpdf_families": qpdf_families,
            }
            artifacts = self._write_qpdf_artifacts(stage_dir, qpdf_families)
            summary = "Skipped renormalization and passed the sample-wise bare qPDF families through unchanged."
            return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

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

    def _write_qpdf_artifacts(self, stage_dir: Path, qpdf_families: list[dict[str, Any]]) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        summary_payload = {
            "mode": "identity",
            "renormalization_applied": False,
            "family_count": len(qpdf_families),
            "families": [
                {
                    "metadata": dict(family["metadata"]),
                    "z_axis": np.asarray(family["z_axis"], dtype=float).tolist(),
                    "sample_count": int(family["sample_count"]),
                    "real": {
                        "mean": np.asarray(family["real_mean"], dtype=float).tolist(),
                        "error": np.asarray(family["real_error"], dtype=float).tolist(),
                    },
                    "imag": {
                        "mean": np.asarray(family["imag_mean"], dtype=float).tolist(),
                        "error": np.asarray(family["imag_error"], dtype=float).tolist(),
                    },
                    "sample_artifact": family.get("sample_artifact"),
                }
                for family in qpdf_families
            ],
        }
        summary_path = stage_dir / "renormalization_qpdf_summary.json"
        write_json(summary_path, summary_payload)
        artifacts.append(
            ArtifactRecord(
                name="renormalization_qpdf_summary_json",
                kind="report",
                path=summary_path,
                description="Identity pass-through summary for sample-wise qPDF families.",
                format="json",
            )
        )
        return artifacts
