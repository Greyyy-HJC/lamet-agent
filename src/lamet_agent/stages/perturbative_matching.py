"""Perturbative matching stage for applying the user-provided hard kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        transformed_families = previous.get("_transformed_families")
        if transformed_families:
            matched_families: list[dict[str, Any]] = []
            artifacts: list[ArtifactRecord] = []
            for family in transformed_families:
                axis = np.asarray(family["x_axis"], dtype=float)
                magnitude = np.sqrt(
                    np.asarray(family["real_mean"], dtype=float) ** 2
                    + np.asarray(family["imag_mean"], dtype=float) ** 2
                )
                matched = np.asarray(context.kernel(axis, magnitude, family["metadata"]))
                if matched.shape != magnitude.shape:
                    raise ValueError(
                        "Kernel output must have the same shape as the Fourier-space magnitude input."
                    )
                matched_family = {
                    "metadata": dict(family["metadata"]),
                    "x_axis": axis,
                    "matched_values": matched,
                    "input_magnitude": magnitude,
                }
                matched_families.append(matched_family)
                artifacts.extend(self._write_family_artifacts(stage_dir, context, matched_family))
            payload = {
                "matched_families": [self._serialize_family(family) for family in matched_families],
                "_matched_families": matched_families,
                "family_count": len(matched_families),
            }
            if len(matched_families) == 1:
                payload["axis"] = np.asarray(matched_families[0]["x_axis"], dtype=float)
                payload["values"] = np.asarray(matched_families[0]["matched_values"], dtype=float)
            summary = f"Applied the inline hard kernel to {len(matched_families)} Fourier-space family/families."
            return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

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

    def _serialize_family(self, family: dict[str, Any]) -> dict[str, Any]:
        return {
            "metadata": dict(family["metadata"]),
            "x_axis": np.asarray(family["x_axis"], dtype=float).tolist(),
            "matched_values": np.asarray(family["matched_values"], dtype=float).tolist(),
            "input_magnitude": np.asarray(family["input_magnitude"], dtype=float).tolist(),
        }

    def _family_slug(self, metadata: dict[str, Any]) -> str:
        return (
            f"{metadata['observable']}_{metadata['setup_id']}_{metadata['fit_mode']}"
            f"_b{metadata['b']}_p{metadata['px']}{metadata['py']}{metadata['pz']}"
            f"_{metadata['gamma']}_{str(metadata['flavor']).replace('-', '_')}_{metadata['smearing']}"
        )

    def _write_family_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        family: dict[str, Any],
    ) -> list[ArtifactRecord]:
        slug = self._family_slug(family["metadata"])
        return self._write_artifacts(
            stage_dir=stage_dir,
            context=context,
            axis=np.asarray(family["x_axis"], dtype=float),
            values=np.asarray(family["matched_values"], dtype=float),
            stem=slug,
            description=f"Kernel-matched momentum-space data for {family['metadata']['observable']}.",
        )

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        stem: str = "perturbative_matching",
        description: str = "Kernel-matched momentum-space data.",
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / stem,
            {"momentum": axis, "matched_value": values},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"{stem}_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description=description,
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"{stem}.{plot_format}"
            save_line_plot(axis, values, plot_path, "Perturbative Matching", "Momentum", "Matched value")
            artifacts.append(
                ArtifactRecord(
                    name=f"{stem}_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description=f"{description} Plot.",
                    format=plot_format,
                )
            )
        return artifacts
