"""Physical limit stage for producing final distribution-like observables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json


@register_stage
class PhysicalLimitStage:
    """Map matched momentum-space data to a simple final distribution estimate."""

    name = "physical_limit"
    description = "Produce a goal-aware final distribution estimate."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["perturbative_matching"]
        momentum = np.asarray(previous["axis"], dtype=float)
        matched = np.asarray(previous["values"], dtype=float)
        if context.manifest.goal == "distribution_amplitude":
            x_grid = np.linspace(0.0, 1.0, 128)
            target_grid = np.linspace(float(np.min(momentum)), float(np.max(momentum)), len(x_grid))
            label = "Distribution amplitude"
        else:
            x_grid = np.linspace(-1.0, 1.0, 128)
            target_grid = np.linspace(float(np.min(momentum)), float(np.max(momentum)), len(x_grid))
            label = "Parton distribution function"
        interpolator = interp1d(momentum, matched, kind="linear", fill_value="extrapolate")
        raw_distribution = np.asarray(interpolator(target_grid), dtype=float)
        normalized = np.abs(raw_distribution)
        norm = np.trapezoid(normalized, x_grid) or 1.0
        final_distribution = normalized / norm
        payload = {
            "axis": x_grid,
            "values": final_distribution,
            "observable": label,
            "normalization": norm,
        }
        artifacts = self._write_artifacts(stage_dir, context, x_grid, final_distribution, label)
        write_json(
            stage_dir / "final_observable.json",
            {
                "observable": label,
                "normalization": float(norm),
                "axis_points": len(x_grid),
            },
        )
        summary = f"Interpolated the matched signal onto the physical x-grid and normalized the final {label.lower()}."
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        label: str,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "physical_limit",
            {"x": axis, "value": values},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"physical_limit_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description=f"Final {label.lower()} values.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"physical_limit.{plot_format}"
            save_line_plot(axis, values, plot_path, label, "x", label)
            artifacts.append(
                ArtifactRecord(
                    name=f"physical_limit_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description=f"Final {label.lower()} plot.",
                    format=plot_format,
                )
            )
        return artifacts
