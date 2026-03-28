"""Physical limit stage for producing final distribution-like observables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
        matched_families = previous.get("_matched_families")
        if matched_families:
            outputs: list[dict[str, Any]] = []
            artifacts: list[ArtifactRecord] = []
            available_axes = {
                "setup_ids": sorted({family["metadata"]["setup_id"] for family in matched_families}),
                "momenta": sorted({tuple(family["metadata"]["momentum"]) for family in matched_families}),
                "b_values": sorted({int(family["metadata"]["b"]) for family in matched_families}),
            }
            for family in matched_families:
                family_output = self._physical_limit_for_family(context=context, family=family)
                outputs.append(family_output["payload"])
                artifacts.extend(self._write_family_artifacts(stage_dir, context, family_output))
            payload = {
                "family_count": len(outputs),
                "families": outputs,
                "available_extrapolation_axes": {
                    "setup_ids": list(available_axes["setup_ids"]),
                    "momenta": [list(momentum) for momentum in available_axes["momenta"]],
                    "b_values": list(available_axes["b_values"]),
                },
            }
            if len(outputs) == 1:
                payload.update(
                    {
                        "axis": np.asarray(outputs[0]["axis"], dtype=float).tolist(),
                        "values": np.asarray(outputs[0]["values"], dtype=float).tolist(),
                        "observable": outputs[0]["observable"],
                        "normalization": outputs[0]["normalization"],
                    }
                )
            summary_path = stage_dir / "physical_limit_summary.json"
            write_json(summary_path, payload)
            artifacts.append(
                ArtifactRecord(
                    name="physical_limit_summary_json",
                    kind="report",
                    path=summary_path,
                    description="Metadata-rich summary of per-family physical-limit outputs.",
                    format="json",
                )
            )
            summary = (
                f"Generated per-family physical-limit outputs for {len(outputs)} matched family/families "
                "and recorded the available setup, momentum, and transverse-separation axes."
            )
            return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)
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

    def _physical_limit_for_family(self, *, context: StageContext, family: dict[str, Any]) -> dict[str, Any]:
        momentum = np.asarray(family["x_axis"], dtype=float)
        matched = np.asarray(family["matched_values"], dtype=float)
        observable = str(family["metadata"]["observable"])
        if observable in {"qda", "qtmdwf"}:
            x_grid = np.linspace(0.0, 1.0, 128)
            label = "Distribution amplitude" if observable == "qda" else "TMD wave function"
        else:
            x_grid = np.linspace(-1.0, 1.0, 128)
            label = "Parton distribution function" if observable == "qpdf" else "TMDPDF"
        target_grid = np.linspace(float(np.min(momentum)), float(np.max(momentum)), len(x_grid))
        interpolator = interp1d(momentum, matched, kind="linear", fill_value="extrapolate")
        raw_distribution = np.asarray(interpolator(target_grid), dtype=float)
        normalized = np.abs(raw_distribution)
        norm = np.trapezoid(normalized, x_grid) or 1.0
        final_distribution = normalized / norm
        payload = {
            "axis": x_grid.tolist(),
            "values": final_distribution.tolist(),
            "observable": label,
            "effective_observable": observable,
            "normalization": float(norm),
            "metadata": dict(family["metadata"]),
        }
        return {
            "metadata": dict(family["metadata"]),
            "payload": payload,
            "axis": x_grid,
            "values": final_distribution,
            "label": label,
            "normalization": norm,
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
        family_output: dict[str, Any],
    ) -> list[ArtifactRecord]:
        metadata = dict(family_output["metadata"])
        slug = self._family_slug(metadata)
        return self._write_artifacts(
            stage_dir=stage_dir,
            context=context,
            axis=np.asarray(family_output["axis"], dtype=float),
            values=np.asarray(family_output["values"], dtype=float),
            label=str(family_output["label"]),
            stem=slug,
        )

    def _write_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        label: str,
        stem: str = "physical_limit",
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / stem,
            {"x": axis, "value": values},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"{stem}_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description=f"Final {label.lower()} values.",
                    format=data_path.suffix[1:],
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"{stem}.{plot_format}"
            save_line_plot(axis, values, plot_path, label, "x", label)
            artifacts.append(
                ArtifactRecord(
                    name=f"{stem}_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description=f"Final {label.lower()} plot.",
                    format=plot_format,
                )
            )
        return artifacts
