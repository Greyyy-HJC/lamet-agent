"""Correlator analysis stage for two-point resampling, diagnostics, and fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.errors import OptionalDependencyError
from lamet_agent.extensions.two_point import (
    effective_mass_from_correlator,
    fit_two_point_correlator,
    resample_two_point_correlator,
    summarize_two_point_fit,
    two_point_fit_function,
)
from lamet_agent.plotting import save_line_plot, save_uncertainty_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json


@register_stage
class CorrelatorAnalysisStage:
    """Analyze two-point correlator inputs for downstream workflow stages."""

    name = "correlator_analysis"
    description = "Resample two-point data, inspect the effective mass, and run a configurable n-state fit."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        dataset = self._select_two_point_dataset(context)
        parameters = self._resolve_stage_parameters(context, dataset.metadata)
        if parameters["temporal_extent"] <= 0:
            parameters["temporal_extent"] = int(len(dataset.axis))

        if dataset.samples is not None:
            resampled = resample_two_point_correlator(
                dataset.samples,
                method=parameters["resampling_method"],
                bootstrap_samples=parameters["bootstrap_samples"],
                bootstrap_sample_size=parameters["bootstrap_sample_size"],
                bin_size=parameters["bin_size"],
                seed=parameters["seed"],
            )
        else:
            resampled = None

        values = np.asarray(dataset.values, dtype=float) if resampled is None else resampled.mean
        spread = np.zeros_like(values) if resampled is None else resampled.error
        effective_mass = effective_mass_from_correlator(
            resampled if resampled is not None else values,
            method=parameters["effective_mass_method"],
            boundary=parameters["boundary"],
        )
        fit_summary = self._fit_two_point(
            resampled=resampled,
            parameters=parameters,
            temporal_extent=parameters["temporal_extent"],
        )

        payload = {
            "axis": np.asarray(dataset.axis, dtype=float),
            "values": values,
            "spread": spread,
            "input_label": dataset.label,
            "resampling": {
                "method": parameters["resampling_method"],
                "bin_size": parameters["bin_size"],
                "configuration_count": int(dataset.samples.shape[1]) if dataset.samples is not None else 0,
                "resample_count": 0 if resampled is None else int(resampled.resample_count),
            },
            "effective_mass": {
                "method": effective_mass.method,
                "axis": effective_mass.times,
                "mean": effective_mass.mean,
                "error": effective_mass.error,
            },
            "fit": fit_summary,
        }
        artifacts = self._write_artifacts(
            stage_dir=stage_dir,
            context=context,
            axis=np.asarray(dataset.axis, dtype=float),
            values=values,
            spread=spread,
            effective_mass=effective_mass,
            fit_summary=fit_summary,
            fit_curve=self._build_fit_curve(dataset.axis, fit_summary, parameters),
        )
        summary = self._build_summary(dataset.label, resampled, fit_summary)
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _select_two_point_dataset(self, context: StageContext):
        candidates = [dataset for dataset in context.datasets.values() if dataset.kind == "two_point"]
        if not candidates:
            raise ValueError("Correlator analysis requires at least one two-point correlator input.")
        return candidates[0]

    def _resolve_stage_parameters(self, context: StageContext, metadata: dict[str, Any]) -> dict[str, Any]:
        parameters = context.parameters_for(self.name)
        two_point_parameters = dict(parameters.get("two_point", parameters))
        resampling_parameters = dict(two_point_parameters.get("resampling", {}))
        effective_mass_parameters = dict(two_point_parameters.get("effective_mass", {}))
        fit_parameters = dict(two_point_parameters.get("fit", {}))

        temporal_extent = int(
            fit_parameters.get(
                "temporal_extent",
                two_point_parameters.get("temporal_extent", metadata.get("temporal_extent", metadata.get("Lt", 0))),
            )
            or 0
        )
        if temporal_extent <= 0:
            temporal_extent = int(metadata.get("configuration_time_extent", 0) or 0)
        if temporal_extent <= 0:
            temporal_extent = int(metadata.get("n_t", 0) or 0)

        return {
            "boundary": str(two_point_parameters.get("boundary", metadata.get("boundary", "periodic"))),
            "resampling_method": str(resampling_parameters.get("method", "jackknife")),
            "bootstrap_samples": int(resampling_parameters.get("sample_count", 500)),
            "bootstrap_sample_size": resampling_parameters.get("sample_size"),
            "bin_size": int(resampling_parameters.get("bin_size", 1)),
            "seed": int(resampling_parameters.get("seed", 1984)),
            "effective_mass_method": effective_mass_parameters.get("method"),
            "fit_enabled": bool(fit_parameters.get("enabled", True)),
            "state_count": int(fit_parameters.get("state_count", 2)),
            "tmin": int(fit_parameters.get("tmin", 4)),
            "tmax": int(fit_parameters.get("tmax", min(temporal_extent // 2 if temporal_extent else 20, 20))),
            "normalize": bool(fit_parameters.get("normalize", True)),
            "prior_overrides": dict(fit_parameters.get("prior_overrides", {})),
            "temporal_extent": temporal_extent,
        }

    def _fit_two_point(
        self,
        *,
        resampled,
        parameters: dict[str, Any],
        temporal_extent: int,
    ) -> dict[str, Any]:
        if resampled is None or resampled.average is None:
            return {
                "performed": False,
                "reason": "Fit skipped because no resampled ensemble average is available.",
            }
        if not parameters["fit_enabled"]:
            return {
                "performed": False,
                "reason": "Fit disabled by stage parameters.",
            }
        if temporal_extent <= 0:
            return {
                "performed": False,
                "reason": "Fit skipped because the temporal lattice extent is not known.",
            }
        try:
            fit_result = fit_two_point_correlator(
                resampled.average,
                temporal_extent=temporal_extent,
                tmin=parameters["tmin"],
                tmax=parameters["tmax"],
                state_count=parameters["state_count"],
                boundary=parameters["boundary"],
                normalize=parameters["normalize"],
                prior_overrides=parameters["prior_overrides"],
            )
        except (OptionalDependencyError, ValueError, KeyError) as exc:
            return {
                "performed": False,
                "reason": str(exc),
            }

        summary = summarize_two_point_fit(fit_result, state_count=parameters["state_count"])
        summary.update(
            {
                "performed": True,
                "tmin": int(parameters["tmin"]),
                "tmax": int(parameters["tmax"]),
                "boundary": parameters["boundary"],
                "normalize": bool(parameters["normalize"]),
                "quality": "good" if fit_result.Q >= 0.05 else "poor",
            }
        )
        return summary

    def _build_fit_curve(
        self,
        axis: np.ndarray,
        fit_summary: dict[str, Any],
        parameters: dict[str, Any],
    ) -> dict[str, np.ndarray] | None:
        if not fit_summary.get("performed"):
            return None
        fit_parameters = {
            key: value["mean"]
            for key, value in fit_summary.get("parameters", {}).items()
        }
        fit_values = two_point_fit_function(
            axis,
            fit_parameters,
            temporal_extent=parameters["temporal_extent"],
            state_count=parameters["state_count"],
            boundary=parameters["boundary"],
        )
        return {
            "axis": np.asarray(axis, dtype=float),
            "fit_value": np.asarray(fit_values, dtype=float),
        }

    def _write_artifacts(
        self,
        *,
        stage_dir: Path,
        context: StageContext,
        axis: np.ndarray,
        values: np.ndarray,
        spread: np.ndarray,
        effective_mass,
        fit_summary: dict[str, Any],
        fit_curve: dict[str, np.ndarray] | None,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "correlator_analysis",
            {"t": axis, "value": values, "error": spread},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"correlator_analysis_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Resampled two-point correlator mean values and uncertainties.",
                    format=data_path.suffix[1:],
                )
            )
        for data_path in write_columnar_data(
            stage_dir / "effective_mass",
            {"t": effective_mass.times, "meff": effective_mass.mean, "error": effective_mass.error},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"effective_mass_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Effective-mass estimates derived from the two-point correlator.",
                    format=data_path.suffix[1:],
                )
            )
        if fit_curve is not None:
            for data_path in write_columnar_data(
                stage_dir / "two_point_fit_curve",
                fit_curve,
                context.manifest.outputs.data_formats,
            ):
                artifacts.append(
                    ArtifactRecord(
                        name=f"two_point_fit_curve_{data_path.suffix[1:]}",
                        kind="data",
                        path=data_path,
                        description="Best-fit two-point correlator curve evaluated on the analysis time axis.",
                        format=data_path.suffix[1:],
                    )
                )
        write_json(stage_dir / "two_point_fit_summary.json", fit_summary)
        artifacts.append(
            ArtifactRecord(
                name="two_point_fit_summary_json",
                kind="report",
                path=stage_dir / "two_point_fit_summary.json",
                description="Fit diagnostics and posterior parameter summaries for the two-point correlator.",
                format="json",
            )
        )
        for plot_format in context.manifest.outputs.plot_formats:
            correlator_plot = stage_dir / f"correlator_analysis.{plot_format}"
            effective_mass_plot = stage_dir / f"effective_mass.{plot_format}"
            if np.any(spread > 0):
                save_uncertainty_plot(
                    axis,
                    values,
                    spread,
                    correlator_plot,
                    "Two-Point Correlator",
                    "tsep",
                    "C2pt",
                    fit_x=None if fit_curve is None else fit_curve["axis"],
                    fit_y=None if fit_curve is None else fit_curve["fit_value"],
                )
                save_uncertainty_plot(
                    effective_mass.times,
                    effective_mass.mean,
                    effective_mass.error,
                    effective_mass_plot,
                    "Effective Mass",
                    "tsep",
                    "m_eff",
                )
            else:
                save_line_plot(axis, values, correlator_plot, "Two-Point Correlator", "t", "C2pt")
                save_line_plot(
                    effective_mass.times,
                    effective_mass.mean,
                    effective_mass_plot,
                    "Effective Mass",
                    "tsep",
                    "m_eff",
                )
            artifacts.append(
                ArtifactRecord(
                    name=f"correlator_analysis_plot_{plot_format}",
                    kind="plot",
                    path=correlator_plot,
                    description="Two-point correlator mean signal plot.",
                    format=plot_format,
                )
            )
            artifacts.append(
                ArtifactRecord(
                    name=f"effective_mass_plot_{plot_format}",
                    kind="plot",
                    path=effective_mass_plot,
                    description="Effective-mass diagnostic plot derived from the two-point correlator.",
                    format=plot_format,
                )
            )
        return artifacts

    def _build_summary(self, label: str, resampled, fit_summary: dict[str, Any]) -> str:
        if resampled is None:
            return f"Analyzed two-point correlator {label!r} from pre-averaged data and computed an effective-mass diagnostic."
        fit_clause = "fit skipped"
        if fit_summary.get("performed"):
            quality_prefix = "acceptable " if fit_summary.get("quality") == "good" else "poor "
            fit_clause = (
                f"{quality_prefix}{fit_summary['state_count']}-state fit completed with "
                f"chi2/dof = {fit_summary['chi2_per_dof']:.3f} and Q = {fit_summary['Q']:.3f}"
            )
        elif "reason" in fit_summary:
            fit_clause = f"fit skipped ({fit_summary['reason']})"
        return (
            f"Resampled two-point correlator {label!r} with {resampled.method} over "
            f"{resampled.configuration_count} configuration(s), computed the effective mass, and {fit_clause}."
        )
