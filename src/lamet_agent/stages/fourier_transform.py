"""Fourier transform stage for demo signals and sample-wise qPDF workflows."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.extensions.qpdf_fourier import (
    batch_fourier_transform_qpdf,
    build_fourier_kernel,
    build_lambda_axis,
    build_x_grid,
    extrapolate_asymptotic_qpdf,
    mirror_qpdf_coordinate_space_samples,
)
from lamet_agent.extensions.statistics import gv
from lamet_agent.plotting import save_uncertainty_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json

_QPDF_FT_POOL_CONTEXT: dict[str, Any] | None = None


def _set_qpdf_ft_pool_context(context: dict[str, Any]) -> None:
    global _QPDF_FT_POOL_CONTEXT
    _QPDF_FT_POOL_CONTEXT = context


def _extrapolate_qpdf_sample_from_context(sample_index: int) -> dict[str, np.ndarray]:
    if _QPDF_FT_POOL_CONTEXT is None:
        raise RuntimeError("QPDF FT pool context is not initialized.")
    context = _QPDF_FT_POOL_CONTEXT
    extrapolated = extrapolate_asymptotic_qpdf(
        context["lambda_axis"],
        context["real_samples"][sample_index],
        context["imag_samples"][sample_index],
        context["real_errors"],
        context["imag_errors"],
        fit_idx_range=context["fit_idx_range"],
        extrapolated_length=context["extrapolated_length"],
        weight_ini=context["weight_ini"],
        m0=context["m0"],
        gauge_type=context["gauge_type"],
        real_prior_overrides=context["real_prior_overrides"],
        imag_prior_overrides=context["imag_prior_overrides"],
    )
    return {
        "real": np.asarray(extrapolated["real"], dtype=float),
        "imag": np.asarray(extrapolated["imag"], dtype=float),
    }


@register_stage
class FourierTransformStage:
    """Compute either the legacy demo FFT or the sample-wise qPDF Fourier transform."""

    name = "fourier_transform"
    description = "Transform coordinate-space data into momentum space."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["renormalization"]
        qpdf_families = previous.get("_qpdf_families")
        if not qpdf_families:
            raise ValueError("fourier_transform now requires sample-wise qPDF families from the renormalization stage.")
        return self._run_qpdf_ft(context, stage_dir, qpdf_families)

    def _run_qpdf_ft(self, context: StageContext, stage_dir: Path, qpdf_families: list[dict[str, Any]]) -> StageResult:
        parameters = self._resolve_qpdf_parameters(context)
        family = self._select_family(qpdf_families, parameters["family_selector"])
        lambda_axis = build_lambda_axis(
            family["z_axis"],
            lattice_spacing_fm=parameters["physics"]["lattice_spacing_fm"],
            spatial_extent=parameters["physics"]["spatial_extent"],
            momentum_vector=parameters["physics"]["momentum_vector"],
            coordinate_direction=parameters["physics"]["coordinate_direction"],
            coordinate_step_multiplier=parameters["physics"]["coordinate_step_multiplier"],
        )
        x_grid = build_x_grid(parameters["x_grid"])

        real_samples = np.asarray(family["real_samples"], dtype=float)
        imag_samples = parameters["imaginary_sign"] * np.asarray(family["imag_samples"], dtype=float)
        real_average = self._resampled_average(real_samples, family["metadata"]["resampling_method"])
        imag_average = self._resampled_average(imag_samples, family["metadata"]["resampling_method"])
        real_errors = np.asarray(gv.sdev(real_average), dtype=float)
        imag_errors = np.asarray(gv.sdev(imag_average), dtype=float)

        extrapolated_real_samples: list[np.ndarray] = []
        extrapolated_imag_samples: list[np.ndarray] = []
        representative_real_fit = None
        representative_imag_fit = None
        representative_lambda = None
        representative = extrapolate_asymptotic_qpdf(
            lambda_axis,
            real_samples[0],
            imag_samples[0],
            real_errors,
            imag_errors,
            fit_idx_range=parameters["extrapolation"]["fit_idx_range"],
            extrapolated_length=parameters["extrapolation"]["extrapolated_length"],
            weight_ini=parameters["extrapolation"]["weight_ini"],
            m0=parameters["extrapolation"]["m0"],
            gauge_type=parameters["gauge_type"],
            real_prior_overrides=parameters["extrapolation"]["real_prior_overrides"],
            imag_prior_overrides=parameters["extrapolation"]["imag_prior_overrides"],
        )
        representative_real_fit = representative["fit_result_real"]
        representative_imag_fit = representative["fit_result_imag"]
        representative_lambda = np.asarray(representative["lambda_axis"], dtype=float)
        extrapolated_real_samples.append(np.asarray(representative["real"], dtype=float))
        extrapolated_imag_samples.append(np.asarray(representative["imag"], dtype=float))

        remaining_indices = list(range(1, real_samples.shape[0]))
        if remaining_indices:
            pool_context = {
                "lambda_axis": lambda_axis,
                "real_samples": real_samples,
                "imag_samples": imag_samples,
                "real_errors": real_errors,
                "imag_errors": imag_errors,
                "fit_idx_range": parameters["extrapolation"]["fit_idx_range"],
                "extrapolated_length": parameters["extrapolation"]["extrapolated_length"],
                "weight_ini": parameters["extrapolation"]["weight_ini"],
                "m0": parameters["extrapolation"]["m0"],
                "gauge_type": parameters["gauge_type"],
                "real_prior_overrides": parameters["extrapolation"]["real_prior_overrides"],
                "imag_prior_overrides": parameters["extrapolation"]["imag_prior_overrides"],
            }
            worker_count = min(parameters["sample_transform_workers"], real_samples.shape[0])
            if worker_count > 1:
                ctx = mp.get_context("fork")
                chunksize = max(1, len(remaining_indices) // (worker_count * 4))
                with ctx.Pool(
                    processes=worker_count,
                    initializer=_set_qpdf_ft_pool_context,
                    initargs=(pool_context,),
                ) as pool:
                    remaining_results = pool.map(_extrapolate_qpdf_sample_from_context, remaining_indices, chunksize=chunksize)
            else:
                _set_qpdf_ft_pool_context(pool_context)
                remaining_results = [_extrapolate_qpdf_sample_from_context(index) for index in remaining_indices]
            extrapolated_real_samples.extend([result["real"] for result in remaining_results])
            extrapolated_imag_samples.extend([result["imag"] for result in remaining_results])

        extrapolated_real_array = np.asarray(extrapolated_real_samples, dtype=float)
        extrapolated_imag_array = np.asarray(extrapolated_imag_samples, dtype=float)
        mirrored_lambda, mirrored_real_array, mirrored_imag_array = mirror_qpdf_coordinate_space_samples(
            representative_lambda,
            extrapolated_real_array,
            extrapolated_imag_array,
        )
        fourier_kernel = build_fourier_kernel(mirrored_lambda, x_grid)
        ft_real_array, ft_imag_array = batch_fourier_transform_qpdf(
            fourier_kernel,
            mirrored_real_array,
            mirrored_imag_array,
        )
        extrapolated_real_avg = self._resampled_average(extrapolated_real_array, family["metadata"]["resampling_method"])
        extrapolated_imag_avg = self._resampled_average(extrapolated_imag_array, family["metadata"]["resampling_method"])
        ft_real_avg = self._resampled_average(ft_real_array, family["metadata"]["resampling_method"])
        ft_imag_avg = self._resampled_average(ft_imag_array, family["metadata"]["resampling_method"])

        payload = {
            "axis": np.asarray(x_grid, dtype=float),
            "values": np.asarray(gv.mean(ft_real_avg), dtype=float),
            "magnitude": np.sqrt(
                np.asarray(gv.mean(ft_real_avg), dtype=float) ** 2
                + np.asarray(gv.mean(ft_imag_avg), dtype=float) ** 2
            ),
            "family": dict(family["metadata"]),
            "lambda_axis": np.asarray(lambda_axis, dtype=float),
            "x_axis": np.asarray(x_grid, dtype=float),
            "extrapolation": {
                "gauge_type": parameters["gauge_type"],
                "fit_idx_range": list(parameters["extrapolation"]["fit_idx_range"]),
                "extrapolated_length": float(parameters["extrapolation"]["extrapolated_length"]),
                "weight_ini": float(parameters["extrapolation"]["weight_ini"]),
                "m0": float(parameters["extrapolation"]["m0"]),
                "imaginary_sign": int(parameters["imaginary_sign"]),
                "sample_transform_workers": int(parameters["sample_transform_workers"]),
            },
            "coordinate_space": {
                "real": {
                    "mean": np.asarray(gv.mean(extrapolated_real_avg), dtype=float).tolist(),
                    "error": np.asarray(gv.sdev(extrapolated_real_avg), dtype=float).tolist(),
                },
                "imag": {
                    "mean": np.asarray(gv.mean(extrapolated_imag_avg), dtype=float).tolist(),
                    "error": np.asarray(gv.sdev(extrapolated_imag_avg), dtype=float).tolist(),
                },
            },
            "momentum_space": {
                "real": {
                    "mean": np.asarray(gv.mean(ft_real_avg), dtype=float).tolist(),
                    "error": np.asarray(gv.sdev(ft_real_avg), dtype=float).tolist(),
                },
                "imag": {
                    "mean": np.asarray(gv.mean(ft_imag_avg), dtype=float).tolist(),
                    "error": np.asarray(gv.sdev(ft_imag_avg), dtype=float).tolist(),
                },
            },
            "_qpdf_ft_samples": {
                "x_axis": np.asarray(x_grid, dtype=float),
                "real_samples": ft_real_array,
                "imag_samples": ft_imag_array,
            },
        }
        artifacts = self._write_qpdf_artifacts(
            stage_dir=stage_dir,
            context=context,
            family=family,
            lambda_axis=np.asarray(lambda_axis, dtype=float),
            representative_lambda=np.asarray(representative_lambda, dtype=float),
            extrapolated_real_avg=extrapolated_real_avg,
            extrapolated_imag_avg=extrapolated_imag_avg,
            ft_real_avg=ft_real_avg,
            ft_imag_avg=ft_imag_avg,
            x_grid=np.asarray(x_grid, dtype=float),
            ft_real_array=ft_real_array,
            ft_imag_array=ft_imag_array,
            representative_real_fit=representative_real_fit,
            representative_imag_fit=representative_imag_fit,
            extrapolation_payload=payload["extrapolation"],
        )
        summary = (
            "Performed sample-wise asymptotic extrapolation and Fourier transforms for "
            f"qPDF family b={family['metadata']['b']}, p=({family['metadata']['px']},"
            f"{family['metadata']['py']},{family['metadata']['pz']})."
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _resolve_qpdf_parameters(self, context: StageContext) -> dict[str, Any]:
        parameters = dict(context.parameters_for(self.name))
        physics = dict(parameters.get("physics", {}))
        extrapolation = dict(parameters.get("extrapolation", {}))
        return {
            "family_selector": dict(parameters.get("family_selector", {})),
            "gauge_type": str(parameters.get("gauge_type", "cg")).lower(),
            "imaginary_sign": int(parameters.get("imaginary_sign", -1)),
            "physics": {
                "lattice_spacing_fm": float(physics.get("lattice_spacing_fm", 0.09)),
                "spatial_extent": int(physics.get("spatial_extent", 64)),
                "momentum_vector": list(physics.get("momentum_vector", [0, 0, 4])),
                "coordinate_direction": list(physics.get("coordinate_direction", [0, 0, 1])),
                "coordinate_step_multiplier": float(physics.get("coordinate_step_multiplier", 1.0)),
            },
            "sample_transform_workers": max(1, int(parameters.get("sample_transform_workers", 1))),
            "x_grid": dict(parameters.get("x_grid", {"start": -2.0, "stop": 2.0, "num": 4000, "endpoint": False})),
            "extrapolation": {
                "fit_idx_range": [int(value) for value in extrapolation.get("fit_idx_range", [2, 6])],
                "extrapolated_length": float(extrapolation.get("extrapolated_length", 50.0)),
                "weight_ini": float(extrapolation.get("weight_ini", 0.0)),
                "m0": float(extrapolation.get("m0", 0.0)),
                "real_prior_overrides": dict(extrapolation.get("real_prior_overrides", extrapolation.get("prior_overrides", {}))),
                "imag_prior_overrides": dict(extrapolation.get("imag_prior_overrides", extrapolation.get("prior_overrides", {}))),
            },
        }

    def _select_family(self, families: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
        if not selector:
            if len(families) != 1:
                raise ValueError("fourier_transform requires family_selector when multiple qPDF families are available.")
            return families[0]
        matches = []
        for family in families:
            metadata = family["metadata"]
            if all(metadata.get(key) == value for key, value in selector.items()):
                matches.append(family)
        if not matches:
            raise ValueError(f"fourier_transform family_selector={selector!r} did not match any qPDF family.")
        if len(matches) > 1:
            raise ValueError(f"fourier_transform family_selector={selector!r} matched multiple qPDF families.")
        return matches[0]

    def _resampled_average(self, samples: np.ndarray, method: str):
        array = np.asarray(samples, dtype=float)
        mean = np.mean(array, axis=0)
        if method == "jackknife":
            error = np.std(array, axis=0, ddof=1) * np.sqrt(max(array.shape[0] - 1, 1))
        else:
            error = np.std(array, axis=0, ddof=1)
        return gv.gvar(mean, error)

    def _write_qpdf_artifacts(
        self,
        *,
        stage_dir: Path,
        context: StageContext,
        family: dict[str, Any],
        lambda_axis: np.ndarray,
        representative_lambda: np.ndarray,
        extrapolated_real_avg,
        extrapolated_imag_avg,
        ft_real_avg,
        ft_imag_avg,
        x_grid: np.ndarray,
        ft_real_array: np.ndarray,
        ft_imag_array: np.ndarray,
        representative_real_fit,
        representative_imag_fit,
        extrapolation_payload: dict[str, Any],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        summary_path = stage_dir / "qpdf_ft_summary.json"
        write_json(
            summary_path,
            {
                "family": dict(family["metadata"]),
                "lambda_axis": lambda_axis.tolist(),
                "x_axis": x_grid.tolist(),
                "extrapolation": extrapolation_payload,
                "momentum_space": {
                    "real": {
                        "mean": np.asarray(gv.mean(ft_real_avg), dtype=float).tolist(),
                        "error": np.asarray(gv.sdev(ft_real_avg), dtype=float).tolist(),
                    },
                    "imag": {
                        "mean": np.asarray(gv.mean(ft_imag_avg), dtype=float).tolist(),
                        "error": np.asarray(gv.sdev(ft_imag_avg), dtype=float).tolist(),
                    },
                },
                "representative_fit_quality": {
                    "real_Q": float(representative_real_fit.Q),
                    "imag_Q": float(representative_imag_fit.Q),
                },
            },
        )
        artifacts.append(
            ArtifactRecord(
                name="qpdf_ft_summary_json",
                kind="report",
                path=summary_path,
                description="Summary of the sample-wise qPDF extrapolation and Fourier transform.",
                format="json",
            )
        )
        sample_path = stage_dir / "qpdf_ft_samples.npz"
        np.savez(
            sample_path,
            x_axis=x_grid,
            real_samples=ft_real_array,
            imag_samples=ft_imag_array,
        )
        artifacts.append(
            ArtifactRecord(
                name="qpdf_ft_samples_npz",
                kind="data",
                path=sample_path,
                description="Sample-wise x-space qPDF values after extrapolation and Fourier transform.",
                format="npz",
            )
        )
        for data_path in write_columnar_data(
            stage_dir / "qpdf_ft",
            {
                "x": x_grid,
                "real": np.asarray(gv.mean(ft_real_avg), dtype=float),
                "real_error": np.asarray(gv.sdev(ft_real_avg), dtype=float),
                "imag": np.asarray(gv.mean(ft_imag_avg), dtype=float),
                "imag_error": np.asarray(gv.sdev(ft_imag_avg), dtype=float),
            },
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"qpdf_ft_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Final x-space qPDF after sample-wise extrapolation and Fourier transform.",
                    format=data_path.suffix[1:],
                )
            )
        real_fit_path = stage_dir / "qpdf_extrapolation_fit_real.txt"
        imag_fit_path = stage_dir / "qpdf_extrapolation_fit_imag.txt"
        real_fit_path.write_text(representative_real_fit.format(100) + "\n", encoding="utf-8")
        imag_fit_path.write_text(representative_imag_fit.format(100) + "\n", encoding="utf-8")
        artifacts.extend(
            [
                ArtifactRecord(
                    name="qpdf_extrapolation_fit_real_txt",
                    kind="report",
                    path=real_fit_path,
                    description="Representative real-part asymptotic extrapolation fit summary.",
                    format="txt",
                ),
                ArtifactRecord(
                    name="qpdf_extrapolation_fit_imag_txt",
                    kind="report",
                    path=imag_fit_path,
                    description="Representative imaginary-part asymptotic extrapolation fit summary.",
                    format="txt",
                ),
            ]
        )
        for plot_format in context.manifest.outputs.plot_formats:
            coordinate_real_plot = stage_dir / f"qpdf_coordinate_space_real.{plot_format}"
            coordinate_imag_plot = stage_dir / f"qpdf_coordinate_space_imag.{plot_format}"
            ft_real_plot = stage_dir / f"qpdf_ft_real.{plot_format}"
            ft_imag_plot = stage_dir / f"qpdf_ft_imag.{plot_format}"
            save_uncertainty_plot(
                lambda_axis,
                np.asarray(family["real_mean"], dtype=float),
                np.asarray(family["real_error"], dtype=float),
                coordinate_real_plot,
                "qPDF Coordinate Space (Real)",
                r"$\lambda$",
                "Re qPDF",
                fit_x=representative_lambda,
                fit_y=np.asarray(gv.mean(extrapolated_real_avg), dtype=float),
                fit_error=np.asarray(gv.sdev(extrapolated_real_avg), dtype=float),
                data_label="Data",
                fit_label="Extrapolated band",
            )
            save_uncertainty_plot(
                lambda_axis,
                np.asarray(family["imag_mean"], dtype=float),
                np.asarray(family["imag_error"], dtype=float),
                coordinate_imag_plot,
                "qPDF Coordinate Space (Imag)",
                r"$\lambda$",
                "Im qPDF",
                fit_x=representative_lambda,
                fit_y=np.asarray(gv.mean(extrapolated_imag_avg), dtype=float),
                fit_error=np.asarray(gv.sdev(extrapolated_imag_avg), dtype=float),
                data_label="Data",
                fit_label="Extrapolated band",
            )
            save_uncertainty_plot(
                x_grid,
                np.asarray(gv.mean(ft_real_avg), dtype=float),
                np.asarray(gv.sdev(ft_real_avg), dtype=float),
                ft_real_plot,
                "qPDF Fourier Transform (Real)",
                "x",
                "Re qPDF(x)",
                data_label="FT result",
            )
            save_uncertainty_plot(
                x_grid,
                np.asarray(gv.mean(ft_imag_avg), dtype=float),
                np.asarray(gv.sdev(ft_imag_avg), dtype=float),
                ft_imag_plot,
                "qPDF Fourier Transform (Imag)",
                "x",
                "Im qPDF(x)",
                data_label="FT result",
            )
            artifacts.extend(
                [
                    ArtifactRecord(
                        name=f"qpdf_coordinate_space_real_plot_{plot_format}",
                        kind="plot",
                        path=coordinate_real_plot,
                        description="Real-part qPDF data and extrapolated tail in lambda space.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"qpdf_coordinate_space_imag_plot_{plot_format}",
                        kind="plot",
                        path=coordinate_imag_plot,
                        description="Imaginary-part qPDF data and extrapolated tail in lambda space.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"qpdf_ft_real_plot_{plot_format}",
                        kind="plot",
                        path=ft_real_plot,
                        description="Real-part x-space qPDF after Fourier transform.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"qpdf_ft_imag_plot_{plot_format}",
                        kind="plot",
                        path=ft_imag_plot,
                        description="Imaginary-part x-space qPDF after Fourier transform.",
                        format=plot_format,
                    ),
                ]
        )
        return artifacts
