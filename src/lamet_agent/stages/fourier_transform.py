"""Fourier transform stage for demo signals and sample-wise qPDF workflows."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.constants import lattice_unit_to_physical
from lamet_agent.extensions.qpdf_fourier import (
    batch_fourier_transform_qpdf,
    build_fourier_kernel,
    build_lambda_axis,
    build_x_grid,
    extrapolate_asymptotic_qpdf,
    mirror_qpdf_coordinate_space_samples,
)
from lamet_agent.extensions.statistics import gv
from lamet_agent.plotting import save_series_collection_plot, save_uncertainty_plot
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
        hadron=context["hadron"],
        gauge_type=context["gauge_type"],
        quark_sector=context["quark_sector"],
        joint_re_im_fit=context["joint_re_im_fit"],
        joint_prior_overrides=context["joint_prior_overrides"],
        real_prior_overrides=context["real_prior_overrides"],
        imag_prior_overrides=context["imag_prior_overrides"],
    )
    return {
        "sample_index": int(sample_index),
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
        matrix_element_families = previous.get("_renormalized_families") or previous.get("_matrix_element_families")
        if not matrix_element_families:
            raise ValueError(
                "fourier_transform requires sample-wise coordinate-space matrix-element families from the renormalization stage."
            )
        parameters = self._resolve_qpdf_parameters(context)
        selected_families = self._select_families(matrix_element_families, parameters["family_selector"])
        selected_families = [
            f for f in selected_families
            if any(v != 0 for v in (f["metadata"].get("px", 0), f["metadata"].get("py", 0), f["metadata"].get("pz", 0)))
        ]
        if not selected_families:
            raise ValueError("fourier_transform: no families with non-zero momentum remain after filtering.")
        transformed_families: list[dict[str, Any]] = []
        artifacts: list[ArtifactRecord] = []
        for family in selected_families:
            transformed_family, family_artifacts = self._run_qpdf_ft_for_family(
                context=context,
                stage_dir=stage_dir,
                family=family,
                parameters=parameters,
            )
            transformed_families.append(transformed_family)
            artifacts.extend(family_artifacts)
        artifacts.extend(self._write_grouped_b_x_dependence_plots(stage_dir, context, transformed_families))
        payload = {
            "family_count": len(transformed_families),
            "transformed_families": [self._serialize_transformed_family(family) for family in transformed_families],
            "_transformed_families": transformed_families,
        }
        if len(transformed_families) == 1:
            family = transformed_families[0]
            payload.update(
                {
                    "axis": np.asarray(family["x_axis"], dtype=float),
                    "values": np.asarray(family["real_mean"], dtype=float),
                    "magnitude": np.sqrt(
                        np.asarray(family["real_mean"], dtype=float) ** 2
                        + np.asarray(family["imag_mean"], dtype=float) ** 2
                    ),
                    "family": dict(family["metadata"]),
                    "lambda_axis": np.asarray(family["lambda_axis"], dtype=float),
                    "x_axis": np.asarray(family["x_axis"], dtype=float),
                    "extrapolation": dict(family["extrapolation"]),
                    "coordinate_space": {
                        "real": {
                            "mean": np.asarray(family["coordinate_real_mean"], dtype=float).tolist(),
                            "error": np.asarray(family["coordinate_real_error"], dtype=float).tolist(),
                        },
                        "imag": {
                            "mean": np.asarray(family["coordinate_imag_mean"], dtype=float).tolist(),
                            "error": np.asarray(family["coordinate_imag_error"], dtype=float).tolist(),
                        },
                    },
                    "momentum_space": {
                        "real": {
                            "mean": np.asarray(family["real_mean"], dtype=float).tolist(),
                            "error": np.asarray(family["real_error"], dtype=float).tolist(),
                        },
                        "imag": {
                            "mean": np.asarray(family["imag_mean"], dtype=float).tolist(),
                            "error": np.asarray(family["imag_error"], dtype=float).tolist(),
                        },
                    },
                }
            )
        summary = (
            f"Performed sample-wise asymptotic extrapolation and Fourier transforms for "
            f"{len(transformed_families)} family/families."
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _run_qpdf_ft_for_family(
        self,
        *,
        context: StageContext,
        stage_dir: Path,
        family: dict[str, Any],
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], list[ArtifactRecord]]:
        setup = context.manifest.setup_metadata(str(family["metadata"]["setup_id"]))
        momentum_vector = list(family["metadata"]["momentum"])
        momentum_components_gev = self._momentum_components_gev(setup=setup, momentum_vector=momentum_vector)
        total_momentum_gev = float(np.linalg.norm(momentum_components_gev))
        coordinate_direction = list(parameters["physics"].get("coordinate_direction") or momentum_vector)
        m0_gev = float(parameters["extrapolation"]["m0_gev"])
        m0_dimensionless = self._resolve_dimensionless_m0(m0_gev=m0_gev, total_momentum_gev=total_momentum_gev)
        context.report_progress(
            self.name,
            "stage_message",
            message=(
                f"FT family {self._family_slug(family['metadata'])}: "
                f"momentum[GeV]={np.array2string(momentum_components_gev, precision=6, separator=', ')}, "
                f"|P|={total_momentum_gev:.6f} GeV, "
                f"m0={m0_gev:.6f} GeV -> m0/|P|={m0_dimensionless:.6f}"
            ),
        )
        lambda_axis = build_lambda_axis(
            family["z_axis"],
            lattice_spacing_fm=setup["lattice_spacing_fm"],
            spatial_extent=setup["spatial_extent"],
            momentum_vector=momentum_vector,
            coordinate_direction=coordinate_direction,
            coordinate_step_multiplier=parameters["physics"]["coordinate_step_multiplier"],
        )
        x_grid_raw = build_x_grid(parameters["x_grid"])
        x_shift = parameters["x_shift"]
        x_grid = x_grid_raw + x_shift if x_shift != 0.0 else x_grid_raw

        real_samples = np.asarray(family["real_samples"], dtype=float)
        imag_samples = parameters["imaginary_sign"] * np.asarray(family["imag_samples"], dtype=float)
        if parameters["imaginary_zeroing"]:
            imag_samples = np.zeros_like(imag_samples)
        elif imag_samples.ndim == 2 and imag_samples.shape[1] > 0 and np.isclose(float(lambda_axis[0]), 0.0):
            imag_samples = imag_samples.copy()
            imag_samples[:, 0] = 0.0
        real_average = self._resampled_average(real_samples, family["metadata"]["resampling_method"])
        imag_average = self._resampled_average(imag_samples, family["metadata"]["resampling_method"])
        real_errors = np.asarray(gv.sdev(real_average), dtype=float)
        imag_errors = np.asarray(gv.sdev(imag_average), dtype=float)

        context.report_progress(
            self.name,
            "stage_progress_start",
            description="sample-wise asymptotic extrapolation",
            total=real_samples.shape[0],
            unit="sample",
        )
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
            m0=m0_dimensionless,
            hadron=str(family["metadata"]["hadron"]),
            gauge_type=parameters["gauge_type"],
            quark_sector=parameters["extrapolation"]["quark_sector"],
            joint_re_im_fit=parameters["extrapolation"]["joint_re_im_fit"],
            joint_prior_overrides=parameters["extrapolation"]["joint_prior_overrides"],
            real_prior_overrides=parameters["extrapolation"]["real_prior_overrides"],
            imag_prior_overrides=parameters["extrapolation"]["imag_prior_overrides"],
        )
        representative_real_fit = representative["fit_result_real"]
        representative_imag_fit = representative["fit_result_imag"]
        representative_lambda = np.asarray(representative["lambda_axis"], dtype=float)
        extrapolated_real_samples.append(np.asarray(representative["real"], dtype=float))
        extrapolated_imag_samples.append(np.asarray(representative["imag"], dtype=float))
        context.report_progress(self.name, "stage_progress_update", advance=1)

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
                "m0": m0_dimensionless,
                "hadron": str(family["metadata"]["hadron"]),
                "gauge_type": parameters["gauge_type"],
                "quark_sector": parameters["extrapolation"]["quark_sector"],
                "joint_re_im_fit": parameters["extrapolation"]["joint_re_im_fit"],
                "joint_prior_overrides": parameters["extrapolation"]["joint_prior_overrides"],
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
                    remaining_results = []
                    for result in pool.imap_unordered(
                        _extrapolate_qpdf_sample_from_context,
                        remaining_indices,
                        chunksize=chunksize,
                    ):
                        remaining_results.append(result)
                        context.report_progress(self.name, "stage_progress_update", advance=1)
            else:
                _set_qpdf_ft_pool_context(pool_context)
                remaining_results = []
                for index in remaining_indices:
                    remaining_results.append(_extrapolate_qpdf_sample_from_context(index))
                    context.report_progress(self.name, "stage_progress_update", advance=1)
            remaining_results.sort(key=lambda item: int(item["sample_index"]))
            extrapolated_real_samples.extend([result["real"] for result in remaining_results])
            extrapolated_imag_samples.extend([result["imag"] for result in remaining_results])
        context.report_progress(self.name, "stage_progress_end")

        extrapolated_real_array = np.asarray(extrapolated_real_samples, dtype=float)
        extrapolated_imag_array = np.asarray(extrapolated_imag_samples, dtype=float)
        mirrored_lambda, mirrored_real_array, mirrored_imag_array = mirror_qpdf_coordinate_space_samples(
            representative_lambda,
            extrapolated_real_array,
            extrapolated_imag_array,
        )
        fourier_kernel = build_fourier_kernel(mirrored_lambda, x_grid_raw)
        ft_real_array, ft_imag_array = batch_fourier_transform_qpdf(
            fourier_kernel,
            mirrored_real_array,
            mirrored_imag_array,
            separate_re_im=parameters["separate_re_im"],
        )
        extrapolated_real_avg = self._resampled_average(extrapolated_real_array, family["metadata"]["resampling_method"])
        extrapolated_imag_avg = self._resampled_average(extrapolated_imag_array, family["metadata"]["resampling_method"])
        ft_real_avg = self._resampled_average(ft_real_array, family["metadata"]["resampling_method"])
        ft_imag_avg = self._resampled_average(ft_imag_array, family["metadata"]["resampling_method"])

        transformed_family = {
            "metadata": dict(family["metadata"]),
            "momentum": {
                "lattice_units": [int(value) for value in momentum_vector],
                "components_gev": momentum_components_gev.tolist(),
                "total_gev": total_momentum_gev,
            },
            "lambda_axis": np.asarray(lambda_axis, dtype=float),
            "x_axis": np.asarray(x_grid, dtype=float),
            "coordinate_real_mean": np.asarray(gv.mean(extrapolated_real_avg), dtype=float),
            "coordinate_real_error": np.asarray(gv.sdev(extrapolated_real_avg), dtype=float),
            "coordinate_imag_mean": np.asarray(gv.mean(extrapolated_imag_avg), dtype=float),
            "coordinate_imag_error": np.asarray(gv.sdev(extrapolated_imag_avg), dtype=float),
            "real_mean": np.asarray(gv.mean(ft_real_avg), dtype=float),
            "real_error": np.asarray(gv.sdev(ft_real_avg), dtype=float),
            "imag_mean": np.asarray(gv.mean(ft_imag_avg), dtype=float),
            "imag_error": np.asarray(gv.sdev(ft_imag_avg), dtype=float),
            "real_samples": ft_real_array,
            "imag_samples": ft_imag_array,
            "sample_count": int(ft_real_array.shape[0]),
            "extrapolation": {
                "hadron": str(family["metadata"]["hadron"]),
                "gauge_type": parameters["gauge_type"],
                "fit_idx_range": list(parameters["extrapolation"]["fit_idx_range"]),
                "extrapolated_length": float(parameters["extrapolation"]["extrapolated_length"]),
                "weight_ini": float(parameters["extrapolation"]["weight_ini"]),
                "m0_gev": m0_gev,
                "m0_dimensionless": float(m0_dimensionless),
                "quark_sector": str(parameters["extrapolation"]["quark_sector"]),
                "joint_re_im_fit": bool(parameters["extrapolation"]["joint_re_im_fit"]),
                "imaginary_sign": int(parameters["imaginary_sign"]),
                "imaginary_zeroing": bool(parameters["imaginary_zeroing"]),
                "sample_transform_workers": int(parameters["sample_transform_workers"]),
                "separate_re_im": bool(parameters["separate_re_im"]),
                "coordinate_direction": list(coordinate_direction),
                "coordinate_step_multiplier": float(parameters["physics"]["coordinate_step_multiplier"]),
                "x_shift": float(x_shift),
            },
        }
        artifacts = self._write_qpdf_artifacts(
            stage_dir=stage_dir,
            context=context,
            family=transformed_family,
            lambda_axis=np.asarray(lambda_axis, dtype=float),
            representative_lambda=np.asarray(representative_lambda, dtype=float),
            coordinate_real_mean=np.asarray(gv.mean(real_average), dtype=float),
            coordinate_real_error=np.asarray(gv.sdev(real_average), dtype=float),
            coordinate_imag_mean=np.asarray(gv.mean(imag_average), dtype=float),
            coordinate_imag_error=np.asarray(gv.sdev(imag_average), dtype=float),
            extrapolated_real_avg=extrapolated_real_avg,
            extrapolated_imag_avg=extrapolated_imag_avg,
            ft_real_avg=ft_real_avg,
            ft_imag_avg=ft_imag_avg,
            x_grid=np.asarray(x_grid, dtype=float),
            ft_real_array=ft_real_array,
            ft_imag_array=ft_imag_array,
            representative_real_fit=representative_real_fit,
            representative_imag_fit=representative_imag_fit,
            extrapolation_payload=transformed_family["extrapolation"],
        )
        return transformed_family, artifacts

    def _resolve_qpdf_parameters(self, context: StageContext) -> dict[str, Any]:
        parameters = dict(context.parameters_for(self.name))
        physics = dict(parameters.get("physics", {}))
        extrapolation = dict(parameters.get("extrapolation", {}))
        return {
            "family_selector": dict(parameters.get("family_selector", {})),
            "gauge_type": str(parameters.get("gauge_type", "cg")).lower(),
            "imaginary_sign": int(parameters.get("imaginary_sign", -1)),
            "imaginary_zeroing": bool(parameters.get("imaginary_zeroing", False)),
            "physics": {
                "coordinate_direction": None if "coordinate_direction" not in physics else list(physics.get("coordinate_direction", [0, 0, 1])),
                "coordinate_step_multiplier": float(physics.get("coordinate_step_multiplier", 1.0)),
            },
            "sample_transform_workers": max(1, int(parameters.get("sample_transform_workers", 1))),
            "separate_re_im": bool(parameters.get("separate_re_im", False)),
            "x_grid": dict(parameters.get("x_grid", {"start": -2.0, "stop": 2.0, "num": 4000, "endpoint": False})),
            "x_shift": float(parameters.get("x_shift", 0.0)),
            "extrapolation": {
                "fit_idx_range": [int(value) for value in extrapolation.get("fit_idx_range", [2, 6])],
                "extrapolated_length": float(extrapolation.get("extrapolated_length", 50.0)),
                "weight_ini": float(extrapolation.get("weight_ini", 0.0)),
                "m0_gev": float(extrapolation.get("m0_gev", 0.0)),
                "quark_sector": str(extrapolation.get("quark_sector", "valence")).lower(),
                "joint_re_im_fit": bool(extrapolation.get("joint_re_im_fit", True)),
                "joint_prior_overrides": dict(extrapolation.get("joint_prior_overrides", {})),
                "real_prior_overrides": dict(extrapolation.get("real_prior_overrides", extrapolation.get("prior_overrides", {}))),
                "imag_prior_overrides": dict(extrapolation.get("imag_prior_overrides", extrapolation.get("prior_overrides", {}))),
            },
        }

    def _select_families(self, families: list[dict[str, Any]], selector: dict[str, Any]) -> list[dict[str, Any]]:
        if not selector:
            return list(families)
        matches = []
        for family in families:
            metadata = family["metadata"]
            if all(metadata.get(key) == value for key, value in selector.items()):
                matches.append(family)
        if not matches:
            raise ValueError(f"fourier_transform family_selector={selector!r} did not match any qPDF family.")
        return matches

    def _family_slug(self, metadata: dict[str, Any]) -> str:
        return (
            f"{metadata['observable']}_{metadata['setup_id']}_{metadata['fit_mode']}"
            f"_b{metadata['b']}_p{metadata['px']}{metadata['py']}{metadata['pz']}"
            f"_{metadata['gamma']}_{str(metadata['flavor']).replace('-', '_')}_{metadata['smearing']}"
        )

    def _group_families_for_b_plots(self, families: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for family in families:
            metadata = family["metadata"]
            key = (
                str(metadata["setup_id"]),
                str(metadata["fit_mode"]),
                str(metadata["gamma"]),
                str(metadata["flavor"]),
                str(metadata["smearing"]),
                int(metadata["px"]),
                int(metadata["py"]),
                int(metadata["pz"]),
                str(metadata["hadron"]),
                str(metadata["analysis_channel"]),
                str(metadata["gauge"]),
            )
            grouped.setdefault(key, []).append(family)
        return grouped

    def _b_plot_slug(self, metadata: dict[str, Any]) -> str:
        return (
            f"{metadata['hadron']}_{metadata['analysis_channel']}_{metadata['setup_id']}_{metadata['fit_mode']}"
            f"_p{metadata['px']}{metadata['py']}{metadata['pz']}_{metadata['gamma']}"
            f"_{str(metadata['flavor']).replace('-', '_')}_{metadata['smearing']}_all_b"
        )

    def _serialize_transformed_family(self, family: dict[str, Any]) -> dict[str, Any]:
        return {
            "metadata": dict(family["metadata"]),
            "momentum": dict(family.get("momentum", {})),
            "lambda_axis": np.asarray(family["lambda_axis"], dtype=float).tolist(),
            "x_axis": np.asarray(family["x_axis"], dtype=float).tolist(),
            "sample_count": int(family["sample_count"]),
            "extrapolation": dict(family["extrapolation"]),
            "coordinate_space": {
                "real": {
                    "mean": np.asarray(family["coordinate_real_mean"], dtype=float).tolist(),
                    "error": np.asarray(family["coordinate_real_error"], dtype=float).tolist(),
                },
                "imag": {
                    "mean": np.asarray(family["coordinate_imag_mean"], dtype=float).tolist(),
                    "error": np.asarray(family["coordinate_imag_error"], dtype=float).tolist(),
                },
            },
            "momentum_space": {
                "real": {
                    "mean": np.asarray(family["real_mean"], dtype=float).tolist(),
                    "error": np.asarray(family["real_error"], dtype=float).tolist(),
                },
                "imag": {
                    "mean": np.asarray(family["imag_mean"], dtype=float).tolist(),
                    "error": np.asarray(family["imag_error"], dtype=float).tolist(),
                },
            },
            "sample_artifact": family.get("sample_artifact"),
        }

    def _momentum_components_gev(self, *, setup: dict[str, Any], momentum_vector: list[float]) -> np.ndarray:
        if len(momentum_vector) != 3:
            raise ValueError("fourier_transform momentum_vector must have length 3.")
        return np.asarray(
            [
                float(
                    np.asarray(
                        lattice_unit_to_physical(
                            component,
                            a_fm=float(setup["lattice_spacing_fm"]),
                            spatial_extent=int(setup["spatial_extent"]),
                            dimension="P",
                        ),
                        dtype=float,
                    )
                )
                for component in momentum_vector
            ],
            dtype=float,
        )

    def _resolve_dimensionless_m0(self, *, m0_gev: float, total_momentum_gev: float) -> float:
        if np.isclose(m0_gev, 0.0):
            return 0.0
        if total_momentum_gev <= 0.0:
            raise ValueError("fourier_transform requires non-zero total momentum to convert m0_gev into a dimensionless tail parameter.")
        return float(m0_gev / total_momentum_gev)

    def _write_grouped_b_x_dependence_plots(
        self,
        stage_dir: Path,
        context: StageContext,
        families: list[dict[str, Any]],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for grouped_families in self._group_families_for_b_plots(families).values():
            if len(grouped_families) < 2:
                continue
            ordered = sorted(grouped_families, key=lambda family: int(family["metadata"]["b"]))
            reference = ordered[0]["metadata"]
            slug = self._b_plot_slug(reference)
            for plot_format in context.manifest.outputs.plot_formats:
                real_plot_path = stage_dir / f"{slug}_x_real.{plot_format}"
                imag_plot_path = stage_dir / f"{slug}_x_imag.{plot_format}"
                real_series = [
                    {
                        "x": np.asarray(family["x_axis"], dtype=float),
                        "y": np.asarray(family["real_mean"], dtype=float),
                        "error": np.asarray(family["real_error"], dtype=float),
                        "label": f"b={family['metadata']['b']} ({family['metadata']['observable']})",
                        "style": "fill_between",
                    }
                    for family in ordered
                ]
                imag_series = [
                    {
                        "x": np.asarray(family["x_axis"], dtype=float),
                        "y": np.asarray(family["imag_mean"], dtype=float),
                        "error": np.asarray(family["imag_error"], dtype=float),
                        "label": f"b={family['metadata']['b']} ({family['metadata']['observable']})",
                        "style": "fill_between",
                    }
                    for family in ordered
                ]
                save_series_collection_plot(
                    real_series,
                    real_plot_path,
                    f"{reference['hadron']} {reference['analysis_channel']} x dependence (real, {reference['fit_mode']}, {reference['setup_id']})",
                    r"$x$",
                    "Real x-space signal",
                )
                save_series_collection_plot(
                    imag_series,
                    imag_plot_path,
                    f"{reference['hadron']} {reference['analysis_channel']} x dependence (imag, {reference['fit_mode']}, {reference['setup_id']})",
                    r"$x$",
                    "Imag x-space signal",
                )
                artifacts.extend(
                    [
                        ArtifactRecord(
                            name=f"{slug}_x_real_plot_{plot_format}",
                            kind="plot",
                            path=real_plot_path,
                            description="Real-part x-space observables with multiple b families overlaid.",
                            format=plot_format,
                        ),
                        ArtifactRecord(
                            name=f"{slug}_x_imag_plot_{plot_format}",
                            kind="plot",
                            path=imag_plot_path,
                            description="Imaginary-part x-space observables with multiple b families overlaid.",
                            format=plot_format,
                        ),
                    ]
                )
        return artifacts

    def _resampled_average(self, samples: np.ndarray, method: str):
        array = np.asarray(samples, dtype=float)
        mean = np.mean(array, axis=0)
        if method == "jackknife":
            error = np.std(array, axis=0, ddof=1) * np.sqrt(max(array.shape[0] - 1, 1))
        else:
            error = np.std(array, axis=0, ddof=1)
        return gv.gvar(mean, error)

    def _coordinate_plot_xlim(self, lambda_axis: np.ndarray, extrapolated_lambda: np.ndarray) -> tuple[float, float]:
        """Match the proton_cg_pdf comparison plot: cover the data region and a moderate tail extension."""
        data_lambda_max = float(np.max(lambda_axis))
        lam_gap = float(abs(lambda_axis[1] - lambda_axis[0])) if len(lambda_axis) > 1 else 1.0
        extension = max(12.0 * lam_gap, 0.9 * data_lambda_max)
        left_edge = -max(lam_gap, 0.5)
        right_edge = min(float(np.max(extrapolated_lambda)), data_lambda_max + extension)
        return left_edge, right_edge

    def _dense_tail_band(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_errors: np.ndarray,
        *,
        start_x: float,
        stop_x: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate the extrapolated tail onto a denser grid for smoother plotting."""
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        error_array = np.asarray(y_errors, dtype=float)
        mask = (x_array >= float(start_x)) & (x_array <= float(stop_x))
        masked_x = x_array[mask]
        masked_y = y_array[mask]
        masked_error = error_array[mask]
        if masked_x.size < 2:
            return masked_x, masked_y, masked_error
        dense_x = np.linspace(masked_x[0], masked_x[-1], max(200, masked_x.size * 10))
        dense_y = np.interp(dense_x, masked_x, masked_y)
        dense_error = np.interp(dense_x, masked_x, masked_error)
        return dense_x, dense_y, dense_error

    def _fit_window_markers(
        self,
        lambda_axis: np.ndarray,
        y_values: np.ndarray,
        y_errors: np.ndarray,
        *,
        fit_start: int,
        fit_stop: int,
    ) -> list[dict[str, float]]:
        """Draw short dashed markers near the corresponding data values."""
        y_array = np.asarray(y_values, dtype=float)
        error_array = np.asarray(y_errors, dtype=float)
        ymin = np.min(y_array - error_array)
        ymax = np.max(y_array + error_array)
        span = max(float(ymax - ymin), 1.0e-6)
        markers: list[dict[str, float]] = []
        for index in (fit_start, max(fit_start, fit_stop)):
            half_height = max(2.0 * float(error_array[index]), 0.08 * span)
            center = float(y_array[index])
            markers.append(
                {
                    "x": float(lambda_axis[index]),
                    "ymin": center - half_height,
                    "ymax": center + half_height,
                }
            )
        return markers

    def _write_qpdf_artifacts(
        self,
        *,
        stage_dir: Path,
        context: StageContext,
        family: dict[str, Any],
        lambda_axis: np.ndarray,
        representative_lambda: np.ndarray,
        coordinate_real_mean: np.ndarray,
        coordinate_real_error: np.ndarray,
        coordinate_imag_mean: np.ndarray,
        coordinate_imag_error: np.ndarray,
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
        family_slug = self._family_slug(family["metadata"])
        summary_path = stage_dir / f"{family_slug}_summary.json"
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
                    "joint_re_im_fit": bool(extrapolation_payload.get("joint_re_im_fit", False)),
                },
            },
        )
        artifacts.append(
            ArtifactRecord(
                name=f"{family_slug}_summary_json",
                kind="report",
                path=summary_path,
                description="Summary of the sample-wise Fourier transform for one family.",
                format="json",
            )
        )
        sample_path = stage_dir / f"{family_slug}_samples.npz"
        np.savez(
            sample_path,
            x_axis=x_grid,
            real_samples=ft_real_array,
            imag_samples=ft_imag_array,
        )
        family["sample_artifact"] = str(sample_path)
        artifacts.append(
            ArtifactRecord(
                name=f"{family_slug}_samples_npz",
                kind="data",
                path=sample_path,
                description="Sample-wise x-space values after extrapolation and Fourier transform.",
                format="npz",
            )
        )
        for data_path in write_columnar_data(
            stage_dir / family_slug,
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
                    name=f"{family_slug}_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Final x-space observable after sample-wise extrapolation and Fourier transform.",
                    format=data_path.suffix[1:],
                )
            )
        real_fit_path = stage_dir / f"{family_slug}_fit_real.txt"
        imag_fit_path = stage_dir / f"{family_slug}_fit_imag.txt"
        real_fit_path.write_text(representative_real_fit.format(100) + "\n", encoding="utf-8")
        imag_fit_path.write_text(representative_imag_fit.format(100) + "\n", encoding="utf-8")
        artifacts.extend(
            [
                ArtifactRecord(
                    name=f"{family_slug}_fit_real_txt",
                    kind="report",
                    path=real_fit_path,
                    description="Representative real-part asymptotic extrapolation fit summary.",
                    format="txt",
                ),
                ArtifactRecord(
                    name=f"{family_slug}_fit_imag_txt",
                    kind="report",
                    path=imag_fit_path,
                    description="Representative imaginary-part asymptotic extrapolation fit summary.",
                    format="txt",
                ),
            ]
        )
        for plot_format in context.manifest.outputs.plot_formats:
            coordinate_real_plot = stage_dir / f"{family_slug}_coordinate_real.{plot_format}"
            coordinate_imag_plot = stage_dir / f"{family_slug}_coordinate_imag.{plot_format}"
            ft_real_plot = stage_dir / f"{family_slug}_ft_real.{plot_format}"
            ft_imag_plot = stage_dir / f"{family_slug}_ft_imag.{plot_format}"
            fit_start = int(extrapolation_payload["fit_idx_range"][0])
            fit_stop = int(extrapolation_payload["fit_idx_range"][1]) - 1
            coordinate_xlim = self._coordinate_plot_xlim(lambda_axis, representative_lambda)
            real_markers = self._fit_window_markers(
                lambda_axis,
                coordinate_real_mean,
                coordinate_real_error,
                fit_start=fit_start,
                fit_stop=fit_stop,
            )
            imag_markers = self._fit_window_markers(
                lambda_axis,
                coordinate_imag_mean,
                coordinate_imag_error,
                fit_start=fit_start,
                fit_stop=fit_stop,
            )
            real_tail_x, real_tail_y, real_tail_error = self._dense_tail_band(
                representative_lambda,
                np.asarray(gv.mean(extrapolated_real_avg), dtype=float),
                np.asarray(gv.sdev(extrapolated_real_avg), dtype=float),
                start_x=float(lambda_axis[fit_start]),
                stop_x=coordinate_xlim[1],
            )
            imag_tail_x, imag_tail_y, imag_tail_error = self._dense_tail_band(
                representative_lambda,
                np.asarray(gv.mean(extrapolated_imag_avg), dtype=float),
                np.asarray(gv.sdev(extrapolated_imag_avg), dtype=float),
                start_x=float(lambda_axis[fit_start]),
                stop_x=coordinate_xlim[1],
            )
            save_uncertainty_plot(
                lambda_axis,
                coordinate_real_mean,
                coordinate_real_error,
                coordinate_real_plot,
                f"{family['metadata']['observable']} Coordinate Space (Real)",
                r"$\lambda$",
                f"Re {family['metadata']['observable']}",
                fit_x=real_tail_x,
                fit_y=real_tail_y,
                fit_error=real_tail_error,
                data_label="Data",
                fit_label="Extrapolated band",
                xlim=coordinate_xlim,
                vertical_markers=real_markers,
            )
            save_uncertainty_plot(
                lambda_axis,
                coordinate_imag_mean,
                coordinate_imag_error,
                coordinate_imag_plot,
                f"{family['metadata']['observable']} Coordinate Space (Imag)",
                r"$\lambda$",
                f"Im {family['metadata']['observable']}",
                fit_x=imag_tail_x,
                fit_y=imag_tail_y,
                fit_error=imag_tail_error,
                data_label="Data",
                fit_label="Extrapolated band",
                xlim=coordinate_xlim,
                vertical_markers=imag_markers,
            )
            save_series_collection_plot(
                [
                    {
                        "x": x_grid,
                        "y": np.asarray(gv.mean(ft_real_avg), dtype=float),
                        "error": np.asarray(gv.sdev(ft_real_avg), dtype=float),
                        "label": "Sample average",
                        "style": "fill_between",
                    }
                ],
                ft_real_plot,
                f"{family['metadata']['observable']} Fourier Transform (Real)",
                r"$x$",
                f"Re {family['metadata']['observable']}(x)",
            )
            save_series_collection_plot(
                [
                    {
                        "x": x_grid,
                        "y": np.asarray(gv.mean(ft_imag_avg), dtype=float),
                        "error": np.asarray(gv.sdev(ft_imag_avg), dtype=float),
                        "label": "Sample average",
                        "style": "fill_between",
                    }
                ],
                ft_imag_plot,
                f"{family['metadata']['observable']} Fourier Transform (Imag)",
                r"$x$",
                f"Im {family['metadata']['observable']}(x)",
            )
            artifacts.extend(
                [
                    ArtifactRecord(
                        name=f"{family_slug}_coordinate_real_plot_{plot_format}",
                        kind="plot",
                        path=coordinate_real_plot,
                        description="Real-part coordinate-space data and extrapolated tail in lambda space.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{family_slug}_coordinate_imag_plot_{plot_format}",
                        kind="plot",
                        path=coordinate_imag_plot,
                        description="Imaginary-part coordinate-space data and extrapolated tail in lambda space.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{family_slug}_ft_real_plot_{plot_format}",
                        kind="plot",
                        path=ft_real_plot,
                        description="Real-part x-space observable after Fourier transform.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{family_slug}_ft_imag_plot_{plot_format}",
                        kind="plot",
                        path=ft_imag_plot,
                        description="Imaginary-part x-space observable after Fourier transform.",
                        format=plot_format,
                    ),
                ]
        )
        return artifacts
