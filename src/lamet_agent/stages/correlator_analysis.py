"""Correlator analysis stage for two-point and three-point diagnostics."""

from __future__ import annotations

import multiprocessing as mp
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.errors import OptionalDependencyError
from lamet_agent.extensions.statistics import gv
from lamet_agent.extensions.three_point import (
    add_error_to_resampled_samples,
    average_resampled_samples,
    build_fh_samples,
    build_ratio_samples,
    build_summed_ratio_samples,
    evaluate_fh_band,
    evaluate_ratio_band,
    filter_bad_points,
    fit_fh_correlator,
    fit_joint_ratio_fh_correlator,
    fit_ratio_correlator,
    resample_observable,
    summarize_three_point_fit_samples,
)
from lamet_agent.extensions.two_point import (
    EffectiveMassResult,
    ResampledCorrelator,
    effective_mass_from_correlator,
    fit_two_point_correlator,
    summarize_two_point_fit,
    two_point_fit_function,
)
from lamet_agent.plotting import save_series_collection_plot, save_uncertainty_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json


DEFAULT_THREE_POINT_FIT_MODES = ["joint_ratio_fh"]
_THREE_POINT_POOL_CONTEXT: dict[str, Any] | None = None


def _serialize_three_point_fit_result(fit_result) -> dict[str, Any]:
    return {
        "Q": float(fit_result.Q),
        "chi2": float(fit_result.chi2),
        "dof": int(fit_result.dof),
        "loggbf": float(fit_result.logGBF),
        "parameter_means": {key: float(gv.mean(value)) for key, value in fit_result.p.items()},
        "bare_real": float(gv.mean(fit_result.p["O00_re"] / (2 * fit_result.p["E0"]))),
        "bare_imag": float(gv.mean(fit_result.p["O00_im"] / (2 * fit_result.p["E0"]))),
    }


def _fit_three_point_sample_from_context(context: dict[str, Any], sample_index: int):
    mode = context["mode"]
    if mode in {"ratio", "joint_ratio_fh"}:
        sample_two_point_fit = fit_two_point_correlator(
            context["two_point_fit_samples"][sample_index],
            temporal_extent=context["two_point_fit_parameters"]["temporal_extent"],
            tmin=context["two_point_fit_parameters"]["tmin"],
            tmax=context["two_point_fit_parameters"]["tmax"],
            state_count=context["two_point_fit_parameters"]["state_count"],
            boundary=context["two_point_fit_parameters"]["boundary"],
            normalize=context["two_point_fit_parameters"]["normalize"],
            prior_overrides=context["two_point_fit_parameters"].get("prior_overrides", {}),
        )
    else:
        sample_two_point_fit = SimpleNamespace(p={"E0": context["two_point_reference_E0"]})

    if mode == "ratio":
        return fit_ratio_correlator(
            {tsep: context["ratio_real_gv_by_tsep"][tsep][sample_index] for tsep in context["fit_tsep"]},
            {tsep: context["ratio_imag_gv_by_tsep"][tsep][sample_index] for tsep in context["fit_tsep"]},
            fit_window=context["fit_windows"]["ratio"],
            tau_axis=context["tau_axis"],
            temporal_extent=context["temporal_extent"],
            two_point_fit_result=sample_two_point_fit,
            prior_overrides=context["prior_overrides"],
        )
    if mode == "fh":
        return fit_fh_correlator(
            context["fh_real_gv_samples"][sample_index],
            context["fh_imag_gv_samples"][sample_index],
            fh_tsep=context["fh_tsep_by_part"],
            two_point_fit_result=sample_two_point_fit,
            prior_overrides=context["prior_overrides"],
        )
    if mode == "joint_ratio_fh":
        return fit_joint_ratio_fh_correlator(
            {tsep: context["ratio_real_gv_by_tsep"][tsep][sample_index] for tsep in context["fit_tsep"]},
            {tsep: context["ratio_imag_gv_by_tsep"][tsep][sample_index] for tsep in context["fit_tsep"]},
            context["fh_real_gv_samples"][sample_index],
            context["fh_imag_gv_samples"][sample_index],
            ratio_fit_window=context["fit_windows"]["ratio"],
            tau_axis=context["tau_axis"],
            temporal_extent=context["temporal_extent"],
            fh_tsep=context["fh_tsep_by_part"],
            two_point_fit_result=sample_two_point_fit,
            prior_overrides=context["prior_overrides"],
        )
    raise ValueError(f"Unsupported three-point fit mode: {mode}")


def _set_three_point_pool_context(context: dict[str, Any]) -> None:
    global _THREE_POINT_POOL_CONTEXT
    _THREE_POINT_POOL_CONTEXT = context


def _fit_three_point_sample_record_from_index(sample_index: int) -> dict[str, Any]:
    if _THREE_POINT_POOL_CONTEXT is None:
        raise RuntimeError("Three-point pool context is not initialized.")
    fit_result = _fit_three_point_sample_from_context(_THREE_POINT_POOL_CONTEXT, sample_index)
    return _serialize_three_point_fit_result(fit_result)


@register_stage
class CorrelatorAnalysisStage:
    """Analyze correlator inputs for downstream workflow stages."""

    name = "correlator_analysis"
    description = "Resample correlators, inspect effective masses, and run configurable 2pt/3pt fits."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        two_point_dataset = self._select_dataset(context, kind="two_point")
        three_point_datasets = self._select_datasets(context, kind="three_point")
        analyzable_three_point_datasets = [
            dataset
            for dataset in three_point_datasets
            if dataset.samples is not None and getattr(dataset.samples, "ndim", 0) == 3 and "tau" in dataset.extra_axes
        ]
        parameters = self._resolve_stage_parameters(context, two_point_dataset.metadata)

        two_point_result = self._analyze_two_point(two_point_dataset, parameters["two_point"])
        artifacts = self._write_two_point_artifacts(
            stage_dir=stage_dir,
            context=context,
            result=two_point_result,
        )

        payload: dict[str, Any] = {
            "axis": np.asarray(two_point_result["axis"], dtype=float),
            "values": np.asarray(two_point_result["values"], dtype=float),
            "spread": np.asarray(two_point_result["spread"], dtype=float),
            "input_label": two_point_dataset.label,
            "resampling": two_point_result["payload_resampling"],
            "bad_point_filter": two_point_result["bad_point_filter"],
            "effective_mass": two_point_result["payload_effective_mass"],
            "fit": two_point_result["fit_summary"],
        }

        three_point_payloads = []
        three_point_results = []
        if analyzable_three_point_datasets:
            for dataset in sorted(analyzable_three_point_datasets, key=self._dataset_sort_key):
                dataset_result = self._analyze_three_point_dataset(
                    dataset=dataset,
                    two_point_result=two_point_result,
                    parameters=parameters["three_point"],
                )
                three_point_results.append(dataset_result)
                three_point_payloads.append(dataset_result["payload"])
                artifacts.extend(
                    self._write_three_point_artifacts(
                        stage_dir=stage_dir,
                        context=context,
                        result=dataset_result,
                    )
                )
            payload["three_point"] = three_point_payloads
        if len(analyzable_three_point_datasets) != len(three_point_datasets):
            payload["ignored_three_point_inputs"] = [
                dataset.label for dataset in three_point_datasets if dataset not in analyzable_three_point_datasets
            ]

        if analyzable_three_point_datasets:
            qpdf_families = self._build_qpdf_families(
                payloads=three_point_results,
                preferred_fit_mode=parameters["three_point"]["primary_fit_mode"],
            )
            if qpdf_families:
                artifacts.extend(self._write_qpdf_sample_artifacts(stage_dir, qpdf_families))
                payload["qpdf_families"] = [self._serialize_qpdf_family(family) for family in qpdf_families]
                payload["_qpdf_families"] = qpdf_families
                bare_qpdf = self._build_legacy_bare_qpdf_alias(qpdf_families, parameters["three_point"]["primary_fit_mode"])
                if bare_qpdf is not None:
                    payload["bare_qpdf"] = bare_qpdf
                    artifacts.extend(self._write_bare_qpdf_artifacts(stage_dir, context, bare_qpdf))

        summary = self._build_summary(
            label=two_point_dataset.label,
            two_point_result=two_point_result,
            three_point_results=three_point_results,
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    def _select_dataset(self, context: StageContext, *, kind: str):
        candidates = [dataset for dataset in context.datasets.values() if dataset.kind == kind]
        if not candidates:
            raise ValueError(f"Correlator analysis requires at least one {kind.replace('_', '-')!s} correlator input.")
        return candidates[0]

    def _select_datasets(self, context: StageContext, *, kind: str):
        return [dataset for dataset in context.datasets.values() if dataset.kind == kind]

    def _resolve_stage_parameters(self, context: StageContext, metadata: dict[str, Any]) -> dict[str, Any]:
        parameters = context.parameters_for(self.name)
        two_point_parameters = dict(parameters.get("two_point", parameters))
        three_point_parameters = dict(parameters.get("three_point", {}))
        two_point_resampling = dict(two_point_parameters.get("resampling", {}))
        two_point_bad_filter = dict(two_point_parameters.get("bad_point_filter", {}))
        effective_mass_parameters = dict(two_point_parameters.get("effective_mass", {}))
        two_point_fit = dict(two_point_parameters.get("fit", {}))

        temporal_extent = int(
            two_point_fit.get(
                "temporal_extent",
                two_point_parameters.get("temporal_extent", metadata.get("temporal_extent", metadata.get("Lt", 0))),
            )
            or 0
        )
        if temporal_extent <= 0:
            temporal_extent = int(metadata.get("configuration_time_extent", 0) or metadata.get("n_t", 0) or 0)

        three_point_bad_filter = dict(three_point_parameters.get("bad_point_filter", two_point_bad_filter))
        fit_modes = list(three_point_parameters.get("fit_modes", DEFAULT_THREE_POINT_FIT_MODES))
        primary_fit_mode = str(
            three_point_parameters.get(
                "primary_fit_mode",
                "joint_ratio_fh" if "joint_ratio_fh" in fit_modes else fit_modes[0],
            )
        )

        return {
            "two_point": {
                "boundary": str(two_point_parameters.get("boundary", metadata.get("boundary", "periodic"))),
                "resampling_method": str(two_point_resampling.get("method", "jackknife")),
                "bootstrap_samples": int(two_point_resampling.get("sample_count", 500)),
                "bootstrap_sample_size": two_point_resampling.get("sample_size"),
                "bin_size": int(two_point_resampling.get("bin_size", 1)),
                "seed": int(two_point_resampling.get("seed", 1984)),
                "effective_mass_method": effective_mass_parameters.get("method"),
                "fit_enabled": bool(two_point_fit.get("enabled", True)),
                "state_count": int(two_point_fit.get("state_count", 2)),
                "tmin": int(two_point_fit.get("tmin", 4)),
                "tmax": int(two_point_fit.get("tmax", min(temporal_extent // 2 if temporal_extent else 20, 20))),
                "normalize": bool(two_point_fit.get("normalize", True)),
                "prior_overrides": dict(two_point_fit.get("prior_overrides", {})),
                "temporal_extent": temporal_extent,
                "bad_point_filter": self._resolve_bad_point_filter(two_point_bad_filter),
            },
            "three_point": {
                "fit_modes": fit_modes,
                "primary_fit_mode": primary_fit_mode,
                "fit_windows": self._resolve_three_point_fit_windows(three_point_parameters),
                "sample_fit_workers": max(1, int(three_point_parameters.get("sample_fit_workers", 1))),
                "gamma": str(three_point_parameters.get("gamma", "gt")),
                "flavor": str(three_point_parameters.get("flavor", "u-d")),
                "b": int(three_point_parameters.get("b", 0)),
                "prior_overrides": dict(three_point_parameters.get("prior_overrides", {})),
                "bad_point_filter": self._resolve_bad_point_filter(three_point_bad_filter),
            },
        }

    def _resolve_three_point_fit_windows(self, parameters: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
        legacy_fit_tsep = [int(value) for value in parameters.get("fit_tsep", [8, 10, 12])]
        legacy_tau_cut = int(parameters.get("tau_cut", 2))
        ratio_parameters = dict(parameters.get("ratio", {}))
        fh_parameters = dict(parameters.get("fh", {}))
        return {
            "ratio": self._resolve_three_point_part_windows(ratio_parameters, legacy_fit_tsep, legacy_tau_cut),
            "fh": self._resolve_three_point_part_windows(fh_parameters, legacy_fit_tsep, legacy_tau_cut),
        }

    def _resolve_three_point_part_windows(
        self,
        parameters: dict[str, Any],
        default_fit_tsep: list[int],
        default_tau_cut: int,
    ) -> dict[str, dict[str, Any]]:
        base_fit_tsep = [int(value) for value in parameters.get("fit_tsep", default_fit_tsep)]
        base_tau_cut = int(parameters.get("tau_cut", default_tau_cut))
        resolved: dict[str, dict[str, Any]] = {}
        for part in ("real", "imag"):
            part_parameters = dict(parameters.get(part, {}))
            resolved[part] = {
                "fit_tsep": [int(value) for value in part_parameters.get("fit_tsep", base_fit_tsep)],
                "tau_cut": int(part_parameters.get("tau_cut", base_tau_cut)),
            }
        return resolved

    def _resolve_bad_point_filter(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": str(parameters.get("mode", "mad")),
            "mad_zcut": float(parameters.get("mad_zcut", 8.0)),
            "ratio_cut": float(parameters.get("ratio_cut", 50.0)),
            "absolute_threshold": float(parameters.get("absolute_threshold", 1e12)),
            "replacement": str(parameters.get("replacement", "median")),
        }

    def _analyze_two_point(self, dataset, parameters: dict[str, Any]) -> dict[str, Any]:
        filtered_samples = None
        filter_payload = {"mode": "disabled", "flagged_count": 0}
        if dataset.samples is not None:
            filtered_samples, filter_info = filter_bad_points(
                dataset.samples,
                axis=-1,
                **parameters["bad_point_filter"],
            )
            filter_payload = {
                "mode": filter_info.mode,
                "replacement": filter_info.replacement,
                "flagged_count": filter_info.flagged_count,
                "total_count": filter_info.total_count,
                "mad_zcut": filter_info.mad_zcut,
                "ratio_cut": filter_info.ratio_cut,
                "absolute_threshold": filter_info.absolute_threshold,
            }
            generic_resampled = resample_observable(
                filtered_samples,
                method=parameters["resampling_method"],
                axis=-1,
                bootstrap_samples=parameters["bootstrap_samples"],
                bootstrap_sample_size=parameters["bootstrap_sample_size"],
                bin_size=parameters["bin_size"],
                seed=parameters["seed"],
            )
            resampled = ResampledCorrelator(
                method=generic_resampled.method,
                sample_means=generic_resampled.sample_means,
                mean=generic_resampled.mean,
                error=generic_resampled.error,
                average=generic_resampled.average,
                configuration_count=generic_resampled.configuration_count,
                resample_count=generic_resampled.resample_count,
                bin_size=generic_resampled.bin_size,
            )
        else:
            resampled = None

        values = np.asarray(dataset.values, dtype=float) if resampled is None else np.asarray(resampled.mean, dtype=float)
        spread = np.zeros_like(values) if resampled is None else np.asarray(resampled.error, dtype=float)
        normalization_factor = 1.0
        if resampled is not None and resampled.average is not None and parameters["normalize"]:
            normalization_factor = abs(float(gv.mean(resampled.average[0]))) or 1.0
        plotting_parameters = dict(parameters)
        plotting_parameters["normalization_factor"] = normalization_factor
        effective_mass = effective_mass_from_correlator(
            resampled if resampled is not None else values,
            method=parameters["effective_mass_method"],
            boundary=parameters["boundary"],
        )
        fit_summary, fit_result = self._fit_two_point(resampled, parameters)
        fit_curve = self._build_two_point_fit_curve(np.asarray(dataset.axis, dtype=float), fit_result, plotting_parameters)
        fit_effective_mass = self._build_two_point_fit_effective_mass(fit_curve, fit_result, plotting_parameters)
        analysis_settings = self._build_two_point_settings(plotting_parameters)

        return {
            "axis": np.asarray(dataset.axis, dtype=float),
            "values": values,
            "spread": spread,
            "resampled": resampled,
            "effective_mass": effective_mass,
            "fit_summary": fit_summary,
            "fit_result": fit_result,
            "fit_curve": fit_curve,
            "fit_effective_mass": fit_effective_mass,
            "analysis_settings": analysis_settings,
            "bad_point_filter": filter_payload,
            "payload_resampling": {
                "method": parameters["resampling_method"],
                "bin_size": parameters["bin_size"],
                "configuration_count": int(dataset.samples.shape[-1]) if dataset.samples is not None else 0,
                "resample_count": 0 if resampled is None else int(resampled.resample_count),
            },
            "payload_effective_mass": {
                "method": effective_mass.method,
                "axis": effective_mass.times.tolist(),
                "mean": np.asarray(effective_mass.mean, dtype=float).tolist(),
                "error": np.asarray(effective_mass.error, dtype=float).tolist(),
            },
        }

    def _fit_two_point(self, resampled: ResampledCorrelator | None, parameters: dict[str, Any]) -> tuple[dict[str, Any], Any | None]:
        if resampled is None or resampled.average is None:
            return {"performed": False, "reason": "Fit skipped because no resampled ensemble average is available."}, None
        if not parameters["fit_enabled"]:
            return {"performed": False, "reason": "Fit disabled by stage parameters."}, None
        if parameters["temporal_extent"] <= 0:
            return {"performed": False, "reason": "Fit skipped because the temporal lattice extent is not known."}, None
        try:
            fit_result = fit_two_point_correlator(
                resampled.average,
                temporal_extent=parameters["temporal_extent"],
                tmin=parameters["tmin"],
                tmax=parameters["tmax"],
                state_count=parameters["state_count"],
                boundary=parameters["boundary"],
                normalize=parameters["normalize"],
                prior_overrides=parameters["prior_overrides"],
            )
        except (OptionalDependencyError, ValueError, KeyError) as exc:
            return {"performed": False, "reason": str(exc)}, None

        summary = summarize_two_point_fit(fit_result, state_count=parameters["state_count"])
        summary.update(
            {
                "performed": True,
                "tmin": int(parameters["tmin"]),
                "tmax": int(parameters["tmax"]),
                "boundary": parameters["boundary"],
                "normalize": bool(parameters["normalize"]),
                "quality": "good" if fit_result.Q >= 0.05 else "poor",
                "resampling_method": parameters["resampling_method"],
            }
        )
        return summary, fit_result

    def _build_two_point_fit_curve(self, axis: np.ndarray, fit_result, parameters: dict[str, Any]) -> dict[str, np.ndarray] | None:
        if fit_result is None:
            return None
        fit_values = two_point_fit_function(
            axis,
            fit_result.p,
            temporal_extent=parameters["temporal_extent"],
            state_count=parameters["state_count"],
            boundary=parameters["boundary"],
        ) * float(parameters.get("normalization_factor", 1.0))
        return {
            "axis": np.asarray(axis, dtype=float),
            "fit_value": np.asarray(gv.mean(fit_values), dtype=float) if gv is not None else np.asarray(fit_values, dtype=float),
            "fit_error": np.asarray(gv.sdev(fit_values), dtype=float) if gv is not None else np.zeros_like(axis, dtype=float),
        }

    def _build_two_point_fit_effective_mass(
        self,
        fit_curve: dict[str, np.ndarray] | None,
        fit_result,
        parameters: dict[str, Any],
    ) -> dict[str, np.ndarray] | None:
        if fit_curve is None or fit_result is None or gv is None:
            return None
        parameter_keys = list(fit_result.p.keys())
        parameter_values = [fit_result.p[key] for key in parameter_keys]
        mean = np.asarray([gv.mean(value) for value in parameter_values], dtype=float)
        covariance = np.asarray(gv.evalcov(parameter_values), dtype=float)
        rng = np.random.default_rng(1984)
        sampled_parameters = rng.multivariate_normal(mean, covariance, size=200)

        meff_samples: list[np.ndarray] = []
        meff_axis: np.ndarray | None = None
        for sample in sampled_parameters:
            parameter_sample = {key: value for key, value in zip(parameter_keys, sample, strict=True)}
            curve_sample = np.asarray(
                two_point_fit_function(
                    fit_curve["axis"],
                    parameter_sample,
                    temporal_extent=parameters["temporal_extent"],
                    state_count=parameters["state_count"],
                    boundary=parameters["boundary"],
                ),
                dtype=float,
            )
            meff_sample = effective_mass_from_correlator(
                curve_sample,
                method=parameters["effective_mass_method"],
                boundary=parameters["boundary"],
            )
            meff_axis = meff_sample.times
            meff_samples.append(meff_sample.mean)
        if not meff_samples or meff_axis is None:
            return None
        sample_array = np.asarray(meff_samples, dtype=float)
        return {
            "axis": meff_axis,
            "fit_value": np.mean(sample_array, axis=0),
            "fit_error": np.std(sample_array, axis=0, ddof=1),
        }

    def _build_two_point_settings(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "resampling_method": parameters["resampling_method"],
            "bootstrap_samples": parameters["bootstrap_samples"],
            "bootstrap_sample_size": parameters["bootstrap_sample_size"],
            "bin_size": parameters["bin_size"],
            "seed": parameters["seed"],
            "boundary": parameters["boundary"],
            "effective_mass_method": parameters["effective_mass_method"],
            "fit_enabled": parameters["fit_enabled"],
            "state_count": parameters["state_count"],
            "tmin": parameters["tmin"],
            "tmax": parameters["tmax"],
            "normalize": parameters["normalize"],
            "prior_overrides": parameters["prior_overrides"],
            "temporal_extent": parameters["temporal_extent"],
            "normalization_factor": float(parameters.get("normalization_factor", 1.0)),
            "fit_band_sample_count": 200,
            "bad_point_filter": parameters["bad_point_filter"],
        }

    def _analyze_three_point_dataset(self, *, dataset, two_point_result: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
        if dataset.samples is None:
            raise ValueError(f"Three-point dataset {dataset.label!r} must provide raw samples.")
        if dataset.samples.ndim != 3:
            raise ValueError(f"Three-point dataset {dataset.label!r} must have shape (n_tsep, n_tau, n_cfg).")
        tau_axis = dataset.extra_axes.get("tau")
        if tau_axis is None:
            raise ValueError(f"Three-point dataset {dataset.label!r} is missing its tau axis.")
        if two_point_result["fit_result"] is None:
            raise ValueError("Three-point analysis requires a successful two-point fit result.")
        resampled_two_point = two_point_result["resampled"]
        if resampled_two_point is None or resampled_two_point.sample_means is None:
            raise ValueError("Three-point analysis requires resampled two-point samples.")

        raw_complex = np.asarray(dataset.samples)
        filtered_real, filter_real = filter_bad_points(np.real(raw_complex), axis=-1, **parameters["bad_point_filter"])
        filtered_imag, filter_imag = filter_bad_points(np.imag(raw_complex), axis=-1, **parameters["bad_point_filter"])
        resampled_real = resample_observable(
            filtered_real,
            method=two_point_result["analysis_settings"]["resampling_method"],
            axis=-1,
            bootstrap_samples=two_point_result["analysis_settings"]["bootstrap_samples"],
            bootstrap_sample_size=two_point_result["analysis_settings"]["bootstrap_sample_size"],
            bin_size=two_point_result["analysis_settings"]["bin_size"],
            seed=two_point_result["analysis_settings"]["seed"],
        )
        resampled_imag = resample_observable(
            filtered_imag,
            method=two_point_result["analysis_settings"]["resampling_method"],
            axis=-1,
            bootstrap_samples=two_point_result["analysis_settings"]["bootstrap_samples"],
            bootstrap_sample_size=two_point_result["analysis_settings"]["bootstrap_sample_size"],
            bin_size=two_point_result["analysis_settings"]["bin_size"],
            seed=two_point_result["analysis_settings"]["seed"],
        )

        zero_imag_two_point = np.zeros_like(resampled_two_point.sample_means)
        ratio_real_samples, ratio_imag_samples = build_ratio_samples(
            np.asarray(resampled_two_point.sample_means, dtype=float),
            zero_imag_two_point,
            np.asarray(resampled_real.sample_means, dtype=float),
            np.asarray(resampled_imag.sample_means, dtype=float),
            dataset.axis,
        )
        ratio_real_average = average_resampled_samples(ratio_real_samples, resampled_real.method)
        ratio_imag_average = average_resampled_samples(ratio_imag_samples, resampled_real.method)
        ratio_real_by_tsep = {int(tsep): ratio_real_average[index] for index, tsep in enumerate(np.asarray(dataset.axis, dtype=int))}
        ratio_imag_by_tsep = {int(tsep): ratio_imag_average[index] for index, tsep in enumerate(np.asarray(dataset.axis, dtype=int))}

        ratio_fit_windows = parameters["fit_windows"]["ratio"]
        fh_fit_windows = parameters["fit_windows"]["fh"]

        summed_tsep_real, summed_real = build_summed_ratio_samples(
            ratio_real_samples,
            dataset.axis,
            tau_axis,
            tau_cut=fh_fit_windows["real"]["tau_cut"],
            fit_tsep=fh_fit_windows["real"]["fit_tsep"],
        )
        summed_tsep_imag, summed_imag = build_summed_ratio_samples(
            ratio_imag_samples,
            dataset.axis,
            tau_axis,
            tau_cut=fh_fit_windows["imag"]["tau_cut"],
            fit_tsep=fh_fit_windows["imag"]["fit_tsep"],
        )
        fh_tsep_real, fh_real_samples = build_fh_samples(summed_real, summed_tsep_real)
        fh_tsep_imag, fh_imag_samples = build_fh_samples(summed_imag, summed_tsep_imag)
        fh_real_average = average_resampled_samples(fh_real_samples, resampled_real.method)
        fh_imag_average = average_resampled_samples(fh_imag_samples, resampled_real.method)
        two_point_gv_samples = add_error_to_resampled_samples(
            np.asarray(resampled_two_point.sample_means, dtype=float),
            resampled_real.method,
        )
        ratio_real_gv_by_tsep = {
            int(tsep): add_error_to_resampled_samples(np.asarray(ratio_real_samples[:, index, :], dtype=float), resampled_real.method)
            for index, tsep in enumerate(np.asarray(dataset.axis, dtype=int))
        }
        ratio_imag_gv_by_tsep = {
            int(tsep): add_error_to_resampled_samples(np.asarray(ratio_imag_samples[:, index, :], dtype=float), resampled_real.method)
            for index, tsep in enumerate(np.asarray(dataset.axis, dtype=int))
        }
        fh_real_gv_samples = add_error_to_resampled_samples(np.asarray(fh_real_samples, dtype=float), resampled_real.method)
        fh_imag_gv_samples = add_error_to_resampled_samples(np.asarray(fh_imag_samples, dtype=float), resampled_real.method)

        fits = self._fit_three_point_observables(
            ratio_real_by_tsep=ratio_real_by_tsep,
            ratio_imag_by_tsep=ratio_imag_by_tsep,
            ratio_real_gv_by_tsep=ratio_real_gv_by_tsep,
            ratio_imag_gv_by_tsep=ratio_imag_gv_by_tsep,
            fh_real_average=fh_real_average,
            fh_imag_average=fh_imag_average,
            fh_real_gv_samples=fh_real_gv_samples,
            fh_imag_gv_samples=fh_imag_gv_samples,
            tau_axis=tau_axis,
            temporal_extent=two_point_result["analysis_settings"]["temporal_extent"],
            two_point_fit_result=two_point_result["fit_result"],
            two_point_fit_samples=two_point_gv_samples,
            two_point_fit_parameters=two_point_result["analysis_settings"],
            parameters=parameters,
            fh_tsep_by_part={
                "real": np.asarray(fh_tsep_real, dtype=float),
                "imag": np.asarray(fh_tsep_imag, dtype=float),
            },
            resampling_method=two_point_result["analysis_settings"]["resampling_method"],
        )
        dataset_channel_metadata = {
            "gamma": str(dataset.metadata.get("gamma", parameters["gamma"])),
            "flavor": str(dataset.metadata.get("flavor", parameters["flavor"])),
            "b": int(dataset.metadata.get("b", parameters["b"])),
            "ss_sp": dataset.metadata.get("ss_sp"),
            "px": int(dataset.metadata.get("px", 0)),
            "py": int(dataset.metadata.get("py", 0)),
            "pz": int(dataset.metadata.get("pz", 0)),
        }
        for fit_payload in fits.values():
            fit_payload["summary"].update(dataset_channel_metadata)
        preferred_fit = fits.get(parameters["primary_fit_mode"])

        return {
            "label": dataset.label,
            "slug": self._slugify(dataset.label),
            "z": int(dataset.metadata.get("z", -1)),
            "b": dataset_channel_metadata["b"],
            "gamma": dataset_channel_metadata["gamma"],
            "flavor": dataset_channel_metadata["flavor"],
            "ss_sp": dataset_channel_metadata["ss_sp"],
            "px": dataset_channel_metadata["px"],
            "py": dataset_channel_metadata["py"],
            "pz": dataset_channel_metadata["pz"],
            "dataset_axis": np.asarray(dataset.axis, dtype=float),
            "tau_axis": np.asarray(tau_axis, dtype=float),
            "temporal_extent": int(two_point_result["analysis_settings"]["temporal_extent"]),
            "fit_windows": parameters["fit_windows"],
            "ratio_real_average": ratio_real_average,
            "ratio_imag_average": ratio_imag_average,
            "fh_tsep_by_part": {
                "real": np.asarray(fh_tsep_real, dtype=float),
                "imag": np.asarray(fh_tsep_imag, dtype=float),
            },
            "fh_real_average": fh_real_average,
            "fh_imag_average": fh_imag_average,
            "ratio_real_by_tsep": ratio_real_by_tsep,
            "ratio_imag_by_tsep": ratio_imag_by_tsep,
            "fits": fits,
            "preferred_fit_mode": parameters["primary_fit_mode"],
            "preferred_bare_matrix_element": None if preferred_fit is None else preferred_fit["summary"]["bare_matrix_element"],
            "bad_point_filter": {
                "mode": filter_real.mode,
                "replacement": filter_real.replacement,
                "flagged_real_count": filter_real.flagged_count,
                "flagged_imag_count": filter_imag.flagged_count,
                "total_count": filter_real.total_count + filter_imag.total_count,
            },
            "resampling": {
                "method": resampled_real.method,
                "configuration_count": resampled_real.configuration_count,
                "resample_count": resampled_real.resample_count,
                "bin_size": resampled_real.bin_size,
            },
            "payload": {
                "label": dataset.label,
                "z": int(dataset.metadata.get("z", -1)),
                "b": dataset_channel_metadata["b"],
                "gamma": dataset_channel_metadata["gamma"],
                "flavor": dataset_channel_metadata["flavor"],
                "ss_sp": dataset_channel_metadata["ss_sp"],
                "px": dataset_channel_metadata["px"],
                "py": dataset_channel_metadata["py"],
                "pz": dataset_channel_metadata["pz"],
                "tsep_axis": np.asarray(dataset.axis, dtype=float).tolist(),
                "tau_axis": np.asarray(tau_axis, dtype=float).tolist(),
                "fit_windows": parameters["fit_windows"],
                "resampling": {
                    "method": resampled_real.method,
                    "configuration_count": resampled_real.configuration_count,
                    "resample_count": resampled_real.resample_count,
                    "bin_size": resampled_real.bin_size,
                },
                "bad_point_filter": {
                    "mode": filter_real.mode,
                    "replacement": filter_real.replacement,
                    "flagged_real_count": filter_real.flagged_count,
                    "flagged_imag_count": filter_imag.flagged_count,
                },
                "ratio": self._serialize_matrix_observable(dataset.axis, tau_axis, ratio_real_average, ratio_imag_average),
                "fh": self._serialize_vector_observable(
                    {"real": fh_tsep_real, "imag": fh_tsep_imag},
                    fh_real_average,
                    fh_imag_average,
                ),
                "fits": {name: fit["summary"] for name, fit in fits.items()},
            },
        }

    def _fit_three_point_observables(
        self,
        *,
        ratio_real_by_tsep: dict[int, Sequence[Any]],
        ratio_imag_by_tsep: dict[int, Sequence[Any]],
        ratio_real_gv_by_tsep: dict[int, np.ndarray],
        ratio_imag_gv_by_tsep: dict[int, np.ndarray],
        fh_real_average,
        fh_imag_average,
        fh_real_gv_samples: np.ndarray,
        fh_imag_gv_samples: np.ndarray,
        tau_axis: np.ndarray,
        temporal_extent: int,
        two_point_fit_result,
        two_point_fit_samples: np.ndarray,
        two_point_fit_parameters: dict[str, Any],
        parameters: dict[str, Any],
        fh_tsep_by_part: dict[str, np.ndarray],
        resampling_method: str,
    ) -> dict[str, dict[str, Any]]:
        fits: dict[str, dict[str, Any]] = {}
        for mode in parameters["fit_modes"]:
            mode_fit_windows = self._fit_windows_for_mode(parameters["fit_windows"], mode)
            fit_context = {
                "mode": mode,
                "two_point_fit_samples": two_point_fit_samples,
                "two_point_fit_parameters": two_point_fit_parameters,
                "two_point_reference_E0": two_point_fit_result.p["E0"],
                "ratio_real_gv_by_tsep": ratio_real_gv_by_tsep,
                "ratio_imag_gv_by_tsep": ratio_imag_gv_by_tsep,
                "fh_real_gv_samples": fh_real_gv_samples,
                "fh_imag_gv_samples": fh_imag_gv_samples,
                "fit_tsep": sorted(
                    {
                        *parameters["fit_windows"]["ratio"]["real"]["fit_tsep"],
                        *parameters["fit_windows"]["ratio"]["imag"]["fit_tsep"],
                    }
                ),
                "fit_windows": mode_fit_windows,
                "tau_axis": tau_axis,
                "temporal_extent": temporal_extent,
                "fh_tsep_by_part": fh_tsep_by_part,
                "prior_overrides": parameters["prior_overrides"],
            }
            fit_result = _fit_three_point_sample_from_context(fit_context, 0)
            sample_fit_records = [_serialize_three_point_fit_result(fit_result)]

            remaining_indices = list(range(1, len(two_point_fit_samples)))
            worker_count = min(int(parameters["sample_fit_workers"]), len(two_point_fit_samples))
            if remaining_indices:
                if worker_count > 1:
                    ctx = mp.get_context("fork")
                    chunksize = max(1, len(remaining_indices) // (worker_count * 4))
                    with ctx.Pool(
                        processes=worker_count,
                        initializer=_set_three_point_pool_context,
                        initargs=(fit_context,),
                    ) as pool:
                        sample_fit_records.extend(
                            pool.map(_fit_three_point_sample_record_from_index, remaining_indices, chunksize=chunksize)
                        )
                else:
                    sample_fit_records.extend(
                        [
                            _serialize_three_point_fit_result(
                                _fit_three_point_sample_from_context(fit_context, sample_index)
                            )
                            for sample_index in remaining_indices
                        ]
                    )
            summary = summarize_three_point_fit_samples(
                sample_fit_records,
                fit_result,
                method=resampling_method,
                mode=mode,
                fit_windows=mode_fit_windows,
            )
            summary.update(
                {
                    "resampling_method": resampling_method,
                }
            )
            bare_real_samples = np.asarray([float(record["bare_real"]) for record in sample_fit_records], dtype=float)
            bare_imag_samples = np.asarray([float(record["bare_imag"]) for record in sample_fit_records], dtype=float)
            fits[mode] = {
                "result": fit_result,
                "summary": summary,
                "text": fit_result.format(100),
                "sample_fit_count": len(sample_fit_records),
                "fit_windows": mode_fit_windows,
                "bare_real_samples": bare_real_samples,
                "bare_imag_samples": bare_imag_samples,
            }
        return fits

    def _fit_windows_for_mode(self, fit_windows: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "ratio":
            return {"ratio": fit_windows["ratio"]}
        if mode == "fh":
            return {"fh": fit_windows["fh"]}
        if mode == "joint_ratio_fh":
            return {"ratio": fit_windows["ratio"], "fh": fit_windows["fh"]}
        raise ValueError(f"Unsupported three-point fit mode: {mode}")

    def _write_two_point_artifacts(self, *, stage_dir: Path, context: StageContext, result: dict[str, Any]) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        effective_mass: EffectiveMassResult = result["effective_mass"]
        if result["fit_curve"] is not None:
            for data_path in write_columnar_data(
                stage_dir / "two_point_fit_curve",
                result["fit_curve"],
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
        write_json(stage_dir / "two_point_fit_summary.json", result["fit_summary"])
        write_json(stage_dir / "correlator_analysis_settings.json", result["analysis_settings"])
        artifacts.append(
            ArtifactRecord(
                name="two_point_fit_summary_json",
                kind="report",
                path=stage_dir / "two_point_fit_summary.json",
                description="Fit diagnostics and posterior parameter summaries for the two-point correlator.",
                format="json",
            )
        )
        artifacts.append(
            ArtifactRecord(
                name="correlator_analysis_settings_json",
                kind="report",
                path=stage_dir / "correlator_analysis_settings.json",
                description="Analysis settings including bad-point filtering, resampling, effective-mass, and fit options.",
                format="json",
            )
        )
        if result["fit_result"] is not None:
            fit_result_path = stage_dir / "two_point_fit_result.txt"
            fit_result_path.write_text(result["fit_result"].format(100) + "\n", encoding="utf-8")
            artifacts.append(
                ArtifactRecord(
                    name="two_point_fit_result_txt",
                    kind="report",
                    path=fit_result_path,
                    description="Verbatim lsqfit nonlinear_fit summary produced by fit_result.format(100).",
                    format="txt",
                )
            )
        for plot_format in context.manifest.outputs.plot_formats:
            correlator_plot = stage_dir / f"correlator_analysis.{plot_format}"
            effective_mass_plot = stage_dir / f"effective_mass.{plot_format}"
            effective_mass_comparison_plot = stage_dir / f"effective_mass_comparison.{plot_format}"
            save_uncertainty_plot(
                result["axis"],
                result["values"],
                result["spread"],
                correlator_plot,
                "Two-Point Correlator",
                "tsep",
                "C2pt",
                fit_x=None if result["fit_curve"] is None else result["fit_curve"]["axis"],
                fit_y=None if result["fit_curve"] is None else result["fit_curve"]["fit_value"],
                fit_error=None if result["fit_curve"] is None else result["fit_curve"]["fit_error"],
                yscale="log",
                data_label=result["analysis_settings"]["resampling_method"].capitalize(),
                fit_label="Fit band",
            )
            save_uncertainty_plot(
                effective_mass.times,
                effective_mass.mean,
                effective_mass.error,
                effective_mass_plot,
                "Effective Mass",
                "tsep",
                r"$m_{\rm eff}$",
                data_label=result["analysis_settings"]["resampling_method"].capitalize(),
            )
            if result["fit_effective_mass"] is not None:
                save_uncertainty_plot(
                    effective_mass.times,
                    effective_mass.mean,
                    effective_mass.error,
                    effective_mass_comparison_plot,
                    "Effective Mass Comparison",
                    "tsep",
                    r"$m_{\rm eff}$",
                    fit_x=result["fit_effective_mass"]["axis"],
                    fit_y=result["fit_effective_mass"]["fit_value"],
                    fit_error=result["fit_effective_mass"]["fit_error"],
                    data_label=result["analysis_settings"]["resampling_method"].capitalize(),
                    fit_label="Fit band",
                )
            artifacts.extend(
                [
                    ArtifactRecord(
                        name=f"correlator_analysis_plot_{plot_format}",
                        kind="plot",
                        path=correlator_plot,
                        description="Two-point correlator mean signal plot.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"effective_mass_plot_{plot_format}",
                        kind="plot",
                        path=effective_mass_plot,
                        description="Effective-mass diagnostic plot derived from the two-point correlator.",
                        format=plot_format,
                    ),
                ]
            )
            if result["fit_effective_mass"] is not None:
                artifacts.append(
                    ArtifactRecord(
                        name=f"effective_mass_comparison_plot_{plot_format}",
                        kind="plot",
                        path=effective_mass_comparison_plot,
                        description="Comparison of the resampled effective mass and the fit-derived effective mass.",
                        format=plot_format,
                    )
                )
        return artifacts

    def _write_three_point_artifacts(self, *, stage_dir: Path, context: StageContext, result: dict[str, Any]) -> list[ArtifactRecord]:
        dataset_dir = ensure_directory(stage_dir / result["slug"])
        artifacts: list[ArtifactRecord] = []
        settings_path = dataset_dir / "analysis_settings.json"
        write_json(
            settings_path,
            {
                "label": result["label"],
                "z": result["z"],
                "resampling": result["resampling"],
                "bad_point_filter": result["bad_point_filter"],
                "fit_windows": result["fit_windows"],
                "fit_modes": list(result["fits"].keys()),
                "preferred_fit_mode": result["preferred_fit_mode"],
            },
        )
        artifacts.append(
            ArtifactRecord(
                name=f"{result['slug']}_analysis_settings_json",
                kind="report",
                path=settings_path,
                description=f"Three-point analysis settings for {result['label']}.",
                format="json",
            )
        )

        for fit_mode, fit_payload in result["fits"].items():
            summary_path = dataset_dir / f"{fit_mode}_fit_summary.json"
            write_json(summary_path, fit_payload["summary"])
            text_path = dataset_dir / f"{fit_mode}_fit_result.txt"
            text_path.write_text(fit_payload["text"] + "\n", encoding="utf-8")
            artifacts.extend(
                [
                    ArtifactRecord(
                        name=f"{result['slug']}_{fit_mode}_fit_summary_json",
                        kind="report",
                        path=summary_path,
                        description=f"{fit_mode} fit diagnostics for {result['label']}.",
                        format="json",
                    ),
                    ArtifactRecord(
                        name=f"{result['slug']}_{fit_mode}_fit_result_txt",
                        kind="report",
                        path=text_path,
                        description=f"Verbatim lsqfit summary for the {fit_mode} fit of {result['label']}.",
                        format="txt",
                    ),
                ]
            )

        for plot_format in context.manifest.outputs.plot_formats:
            ratio_real_plot = dataset_dir / f"ratio_real.{plot_format}"
            ratio_imag_plot = dataset_dir / f"ratio_imag.{plot_format}"
            fh_real_plot = dataset_dir / f"fh_real.{plot_format}"
            fh_imag_plot = dataset_dir / f"fh_imag.{plot_format}"
            save_series_collection_plot(
                self._ratio_series(result, part="real", include_fit_mode=None),
                ratio_real_plot,
                f"Ratio Data ({result['label']})",
                r"$\tau - t_{\rm sep}/2$",
                "ratio_re",
            )
            save_series_collection_plot(
                self._ratio_series(result, part="imag", include_fit_mode=None),
                ratio_imag_plot,
                f"Ratio Data ({result['label']})",
                r"$\tau - t_{\rm sep}/2$",
                "ratio_im",
            )
            save_series_collection_plot(
                [
                    {
                        "x": result["fh_tsep_by_part"]["real"],
                        "y": np.atleast_1d(np.asarray(gv.mean(result["fh_real_average"]), dtype=float)),
                        "error": np.atleast_1d(np.asarray(gv.sdev(result["fh_real_average"]), dtype=float)),
                        "label": "FH data",
                        "style": "errorbar",
                    }
                ],
                fh_real_plot,
                f"FH Data ({result['label']})",
                "tsep",
                "fh_re",
            )
            save_series_collection_plot(
                [
                    {
                        "x": result["fh_tsep_by_part"]["imag"],
                        "y": np.atleast_1d(np.asarray(gv.mean(result["fh_imag_average"]), dtype=float)),
                        "error": np.atleast_1d(np.asarray(gv.sdev(result["fh_imag_average"]), dtype=float)),
                        "label": "FH data",
                        "style": "errorbar",
                    }
                ],
                fh_imag_plot,
                f"FH Data ({result['label']})",
                "tsep",
                "fh_im",
            )
            artifacts.extend(
                [
                    ArtifactRecord(
                        name=f"{result['slug']}_ratio_real_plot_{plot_format}",
                        kind="plot",
                        path=ratio_real_plot,
                        description=f"Real-part ratio data plot for {result['label']}.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{result['slug']}_ratio_imag_plot_{plot_format}",
                        kind="plot",
                        path=ratio_imag_plot,
                        description=f"Imaginary-part ratio data plot for {result['label']}.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{result['slug']}_fh_real_plot_{plot_format}",
                        kind="plot",
                        path=fh_real_plot,
                        description=f"Real-part FH data plot for {result['label']}.",
                        format=plot_format,
                    ),
                    ArtifactRecord(
                        name=f"{result['slug']}_fh_imag_plot_{plot_format}",
                        kind="plot",
                        path=fh_imag_plot,
                        description=f"Imaginary-part FH data plot for {result['label']}.",
                        format=plot_format,
                    ),
                ]
            )
            for fit_mode in result["fits"]:
                if fit_mode in {"ratio", "joint_ratio_fh"}:
                    ratio_fit_real_plot = dataset_dir / f"ratio_fit_{fit_mode}_real.{plot_format}"
                    ratio_fit_imag_plot = dataset_dir / f"ratio_fit_{fit_mode}_imag.{plot_format}"
                    save_series_collection_plot(
                        self._ratio_series(result, part="real", include_fit_mode=fit_mode),
                        ratio_fit_real_plot,
                        f"Ratio + Fit ({fit_mode}, {result['label']})",
                        r"$\tau - t_{\rm sep}/2$",
                        "ratio_re",
                    )
                    save_series_collection_plot(
                        self._ratio_series(result, part="imag", include_fit_mode=fit_mode),
                        ratio_fit_imag_plot,
                        f"Ratio + Fit ({fit_mode}, {result['label']})",
                        r"$\tau - t_{\rm sep}/2$",
                        "ratio_im",
                    )
                    artifacts.extend(
                        [
                            ArtifactRecord(
                                name=f"{result['slug']}_{fit_mode}_ratio_fit_real_plot_{plot_format}",
                                kind="plot",
                                path=ratio_fit_real_plot,
                                description=f"Ratio data and {fit_mode} fit band for {result['label']} (real part).",
                                format=plot_format,
                            ),
                            ArtifactRecord(
                                name=f"{result['slug']}_{fit_mode}_ratio_fit_imag_plot_{plot_format}",
                                kind="plot",
                                path=ratio_fit_imag_plot,
                                description=f"Ratio data and {fit_mode} fit band for {result['label']} (imaginary part).",
                                format=plot_format,
                            ),
                        ]
                    )
                if fit_mode in {"fh", "joint_ratio_fh"}:
                    fh_fit_real_plot = dataset_dir / f"fh_fit_{fit_mode}_real.{plot_format}"
                    fh_fit_imag_plot = dataset_dir / f"fh_fit_{fit_mode}_imag.{plot_format}"
                    save_series_collection_plot(
                        self._fh_series(result, part="real", fit_mode=fit_mode),
                        fh_fit_real_plot,
                        f"FH + Fit ({fit_mode}, {result['label']})",
                        "tsep",
                        "fh_re",
                    )
                    save_series_collection_plot(
                        self._fh_series(result, part="imag", fit_mode=fit_mode),
                        fh_fit_imag_plot,
                        f"FH + Fit ({fit_mode}, {result['label']})",
                        "tsep",
                        "fh_im",
                    )
                    artifacts.extend(
                        [
                            ArtifactRecord(
                                name=f"{result['slug']}_{fit_mode}_fh_fit_real_plot_{plot_format}",
                                kind="plot",
                                path=fh_fit_real_plot,
                                description=f"FH data and {fit_mode} fit band for {result['label']} (real part).",
                                format=plot_format,
                            ),
                            ArtifactRecord(
                                name=f"{result['slug']}_{fit_mode}_fh_fit_imag_plot_{plot_format}",
                                kind="plot",
                                path=fh_fit_imag_plot,
                                description=f"FH data and {fit_mode} fit band for {result['label']} (imaginary part).",
                                format=plot_format,
                            ),
                        ]
                    )
        return artifacts

    def _write_bare_qpdf_artifacts(self, stage_dir: Path, context: StageContext, bare_qpdf: dict[str, Any]) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for data_path in write_columnar_data(
            stage_dir / "bare_qpdf_vs_z",
            {
                "z": np.asarray(bare_qpdf["z"], dtype=float),
                "real": np.asarray(bare_qpdf["real"]["mean"], dtype=float),
                "real_error": np.asarray(bare_qpdf["real"]["error"], dtype=float),
                "imag": np.asarray(bare_qpdf["imag"]["mean"], dtype=float),
                "imag_error": np.asarray(bare_qpdf["imag"]["error"], dtype=float),
            },
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(
                ArtifactRecord(
                    name=f"bare_qpdf_vs_z_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description="Bare qPDF matrix elements versus z.",
                    format=data_path.suffix[1:],
                )
            )
        write_json(stage_dir / "bare_qpdf_summary.json", bare_qpdf)
        artifacts.append(
            ArtifactRecord(
                name="bare_qpdf_summary_json",
                kind="report",
                path=stage_dir / "bare_qpdf_summary.json",
                description="Aggregated bare qPDF summaries extracted from the preferred fit mode.",
                format="json",
            )
        )
        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"bare_qpdf_vs_z.{plot_format}"
            save_series_collection_plot(
                [
                    {
                        "x": np.asarray(bare_qpdf["z"], dtype=float),
                        "y": np.asarray(bare_qpdf["real"]["mean"], dtype=float),
                        "error": np.asarray(bare_qpdf["real"]["error"], dtype=float),
                        "label": "Real",
                        "style": "errorbar",
                    },
                    {
                        "x": np.asarray(bare_qpdf["z"], dtype=float),
                        "y": np.asarray(bare_qpdf["imag"]["mean"], dtype=float),
                        "error": np.asarray(bare_qpdf["imag"]["error"], dtype=float),
                        "label": "Imag",
                        "style": "errorbar",
                    },
                ],
                plot_path,
                f"Bare qPDF vs z ({bare_qpdf['fit_mode']})",
                "z",
                "bare_qpdf",
            )
            artifacts.append(
                ArtifactRecord(
                    name=f"bare_qpdf_vs_z_plot_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description="Bare qPDF matrix elements versus z.",
                    format=plot_format,
                )
            )
        return artifacts

    def _build_qpdf_families(self, *, payloads: list[dict[str, Any]], preferred_fit_mode: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for payload in payloads:
            z_value = int(payload.get("z", -1))
            if z_value < 0 or preferred_fit_mode not in payload.get("fits", {}):
                continue
            key = (
                preferred_fit_mode,
                int(payload.get("b", 0)),
                str(payload.get("gamma", "")),
                str(payload.get("flavor", "")),
                str(payload.get("ss_sp", "")),
                int(payload.get("px", 0)),
                int(payload.get("py", 0)),
                int(payload.get("pz", 0)),
            )
            grouped.setdefault(key, []).append(payload)

        families: list[dict[str, Any]] = []
        for items in grouped.values():
            ordered = sorted(items, key=lambda item: int(item["z"]))
            sample_count = int(ordered[0]["fits"][preferred_fit_mode]["sample_fit_count"])
            real_samples = np.vstack(
                [np.asarray(item["fits"][preferred_fit_mode]["bare_real_samples"], dtype=float) for item in ordered]
            ).T
            imag_samples = np.vstack(
                [np.asarray(item["fits"][preferred_fit_mode]["bare_imag_samples"], dtype=float) for item in ordered]
            ).T
            method = ordered[0]["resampling"]["method"]
            real_average = average_resampled_samples(real_samples, method)
            imag_average = average_resampled_samples(imag_samples, method)
            family = {
                "metadata": {
                    "fit_mode": preferred_fit_mode,
                    "b": int(ordered[0].get("b", 0)),
                    "gamma": str(ordered[0].get("gamma", "")),
                    "flavor": str(ordered[0].get("flavor", "")),
                    "ss_sp": ordered[0].get("ss_sp"),
                    "px": int(ordered[0].get("px", 0)),
                    "py": int(ordered[0].get("py", 0)),
                    "pz": int(ordered[0].get("pz", 0)),
                    "resampling_method": method,
                },
                "z_axis": np.asarray([int(item["z"]) for item in ordered], dtype=float),
                "real_samples": real_samples,
                "imag_samples": imag_samples,
                "sample_count": sample_count,
                "real_mean": np.atleast_1d(np.asarray(gv.mean(real_average), dtype=float)),
                "real_error": np.atleast_1d(np.asarray(gv.sdev(real_average), dtype=float)),
                "imag_mean": np.atleast_1d(np.asarray(gv.mean(imag_average), dtype=float)),
                "imag_error": np.atleast_1d(np.asarray(gv.sdev(imag_average), dtype=float)),
            }
            families.append(family)
        return sorted(families, key=lambda family: (family["metadata"]["b"], family["metadata"]["px"], family["metadata"]["py"], family["metadata"]["pz"]))

    def _serialize_qpdf_family(self, family: dict[str, Any]) -> dict[str, Any]:
        return {
            "metadata": dict(family["metadata"]),
            "z_axis": np.atleast_1d(np.asarray(family["z_axis"], dtype=float)).tolist(),
            "sample_count": int(family["sample_count"]),
            "real": {
                "mean": np.atleast_1d(np.asarray(family["real_mean"], dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(family["real_error"], dtype=float)).tolist(),
            },
            "imag": {
                "mean": np.atleast_1d(np.asarray(family["imag_mean"], dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(family["imag_error"], dtype=float)).tolist(),
            },
            "sample_artifact": family.get("sample_artifact"),
        }

    def _write_qpdf_sample_artifacts(self, stage_dir: Path, qpdf_families: list[dict[str, Any]]) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for family in qpdf_families:
            metadata = family["metadata"]
            sample_path = stage_dir / (
                "bare_qpdf_samples"
                f"_{metadata['fit_mode']}"
                f"_b{metadata['b']}"
                f"_p{metadata['px']}{metadata['py']}{metadata['pz']}"
                f"_{metadata['gamma']}"
                f"_{str(metadata['flavor']).replace('-', '_')}"
                f"_{str(metadata.get('ss_sp') or 'na')}.npz"
            )
            np.savez(
                sample_path,
                z_axis=np.asarray(family["z_axis"], dtype=float),
                real_samples=np.asarray(family["real_samples"], dtype=float),
                imag_samples=np.asarray(family["imag_samples"], dtype=float),
            )
            family["sample_artifact"] = str(sample_path)
            artifacts.append(
                ArtifactRecord(
                    name=sample_path.stem,
                    kind="data",
                    path=sample_path,
                    description="Sample-wise bare qPDF matrix elements retained for downstream stages.",
                    format="npz",
                )
            )
        return artifacts

    def _build_legacy_bare_qpdf_alias(self, qpdf_families: list[dict[str, Any]], fit_mode: str) -> dict[str, Any] | None:
        matching = [family for family in qpdf_families if family["metadata"]["fit_mode"] == fit_mode]
        if len(matching) != 1:
            return None
        family = matching[0]
        return {
            "fit_mode": fit_mode,
            "z": np.atleast_1d(np.asarray(family["z_axis"], dtype=float)).astype(int).tolist(),
            "real": {
                "mean": np.atleast_1d(np.asarray(family["real_mean"], dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(family["real_error"], dtype=float)).tolist(),
            },
            "imag": {
                "mean": np.atleast_1d(np.asarray(family["imag_mean"], dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(family["imag_error"], dtype=float)).tolist(),
            },
        }

    def _ratio_series(self, result: dict[str, Any], *, part: str, include_fit_mode: str | None) -> list[dict[str, object]]:
        series: list[dict[str, object]] = []
        tsep_axis = np.asarray(result["dataset_axis"], dtype=int)
        tau_axis = np.asarray(result["tau_axis"], dtype=int)
        averages = result["ratio_real_average"] if part == "real" else result["ratio_imag_average"]
        for index, tsep in enumerate(tsep_axis):
            valid_mask = tau_axis <= tsep
            if include_fit_mode is not None:
                ratio_window = result["fits"][include_fit_mode]["fit_windows"]["ratio"][part]
                if int(tsep) not in ratio_window["fit_tsep"]:
                    continue
                valid_mask = valid_mask & (tau_axis >= ratio_window["tau_cut"]) & (tau_axis <= tsep - ratio_window["tau_cut"])
            x_values = tau_axis[valid_mask] - tsep / 2.0
            series.append(
                {
                    "x": x_values,
                    "y": np.asarray(gv.mean(averages[index][valid_mask]), dtype=float),
                    "error": np.asarray(gv.sdev(averages[index][valid_mask]), dtype=float),
                    "label": f"tsep = {tsep}",
                    "style": "errorbar",
                }
            )
            if include_fit_mode is not None and include_fit_mode in result["fits"]:
                fit_result = result["fits"][include_fit_mode]["result"]
                fit_mean, fit_error = evaluate_ratio_band(
                    fit_result,
                    tsep=int(tsep),
                    tau_values=tau_axis[valid_mask],
                    temporal_extent=result["temporal_extent"],
                    part=part,
                )
                series.append(
                    {
                        "x": x_values,
                        "y": fit_mean,
                        "error": fit_error,
                        "label": f"{include_fit_mode} fit",
                        "style": "fill_between",
                    }
                )
        if include_fit_mode is not None and include_fit_mode in result["fits"]:
            bare = result["fits"][include_fit_mode]["summary"]["bare_matrix_element"]["real" if part == "real" else "imag"]
            max_half_width = np.max(tsep_axis) / 2.0
            series.append(
                {
                    "x": np.linspace(-max_half_width, max_half_width, 100),
                    "y": np.full(100, float(bare["mean"])),
                    "error": np.full(100, float(bare["sdev"])),
                    "label": "Bare matrix element",
                    "style": "fill_between",
                }
            )
        return series

    def _fh_series(self, result: dict[str, Any], *, part: str, fit_mode: str) -> list[dict[str, object]]:
        average = result["fh_real_average"] if part == "real" else result["fh_imag_average"]
        x_values = np.atleast_1d(np.asarray(result["fh_tsep_by_part"][part], dtype=float))
        mean_values = np.atleast_1d(np.asarray(gv.mean(average), dtype=float))
        error_values = np.atleast_1d(np.asarray(gv.sdev(average), dtype=float))
        series: list[dict[str, object]] = [
            {
                "x": x_values,
                "y": mean_values,
                "error": error_values,
                "label": "FH data",
                "style": "errorbar",
            }
        ]
        fit_result = result["fits"][fit_mode]["result"]
        fit_mean, fit_error = evaluate_fh_band(fit_result, tsep_values=x_values, part=part)
        series.append(
            {
                "x": x_values,
                "y": np.atleast_1d(np.asarray(fit_mean, dtype=float)),
                "error": np.atleast_1d(np.asarray(fit_error, dtype=float)),
                "label": "Fit band",
                "style": "fill_between",
            }
        )
        return series

    def _serialize_matrix_observable(self, tsep_axis, tau_axis, real_average, imag_average) -> dict[str, Any]:
        return {
            "tsep_axis": np.asarray(tsep_axis, dtype=float).tolist(),
            "tau_axis": np.asarray(tau_axis, dtype=float).tolist(),
            "real": {
                "mean": np.asarray(gv.mean(real_average), dtype=float).tolist(),
                "error": np.asarray(gv.sdev(real_average), dtype=float).tolist(),
            },
            "imag": {
                "mean": np.asarray(gv.mean(imag_average), dtype=float).tolist(),
                "error": np.asarray(gv.sdev(imag_average), dtype=float).tolist(),
            },
        }

    def _serialize_vector_observable(self, x_axis_by_part, real_average, imag_average) -> dict[str, Any]:
        return {
            "real": {
                "axis": np.atleast_1d(np.asarray(x_axis_by_part["real"], dtype=float)).tolist(),
                "mean": np.atleast_1d(np.asarray(gv.mean(real_average), dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(gv.sdev(real_average), dtype=float)).tolist(),
            },
            "imag": {
                "axis": np.atleast_1d(np.asarray(x_axis_by_part["imag"], dtype=float)).tolist(),
                "mean": np.atleast_1d(np.asarray(gv.mean(imag_average), dtype=float)).tolist(),
                "error": np.atleast_1d(np.asarray(gv.sdev(imag_average), dtype=float)).tolist(),
            },
        }

    def _flatten_matrix_observable(self, tsep_axis, tau_axis, real_average, imag_average) -> dict[str, np.ndarray]:
        tsep_grid, tau_grid = np.meshgrid(np.asarray(tsep_axis, dtype=float), np.asarray(tau_axis, dtype=float), indexing="ij")
        return {
            "tsep": tsep_grid.reshape(-1),
            "tau": tau_grid.reshape(-1),
            "real": np.asarray(gv.mean(real_average), dtype=float).reshape(-1),
            "real_error": np.asarray(gv.sdev(real_average), dtype=float).reshape(-1),
            "imag": np.asarray(gv.mean(imag_average), dtype=float).reshape(-1),
            "imag_error": np.asarray(gv.sdev(imag_average), dtype=float).reshape(-1),
        }

    def _flatten_vector_observable(self, x_axis, real_average, imag_average) -> dict[str, np.ndarray]:
        return {
            "tsep": np.atleast_1d(np.asarray(x_axis, dtype=float)),
            "real": np.atleast_1d(np.asarray(gv.mean(real_average), dtype=float)),
            "real_error": np.atleast_1d(np.asarray(gv.sdev(real_average), dtype=float)),
            "imag": np.atleast_1d(np.asarray(gv.mean(imag_average), dtype=float)),
            "imag_error": np.atleast_1d(np.asarray(gv.sdev(imag_average), dtype=float)),
        }

    def _dataset_sort_key(self, dataset) -> tuple[int, str]:
        return (int(dataset.metadata.get("z", 10_000)), dataset.label)

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "three_point"

    def _build_summary(self, *, label: str, two_point_result: dict[str, Any], three_point_results: list[dict[str, Any]]) -> str:
        fit_summary = two_point_result["fit_summary"]
        if fit_summary.get("performed"):
            fit_clause = (
                f"{fit_summary['state_count']}-state 2pt fit completed with "
                f"chi2/dof = {fit_summary['chi2_per_dof']:.3f} and Q = {fit_summary['Q']:.3f}"
            )
        else:
            fit_clause = f"2pt fit skipped ({fit_summary.get('reason', 'unknown reason')})"
        if not three_point_results:
            return (
                f"Resampled two-point correlator {label!r}, computed the effective mass, "
                f"and {fit_clause}."
            )
        preferred = [item for item in three_point_results if item.get("preferred_bare_matrix_element") is not None]
        if preferred:
            fit_mode = preferred[0]["preferred_fit_mode"]
            return (
                f"Resampled two-point correlator {label!r}, analyzed {len(three_point_results)} three-point dataset(s), "
                f"and built bare-qPDF summaries from the {fit_mode} fits after {fit_clause}."
            )
        return (
            f"Resampled two-point correlator {label!r}, analyzed {len(three_point_results)} three-point dataset(s), "
            f"and {fit_clause}."
        )
