"""Evaluation stage for extracting derived observables from cross-family comparisons.

Supported evaluation methods (selected via ``stage_parameters.evaluation.method``):

* ``cs_kernel_momentum_ratio``: extract the Collins-Soper kernel from the
  logarithmic ratio of x-space quasi distributions at different momenta.
  Optionally reduce the x-dependent kernel to a constant-in-x value per b.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.extensions.statistics import constant_fit, gv
from lamet_agent.plotting import save_series_collection_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json

logger = logging.getLogger(__name__)


@register_stage
class EvaluationStage:
    """Extract derived observables that require cross-family aggregation."""

    name = "evaluation"
    description = "Extract observables such as the Collins-Soper kernel from cross-momentum comparisons."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        parameters = self._resolve_parameters(context)
        method = parameters["method"]

        if method == "cs_kernel_momentum_ratio":
            return self._run_cs_kernel_momentum_ratio(context, stage_dir, parameters)

        raise ValueError(f"Unsupported evaluation method: {method!r}")

    # ------------------------------------------------------------------
    # Parameter resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_parameters(context: StageContext) -> dict[str, Any]:
        params = dict(context.parameters_for("evaluation"))
        x_window = params.get("x_window", [0.3, 0.7])
        plot_ylim = params.get("plot_ylim", [-3.0, 3.0])
        return {
            "method": str(params.get("method", "cs_kernel_momentum_ratio")),
            "x_window": [float(x_window[0]), float(x_window[1])],
            "constant_x_fit": bool(params.get("constant_x_fit", True)),
            "momentum_pairs": params.get("momentum_pairs", "all"),
            "family_selector": dict(params.get("family_selector", {})),
            "plot_ylim": [float(plot_ylim[0]), float(plot_ylim[1])],
        }

    # ------------------------------------------------------------------
    # CS kernel via momentum ratio
    # ------------------------------------------------------------------

    def _run_cs_kernel_momentum_ratio(
        self,
        context: StageContext,
        stage_dir: Path,
        parameters: dict[str, Any],
    ) -> StageResult:
        previous = context.stage_payloads.get("fourier_transform", {})
        families = previous.get("_transformed_families")
        if not families:
            raise ValueError("evaluation requires _transformed_families from the fourier_transform stage.")

        families = self._select_families(families, parameters["family_selector"])
        grouped = self._group_families_by_b(families)

        x_window = parameters["x_window"]
        do_constant_fit = parameters["constant_x_fit"]

        cs_kernel_raw: dict[str, Any] = {}
        cs_kernel_reduced: list[dict[str, Any]] = []
        evaluation_results: list[dict[str, Any]] = []
        artifacts: list[ArtifactRecord] = []

        for (group_key, b_value), b_families in sorted(grouped.items()):
            momenta, momentum_labels, total_momenta_gev = self._extract_momentum_info(b_families)
            if len(b_families) < 2:
                logger.warning("evaluation: b=%d has only %d momentum; skipping CS kernel extraction.", b_value, len(b_families))
                continue

            pairs = self._build_momentum_pairs(b_families, parameters["momentum_pairs"])
            pair_results: list[dict[str, Any]] = []
            for idx1, idx2 in pairs:
                family1 = b_families[idx1]
                family2 = b_families[idx2]
                result = self._cs_kernel_for_pair(family1, family2, x_window, do_constant_fit)
                pair_label = f"p{momentum_labels[idx2]}_p{momentum_labels[idx1]}"
                result["pair_label"] = pair_label
                result["b"] = b_value
                pair_results.append(result)
                cs_kernel_raw[f"b{b_value}_{pair_label}"] = {
                    "x_axis": result["x_axis"].tolist(),
                    "mean": result["mean"].tolist(),
                    "error": result["error"].tolist(),
                }

            if do_constant_fit and pair_results:
                reduced = self._reduce_across_pairs(pair_results, b_value, group_key)
                cs_kernel_reduced.append(reduced)

            evaluation_results.extend(pair_results)
            artifacts.extend(self._write_b_artifacts(stage_dir, context, b_value, pair_results, group_key, x_window, parameters["plot_ylim"]))

        if cs_kernel_reduced:
            artifacts.extend(self._write_reduced_artifacts(stage_dir, context, cs_kernel_reduced))

        payload: dict[str, Any] = {
            "method": "cs_kernel_momentum_ratio",
            "cs_kernel_raw": cs_kernel_raw,
            "cs_kernel_reduced": [
                {k: v for k, v in entry.items() if not str(k).startswith("_")}
                for entry in cs_kernel_reduced
            ],
            "_evaluation_results": evaluation_results,
            "_cs_kernel_reduced": cs_kernel_reduced,
        }
        n_pairs = len(evaluation_results)
        n_b = len(cs_kernel_reduced)
        summary = (
            f"Extracted Collins-Soper kernel via momentum ratio for {n_pairs} momentum pair(s) "
            f"across {n_b} b value(s)."
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    # ------------------------------------------------------------------
    # Family grouping and selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_families(families: list[dict[str, Any]], selector: dict[str, Any]) -> list[dict[str, Any]]:
        if not selector:
            return list(families)
        return [
            f for f in families
            if all(f["metadata"].get(k) == v for k, v in selector.items())
        ]

    @staticmethod
    def _group_families_by_b(
        families: list[dict[str, Any]],
    ) -> dict[tuple[tuple, int], list[dict[str, Any]]]:
        """Group families by (group_key, b) where group_key = (setup_id, gamma, flavor, smearing)."""
        grouped: dict[tuple[tuple, int], list[dict[str, Any]]] = {}
        for family in families:
            meta = family["metadata"]
            group_key = (
                str(meta["setup_id"]),
                str(meta["gamma"]),
                str(meta.get("flavor", "")),
                str(meta["smearing"]),
            )
            b = int(meta["b"])
            key = (group_key, b)
            grouped.setdefault(key, []).append(family)
        for key in grouped:
            grouped[key].sort(key=lambda f: sum(int(c) ** 2 for c in f["metadata"]["momentum"]))
        return grouped

    @staticmethod
    def _extract_momentum_info(
        b_families: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, ...]], list[str], list[float]]:
        momenta = []
        labels = []
        totals_gev = []
        for f in b_families:
            meta = f["metadata"]
            mom = (int(meta["px"]), int(meta["py"]), int(meta["pz"]))
            momenta.append(mom)
            labels.append(f"{mom[0]}{mom[1]}{mom[2]}")
            mom_info = f.get("momentum", {})
            total_gev = float(mom_info.get("total_gev", 0.0))
            if total_gev <= 0.0:
                total_gev = float(np.sqrt(sum(c ** 2 for c in mom)))
            totals_gev.append(total_gev)
        return momenta, labels, totals_gev

    @staticmethod
    def _build_momentum_pairs(
        b_families: list[dict[str, Any]],
        spec: Any,
    ) -> list[tuple[int, int]]:
        n = len(b_families)
        if spec == "all" or spec is None:
            return [(i, j) for i, j in itertools.combinations(range(n), 2)]
        return [(int(p[0]), int(p[1])) for p in spec]

    # ------------------------------------------------------------------
    # CS kernel computation
    # ------------------------------------------------------------------

    def _cs_kernel_for_pair(
        self,
        family_lo: dict[str, Any],
        family_hi: dict[str, Any],
        x_window: list[float],
        do_constant_fit: bool,
    ) -> dict[str, Any]:
        """Compute the CS kernel from one momentum pair using sample-level data."""
        x_axis_lo = np.asarray(family_lo["x_axis"], dtype=float)
        x_axis_hi = np.asarray(family_hi["x_axis"], dtype=float)
        if not np.allclose(x_axis_lo, x_axis_hi):
            raise ValueError("evaluation: x grids for momentum pair do not match.")
        x_axis = x_axis_lo

        real_lo = np.asarray(family_lo["real_samples"], dtype=float)
        real_hi = np.asarray(family_hi["real_samples"], dtype=float)

        mom_lo = family_lo.get("momentum", {})
        mom_hi = family_hi.get("momentum", {})
        p_lo = float(mom_lo.get("total_gev", 0.0))
        p_hi = float(mom_hi.get("total_gev", 0.0))
        if p_lo <= 0 or p_hi <= 0:
            meta_lo = family_lo["metadata"]
            meta_hi = family_hi["metadata"]
            p_lo = float(np.sqrt(int(meta_lo["px"]) ** 2 + int(meta_lo["py"]) ** 2 + int(meta_lo["pz"]) ** 2))
            p_hi = float(np.sqrt(int(meta_hi["px"]) ** 2 + int(meta_hi["py"]) ** 2 + int(meta_hi["pz"]) ** 2))

        ln_p_ratio = float(np.log(p_hi / p_lo)) if p_hi > p_lo > 0 else 1.0

        n_samples = min(real_lo.shape[0], real_hi.shape[0])
        cs_samples = np.full((n_samples, len(x_axis)), np.nan)
        for n in range(n_samples):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = real_hi[n] / real_lo[n]
                cs_samples[n] = np.log(ratio) / ln_p_ratio

        method = str(family_lo["metadata"].get("resampling_method", "jackknife"))
        cs_mean = np.nanmean(cs_samples, axis=0)
        if method == "jackknife":
            cs_error = np.nanstd(cs_samples, axis=0, ddof=1) * np.sqrt(max(n_samples - 1, 1))
        else:
            cs_error = np.nanstd(cs_samples, axis=0, ddof=1)

        result: dict[str, Any] = {
            "x_axis": x_axis,
            "mean": cs_mean,
            "error": cs_error,
            "samples": cs_samples,
            "p_lo_gev": p_lo,
            "p_hi_gev": p_hi,
            "ln_p_ratio": ln_p_ratio,
        }

        if do_constant_fit and gv is not None:
            result["constant_fit"] = self._constant_x_fit(x_axis, cs_samples, x_window, method)

        return result

    @staticmethod
    def _constant_x_fit(
        x_axis: np.ndarray,
        cs_samples: np.ndarray,
        x_window: list[float],
        resampling_method: str,
    ) -> dict[str, Any]:
        """Fit the CS kernel to a constant within the x window for each sample."""
        mask = (x_axis >= x_window[0]) & (x_axis <= x_window[1])
        x_selected = x_axis[mask]
        if len(x_selected) == 0:
            return {"value": 0.0, "error": 0.0, "x_window": x_window, "n_points": 0}

        cs_in_window = cs_samples[:, mask]
        mean_per_x = np.nanmean(cs_in_window, axis=0)
        if resampling_method == "jackknife":
            n = cs_in_window.shape[0]
            error_per_x = np.nanstd(cs_in_window, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))
        else:
            error_per_x = np.nanstd(cs_in_window, axis=0, ddof=1)

        valid = np.isfinite(mean_per_x) & np.isfinite(error_per_x) & (error_per_x > 0)
        if not np.any(valid):
            return {"value": 0.0, "error": 0.0, "x_window": x_window, "n_points": 0}

        gv_values = gv.gvar(mean_per_x[valid], np.clip(error_per_x[valid], 1e-12, None))
        try:
            fit_val = constant_fit(gv_values)
            return {
                "value": float(gv.mean(fit_val)),
                "error": float(gv.sdev(fit_val)),
                "x_window": x_window,
                "n_points": int(np.sum(valid)),
            }
        except Exception:
            logger.warning("evaluation: constant_fit failed in x window; falling back to weighted mean.")
            weights = 1.0 / error_per_x[valid] ** 2
            wmean = float(np.average(mean_per_x[valid], weights=weights))
            werr = float(1.0 / np.sqrt(np.sum(weights)))
            return {"value": wmean, "error": werr, "x_window": x_window, "n_points": int(np.sum(valid))}

    # ------------------------------------------------------------------
    # Cross-pair reduction
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_across_pairs(
        pair_results: list[dict[str, Any]],
        b_value: int,
        group_key: tuple,
    ) -> dict[str, Any]:
        """Combine constant-fit results from multiple momentum pairs into a single value with systematic."""
        values = []
        errors = []
        for pr in pair_results:
            cf = pr.get("constant_fit", {})
            if cf.get("n_points", 0) > 0:
                values.append(cf["value"])
                errors.append(cf["error"])

        if not values:
            return {"b": b_value, "group_key": list(group_key), "value": 0.0, "stat_error": 0.0, "sys_error": 0.0, "total_error": 0.0, "n_pairs": 0}

        values_arr = np.array(values)
        errors_arr = np.array(errors)
        weights = 1.0 / errors_arr ** 2
        central = float(np.average(values_arr, weights=weights))
        stat = float(1.0 / np.sqrt(np.sum(weights)))
        sys = float(np.std(values_arr, ddof=0)) if len(values_arr) > 1 else 0.0
        total = float(np.sqrt(stat ** 2 + sys ** 2))
        return {
            "b": b_value,
            "group_key": list(group_key),
            "value": central,
            "stat_error": stat,
            "sys_error": sys,
            "total_error": total,
            "n_pairs": len(values),
            "per_pair": [
                {"pair_label": pr["pair_label"], "value": pr["constant_fit"]["value"], "error": pr["constant_fit"]["error"]}
                for pr in pair_results if pr.get("constant_fit", {}).get("n_points", 0) > 0
            ],
        }

    # ------------------------------------------------------------------
    # Artifact writing
    # ------------------------------------------------------------------

    def _write_b_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        b_value: int,
        pair_results: list[dict[str, Any]],
        group_key: tuple,
        x_window: list[float],
        plot_ylim: list[float],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        slug = f"cs_kernel_b{b_value}"

        for pr in pair_results:
            pair_slug = f"{slug}_{pr['pair_label']}"
            for data_path in write_columnar_data(
                stage_dir / pair_slug,
                {
                    "x": pr["x_axis"],
                    "cs_kernel": pr["mean"],
                    "cs_kernel_error": pr["error"],
                },
                context.manifest.outputs.data_formats,
            ):
                artifacts.append(ArtifactRecord(
                    name=f"{pair_slug}_data_{data_path.suffix[1:]}",
                    kind="data",
                    path=data_path,
                    description=f"CS kernel gamma(x) for b={b_value}, {pr['pair_label']}.",
                    format=data_path.suffix[1:],
                ))

        series = []
        for pr in pair_results:
            series.append({
                "x": pr["x_axis"],
                "y": pr["mean"],
                "error": pr["error"],
                "label": pr["pair_label"],
                "style": "fill_between",
            })
        if series:
            plot_xlim = (x_window[0], x_window[1])
            plot_ylim_tuple = (plot_ylim[0], plot_ylim[1])
            for plot_format in context.manifest.outputs.plot_formats:
                plot_path = stage_dir / f"{slug}_x_dependence.{plot_format}"
                save_series_collection_plot(
                    series,
                    plot_path,
                    f"Collins-Soper kernel b={b_value}",
                    r"$x$",
                    r"$\gamma_\zeta(b_\perp)$",
                    xlim=plot_xlim,
                    ylim=plot_ylim_tuple,
                )
                artifacts.append(ArtifactRecord(
                    name=f"{slug}_x_dependence_{plot_format}",
                    kind="plot",
                    path=plot_path,
                    description=f"CS kernel x dependence for b={b_value} across momentum pairs.",
                    format=plot_format,
                ))
        return artifacts

    def _write_reduced_artifacts(
        self,
        stage_dir: Path,
        context: StageContext,
        reduced: list[dict[str, Any]],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        summary_path = stage_dir / "cs_kernel_reduced_summary.json"
        write_json(summary_path, {"cs_kernel_reduced": reduced})
        artifacts.append(ArtifactRecord(
            name="cs_kernel_reduced_summary_json",
            kind="report",
            path=summary_path,
            description="Reduced CS kernel gamma(b) with statistical and systematic errors.",
            format="json",
        ))

        b_values = [float(r["b"]) for r in reduced]
        values = [float(r["value"]) for r in reduced]
        errors = [float(r["total_error"]) for r in reduced]
        for data_path in write_columnar_data(
            stage_dir / "cs_kernel_reduced",
            {"b": np.array(b_values), "cs_kernel": np.array(values), "total_error": np.array(errors)},
            context.manifest.outputs.data_formats,
        ):
            artifacts.append(ArtifactRecord(
                name=f"cs_kernel_reduced_data_{data_path.suffix[1:]}",
                kind="data",
                path=data_path,
                description="Reduced CS kernel gamma(b) summary data.",
                format=data_path.suffix[1:],
            ))

        for plot_format in context.manifest.outputs.plot_formats:
            plot_path = stage_dir / f"cs_kernel_reduced.{plot_format}"
            save_series_collection_plot(
                [{
                    "x": np.array(b_values),
                    "y": np.array(values),
                    "error": np.array(errors),
                    "label": "CS kernel (reduced)",
                    "style": "errorbar",
                }],
                plot_path,
                "Collins-Soper kernel (reduced)",
                r"$b_\perp$ [lattice units]",
                r"$\gamma_\zeta(b_\perp)$",
            )
            artifacts.append(ArtifactRecord(
                name=f"cs_kernel_reduced_plot_{plot_format}",
                kind="plot",
                path=plot_path,
                description="Reduced CS kernel gamma(b) plot.",
                format=plot_format,
            ))
        return artifacts
