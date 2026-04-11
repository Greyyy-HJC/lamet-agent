"""Renormalization stage for signal rescaling and scheme-specific normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.plotting import save_line_plot
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import register_stage
from lamet_agent.utils import ensure_directory, write_columnar_data, write_json

logger = logging.getLogger(__name__)


@register_stage
class RenormalizationStage:
    """Apply renormalization to the analyzed signal.

    Supported schemes (selected via ``stage_parameters.renormalization.scheme``):

    * ``identity`` (default): pass families through unchanged.
    * ``cg_ratio``: CG ratio-scheme for qTMDWF / qTMDPDF.
      1. Divide every family by the real-part z=0 value of the b=0 family at
         the same momentum (b=0 normalization).
      2. Divide by the real-part z=0 value of the p=0 family at the same b
         (zero-momentum reference subtraction).
    """

    name = "renormalization"
    description = "Rescale the correlator signal using a simple normalization convention."

    def run(self, context: StageContext) -> StageResult:
        stage_dir = ensure_directory(context.stage_directory(self.name))
        previous = context.stage_payloads["correlator_analysis"]
        matrix_element_families = previous.get("_matrix_element_families")

        parameters = dict(context.parameters_for(self.name))
        scheme = str(parameters.get("scheme", "identity")).lower()

        if matrix_element_families:
            if scheme == "cg_ratio":
                return self._run_cg_ratio(context, stage_dir, matrix_element_families)
            return self._run_identity_families(context, stage_dir, previous, matrix_element_families)

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

    # ------------------------------------------------------------------
    # Identity pass-through (existing behaviour)
    # ------------------------------------------------------------------

    def _run_identity_families(
        self,
        context: StageContext,
        stage_dir: Path,
        previous: dict[str, Any],
        families: list[dict[str, Any]],
    ) -> StageResult:
        payload = {
            "mode": "identity",
            "renormalization_applied": False,
            "renormalized_families": previous.get("matrix_element_families", []),
            "_renormalized_families": families,
        }
        artifacts = self._write_family_artifacts(stage_dir, families, mode="identity")
        summary = "Skipped renormalization and passed the sample-wise coordinate-space families through unchanged."
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    # ------------------------------------------------------------------
    # CG ratio scheme
    # ------------------------------------------------------------------

    def _run_cg_ratio(
        self,
        context: StageContext,
        stage_dir: Path,
        families: list[dict[str, Any]],
    ) -> StageResult:
        """Apply CG ratio-scheme renormalization across families.

        Step 1 — b=0 normalization: for each family, divide all z-dependent
        samples by the mean real-part value at z=0 from the b=0 family that
        shares the same (setup_id, momentum, gamma, flavor, smearing).

        Step 2 — p=0 reference: divide by the mean real-part value at z=0
        from the zero-momentum family that shares the same (setup_id, b,
        gamma, flavor, smearing).
        """
        family_index = self._build_family_index(families)
        renormalized_families: list[dict[str, Any]] = []
        renorm_log: list[dict[str, Any]] = []

        for family in families:
            meta = family["metadata"]
            momentum_tuple = (int(meta["px"]), int(meta["py"]), int(meta["pz"]))
            group_key = self._group_key(meta)

            b0_factor = self._b0_norm_factor(family_index, group_key, momentum_tuple)
            p0_factor = self._p0_norm_factor(family_index, group_key, int(meta["b"]))

            real_samples = np.asarray(family["real_samples"], dtype=float)
            imag_samples = np.asarray(family["imag_samples"], dtype=float)

            combined_factor = b0_factor * p0_factor
            normed_real = real_samples / combined_factor
            normed_imag = imag_samples / combined_factor

            normed_real_mean = np.mean(normed_real, axis=0)
            normed_imag_mean = np.mean(normed_imag, axis=0)
            method = str(meta.get("resampling_method", "jackknife"))
            if method == "jackknife":
                n = normed_real.shape[0]
                normed_real_error = np.std(normed_real, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))
                normed_imag_error = np.std(normed_imag, axis=0, ddof=1) * np.sqrt(max(n - 1, 1))
            else:
                normed_real_error = np.std(normed_real, axis=0, ddof=1)
                normed_imag_error = np.std(normed_imag, axis=0, ddof=1)

            renormalized_family = {
                **family,
                "real_samples": normed_real,
                "imag_samples": normed_imag,
                "real_mean": normed_real_mean,
                "real_error": normed_real_error,
                "imag_mean": normed_imag_mean,
                "imag_error": normed_imag_error,
            }
            renormalized_families.append(renormalized_family)
            renorm_log.append({
                "family": f"b={meta['b']}_p{meta['px']}{meta['py']}{meta['pz']}",
                "b0_factor": float(b0_factor),
                "p0_factor": float(p0_factor),
            })

        logger.info("cg_ratio renormalization applied to %d families", len(renormalized_families))

        payload: dict[str, Any] = {
            "mode": "cg_ratio",
            "renormalization_applied": True,
            "renormalized_families": [self._serialize_family(f) for f in renormalized_families],
            "_renormalized_families": renormalized_families,
            "renorm_factors": renorm_log,
        }
        artifacts = self._write_family_artifacts(stage_dir, renormalized_families, mode="cg_ratio")
        summary = (
            f"Applied cg_ratio renormalization (b=0 normalization + p=0 reference) "
            f"to {len(renormalized_families)} families."
        )
        return StageResult(stage_name=self.name, summary=summary, payload=payload, artifacts=artifacts)

    @staticmethod
    def _group_key(meta: dict[str, Any]) -> tuple[str, str, str, str]:
        """Key that groups families by everything except momentum and b."""
        return (
            str(meta["setup_id"]),
            str(meta["gamma"]),
            str(meta.get("flavor", "")),
            str(meta["smearing"]),
        )

    @staticmethod
    def _build_family_index(
        families: list[dict[str, Any]],
    ) -> dict[tuple, list[dict[str, Any]]]:
        """Index families by (group_key, momentum_tuple, b)."""
        index: dict[tuple, list[dict[str, Any]]] = {}
        for family in families:
            meta = family["metadata"]
            group = RenormalizationStage._group_key(meta)
            momentum = (int(meta["px"]), int(meta["py"]), int(meta["pz"]))
            b = int(meta["b"])
            key = (group, momentum, b)
            index.setdefault(key, []).append(family)
        return index

    @staticmethod
    def _z0_real_mean(family: dict[str, Any]) -> float:
        """Return the mean of the real-part samples at z=0 (index 0)."""
        real_samples = np.asarray(family["real_samples"], dtype=float)
        return float(np.mean(real_samples[:, 0]))

    def _b0_norm_factor(
        self,
        index: dict[tuple, list[dict[str, Any]]],
        group_key: tuple[str, str, str, str],
        momentum: tuple[int, int, int],
    ) -> float:
        """Return the b=0 normalization factor for a given group and momentum."""
        key = (group_key, momentum, 0)
        candidates = index.get(key, [])
        if not candidates:
            logger.warning(
                "cg_ratio: no b=0 family found for group=%s momentum=%s; using 1.0",
                group_key,
                momentum,
            )
            return 1.0
        return self._z0_real_mean(candidates[0])

    def _p0_norm_factor(
        self,
        index: dict[tuple, list[dict[str, Any]]],
        group_key: tuple[str, str, str, str],
        b_value: int,
    ) -> float:
        """Return the p=0 reference factor for a given group and b value."""
        p0 = (0, 0, 0)
        key = (group_key, p0, b_value)
        candidates = index.get(key, [])
        if not candidates:
            logger.warning(
                "cg_ratio: no p=0 family found for group=%s b=%d; using 1.0",
                group_key,
                b_value,
            )
            return 1.0
        return self._z0_real_mean(candidates[0])

    @staticmethod
    def _serialize_family(family: dict[str, Any]) -> dict[str, Any]:
        return {
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

    # ------------------------------------------------------------------
    # Artifact writing
    # ------------------------------------------------------------------

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

    def _write_family_artifacts(
        self,
        stage_dir: Path,
        families: list[dict[str, Any]],
        *,
        mode: str = "identity",
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        summary_payload = {
            "mode": mode,
            "renormalization_applied": mode != "identity",
            "family_count": len(families),
            "families": [self._serialize_family(family) for family in families],
        }
        summary_path = stage_dir / "renormalization_family_summary.json"
        write_json(summary_path, summary_payload)
        artifacts.append(
            ArtifactRecord(
                name="renormalization_family_summary_json",
                kind="report",
                path=summary_path,
                description=f"Renormalization summary ({mode} scheme) for sample-wise coordinate-space families.",
                format="json",
            )
        )
        return artifacts
