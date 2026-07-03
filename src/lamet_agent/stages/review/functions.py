"""Review-stage utilities built from existing stage reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lamet_agent.manifest import AnalysisManifest

from .reporting import write_review_report


def write_review_from_manifest(
    manifest: AnalysisManifest,
    *,
    report_language: str = "en",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write a paper-style review from reports under the manifest artifact tree."""
    artifacts_dir = Path(output_dir) if output_dir is not None else manifest.artifacts_directory
    review_dir = artifacts_dir / "review"
    target = review_dir / "review.md"
    stage_files = {
        "correlator_analysis": ("ca_report.md", "ca_report_CN.md"),
        "renormalization": ("renorm_report.md", "renorm_report_CN.md"),
        "fourier_transform": ("ft_report.md", "ft_report_CN.md"),
        "perturbative_matching": ("matching_report.md", "matching_report_CN.md"),
        "extrapolation": ("extrapolation_report.md", "extrapolation_report_CN.md"),
    }
    reports = []
    missing = []
    language = "zh" if report_language.lower() == "ch" else "en"
    for stage in manifest.metadata.stages:
        if stage not in stage_files:
            continue
        en_name, zh_name = stage_files[stage]
        path = artifacts_dir / stage / (zh_name if language == "zh" else en_name)
        if not path.exists() and language == "zh":
            path = artifacts_dir / stage / en_name
        if path.exists():
            reports.append({"stage": stage, "path": path, "text": path.read_text(encoding="utf-8")})
        else:
            missing.append(stage)
    output = write_review_report(
        reports=reports,
        missing_stages=missing,
        path=target,
        report_language=report_language,
        metadata=manifest.model_dump(mode="json"),
    )
    return {"review": str(output["report"]), "artifact": str(output["report"]), "n_reports": len(reports), "missing_stages": missing}


def write_review(store: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Tool wrapper: write review from ``store['manifest']``."""
    result = write_review_from_manifest(store["manifest"], **kwargs)
    store["output"] = result["review"]
    return result


STAGE_TOOLS = {"write_review": write_review}
