"""Unit tests for shared stage-report formatting and artifact paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
)


def test_report_formatters_handle_scalars_and_list_previews() -> None:
    assert format_report_value(None) == "not set"
    assert format_report_value(1.23456, digits=4) == "1.235"
    assert format_report_value("label") == "label"
    assert format_report_list([]) == "[]"
    assert format_report_list(range(10), max_items=3) == "[0, 1, 2, ...]"


def test_resolve_report_target_selects_one_language_path(tmp_path: Path) -> None:
    path = tmp_path / "stage_report.md"
    assert resolve_report_target(path, "en") == (path, "en")
    assert resolve_report_target(path, "ch") == (tmp_path / "stage_report_CN.md", "zh")
    with pytest.raises(ValueError, match="report_language"):
        resolve_report_target(path, "fr")


def test_markdown_artifact_paths_relativizes_selected_scalar_and_list_paths(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    artifact = tmp_path / "artifacts" / "result.nc"
    plot = tmp_path / "artifacts" / "plot.svg"
    output = markdown_artifact_paths(
        {
            "artifact": artifact,
            "plots": [plot, "relative.svg", None],
            "relative": "already-relative.pdf",
            "untouched": artifact,
        },
        base_dir=report_dir,
        path_keys=("artifact", "relative"),
        list_path_keys=("plots",),
    )
    assert output["artifact"] == "../artifacts/result.nc"
    assert output["plots"] == ["../artifacts/plot.svg", "relative.svg"]
    assert output["relative"] == "already-relative.pdf"
    assert output["untouched"] == artifact
