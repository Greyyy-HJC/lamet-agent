"""Unit tests for core tool helpers."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.core.tools import resolve_plot_save_path


def test_resolve_plot_save_path_strips_suffix_and_uses_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    resolved = resolve_plot_save_path(
        "/elsewhere/plots/fit_on_data.png",
        artifacts_dir=artifacts,
    )
    assert resolved == str(artifacts / "fit_on_data")


def test_resolve_plot_save_path_default_stem(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    resolved = resolve_plot_save_path(None, artifacts_dir=artifacts)
    assert resolved == str(artifacts / "fit_on_data")
