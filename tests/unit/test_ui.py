"""Tests for unified framework UI and progress routing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest

from lamet_agent.agent import create_session
from lamet_agent.banner import BANNER
from lamet_agent.manifest import Manifest
from prompt_toolkit.shortcuts.progress_bar.formatters import IterationsPerSecond, TimeLeft

from lamet_agent.ui import (
    PlainUi,
    ProgressTask,
    TerminalUi,
    UiCancelled,
    _ANSI_RAINBOW,
    _PROGRESS_STYLE,
    _progress_formatters,
    current_ui,
    track,
    use_ui,
    warning,
)


class RecordingUi(PlainUi):
    def __init__(self) -> None:
        self.messages = []
        self.started = []
        self.advanced = []
        self.finished = []
        self.confirmations = []
        self.answers = []

    def log(self, message="", *, level="info", style=None) -> None:
        self.messages.append((str(message), level))

    def start_progress(self, label: str, *, total: int, unit: str) -> ProgressTask:
        task = ProgressTask(label, total, unit)
        self.started.append(task)
        return task

    def advance_progress(self, task: ProgressTask, amount: int = 1) -> None:
        super().advance_progress(task, amount)
        self.advanced.append((task.label, amount, task.completed))

    def finish_progress(self, task: ProgressTask, *, success: bool = True) -> None:
        super().finish_progress(task, success=success)
        self.finished.append((task.label, success, task.completed))

    def confirm(self, question: str) -> bool:
        self.messages.append((question, "question"))
        return self.confirmations.pop(0)

    def ask(self, question: str, _state=None) -> str:
        self.messages.append((question, "question"))
        return self.answers.pop(0)


class _Backend:
    identity = "ui:test"

    def close(self):
        pass


def _manifest(tmp_path: Path, *, include_seed: bool = True) -> Manifest:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    metadata = {
        "run_id": "ui",
        "root_directory": str(tmp_path),
        "artifacts_directory": "runs",
        "workers": 1,
        "target_observable": "pdf",
        "parton": "quark",
        "resample_mode": "jackknife",
        "sample_error_mode": "covariance",
        "bin_size": 1,
    }
    if include_seed:
        metadata["random_seed"] = 7
    document = {
        "metadata": metadata,
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "en",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return Manifest(path, copy.deepcopy(document))


def test_track_routes_progress_through_the_active_ui() -> None:
    ui = RecordingUi()

    with use_ui(ui):
        assert current_ui() is ui
        assert list(track(["a", "b", "c"], label="fits", unit="z")) == ["a", "b", "c"]

    assert [(task.label, task.total, task.unit) for task in ui.started] == [("fits", 3, "z")]
    assert ui.advanced[-1] == ("fits", 1, 3)
    assert ui.finished == [("fits", True, 3)]


def test_terminal_progress_formatters_keep_eta_and_add_iteration_speed() -> None:
    formatters = _progress_formatters()

    assert any(isinstance(formatter, TimeLeft) for formatter in formatters)
    assert any(isinstance(formatter, IterationsPerSecond) for formatter in formatters)
    speed_index = next(
        index for index, formatter in enumerate(formatters) if isinstance(formatter, IterationsPerSecond)
    )
    eta_index = next(index for index, formatter in enumerate(formatters) if isinstance(formatter, TimeLeft))
    assert speed_index < eta_index


def test_progress_color_is_neutral_gray() -> None:
    assert str(_PROGRESS_STYLE.get_attrs_for_style_str("class:percentage").color) == "ansibrightblack"


def test_track_marks_interrupted_progress_unsuccessful() -> None:
    ui = RecordingUi()

    with use_ui(ui):
        iterator = track([1, 2, 3], label="models", unit="model")
        assert next(iterator) == 1
        iterator.close()

    assert ui.finished == [("models", False, 0)]


def test_warning_uses_the_same_active_ui_log_stream() -> None:
    ui = RecordingUi()

    with use_ui(ui):
        warning("fit quality is low")

    assert ui.messages == [("ATTENTION: fit quality is low", "warning")]


def test_terminal_ui_applies_semantic_colors_without_changing_plain_output(capsys) -> None:
    ui = TerminalUi.__new__(TerminalUi)

    ui.warning("low fit quality")
    ui.log("Reasoning: recommendation", style="llm")
    ui.log("LLM usage: 1.00K", style="llm")
    ui.log("Executing: read data", style="running")

    output = capsys.readouterr().out
    assert "\033[91mATTENTION\033[0m: low fit quality" in output
    assert "\033[94mReasoning\033[0m: recommendation" in output
    assert "\033[94mLLM usage\033[0m: 1.00K" in output
    assert "\033[32mExecuting\033[0m: read data" in output
    assert "low fit quality\033[0m" not in output
    assert "recommendation\033[0m" not in output
    assert "read data\033[0m" not in output


def test_terminal_ui_renders_banner_with_left_to_right_rainbow(capsys) -> None:
    ui = TerminalUi.__new__(TerminalUi)

    ui.log(BANNER, style="banner")

    output = capsys.readouterr().out
    assert "\033[38;2;" not in output
    assert "\033[38;5;" not in output
    for color in (31, 91, 33, 93, 32, 92, 36, 96, 34, 94, 35, 95):
        assert f"\033[{color}m" in output
    assert BANNER in re.sub(r"\x1b\[[0-9;]*m", "", output)
    assert _ANSI_RAINBOW == tuple(
        f"\033[{color}m" for color in (91, 31, 33, 93, 92, 32, 36, 96, 94, 34, 35, 95, 91)
    )


def test_plain_ui_ctrl_c_cancels_interaction(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(UiCancelled, match="cancelled by user"):
        PlainUi().ask("Choose another artifacts directory")


def test_banner_is_emitted_before_manifest_validation(tmp_path: Path) -> None:
    ui = RecordingUi()

    with pytest.raises(ValueError, match="random_seed"):
        create_session(_Backend(), ui=ui).run_manifest(_manifest(tmp_path, include_seed=False))

    assert ui.messages[0][0] == BANNER


def test_empty_artifacts_directory_is_reused_without_confirmation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.validate() == []
    (tmp_path / "runs").mkdir()
    ui = RecordingUi()
    session = create_session(_Backend(), ui=ui)

    _jobs, root = session._prepare_artifact_directory(manifest)

    assert root == tmp_path / "runs"
    assert not [message for message in ui.messages if message[1] == "question"]


def test_nonempty_artifacts_directory_is_removed_only_after_confirmation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.validate() == []
    root = tmp_path / "runs"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    ui = RecordingUi()
    ui.confirmations = [True]
    session = create_session(_Backend(), ui=ui)

    _jobs, selected = session._prepare_artifact_directory(manifest)

    assert selected == root
    assert not root.exists()


def test_declined_artifact_overwrite_accepts_a_new_path(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.validate() == []
    root = tmp_path / "runs"
    root.mkdir()
    (root / "old.txt").write_text("old", encoding="utf-8")
    ui = RecordingUi()
    ui.confirmations = [False]
    ui.answers = ["runs-new"]
    session = create_session(_Backend(), ui=ui)

    jobs, selected = session._prepare_artifact_directory(manifest)

    assert selected == tmp_path / "runs-new"
    assert manifest.document["metadata"]["artifacts_directory"] == "runs-new"
    assert all((tmp_path / "runs-new") in job.artifact_directory.parents for job in jobs)
    assert (root / "old.txt").is_file()
