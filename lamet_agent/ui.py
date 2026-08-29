"""Unified terminal UI, logging, interaction, and progress events.

Purpose: route Plan and Run output through one interface with interactive and
plain-terminal renderers.
Inputs: framework log/progress events and optional Plan conversation state.
Outputs: terminal rendering, user answers, confirmations, and progress updates.
Example: ``session = create_session(backend, ui=TerminalUi())``.
"""

from __future__ import annotations

import colorsys
import json
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.shortcuts.progress_bar.formatters import (
    IterationsPerSecond,
    Text,
    TimeLeft,
    create_default_formatters,
)
from prompt_toolkit.styles import Style

_COMMANDS = ("/show", "/issues", "/undo", "/edit", "/save", "/help", "/quit")
_ANSI_STYLES = {
    "attention": "\033[91m",
    "llm": "\033[94m",
    "running": "\033[32m",
}
_ANSI_RESET = "\033[0m"
_STATUS_PREFIXES = {
    "attention": ("ATTENTION", "Execution failed"),
    "llm": ("LLM usage", "Reasoning"),
    "running": ("Executing", "Running"),
}
_SHIFT_ENTER_SEQUENCES = {
    "\x1b[27;2;13~",  # xterm modifyOtherKeys
    "\x1b[13;2u",  # CSI u / extended keyboard protocol
}

_CONVERSATION_KEY_BINDINGS = KeyBindings()


@_CONVERSATION_KEY_BINDINGS.add("enter")
def _submit_or_insert_newline(event) -> None:
    """Submit on Enter while preserving distinct Shift+Enter sequences."""
    key_data = event.key_sequence[-1].data
    if key_data in _SHIFT_ENTER_SEQUENCES:
        event.current_buffer.newline()
    else:
        event.current_buffer.validate_and_handle()


_PROGRESS_STYLE = Style.from_dict(
    {
        "": "#808080",
        "bottom-toolbar": "#808080",
    }
)


def _progress_formatters():
    """Insert iteration speed immediately before the default ETA block."""
    formatters = create_default_formatters()
    time_left = next(index for index, formatter in enumerate(formatters) if isinstance(formatter, TimeLeft))
    eta_label = time_left - 1
    return [
        *formatters[:eta_label],
        IterationsPerSecond(),
        Text(" it/s "),
        *formatters[eta_label:],
    ]


def _render_rainbow_banner(message: str) -> str:
    """Render a smooth cyclic hue gradient with the xterm ANSI-256 cube."""
    lines = message.splitlines()
    width = max((len(line) for line in lines), default=0)
    if width == 0:
        return message
    rendered = []
    for line in lines:
        parts = []
        for column, character in enumerate(line):
            if character == " ":
                parts.append(character)
                continue
            hue = column / max(width - 1, 1)
            red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            red_level, green_level, blue_level = (round(value * 5) for value in (red, green, blue))
            color_index = 16 + 36 * red_level + 6 * green_level + blue_level
            parts.append(f"\033[38;5;{color_index}m{character}")
        rendered.append("".join(parts) + _ANSI_RESET)
    return "\n".join(rendered)


class UiCancelled(RuntimeError):
    """Raised when the user explicitly cancels an interactive workflow."""


@dataclass
class ProgressTask:
    """One renderer-neutral progress counter."""

    label: str
    total: int
    unit: str
    completed: int = 0
    native: Any | None = None


def _pointer_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}/{str(key).replace('~', '~0').replace('/', '~1')}"
            yield path
            yield from _pointer_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}/{index}"
            yield path
            yield from _pointer_paths(child, path)


class _ManifestCompleter(Completer):
    def __init__(self) -> None:
        self.state: Any | None = None
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)
        if text.startswith("/show ") and self.state is not None:
            for path in _pointer_paths(self.state.candidate):
                if path.startswith(word):
                    yield Completion(path, start_position=-len(word))
            return
        if word.startswith((".", "~")) or "/" in word[1:]:
            yield from self.path_completer.get_completions(document, complete_event)
            return
        for command in _COMMANDS:
            if command.startswith(word):
                yield Completion(command, start_position=-len(word))


class PlainUi:
    """Non-full-screen UI for redirected output, tests, and simple terminals."""

    def log(self, message: str = "", *, level: str = "info", style: str | None = None) -> None:
        stream = sys.stderr if level == "error" else sys.stdout
        print(message, file=stream, flush=True)

    def warning(self, message: str) -> None:
        self.log(f"ATTENTION: {message}", level="warning", style="attention")

    def write(self, message: str) -> None:
        self.log(message)

    def manifest_updated(self, edits: list[dict[str, Any]], state: Any) -> None:
        self.log(
            f"Manifest updated ({len(edits)} field{'s' if len(edits) != 1 else ''}); "
            f"{len(state.issues)} validation issue{'s' if len(state.issues) != 1 else ''} remain."
        )

    def show_patch(self, edits: list[dict[str, Any]], state: Any) -> None:
        self.manifest_updated(edits, state)

    def ask(self, question: str, _state: Any | None = None) -> str:
        self.log(f"\n{'Planner' if _state is not None else 'Agent'}: {question}")
        try:
            return input("You> ").strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc

    def confirm(self, question: str) -> bool:
        try:
            return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc

    def review_plan(self, question: str, _state: Any) -> bool | str | None:
        try:
            answer = input(f"{question} [y/N or describe a revision] ").strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc
        if answer.lower() in {"y", "yes"}:
            return True
        if answer.lower() in {"", "n", "no"}:
            return None
        return answer

    def start_progress(self, label: str, *, total: int, unit: str) -> ProgressTask:
        return ProgressTask(label, total, unit)

    def advance_progress(self, task: ProgressTask, amount: int = 1) -> None:
        task.completed = min(task.total, task.completed + amount)

    def finish_progress(self, task: ProgressTask, *, success: bool = True) -> None:
        if success:
            task.completed = task.total

    def close(self) -> None:
        pass


class TerminalUi(PlainUi):
    """Persistent prompt_toolkit conversation and progress renderer."""

    def __init__(self) -> None:
        self.completer = _ManifestCompleter()
        self.session = PromptSession(
            history=InMemoryHistory(),
            completer=self.completer,
            complete_while_typing=False,
            auto_suggest=AutoSuggestFromHistory(),
        )
        self._progress_bar: ProgressBar | None = None
        self._stdout_context: Any | None = None

    def log(self, message: str = "", *, level: str = "info", style: str | None = None) -> None:
        stream = sys.stderr if level == "error" else sys.stdout
        if style == "banner":
            print(_render_rainbow_banner(message), file=stream, flush=True)
            return
        color = _ANSI_STYLES.get(style or "")
        rendered = message
        if color:
            prefix = next(
                (candidate for candidate in _STATUS_PREFIXES.get(style or "", ()) if message.startswith(candidate)),
                None,
            )
            if prefix is not None:
                rendered = f"{color}{prefix}{_ANSI_RESET}{message[len(prefix):]}"
        print(rendered, file=stream, flush=True)

    def _ensure_progress_bar(self) -> ProgressBar:
        if self._progress_bar is None:
            self._stdout_context = patch_stdout(raw=True)
            self._stdout_context.__enter__()
            self._progress_bar = ProgressBar(
                formatters=_progress_formatters(),
                bottom_toolbar=" LaMET Agent running ",
                style=_PROGRESS_STYLE,
                color_depth=ColorDepth.DEPTH_8_BIT,
            )
            self._progress_bar.__enter__()
        return self._progress_bar

    def start_progress(self, label: str, *, total: int, unit: str) -> ProgressTask:
        task = ProgressTask(label, total, unit)
        task.native = self._ensure_progress_bar()(label=f"{label} ({unit})", total=total, remove_when_done=True)
        return task

    def advance_progress(self, task: ProgressTask, amount: int = 1) -> None:
        remaining = min(amount, task.total - task.completed)
        task.completed += remaining
        if task.native is not None:
            for _ in range(remaining):
                task.native.item_completed()

    def finish_progress(self, task: ProgressTask, *, success: bool = True) -> None:
        if success:
            self.advance_progress(task, task.total - task.completed)
        if task.native is not None and not task.native.done:
            task.native.done = success
            task.native.stopped = True

    def _help(self) -> None:
        self.log(
            "Commands: /show [JSON pointer], /issues, /undo, /edit, /save, /help, /quit. "
            "Enter submits; Shift+Enter inserts a newline; Tab opens completion."
        )

    def _edit(self, state: Any) -> None:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        descriptor, name = tempfile.mkstemp(suffix=".json", prefix="lamet-plan-")
        path = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(state.candidate, indent=2, ensure_ascii=False) + "\n")
            subprocess.run([*shlex.split(editor), str(path)], check=True)
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("edited manifest root must be an object")
            state.replace_candidate(document, note="user external editor")
            self.log(f"Edited manifest loaded; {len(state.issues)} validation issues remain.")
        except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
            self.log(f"Editor changes were not applied: {exc}", level="error")
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _command(self, text: str, state: Any) -> bool:
        command, _, argument = text.partition(" ")
        if command == "/show":
            self.log(json.dumps(state.manifest_view(argument.strip()), indent=2, ensure_ascii=False))
        elif command == "/issues":
            if not state.issues:
                self.log("No validation issues remain.")
            for issue in state.issues:
                self.log(f"- {issue.path}: {issue.message}\n  {issue.physics}")
        elif command == "/undo":
            self.log(
                f"Previous manifest update undone; {len(state.issues)} validation issues remain."
                if state.undo()
                else "Nothing to undo."
            )
        elif command == "/edit":
            self._edit(state)
        elif command == "/save":
            self.log(f"Draft saved to {state.save()} with {len(state.issues)} validation issues remaining.")
        elif command == "/help":
            self._help()
        elif command == "/quit":
            raise UiCancelled("planning cancelled by user")
        else:
            return False
        return True

    def ask(self, question: str, state: Any | None = None) -> str:
        self.log(f"\n{'Planner' if state is not None else 'Agent'}: {question}")
        self.completer.state = state
        while True:
            try:
                answer = self.session.prompt(
                    HTML("<b>You</b>&gt; "),
                    multiline=True,
                    key_bindings=_CONVERSATION_KEY_BINDINGS,
                    prompt_continuation="... ",
                    bottom_toolbar=(
                        " Enter submit | Shift+Enter newline | Tab complete | Ctrl+C cancel | /help commands "
                    ),
                ).strip()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            if not answer:
                continue
            if state is not None and answer.startswith("/") and self._command(answer, state):
                continue
            return answer

    def confirm(self, question: str) -> bool:
        while True:
            try:
                answer = self.session.prompt(f"{question} [y/N] ").strip().lower()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            if answer in {"y", "yes"}:
                return True
            if answer in {"", "n", "no"}:
                return False
            self.log("Please answer yes or no.")

    def review_plan(self, question: str, state: Any) -> bool | str | None:
        self.completer.state = state
        try:
            answer = self.session.prompt(
                f"{question} [y/N or describe a revision] ",
                multiline=True,
                key_bindings=_CONVERSATION_KEY_BINDINGS,
                prompt_continuation="... ",
                bottom_toolbar=" Enter submit | Shift+Enter newline | Ctrl+C cancel ",
            ).strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc
        if answer.lower() in {"y", "yes"}:
            return True
        if answer.lower() in {"", "n", "no"}:
            return None
        return answer

    def close(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.__exit__(None, None, None)
            self._progress_bar = None
        if self._stdout_context is not None:
            self._stdout_context.__exit__(None, None, None)
            self._stdout_context = None


_ACTIVE_UI: ContextVar[PlainUi | None] = ContextVar("lamet_agent_ui", default=None)
_FALLBACK_UI = PlainUi()


def current_ui() -> PlainUi:
    return _ACTIVE_UI.get() or _FALLBACK_UI


@contextmanager
def use_ui(ui: PlainUi) -> Iterator[None]:
    token = _ACTIVE_UI.set(ui)
    try:
        yield
    finally:
        _ACTIVE_UI.reset(token)


def create_ui(*, interactive: bool | None = None) -> PlainUi:
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    return TerminalUi() if interactive else PlainUi()


def log(message: str = "", *, level: str = "info", style: str | None = None) -> None:
    if style is None and message.startswith("Running"):
        style = "running"
    current_ui().log(message, level=level, style=style)


def warning(message: str) -> None:
    current_ui().warning(message)


def track(iterable: Iterable[Any], *, label: str, unit: str, enabled: bool = True):
    """Yield an iterable while emitting renderer-neutral progress events."""
    if not enabled:
        yield from iterable
        return
    values = iterable if hasattr(iterable, "__len__") else list(iterable)
    task = current_ui().start_progress(label, total=len(values), unit=unit)
    success = False
    try:
        for value in values:
            yield value
            current_ui().advance_progress(task)
        success = True
    finally:
        current_ui().finish_progress(task, success=success)


__all__ = [
    "PlainUi",
    "ProgressTask",
    "TerminalUi",
    "UiCancelled",
    "create_ui",
    "current_ui",
    "log",
    "track",
    "use_ui",
    "warning",
]
