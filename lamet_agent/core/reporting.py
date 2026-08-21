"""Shared formatting and path helpers for stage Markdown reports."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def translate_markdown_report(
    markdown: str,
    *,
    backend: str,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Translate one English Markdown report to Simplified Chinese with an LLM."""
    from lamet_agent.core.llm import request_llm_text

    translated = request_llm_text(
        backend=backend,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the supplied English LaMET analysis Markdown into Simplified Chinese. "
                    "Translation using physics biased language (especially lattice QCD and large momentum effective theory preference)."
                    "Preserve Markdown structure, tables, image links, file paths, code spans, code blocks, "
                    "identifiers, JSON keys, numbers, units, and all LaTeX math exactly. "
                    "Return only the translated Markdown."
                ),
            },
            {"role": "user", "content": markdown},
        ],
    ).strip()
    if translated.startswith("```"):
        lines = translated.splitlines()
        if lines and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            translated = "\n".join(lines[1:-1]).strip()
    return translated


def format_report_value(value: Any, digits: int = 4) -> str:
    """Format one scalar value for a compact Markdown report."""
    if value is None:
        return "not set"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def format_report_list(values: Any, *, max_items: int = 8, digits: int = 4) -> str:
    """Format a flattened array preview for a compact Markdown report."""
    array = np.asarray(values)
    if array.size == 0:
        return "[]"
    flat = array.reshape(-1)
    items = [format_report_value(item, digits=digits) for item in flat[:max_items]]
    suffix = ", ..." if flat.size > max_items else ""
    return "[" + ", ".join(items) + suffix + "]"


def resolve_report_target(path: Path, report_language: str) -> tuple[Path, str]:
    """Return the language-specific report path and internal language code."""
    language = report_language.lower()
    if language == "en":
        return path, "en"
    if language == "ch":
        target = path.with_name(f"{path.stem}_CN{path.suffix or '.md'}")
        return target, "ch"
    raise ValueError("report_language must be 'en' or 'ch'")


def markdown_artifact_paths(
    artifacts: dict[str, Any] | None,
    *,
    base_dir: Path,
    path_keys: Iterable[str],
    list_path_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Make selected scalar and list artifact paths relative to a report directory."""
    output = dict(artifacts or {})
    for key in path_keys:
        if key not in output:
            continue
        value = output[key]
        if not value:
            output[key] = None
            continue
        path = Path(str(value))
        output[key] = (
            Path(os.path.relpath(path, base_dir)).as_posix()
            if path.is_absolute()
            else str(value)
        )
    for key in list_path_keys:
        values = output.get(key)
        if values is None:
            continue
        relative_paths: list[str] = []
        for value in values:
            if not value:
                continue
            path = Path(str(value))
            relative_paths.append(
                Path(os.path.relpath(path, base_dir)).as_posix()
                if path.is_absolute()
                else str(value)
            )
        output[key] = relative_paths
    return output
