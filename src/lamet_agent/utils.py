"""Shared utility helpers for paths, serialization, and timestamps."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with deterministic formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    """Write a markdown file from a list of lines."""
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv_columns(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write aligned NumPy columns to CSV."""
    headers = list(columns.keys())
    arrays = [np.asarray(columns[name]) for name in headers]
    length = len(arrays[0]) if arrays else 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(length):
            row = []
            for array in arrays:
                value = array[index]
                if np.iscomplexobj(value):
                    row.append(f"{value.real}+{value.imag}j")
                else:
                    row.append(value)
            writer.writerow(row)


def write_columnar_data(base_path: Path, columns: dict[str, np.ndarray], formats: list[str]) -> list[Path]:
    """Export aligned columnar data into one or more requested formats."""
    written_paths: list[Path] = []
    for data_format in formats:
        path = base_path.with_suffix(f".{data_format}")
        if data_format == "csv":
            write_csv_columns(path, columns)
        elif data_format == "npz":
            np.savez(path, **columns)
        elif data_format == "json":
            payload = {key: np.asarray(value).tolist() for key, value in columns.items()}
            write_json(path, payload)
        else:  # pragma: no cover - schema validation should prevent unsupported formats.
            raise ValueError(f"Unsupported data export format: {data_format}")
        written_paths.append(path)
    return written_paths


def resolve_manifest_relative_path(manifest_path: Path, raw_path: str) -> Path:
    """Resolve a data path relative to the manifest file when needed."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()
