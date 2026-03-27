"""Correlator data loaders for built-in CSV and NPZ formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.errors import ManifestValidationError
from lamet_agent.schemas import CorrelatorSpec
from lamet_agent.utils import resolve_manifest_relative_path


@dataclass(slots=True)
class CorrelatorDataset:
    """In-memory representation of a correlator input."""

    kind: str
    label: str
    path: Path
    axis: np.ndarray
    values: np.ndarray
    samples: np.ndarray | None
    metadata: dict[str, Any]


def load_correlator_dataset(spec: CorrelatorSpec, manifest_path: Path) -> CorrelatorDataset:
    """Load one correlator dataset from its declared file."""
    resolved_path = resolve_manifest_relative_path(manifest_path, spec.path)
    samples: np.ndarray | None = None
    if spec.file_format == "csv":
        raw = np.loadtxt(resolved_path, delimiter=",", comments="#")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 2:
            raise ManifestValidationError(
                f"CSV correlator inputs must contain at least two columns, got shape {raw.shape}."
            )
        axis = raw[:, 0]
        if raw.shape[1] == 2:
            values = raw[:, 1]
        else:
            samples = np.asarray(raw[:, 1:], dtype=float)
            values = np.mean(samples, axis=1)
    elif spec.file_format == "npz":
        with np.load(resolved_path) as data:
            if "x" not in data or "y" not in data:
                raise ManifestValidationError(
                    f"NPZ correlator inputs must define 'x' and 'y' arrays: {resolved_path}."
                )
            axis = np.asarray(data["x"])
            values = np.asarray(data["y"])
            if "samples" in data:
                samples = np.asarray(data["samples"])
                if samples.ndim != 2:
                    raise ManifestValidationError(
                        f"NPZ correlator samples must be 2D, got shape {samples.shape}: {resolved_path}."
                    )
                if samples.shape[0] != len(axis) and samples.shape[1] == len(axis):
                    samples = samples.T
                if samples.shape[0] != len(axis):
                    raise ManifestValidationError(
                        "NPZ correlator samples must have one axis matching the correlator time axis."
                    )
    else:
        raise ManifestValidationError(f"Unsupported correlator file format: {spec.file_format}.")
    metadata = dict(spec.metadata)
    if samples is not None:
        metadata.setdefault("configuration_count", int(samples.shape[1]))
    return CorrelatorDataset(
        kind=spec.kind,
        label=spec.label,
        path=resolved_path,
        axis=np.asarray(axis, dtype=float),
        values=np.asarray(values, dtype=float),
        samples=samples,
        metadata=metadata,
    )


def load_all_correlators(correlators: list[CorrelatorSpec], manifest_path: Path) -> dict[str, CorrelatorDataset]:
    """Load all correlators and return them keyed by label."""
    loaded: dict[str, CorrelatorDataset] = {}
    for correlator in correlators:
        dataset = load_correlator_dataset(correlator, manifest_path)
        loaded[dataset.label] = dataset
    return loaded
