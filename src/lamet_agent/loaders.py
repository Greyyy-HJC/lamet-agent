"""Correlator data loaders for built-in CSV, NPZ, and TXT formats."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    extra_axes: dict[str, np.ndarray] = field(default_factory=dict)


def load_correlator_dataset(spec: CorrelatorSpec, manifest_path: Path) -> CorrelatorDataset:
    """Load one correlator dataset from its declared file."""
    resolved_path = resolve_manifest_relative_path(manifest_path, spec.path)
    samples: np.ndarray | None = None
    extra_axes: dict[str, np.ndarray] = {}
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
    elif spec.file_format == "txt":
        raw = np.loadtxt(resolved_path, comments="#")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if spec.kind == "three_point":
            if raw.shape[1] < 4:
                raise ManifestValidationError(
                    f"TXT three-point correlator inputs must contain at least tsep, tau, and one complex sample: {resolved_path}."
                )
            payload_columns = raw.shape[1] - 2
            if payload_columns % 2 != 0:
                raise ManifestValidationError(
                    "TXT three-point correlator inputs must have an even number of sample columns after tsep/tau."
                )
            configuration_count = payload_columns // 2
            tsep_values = np.asarray(raw[:, 0], dtype=int)
            tau_values = np.asarray(raw[:, 1], dtype=int)
            unique_tsep = np.unique(tsep_values)
            unique_tau = np.unique(tau_values)
            complex_rows = raw[:, 2 : 2 + configuration_count] + 1j * raw[:, 2 + configuration_count :]

            samples = np.full((len(unique_tsep), len(unique_tau), configuration_count), np.nan + 0j, dtype=complex)
            values = np.full((len(unique_tsep), len(unique_tau)), np.nan + 0j, dtype=complex)
            tsep_index = {value: index for index, value in enumerate(unique_tsep)}
            tau_index = {value: index for index, value in enumerate(unique_tau)}
            for row_index, (tsep, tau) in enumerate(zip(tsep_values, tau_values, strict=True)):
                i = tsep_index[int(tsep)]
                j = tau_index[int(tau)]
                samples[i, j] = complex_rows[row_index]
                values[i, j] = np.mean(complex_rows[row_index])
            if np.isnan(np.real(samples)).any():
                raise ManifestValidationError(
                    f"TXT three-point correlator input is missing some (tsep, tau) rows: {resolved_path}."
                )
            axis = np.asarray(unique_tsep, dtype=float)
            extra_axes["tau"] = np.asarray(unique_tau, dtype=float)
        else:
            if raw.shape[1] < 2:
                raise ManifestValidationError(
                    f"TXT correlator inputs must contain at least two columns, got shape {raw.shape}."
                )
            axis = raw[:, 0]
            if raw.shape[1] == 2:
                values = raw[:, 1]
            else:
                samples = np.asarray(raw[:, 1:], dtype=float)
                values = np.mean(samples, axis=1)
    else:
        raise ManifestValidationError(f"Unsupported correlator file format: {spec.file_format}.")
    metadata = dict(spec.metadata)
    if samples is not None:
        metadata.setdefault("configuration_count", int(samples.shape[-1]))
    if extra_axes:
        metadata.setdefault("extra_axes", list(extra_axes))
    return CorrelatorDataset(
        kind=spec.kind,
        label=spec.label,
        path=resolved_path,
        axis=np.asarray(axis, dtype=float),
        values=np.asarray(values),
        samples=samples,
        extra_axes=extra_axes,
        metadata=metadata,
    )


def load_all_correlators(correlators: list[CorrelatorSpec], manifest_path: Path) -> dict[str, CorrelatorDataset]:
    """Load all correlators and return them keyed by label."""
    loaded: dict[str, CorrelatorDataset] = {}
    for correlator in correlators:
        dataset = load_correlator_dataset(correlator, manifest_path)
        loaded[dataset.label] = dataset
    return loaded
