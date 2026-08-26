"""Compile Fourier tail-window systematics into concrete jobs."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np


_SAFE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _merge(defaults: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(defaults))
    for key, value in params.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append one Fourier clone per declared lattice-step offset."""
    if set(config) != {"tail_window_variants"}:
        raise ValueError("fourier_transform systematics keys must be exactly ['tail_window_variants']")
    variants = config["tail_window_variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("fourier_transform.tail_window_variants must be nonempty")
    parsed: list[tuple[str, int]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping) or set(variant) != {"id", "step_offset"}:
            raise ValueError(f"fourier_transform.tail_window_variants[{index}] must contain id and step_offset")
        label = variant["id"]
        offset = variant["step_offset"]
        if not isinstance(label, str) or not _SAFE_LABEL.fullmatch(label):
            raise ValueError("Fourier systematics ids must match [a-z][a-z0-9_]*")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset == 0:
            raise ValueError("Fourier systematics step_offset must be a nonzero integer")
        parsed.append((label, offset))
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("Fourier systematics ids must be unique")
    if len({offset for _, offset in parsed}) != len(parsed):
        raise ValueError("Fourier systematics step offsets must be unique")

    block = document["stages"]["fourier_transform"]
    defaults = block.get("defaults", {})
    central = list(block["jobs"])
    suffixes = tuple(f"_{label}" for label, _ in parsed)
    if any(str(job.get("id", "")).endswith(suffixes) for job in central):
        raise ValueError("Fourier systematics cannot be combined with explicitly authored variation jobs")
    known_ids = {job["id"] for stage in document["stages"].values() for job in stage["jobs"]}
    generated: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {}
    for job in central:
        effective = _merge(
            defaults,
            {key: value for key, value in job.items() if key not in {"id", "inputs"}},
        )
        zmin = effective.get("zmin_fm")
        if (
            not isinstance(zmin, list)
            or len(zmin) < 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
                for value in zmin
            )
        ):
            raise ValueError(
                f"Fourier job '{job['id']}' needs at least two finite zmin_fm values to derive one lattice step"
            )
        differences = np.diff(np.asarray(zmin, dtype=float))
        spacing = float(differences[0])
        if spacing <= 0 or not np.allclose(differences, spacing, rtol=0.0, atol=1e-12):
            raise ValueError(f"Fourier job '{job['id']}' zmin_fm must be a strictly increasing uniform lattice grid")
        mapping[job["id"]] = {}
        for label, offset in parsed:
            job_id = f"{job['id']}_{label}"
            if job_id in known_ids:
                raise ValueError(f"generated Fourier job id collides with '{job_id}'")
            shifted = [round(float(value) + offset * spacing, 12) for value in zmin]
            if min(shifted) < 0:
                raise ValueError(f"Fourier variation '{job_id}' shifts zmin_fm below zero")
            clone = copy.deepcopy(job)
            clone["id"] = job_id
            clone["zmin_fm"] = shifted
            generated.append(clone)
            mapping[job["id"]][label] = job_id
            known_ids.add(job_id)
    block["jobs"] = central + generated
    state["fourier_variants"] = mapping
    state["fourier_variant_labels"] = [label for label, _ in parsed]
