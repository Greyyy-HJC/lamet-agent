"""Compile Fourier tail-window systematics into concrete jobs."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lamet_agent.data import lattice_spacing_from_path

_PREFERRED_INPUT_ROLES = ("input", "target", "quasi", "correlators")
_SKIPPED_INPUT_ROLES = {"zR", "denominator", "reference"}


def _merge(defaults: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(defaults))
    for key, value in params.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _job_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs: dict[str, Mapping[str, Any]] = {}
    for block in document["stages"].values():
        if not isinstance(block, Mapping) or not isinstance(block.get("jobs"), list):
            continue
        for job in block["jobs"]:
            if isinstance(job, Mapping) and isinstance(job.get("id"), str):
                jobs[job["id"]] = job
    return jobs


def _upstream_sources(job: Mapping[str, Any]) -> list[Any]:
    inputs = job.get("inputs")
    if not isinstance(inputs, Mapping):
        return []
    preferred = [inputs[role] for role in _PREFERRED_INPUT_ROLES if role in inputs]
    if preferred:
        return preferred
    return [value for role, value in inputs.items() if role not in _SKIPPED_INPUT_ROLES]


def _lattice_spacing(
    source: Any,
    *,
    jobs: Mapping[str, Mapping[str, Any]],
    root: Path,
    seen: set[str],
) -> float | None:
    if isinstance(source, list):
        for item in source:
            spacing = _lattice_spacing(item, jobs=jobs, root=root, seen=seen)
            if spacing is not None:
                return spacing
        return None
    if isinstance(source, Mapping) and set(source) == {"file"} and isinstance(source.get("file"), str):
        path = Path(source["file"]).expanduser()
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        return lattice_spacing_from_path(resolved)
    if isinstance(source, str):
        if source in seen:
            return None
        job = jobs.get(source)
        if job is None:
            return None
        seen.add(source)
        for upstream in _upstream_sources(job):
            spacing = _lattice_spacing(upstream, jobs=jobs, root=root, seen=seen)
            if spacing is not None:
                return spacing
    return None


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append one Fourier clone per declared lattice-step offset."""
    parsed = [(str(variant["id"]), int(variant["tail_window_step_offset"])) for variant in config["variants"]]
    if not parsed:
        return

    root = state.get("root_directory")
    if not isinstance(root, Path):
        raise ValueError("Fourier systematics need a resolved root_directory to read lattice spacing")
    jobs = _job_index(document)
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
            or not zmin
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
                for value in zmin
            )
        ):
            raise ValueError(f"Fourier job '{job['id']}' systematics require authored finite zmin_fm candidates")
        spacing = _lattice_spacing(job.get("inputs", {}).get("input"), jobs=jobs, root=root, seen=set())
        if spacing is None:
            raise ValueError(
                f"Fourier job '{job['id']}' needs lattice spacing from input .nc attrs "
                "or an upstream correlator ensemble"
            )
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
