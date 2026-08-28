"""Compile Fourier tail-window systematics into concrete jobs."""

from __future__ import annotations

import copy
from typing import Any


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append one Fourier clone per declared lattice-step offset."""
    parsed = [(str(variant["id"]), int(variant["tail_window_step_offset"])) for variant in config["variants"]]
    if not parsed:
        return

    block = document["stages"]["fourier_transform"]
    central = list(block["jobs"])
    suffixes = tuple(f"_{label}" for label, _ in parsed)
    if any(str(job.get("id", "")).endswith(suffixes) for job in central):
        raise ValueError("Fourier systematics cannot be combined with explicitly authored variation jobs")
    known_ids = {job["id"] for stage in document["stages"].values() for job in stage["jobs"]}
    generated: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {}
    for job in central:
        mapping[job["id"]] = {}
        for label, offset in parsed:
            job_id = f"{job['id']}_{label}"
            if job_id in known_ids:
                raise ValueError(f"generated Fourier job id collides with '{job_id}'")
            clone = copy.deepcopy(job)
            clone["id"] = job_id
            clone["tail_window_step_offset"] = offset
            generated.append(clone)
            mapping[job["id"]][label] = job_id
            known_ids.add(job_id)
    block["jobs"] = central + generated
    state["fourier_variants"] = mapping
    state["fourier_variant_labels"] = [label for label, _ in parsed]
