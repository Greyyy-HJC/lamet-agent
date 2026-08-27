"""Compile propagated Fourier and matching-scale systematics."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _merge(defaults: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(defaults))
    for key, value in params.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append Fourier-propagated and scale-varied matching jobs."""
    scale_variants = [(str(variant["id"]), float(variant["mu_factor"])) for variant in config["variants"]]

    fourier_mapping = state.get("fourier_variants", {})
    fourier_labels = state.get("fourier_variant_labels", [])
    propagate = bool(fourier_mapping and fourier_labels)
    if not propagate and not scale_variants:
        return
    labels = [*fourier_labels, *[label for label, _ in scale_variants]]
    if len(set(labels)) != len(labels):
        raise ValueError("Fourier-propagated and matching scale ids must be disjoint")

    block = document["stages"]["perturbative_matching"]
    defaults = block.get("defaults", {})
    central = list(block["jobs"])
    suffixes = tuple(f"_{label}" for label in labels)
    if suffixes and any(str(job.get("id", "")).endswith(suffixes) for job in central):
        raise ValueError("Matching systematics cannot be combined with explicitly authored variation jobs")
    known_ids = {job["id"] for stage in document["stages"].values() for job in stage["jobs"]}
    generated: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {}
    groups = {
        "lambda_extrapolation": list(fourier_labels),
        "lamet_scale": [label for label, _ in scale_variants],
    }
    for job in central:
        central_fourier_id = job.get("inputs", {}).get("quasi")
        if not isinstance(central_fourier_id, str):
            raise ValueError(f"Matching job '{job['id']}' systematics require one upstream Fourier job input")
        mapping[job["id"]] = {}
        if propagate:
            if central_fourier_id not in fourier_mapping:
                raise ValueError(f"Matching job '{job['id']}' references Fourier job without compiled variants")
            for label in fourier_labels:
                clone = copy.deepcopy(job)
                clone["id"] = f"{job['id']}_{label}"
                clone["inputs"]["quasi"] = fourier_mapping[central_fourier_id][label]
                if clone["id"] in known_ids:
                    raise ValueError(f"generated Matching job id collides with '{clone['id']}'")
                generated.append(clone)
                mapping[job["id"]][label] = clone["id"]
                known_ids.add(clone["id"])
        effective = _merge(
            defaults,
            {key: value for key, value in job.items() if key not in {"id", "inputs"}},
        )
        mu = float(effective["mu"])
        for label, factor in scale_variants:
            clone = copy.deepcopy(job)
            clone["id"] = f"{job['id']}_{label}"
            clone["mu"] = float(mu) * factor
            if clone["id"] in known_ids:
                raise ValueError(f"generated Matching job id collides with '{clone['id']}'")
            generated.append(clone)
            mapping[job["id"]][label] = clone["id"]
            known_ids.add(clone["id"])
    block["jobs"] = central + generated
    state["matching_variants"] = mapping
    state["matching_variant_groups"] = groups
