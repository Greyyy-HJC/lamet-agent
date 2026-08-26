"""Compile propagated Fourier and matching-scale systematics."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any


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
    """Append Fourier-propagated and scale-varied matching jobs."""
    if set(config) != {"propagate_fourier_variants", "scale_variants"}:
        raise ValueError("perturbative_matching systematics keys must be propagate_fourier_variants and scale_variants")
    propagate = config["propagate_fourier_variants"]
    if not isinstance(propagate, bool):
        raise ValueError("propagate_fourier_variants must be boolean")
    raw_variants = config["scale_variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("perturbative_matching.scale_variants must be nonempty")
    scale_variants: list[tuple[str, float]] = []
    for index, variant in enumerate(raw_variants):
        if not isinstance(variant, Mapping) or set(variant) != {"id", "factor"}:
            raise ValueError(f"perturbative_matching.scale_variants[{index}] must contain id and factor")
        label = variant["id"]
        factor = variant["factor"]
        if not isinstance(label, str) or not _SAFE_LABEL.fullmatch(label):
            raise ValueError("Matching systematics ids must match [a-z][a-z0-9_]*")
        if (
            not isinstance(factor, (int, float))
            or isinstance(factor, bool)
            or not math.isfinite(float(factor))
            or float(factor) <= 0
            or math.isclose(float(factor), 1.0, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise ValueError("Matching scale factors must be finite, positive, and not one")
        scale_variants.append((label, float(factor)))
    if len({label for label, _ in scale_variants}) != len(scale_variants):
        raise ValueError("Matching systematics ids must be unique")

    fourier_mapping = state.get("fourier_variants", {})
    fourier_labels = state.get("fourier_variant_labels", [])
    if propagate and (not fourier_mapping or not fourier_labels):
        raise ValueError("Matching requested Fourier propagation but no Fourier variants were compiled")
    labels = ([*fourier_labels] if propagate else []) + [label for label, _ in scale_variants]
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
        "lambda_extrapolation": list(fourier_labels) if propagate else [],
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
        mu = effective.get("mu")
        if not isinstance(mu, (int, float)) or isinstance(mu, bool):
            raise ValueError(f"Matching job '{job['id']}' requires a numeric central mu")
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
