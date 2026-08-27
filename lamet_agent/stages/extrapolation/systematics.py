"""Compile extrapolation branches and the final systematics budget."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _variant_job_id(central_id: str, label: str) -> str:
    stem = central_id[:-4] if central_id.endswith("_all") else central_id
    return f"{stem}_{label}"


def _changed_terms(base: list[str], added: list[str], removed: list[str], *, label: str, role: str) -> list[str]:
    missing = [term for term in removed if term not in base]
    if missing:
        raise ValueError(f"Extrapolation variation '{label}' cannot remove absent {role} terms {missing}")
    redundant = [term for term in added if term in base]
    if redundant:
        raise ValueError(f"Extrapolation variation '{label}' cannot add existing {role} terms {redundant}")
    return [term for term in base if term not in removed] + list(added)


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append propagated fits, model variants, and one terminal budget job."""
    variants = config["variants"]

    matching_mapping = state.get("matching_variants", {})
    groups = state.get(
        "matching_variant_groups",
        {"lambda_extrapolation": [], "lamet_scale": []},
    )
    if not variants and not matching_mapping:
        return
    if not isinstance(matching_mapping, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("Extrapolation systematics require compiled Matching variants")
    block = document["stages"]["extrapolation"]
    central_jobs = list(block["jobs"])
    if len(central_jobs) != 1:
        raise ValueError("Extrapolation systematics require exactly one authored central job")
    central = central_jobs[0]
    if central["operation"] != "fit":
        raise ValueError("the authored extrapolation job must be operation='fit'")
    effective = copy.deepcopy(block.get("defaults", {}))
    effective.update({key: value for key, value in central.items() if key not in {"id", "inputs"}})
    central_independent = list(effective.get("x_independent_terms", []))
    central_dependent = list(effective.get("x_dependent_terms", []))
    distributions = central.get("inputs", {}).get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("the authored extrapolation job needs a nonempty distributions list")
    central_matching_ids: list[str] = []
    for source in distributions:
        if not isinstance(source, str):
            raise ValueError("extrapolation systematics require upstream Matching job sources")
        central_matching_ids.append(source)

    labels_by_group = {
        "lambda_extrapolation": list(groups["lambda_extrapolation"]),
        "lamet_scale": list(groups["lamet_scale"]),
    }
    all_labels = [
        *labels_by_group["lambda_extrapolation"],
        *labels_by_group["lamet_scale"],
        *[variant["id"] for variant in variants],
    ]
    suffixes = tuple(f"_{label}" for label in all_labels)
    if suffixes and str(central["id"]).endswith(suffixes):
        raise ValueError("Extrapolation systematics cannot be combined with an authored variation job")
    known_ids = {job["id"] for stage in document["stages"].values() for job in stage["jobs"]}
    generated: list[dict[str, Any]] = []
    generated_by_label: dict[str, str] = {}
    for label in [
        *labels_by_group["lambda_extrapolation"],
        *labels_by_group["lamet_scale"],
    ]:
        clone = copy.deepcopy(central)
        clone["id"] = _variant_job_id(central["id"], label)
        clone["inputs"]["distributions"] = []
        for matching_id in central_matching_ids:
            if matching_id not in matching_mapping or label not in matching_mapping[matching_id]:
                raise ValueError(f"Matching job '{matching_id}' has no compiled '{label}' variant")
            clone["inputs"]["distributions"].append(matching_mapping[matching_id][label])
        if clone["id"] in known_ids:
            raise ValueError(f"generated extrapolation job id collides with '{clone['id']}'")
        generated.append(clone)
        generated_by_label[label] = clone["id"]
        known_ids.add(clone["id"])
    for variant in variants:
        label = variant["id"]
        clone = copy.deepcopy(central)
        clone["id"] = _variant_job_id(central["id"], label)
        clone["x_independent_terms"] = _changed_terms(
            central_independent,
            variant["append_x_independent_terms"],
            variant["remove_x_independent_terms"],
            label=label,
            role="x-independent",
        )
        clone["x_dependent_terms"] = _changed_terms(
            central_dependent,
            variant["append_x_dependent_terms"],
            variant["remove_x_dependent_terms"],
            label=label,
            role="x-dependent",
        )
        if set(clone["x_independent_terms"]) & set(clone["x_dependent_terms"]):
            raise ValueError(f"Extrapolation variation '{label}' produces overlapping term classes")
        if not clone["x_independent_terms"] and not clone["x_dependent_terms"]:
            raise ValueError(f"Extrapolation variation '{label}' removes every extrapolation term")
        if clone["id"] in known_ids:
            raise ValueError(f"generated extrapolation job id collides with '{clone['id']}'")
        generated.append(clone)
        generated_by_label[label] = clone["id"]
        known_ids.add(clone["id"])

    if generated:
        budget_id = "extrapolation_systematics_budget"
        if budget_id in known_ids:
            raise ValueError(f"generated budget job id collides with '{budget_id}'")
        ordered_labels = [
            *labels_by_group["lambda_extrapolation"],
            *labels_by_group["lamet_scale"],
            *[variant["id"] for variant in variants],
        ]
        budget = {
            "id": budget_id,
            "inputs": {
                "distributions": [
                    central["id"],
                    *[generated_by_label[label] for label in ordered_labels],
                ]
            },
            "operation": "systematics_budget",
            "systematics_groups": {
                "main": 0,
                "zs": [],
                "lambda_extrapolation": list(
                    range(
                        1,
                        1 + len(labels_by_group["lambda_extrapolation"]),
                    )
                ),
                "lamet_scale": list(
                    range(
                        1 + len(labels_by_group["lambda_extrapolation"]),
                        1 + len(labels_by_group["lambda_extrapolation"]) + len(labels_by_group["lamet_scale"]),
                    )
                ),
                "other_extrapolations": list(
                    range(
                        1 + len(labels_by_group["lambda_extrapolation"]) + len(labels_by_group["lamet_scale"]),
                        1 + len(ordered_labels),
                    )
                ),
            },
        }
        generated.append(budget)
    block["jobs"] = [central, *generated]
