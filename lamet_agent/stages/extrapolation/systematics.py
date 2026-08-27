"""Compile extrapolation branches and the final systematics budget."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


_MODEL_VARIANTS = {
    "a_sym": {
        "x_independent_terms": ["a", "a2"],
        "x_dependent_terms": ["inv_p2", "inv_p4", "ap2"],
    },
    "p_sym": {
        "x_independent_terms": ["a"],
        "x_dependent_terms": ["inv_p2", "ap2"],
    },
    "ap_sym": {
        "x_independent_terms": ["a"],
        "x_dependent_terms": ["inv_p2", "inv_p4", "ap2", "ap4"],
    },
}


def _variant_job_id(central_id: str, label: str) -> str:
    stem = central_id[:-4] if central_id.endswith("_all") else central_id
    return f"{stem}_{label}"


def expand(document: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> None:
    """Append propagated fits, model variants, and one terminal budget job."""
    if set(config) != {"model_variants", "publish_budget"}:
        raise ValueError("extrapolation systematics keys must be model_variants and publish_budget")
    models = config["model_variants"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(name, str) or name not in _MODEL_VARIANTS for name in models)
    ):
        raise ValueError(f"extrapolation.model_variants must use {sorted(_MODEL_VARIANTS)}")
    if len(set(models)) != len(models):
        raise ValueError("extrapolation model variants must be unique")
    publish_budget = config["publish_budget"]
    if not isinstance(publish_budget, bool):
        raise ValueError("extrapolation.publish_budget must be boolean")

    matching_mapping = state.get("matching_variants")
    groups = state.get("matching_variant_groups")
    if not isinstance(matching_mapping, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("Extrapolation systematics require compiled Matching variants")
    block = document["stages"]["extrapolation"]
    central_jobs = list(block["jobs"])
    if len(central_jobs) != 1:
        raise ValueError("Extrapolation systematics require exactly one authored central job")
    central = central_jobs[0]
    if central.get("operation", block.get("defaults", {}).get("operation", "fit")) != "fit":
        raise ValueError("the authored extrapolation job must be operation='fit'")
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
        *models,
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
    for label in models:
        clone = copy.deepcopy(central)
        clone["id"] = _variant_job_id(central["id"], label)
        clone.update(copy.deepcopy(_MODEL_VARIANTS[label]))
        if clone["id"] in known_ids:
            raise ValueError(f"generated extrapolation job id collides with '{clone['id']}'")
        generated.append(clone)
        generated_by_label[label] = clone["id"]
        known_ids.add(clone["id"])

    if publish_budget:
        budget_id = "extrapolation_systematics_budget"
        if budget_id in known_ids:
            raise ValueError(f"generated budget job id collides with '{budget_id}'")
        ordered_labels = [
            *labels_by_group["lambda_extrapolation"],
            *labels_by_group["lamet_scale"],
            *models,
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
