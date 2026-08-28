"""Rank the local LaMET literature index against the inspected run."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.review.tools._catalog import catalog_path, load_catalog


def run(
    context: ToolContext,
    *,
    stages: list[str] | None = None,
    hadrons: list[str] | None = None,
    currents: list[str] | None = None,
    observables: list[str] | None = None,
    methods: list[str] | None = None,
) -> dict[str, object]:
    """Select compact candidates from the canonical literature index."""
    if "consistency" not in context.state:
        raise RuntimeError("check_consistency must run before list_literature")
    records = load_catalog(context)
    path = catalog_path(context)
    filters = {
        "stages": stages,
        "hadrons": hadrons,
        "currents": currents,
        "observables": observables,
        "methods": methods,
    }
    explicit_filter = any(values is not None for values in filters.values())

    target_metadata = {
        "observables": {str(context.manifest["metadata"]["target_observable"]).lower()},
        "partons": {str(context.manifest["metadata"]["parton"]).lower()},
        "hadrons": set(),
        "polarizations": set(),
    }

    wanted = {
        f"target_observable={context.manifest['metadata']['target_observable']}",
        f"parton={context.manifest['metadata']['parton']}",
        "method=lamet",
        "method=quasi_distribution",
    }
    wanted.update(f"stage={stage_id}" for stage_id in context.manifest["stages"] if stage_id != "review")
    for item in context.state["result_summary"]:
        attrs = item.get("attrs", {})
        for field, prefix in (
            ("hadron", "hadron"),
            ("polarization", "polarization"),
            ("sector", "quark_sector"),
            ("renormalization_scheme", "scheme"),
        ):
            if attrs.get(field):
                wanted.add(f"{prefix}={str(attrs[field]).lower()}")
                if field == "hadron":
                    target_metadata["hadrons"].add(str(attrs[field]).lower())
                elif field == "polarization":
                    target_metadata["polarizations"].add(str(attrs[field]).lower())
        gfix = str(attrs.get("gfix", "")).lower()
        if gfix:
            gfix = {"cg": "coulomb", "gi": "gauge_invariant"}.get(gfix, gfix)
            wanted.add(f"gfix={gfix}")
        kernel_tokens = str(attrs.get("kernel_id", "")).lower().split("_")
        for order in ("lo", "nlo", "nnlo", "n3lo"):
            if order in kernel_tokens:
                wanted.add(f"matching={order.upper()}")
    for block in context.manifest["stages"].values():
        for job in block["jobs"]:
            if job.get("scheme"):
                wanted.add(f"scheme={str(job['scheme']).lower()}")
            kernel_tokens = str(job.get("kernel_id", "")).lower().split("_")
            for order in ("lo", "nlo", "nnlo", "n3lo"):
                if order in kernel_tokens:
                    wanted.add(f"matching={order.upper()}")

    weights = {
        "target_observable": 12,
        "parton": 9,
        "hadron": 9,
        "polarization": 7,
        "quark_sector": 5,
        "scheme": 5,
        "matching": 5,
        "gfix": 4,
        "stage": 2,
        "method": 1,
    }
    candidates = []
    for record in records:
        if record["relevance"] == "unrelated":
            continue
        tags = {key: {str(value).lower() for value in values} for key, values in record["tags"].items()}
        if any(
            values is not None and not ({str(value).lower() for value in values} & tags.get(key, set()))
            for key, values in filters.items()
        ):
            continue
        record_metadata = {field: tags.get(field, set()) for field in target_metadata}
        if any(
            expected and observed and expected.isdisjoint(observed)
            for field, expected in target_metadata.items()
            if (observed := record_metadata[field])
        ):
            continue
        if (
            target_metadata["observables"] == {"pdf"}
            and record["uses_lattice_data"]
            and {"off_forward", "tmd"} & tags.get("kinematic_dependence", set())
        ):
            continue
        matched = sorted(set(record["review_topics"]) & wanted)
        score = sum(weights.get(topic.split("=", 1)[0], 1) for topic in matched)
        exact_matches = [
            field for field, expected in target_metadata.items() if expected and record_metadata[field] == expected
        ]
        score += sum(
            {"observables": 12, "partons": 9, "hadrons": 9, "polarizations": 7}[field] for field in exact_matches
        )
        if explicit_filter:
            score = max(score, 1)
        if score:
            candidates.append(
                {**record, "score": score, "matched_topics": matched, "exact_metadata_matches": exact_matches}
            )
    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["relevance"] != "core",
            item["confidence"] != "high",
            item["id"],
        )
    )
    candidates = candidates[: 3 * int(context.params["max_papers"])]
    context.state["literature_candidates"] = candidates
    context.state["literature_catalog_directory"] = str(path.parent)
    return {
        "summary": f"ranked {len(candidates)} related papers from {len(records)} indexed records",
        "metrics": {"paper_ids": [record["id"] for record in candidates]},
        "query_topics": sorted(wanted),
        "candidates": [
            {
                key: record[key]
                for key in (
                    "id",
                    "title",
                    "authors",
                    "year",
                    "summary",
                    "source",
                    "score",
                    "matched_topics",
                    "exact_metadata_matches",
                )
            }
            for record in candidates
        ],
        "state_keys": ["literature_candidates", "literature_catalog_directory"],
        "artifacts": [],
    }
