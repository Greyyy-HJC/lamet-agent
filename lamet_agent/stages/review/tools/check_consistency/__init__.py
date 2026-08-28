"""Audit provenance along the manifest job graph."""

from __future__ import annotations

import json

from lamet_agent.agent import ToolContext


def run(context: ToolContext) -> dict[str, object]:
    """Compare only metadata that adjacent stages are expected to preserve."""
    summaries = context.state["result_summary"]
    bundle = context.state["review_bundle"]
    checks = set(context.params["checks"])
    by_job = {item["job_id"]: item for item in summaries if item["job_id"] is not None}
    jobs = bundle["jobs"]
    findings: list[dict[str, object]] = []

    for consumer_id, consumer in by_job.items():
        stack = list(jobs[consumer_id]["inputs"].items())
        sources = []
        while stack:
            source_role, value = stack.pop(0)
            if isinstance(value, str):
                if value in jobs:
                    sources.append((source_role, value))
            elif isinstance(value, list):
                stack.extend((source_role, item) for item in value)
            elif isinstance(value, dict):
                stack.extend((source_role, item) for item in value.values())

        for source_role, source_id in sources:
            source_stage = jobs[source_id]["stage_id"]
            consumer_stage = jobs[consumer_id]["stage_id"]
            source = by_job.get(source_id)
            if source is None:
                findings.append(
                    {
                        "status": "not_checkable",
                        "group": "provenance",
                        "source_job": source_id,
                        "consumer_job": consumer_id,
                        "field": "output",
                        "message": "source output was not selected for Review inspection",
                    }
                )
                continue
            left = dict(source.get("attrs", {}))
            right = dict(consumer.get("attrs", {}))
            if source.get("ensemble") is not None:
                left["ensemble"] = source["ensemble"]
            if consumer.get("ensemble") is not None:
                right["ensemble"] = consumer["ensemble"]
            if source.get("resample") is not None:
                left["resample"] = source["resample"]
            if consumer.get("resample") is not None:
                right["resample"] = consumer["resample"]
            if source.get("n_sample") is not None:
                left["n_sample"] = source["n_sample"]
            if consumer.get("n_sample") is not None:
                right["n_sample"] = consumer["n_sample"]
            groups = {
                "identity": ("hadron", "kernel_operator", "parton", "target_observable", "polarization", "gfix"),
                "units": ("coord_unit",),
                "kinematics": ("momentum_gev", "initial_momentum_gev", "final_momentum_gev", "t_gev2", "ensemble"),
                "schemes": (
                    ("renormalization_scheme", "kernel_id")
                    if source_stage == "perturbative_matching"
                    else ("renormalization_scheme",)
                ),
                "resampling": ("resample_id", "resample", "n_sample"),
            }
            for group, fields in groups.items():
                if group not in checks:
                    continue
                if group == "kinematics" and consumer_stage == "renormalization" and source_role != "target":
                    if left.get("ensemble") != right.get("ensemble") and "ensemble" in left and "ensemble" in right:
                        findings.append(
                            {
                                "status": "warning",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": "ensemble",
                                "source": left["ensemble"],
                                "consumer": right["ensemble"],
                                "message": "ensemble changed across the renormalization denominator edge",
                            }
                        )
                    continue
                if group == "units" and source_stage == "correlator_analysis" and consumer_stage == "renormalization":
                    if right.get("coord_unit") != "fm":
                        findings.append(
                            {
                                "status": "error",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": "coord_unit",
                                "source": left.get("coord_unit"),
                                "consumer": right.get("coord_unit"),
                                "message": "renormalization output must use fm coordinates",
                            }
                        )
                    continue
                if group in {"kinematics", "resampling"} and consumer_stage == "extrapolation":
                    continue
                compared = 0
                for field in fields:
                    if field not in left or field not in right:
                        continue
                    compared += 1
                    if left[field] != right[field]:
                        findings.append(
                            {
                                "status": "error" if group == "identity" else "warning",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": field,
                                "source": left[field],
                                "consumer": right[field],
                                "message": f"{field} changed across {source_stage} -> {consumer_stage}",
                            }
                        )
                if compared == 0:
                    findings.append(
                        {
                            "status": "not_checkable",
                            "group": group,
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": None,
                            "message": f"no shared {group} metadata across {source_stage} -> {consumer_stage}",
                        }
                    )
            if "grids" in checks and (source_stage, consumer_stage) in {
                ("fourier_transform", "perturbative_matching"),
                ("perturbative_matching", "extrapolation"),
            }:
                if source.get("dims") is None or consumer.get("dims") is None:
                    findings.append(
                        {
                            "status": "not_checkable",
                            "group": "grids",
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": "dims",
                            "message": "one output has no inspectable grid",
                        }
                    )
                elif source.get("dims") != consumer.get("dims") or source.get("coords") != consumer.get("coords"):
                    findings.append(
                        {
                            "status": "warning",
                            "group": "grids",
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": "dims/coords",
                            "source": {"dims": source.get("dims"), "coords": source.get("coords")},
                            "consumer": {"dims": consumer.get("dims"), "coords": consumer.get("coords")},
                            "message": f"distribution grid changed across {source_stage} -> {consumer_stage}",
                        }
                    )

    if "extrapolation" in checks:
        extrapolated = [item for item in summaries if item.get("stage_id") == "extrapolation"]
        for item in extrapolated:
            attrs = item.get("attrs", {})
            if "physical_pion_mass_gev" not in attrs and "physical_point" not in attrs:
                findings.append(
                    {
                        "status": "not_checkable",
                        "group": "extrapolation",
                        "source_job": None,
                        "consumer_job": item["job_id"],
                        "field": "physical_point",
                        "message": "extrapolation output does not identify its physical point",
                    }
                )

    document = {
        "checks": list(context.params["checks"]),
        "findings": findings,
        "counts": {
            status: sum(finding["status"] == status for finding in findings)
            for status in ("error", "warning", "info", "not_checkable")
        },
    }
    (context.artifact_directory / "consistency.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    context.state["consistency"] = document
    return {
        "summary": f"recorded {len(findings)} stage-aware consistency findings",
        "metrics": document["counts"],
        "findings": findings,
        "state_keys": ["consistency"],
        "artifacts": ["consistency.json"],
    }
