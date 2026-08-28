"""Render and publish the final evidence-grounded Review."""

from __future__ import annotations

import json

from lamet_agent.agent import ToolContext


def run(
    context: ToolContext,
    *,
    title: str,
    workflow_summary: str,
    physical_analysis: str,
    systematics_and_limitations: str,
    literature_comparison: str,
    conclusion: str,
) -> dict[str, object]:
    """Combine deterministic evidence with bounded LLM-authored interpretation."""
    bundle = context.state["review_bundle"]
    consistency = context.state["consistency"]
    papers = context.state["selected_papers"]
    chinese = context.params["report_language"] == "ch"
    headings = (
        {
            "scope": "范围与溯源",
            "workflow": "工作流总结",
            "coverage": "数据与参数覆盖",
            "consistency": "一致性检查",
            "physical": "物理解读",
            "systematics": "系统误差与局限",
            "literature": "文献比较",
            "conclusion": "结论",
            "artifacts": "上游报告与产物",
            "references": "参考文献",
        }
        if chinese
        else {
            "scope": "Scope and provenance",
            "workflow": "Workflow summary",
            "coverage": "Data and parameter coverage",
            "consistency": "Consistency findings",
            "physical": "Physical interpretation",
            "systematics": "Systematic uncertainties and limitations",
            "literature": "Literature comparison",
            "conclusion": "Conclusions",
            "artifacts": "Upstream reports and artifacts",
            "references": "References",
        }
    )
    lines = [
        f"# {title}",
        "",
        f"## {headings['scope']}",
        "",
        (
            f"本综述检查了 `{bundle['run_id']}` 中显式选择的 {len(bundle['results'])} 个结果，"
            f"并纳入 Review 之前的 {len(bundle['stage_reports'])} 份阶段报告。"
            if chinese
            else f"This review inspected {len(bundle['results'])} explicitly selected results from "
            f"`{bundle['run_id']}` and included {len(bundle['stage_reports'])} preceding stage reports."
        ),
        "",
        f"## {headings['workflow']}",
        "",
        workflow_summary.strip(),
        "",
        f"## {headings['coverage']}",
        "",
        "| Stage/job | Dimensions | Samples | Physical metadata |",
        "| --- | --- | ---: | --- |",
    ]
    for item in bundle["results"]:
        attrs = item.get("attrs", {})
        metadata = {"ensemble": item["ensemble"]} if item.get("ensemble") is not None else {}
        metadata.update(
            {
                key: attrs[key]
                for key in (
                    "hadron",
                    "parton",
                    "target_observable",
                    "polarization",
                    "momentum_gev",
                    "lattice_spacing_fm",
                    "renormalization_scheme",
                    "kernel_id",
                )
                if key in attrs
            }
        )
        lines.append(
            f"| `{item.get('stage_id')}/{item.get('job_id') or 'external'}` | "
            f"`{json.dumps(item.get('dims'), ensure_ascii=False)}` | {item.get('n_sample', 'n/a')} | "
            f"`{json.dumps(metadata, sort_keys=True, ensure_ascii=False)}` |"
        )
    lines.extend(
        [
            "",
            f"## {headings['consistency']}",
            "",
        ]
    )
    lines.extend(
        [
            f"- **{finding['status']}** `{finding['source_job'] or '-'} -> {finding['consumer_job']}` "
            f"(`{finding['group']}{'/' + str(finding['field']) if finding['field'] else ''}`): "
            f"{finding['message']}"
            for finding in consistency["findings"]
        ]
        or ["- 未发现确定性一致性问题。" if chinese else "- No deterministic consistency findings were recorded."]
    )
    lines.extend(
        [
            "",
            f"## {headings['physical']}",
            "",
            physical_analysis.strip(),
            "",
            f"## {headings['systematics']}",
            "",
            systematics_and_limitations.strip(),
            "",
            f"## {headings['literature']}",
            "",
            literature_comparison.strip(),
            "",
            f"## {headings['conclusion']}",
            "",
            conclusion.strip(),
            "",
            f"## {headings['artifacts']}",
            "",
        ]
    )
    lines.extend(
        f"- `{item['stage_id']}`: [{item['path']}]({item['path']})"
        if item["available"]
        else f"- `{item['stage_id']}`: not available"
        for item in bundle["stage_reports"]
    )
    lines.extend(
        f"- `{item['stage_id']}/{item['job_id']}`: [{item['path']}]({item['path']})"
        for item in bundle["job_reports"]
        if item["available"]
    )
    lines.extend(["", f"## {headings['references']}", ""])
    lines.extend(
        [
            f"- {', '.join(paper['authors']) + '. ' if paper['authors'] else ''}"
            f"[{paper['title']}]({paper['source']}), arXiv:{paper['id']} "
            f"([ar5iv text]({paper['ar5iv_url']}), `{paper['retrieval']}`)."
            for paper in papers
        ]
        or ["- 未选择文献。" if chinese else "- No literature was selected."]
    )
    report = "\n".join(lines).rstrip() + "\n"
    (context.artifact_directory / "review.md").write_text(report, encoding="utf-8")
    artifacts = [
        "review.md",
        "review_bundle.json",
        "consistency.json",
        "literature_selection.json",
        *context.state["literature_artifacts"],
    ]
    diagnostics = {
        "result_count": len(bundle["results"]),
        "stage_report_count": len(bundle["stage_reports"]),
        "finding_counts": consistency["counts"],
        "selected_paper_ids": [paper["id"] for paper in papers],
    }
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "review",
        "decisions": {"title": title, "report_language": context.params["report_language"]},
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    context.finish(report, summary)
    return {"summary": "published review.md", "metrics": diagnostics, "state_keys": [], "artifacts": artifacts}
