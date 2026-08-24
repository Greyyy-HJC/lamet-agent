"""Write the final review report and finish the review job."""

from __future__ import annotations

from lamet_agent.agent import ToolContext


def run(context: ToolContext, *, title: str, analysis: str, conclusion: str) -> dict[str, object]:
    """Write review.md with findings and selected-paper citations."""
    if not context.state.get("result_summary") or "consistency" not in context.state:
        raise RuntimeError("inspect_results and check_consistency are required before write_review")
    if not title.strip() or not analysis.strip() or not conclusion.strip():
        raise ValueError("title, analysis, and conclusion must be nonempty")
    findings = context.state["consistency"]
    papers = context.state.get("selected_papers", [])
    lines = [f"# {title}", "", "## Scoped results", "", f"Inspected {len(context.state['result_summary'])} explicitly referenced results.", "", "## Consistency findings", ""]
    lines.extend([f"- `{finding['field']}` mismatch at result {finding['index']}: {finding['left']!r} != {finding['right']!r}" for finding in findings] or ["- No deterministic mismatches were found."])
    lines.extend(["", "## Physical analysis", "", analysis.strip(), "", "## Selected literature", ""])
    lines.extend([f"- [{paper['title']}]({paper['source']})" for paper in papers] or ["- No papers selected."])
    lines.extend(["", "## Conclusion", "", conclusion.strip()])
    report = "\n".join(lines) + "\n"
    (context.artifact_directory / "review.md").write_text(report, encoding="utf-8")
    diagnostics = {"finding_count": len(findings), "selected_paper_ids": [paper["id"] for paper in papers]}
    summary = {"stage_id": context.stage_id, "job_id": context.job_id, "result": "review", "decisions": {"title": title}, "diagnostics": diagnostics, "artifacts": ["review.md"]}
    context.finish(report, summary)
    return {"summary": "published review.md", "metrics": diagnostics, "state_keys": [], "artifacts": ["review.md"]}
