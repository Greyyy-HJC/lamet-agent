"""Load selected normalized paper bodies after explicit id selection."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.review.tools._catalog import load_catalog, catalog_path


def run(context: ToolContext, *, paper_ids: list[str]) -> dict[str, object]:
    """Read at most the authored maximum number of selected paper ids."""
    if not paper_ids or len(paper_ids) > int(context.params["max_papers"]):
        raise ValueError("paper_ids must be nonempty and respect max_papers")
    records = {record["id"]: record for record in load_catalog(context)}
    candidates = {record["id"] for record in context.state.get("literature_candidates", [])}
    if any(paper_id not in records or paper_id not in candidates for paper_id in paper_ids):
        raise ValueError("paper_ids must be selected from the current literature candidates")
    root = catalog_path(context).parent
    bodies = []
    for paper_id in paper_ids:
        record = records[paper_id]
        body_path = root / record["text_path"]
        bodies.append(
            {
                "id": paper_id,
                "title": record["title"],
                "text": body_path.read_text(encoding="utf-8"),
                "source": record["source"],
            }
        )
    context.state["selected_papers"] = bodies
    return {
        "summary": f"loaded {len(bodies)} selected paper bodies",
        "metrics": {"paper_ids": paper_ids},
        "state_keys": ["selected_papers"],
        "artifacts": [],
        "papers": bodies,
    }
