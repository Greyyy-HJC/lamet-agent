"""Read selected paper bodies from local storage or ar5iv."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from lamet_agent.agent import ToolContext


class _Ar5ivTextParser(HTMLParser):
    """Extract readable paper text from ar5iv HTML with the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignored += 1
        elif not self.ignored and tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.ignored:
            self.ignored -= 1
        elif not self.ignored and tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored and data.strip():
            self.parts.append(data)


def run(context: ToolContext, *, paper_ids: list[str]) -> dict[str, object]:
    """Load selected papers while bounding only the text sent to the LLM."""
    if not paper_ids or len(paper_ids) > int(context.params["max_papers"]):
        raise ValueError("paper_ids must be nonempty and respect max_papers")
    candidates = {record["id"]: record for record in context.state["literature_candidates"]}
    if any(paper_id not in candidates for paper_id in paper_ids):
        raise ValueError("paper_ids must be selected from the current literature candidates")

    catalog_directory = Path(context.state["literature_catalog_directory"])
    literature_directory = context.artifact_directory / "literature"
    literature_directory.mkdir(exist_ok=True)
    selected = []
    artifacts = []
    prompt_limit = max(10000, 120000 // len(paper_ids))

    for paper_id in paper_ids:
        record = candidates[paper_id]
        safe_id = paper_id.replace("/", "_")
        local_body = None
        if record.get("text_path"):
            local_body = catalog_directory / record["text_path"]
        elif record.get("source_file"):
            local_body = catalog_directory / "arxiv" / record["source_file"]
        raw_html = None
        full_text = None
        retrieval = "index_fallback"
        if local_body is not None and local_body.is_file():
            if local_body.suffix.lower() == ".html":
                raw_html = local_body.read_text(encoding="utf-8")
            else:
                full_text = local_body.read_text(encoding="utf-8")
            retrieval = "local"
        elif record.get("text_path") is None:
            request = urllib.request.Request(
                record["ar5iv_url"],
                headers={"User-Agent": "lamet-agent review (local research workflow)"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw_html = response.read().decode("utf-8", errors="replace")
                retrieval = "ar5iv"
            except (OSError, TimeoutError):
                raw_html = None

        if raw_html is not None:
            parser = _Ar5ivTextParser()
            parser.feed(raw_html)
            full_text = "\n".join(
                line for line in (" ".join(part.split()) for part in "".join(parser.parts).splitlines()) if line
            )
            html_name = f"literature/{safe_id}.html"
            (context.artifact_directory / html_name).write_text(raw_html, encoding="utf-8")
            artifacts.append(html_name)
        if full_text is None:
            evidence = "; ".join(record.get("evidence", []))
            full_text = f"{record['summary']}\n\nIndexed evidence: {evidence}".strip()

        text_name = f"literature/{safe_id}.txt"
        (context.artifact_directory / text_name).write_text(full_text + "\n", encoding="utf-8")
        artifacts.append(text_name)
        selected.append(
            {
                "id": paper_id,
                "title": record["title"],
                "authors": record["authors"],
                "year": record["year"],
                "source": record["source"],
                "ar5iv_url": record["ar5iv_url"],
                "matched_topics": record["matched_topics"],
                "retrieval": retrieval,
                "full_text_available": retrieval in {"local", "ar5iv"},
                "full_text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                "full_text_artifact": text_name,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "prompt_truncated": len(full_text) > prompt_limit,
                "text": full_text[:prompt_limit],
            }
        )

    selection = {
        "max_papers": int(context.params["max_papers"]),
        "selected": [{key: value for key, value in paper.items() if key != "text"} for paper in selected],
    }
    (context.artifact_directory / "literature_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    context.state["selected_papers"] = selected
    context.state["literature_artifacts"] = artifacts
    return {
        "summary": f"loaded {len(selected)} selected paper bodies",
        "metrics": {
            "paper_ids": paper_ids,
            "full_text_available": [paper["id"] for paper in selected if paper["full_text_available"]],
        },
        "papers": selected,
        "state_keys": ["selected_papers", "literature_artifacts"],
        "artifacts": ["literature_selection.json", *artifacts],
    }
