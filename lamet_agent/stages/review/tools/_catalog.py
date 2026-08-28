"""Private review catalog reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def catalog_path(context) -> Path:
    configured = context.params["catalog"]
    if configured == "builtin":
        return Path(__file__).parents[3] / "literature" / "arxiv.json"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        root = Path(context.manifest["metadata"]["root_directory"]).expanduser()
        path = (context.manifest_path.parent / root / path).resolve()
    return path


def load_catalog(context) -> list[dict[str, Any]]:
    path = catalog_path(context)
    records = json.loads(path.read_text(encoding="utf-8"))["papers"]
    inspire = {}
    if context.params["catalog"] == "builtin":
        for entry in json.loads((path.parent / "inspirehep.json").read_text(encoding="utf-8")):
            metadata = entry.get("metadata", {})
            for eprint in metadata.get("arxiv_eprints", []):
                if eprint.get("value"):
                    inspire[eprint["value"]] = metadata

    normalized = []
    for record in records:
        paper_id = str(record.get("arxiv_id", record.get("id")))
        metadata = inspire.get(paper_id, {})
        authors = [item["full_name"] for item in metadata.get("authors", []) if item.get("full_name")]
        date = metadata.get("preprint_date") or metadata.get("legacy_creation_date") or ""
        abstracts = metadata.get("abstracts", [])
        abstract = abstracts[0].get("value", "") if abstracts else record.get("abstract", "")
        normalized.append(
            {
                "id": paper_id,
                "title": record.get("title", ""),
                "authors": authors or record.get("authors", []),
                "year": str(record.get("year", date[:4] if date else "")),
                "summary": record.get("review_summary", abstract),
                "source": record.get("source", f"https://arxiv.org/abs/{paper_id}"),
                "ar5iv_url": record.get("ar5iv_url", f"https://ar5iv.labs.arxiv.org/html/{paper_id}"),
                "source_file": record.get("source_file"),
                "text_path": record.get("text_path"),
                "relevance": record.get("relevance", "core"),
                "review_topics": record.get("review_topics", []),
                "evidence": record.get("evidence", []),
                "confidence": record.get("confidence"),
                "tags": record.get("tags", {}),
                "uses_lattice_data": bool(record.get("lattice_setup", {}).get("uses_lattice_data")),
            }
        )
    return normalized
