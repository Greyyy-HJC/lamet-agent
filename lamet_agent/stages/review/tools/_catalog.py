"""Private review catalog reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def catalog_path(context) -> Path:
    configured = context.params["catalog"]
    if configured == "builtin":
        return Path(__file__).parents[3] / "literature" / "catalog.json"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        root = Path(context.manifest["metadata"]["root_directory"]).expanduser()
        path = (context.manifest_path.parent / root / path).resolve()
    return path


def load_catalog(context) -> list[dict[str, Any]]:
    path = catalog_path(context)
    document = json.loads(path.read_text(encoding="utf-8"))
    papers = document.get("papers")
    if not isinstance(papers, list):
        raise ValueError("literature catalog must contain a papers list")
    return papers
