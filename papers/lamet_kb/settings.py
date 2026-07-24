from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "relevance_config.json"
MANUAL_SEEDS_PATH = PROJECT_ROOT / "config" / "manual_seeds.json"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "lamet_arxiv.sqlite3"
JSONL_PATH = DATA_DIR / "papers.jsonl"
STATE_PATH = DATA_DIR / "state.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

