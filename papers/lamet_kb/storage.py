from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    latest_version INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    comment TEXT NOT NULL,
    journal_ref TEXT NOT NULL,
    doi TEXT NOT NULL,
    abs_url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    score INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    query_hits_json TEXT NOT NULL,
    raw_record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS harvest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    from_date TEXT NOT NULL,
    to_date TEXT NOT NULL,
    queries_json TEXT NOT NULL,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    inserted_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def upsert_paper(connection: sqlite3.Connection, record: Dict[str, Any]) -> str:
    cursor = connection.execute(
        "SELECT latest_version, updated, query_hits_json FROM papers WHERE arxiv_id = ?",
        (record["arxiv_id"],),
    )
    existing = cursor.fetchone()
    timestamp = now_utc()

    query_hits = sorted(set(record.get("query_hits", [])))
    if existing is not None:
        previous_query_hits = json.loads(existing["query_hits_json"])
        query_hits = sorted(set(query_hits) | set(previous_query_hits))

    payload = (
        record["arxiv_id"],
        int(record["latest_version"]),
        record["title"],
        record["summary"],
        json_dump(record.get("authors", [])),
        record.get("primary_category", ""),
        json_dump(record.get("categories", [])),
        record.get("published", ""),
        record.get("updated", ""),
        record.get("comment", ""),
        record.get("journal_ref", ""),
        record.get("doi", ""),
        record.get("abs_url", ""),
        record.get("pdf_url", ""),
        int(record.get("score", 0)),
        record.get("label", ""),
        record.get("confidence", ""),
        json_dump(record.get("reasons", [])),
        json_dump(query_hits),
        json_dump(record),
        timestamp,
        timestamp,
    )

    if existing is None:
        connection.execute(
            """
            INSERT INTO papers (
                arxiv_id, latest_version, title, summary, authors_json,
                primary_category, categories_json, published, updated, comment,
                journal_ref, doi, abs_url, pdf_url, score, label, confidence,
                reasons_json, query_hits_json, raw_record_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return "inserted"

    created_at = connection.execute(
        "SELECT created_at FROM papers WHERE arxiv_id = ?",
        (record["arxiv_id"],),
    ).fetchone()["created_at"]
    update_payload = payload[:-2] + (created_at, timestamp)
    connection.execute(
        """
        INSERT OR REPLACE INTO papers (
            arxiv_id, latest_version, title, summary, authors_json,
            primary_category, categories_json, published, updated, comment,
            journal_ref, doi, abs_url, pdf_url, score, label, confidence,
            reasons_json, query_hits_json, raw_record_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        update_payload,
    )
    return "updated"


def create_run(
    connection: sqlite3.Connection,
    run_mode: str,
    from_date: str,
    to_date: str,
    queries: Iterable[str],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO harvest_runs (
            run_mode, started_at, from_date, to_date, queries_json, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_mode, now_utc(), from_date, to_date, json_dump(list(queries)), "running"),
    )
    return int(cursor.lastrowid)


def finish_run(
    connection: sqlite3.Connection,
    run_id: int,
    fetched_count: int,
    accepted_count: int,
    inserted_count: int,
    updated_count: int,
    status: str,
    notes: str = "",
) -> None:
    connection.execute(
        """
        UPDATE harvest_runs
        SET finished_at = ?, fetched_count = ?, accepted_count = ?,
            inserted_count = ?, updated_count = ?, status = ?, notes = ?
        WHERE id = ?
        """,
        (now_utc(), fetched_count, accepted_count, inserted_count, updated_count, status, notes, run_id),
    )


def set_state(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
        (key, value),
    )


def get_state(connection: sqlite3.Connection, key: str) -> Optional[str]:
    row = connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def export_jsonl(connection: sqlite3.Connection, output_path: Path) -> int:
    rows = connection.execute(
        "SELECT * FROM papers WHERE label IN ('core', 'secondary') ORDER BY published, arxiv_id"
    ).fetchall()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "arxiv_id": row["arxiv_id"],
                "latest_version": row["latest_version"],
                "title": row["title"],
                "summary": row["summary"],
                "authors": json.loads(row["authors_json"]),
                "primary_category": row["primary_category"],
                "categories": json.loads(row["categories_json"]),
                "published": row["published"],
                "updated": row["updated"],
                "comment": row["comment"],
                "journal_ref": row["journal_ref"],
                "doi": row["doi"],
                "abs_url": row["abs_url"],
                "pdf_url": row["pdf_url"],
                "score": row["score"],
                "label": row["label"],
                "confidence": row["confidence"],
                "reasons": json.loads(row["reasons_json"]),
                "query_hits": json.loads(row["query_hits_json"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def report_counts(connection: sqlite3.Connection) -> Dict[str, int]:
    total = connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
    core = connection.execute("SELECT COUNT(*) AS count FROM papers WHERE label = 'core'").fetchone()["count"]
    secondary = connection.execute("SELECT COUNT(*) AS count FROM papers WHERE label = 'secondary'").fetchone()["count"]
    return {"total": int(total), "core": int(core), "secondary": int(secondary)}


def get_latest_published_date(connection: sqlite3.Connection) -> Optional[str]:
    row = connection.execute(
        "SELECT substr(max(published), 1, 10) AS published_date FROM papers"
    ).fetchone()
    if row is None:
        return None
    return row["published_date"]


def list_papers(
    connection: sqlite3.Connection,
    limit: int = 50,
    label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = "SELECT arxiv_id, published, label, score, title FROM papers"
    params: List[Any] = []
    if label:
        query += " WHERE label = ?"
        params.append(label)
    query += " ORDER BY published DESC, arxiv_id DESC LIMIT ?"
    params.append(limit)
    rows = connection.execute(query, params).fetchall()
    return [
        {
            "arxiv_id": row["arxiv_id"],
            "published": row["published"][:10],
            "label": row["label"],
            "score": int(row["score"]),
            "title": row["title"],
        }
        for row in rows
    ]


def search_papers(
    connection: sqlite3.Connection,
    query_text: Optional[str] = None,
    year: Optional[int] = None,
    label: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    if query_text:
        clauses.append("(lower(title) LIKE ? OR lower(summary) LIKE ?)")
        pattern = f"%{query_text.lower()}%"
        params.extend([pattern, pattern])
    if year is not None:
        clauses.append("substr(published, 1, 4) = ?")
        params.append(str(year))
    if label:
        clauses.append("label = ?")
        params.append(label)

    query = "SELECT arxiv_id, published, label, score, title FROM papers"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY published DESC, score DESC, arxiv_id DESC LIMIT ?"
    params.append(limit)

    rows = connection.execute(query, params).fetchall()
    return [
        {
            "arxiv_id": row["arxiv_id"],
            "published": row["published"][:10],
            "label": row["label"],
            "score": int(row["score"]),
            "title": row["title"],
        }
        for row in rows
    ]
