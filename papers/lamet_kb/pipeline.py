from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from . import arxiv, scoring, settings, storage


def load_manual_seed_ids() -> Set[str]:
    payload = settings.load_json(settings.MANUAL_SEEDS_PATH)
    return {str(item).strip() for item in payload.get("seed_ids", []) if str(item).strip()}


def write_state_file(connection, extra: Dict[str, Any]) -> None:
    state = {
        "last_harvest_date": storage.get_state(connection, "last_harvest_date"),
        "last_run_mode": storage.get_state(connection, "last_run_mode"),
        "last_completed_at": storage.get_state(connection, "last_completed_at"),
        "last_jsonl_export": storage.get_state(connection, "last_jsonl_export"),
        "db_path": str(settings.DB_PATH),
        "jsonl_path": str(settings.JSONL_PATH),
    }
    state.update(extra)
    with settings.STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)


def compute_update_start(last_harvest_date: str, bootstrap_start_date: str, backfill_days: int) -> str:
    last_day = datetime.strptime(last_harvest_date, "%Y-%m-%d").date()
    bootstrap_day = datetime.strptime(bootstrap_start_date, "%Y-%m-%d").date()
    candidate = last_day - timedelta(days=max(backfill_days, 0))
    if candidate < bootstrap_day:
        candidate = bootstrap_day
    return candidate.isoformat()


def run_harvest(
    run_mode: str,
    from_date: str,
    to_date: str,
    page_size: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
    max_results_per_query: Optional[int] = None,
    window_days: int = 180,
    resume: bool = True,
) -> Dict[str, Any]:
    settings.ensure_data_dir()
    config = settings.load_json(settings.CONFIG_PATH)
    manual_seed_ids = load_manual_seed_ids()

    effective_page_size = int(page_size or config["default_page_size"])
    effective_sleep = float(sleep_seconds if sleep_seconds is not None else config["default_sleep_seconds"])

    connection = storage.connect(settings.DB_PATH)
    connection.execute("PRAGMA journal_mode = WAL")
    query_names = [item["name"] for item in config["query_groups"]]

    effective_from_date = from_date
    if resume and run_mode == "bootstrap":
        checkpoint = storage.get_state(connection, "bootstrap_progress_date")
        if checkpoint and checkpoint >= from_date and checkpoint <= to_date:
            checkpoint_day = datetime.strptime(checkpoint, "%Y-%m-%d").date() + timedelta(days=1)
            if checkpoint_day.isoformat() <= to_date:
                effective_from_date = checkpoint_day.isoformat()

    run_id = storage.create_run(connection, run_mode, from_date, to_date, query_names)

    fetched_count = 0
    accepted_count = 0
    inserted_count = 0
    updated_count = 0
    staged: Dict[str, Dict[str, Any]] = {}

    try:
        for group in config["query_groups"]:
            group_yielded = 0
            for window_start, window_end in arxiv.iter_date_windows(effective_from_date, to_date, window_days):
                remaining_for_group = None
                if max_results_per_query is not None:
                    remaining_for_group = max_results_per_query - group_yielded
                    if remaining_for_group <= 0:
                        break

                for record in arxiv.iter_query_results(
                    api_url=config["api_url"],
                    raw_query=group["query"],
                    start_date=window_start,
                    end_date=window_end,
                    page_size=effective_page_size,
                    sleep_seconds=effective_sleep,
                    max_results_per_query=remaining_for_group,
                ):
                    fetched_count += 1
                    group_yielded += 1
                    current = staged.get(record["arxiv_id"])
                    if current is None:
                        record["query_hits"] = [group["name"]]
                        staged[record["arxiv_id"]] = record
                        continue

                    merged_hits = sorted(set(current.get("query_hits", [])) | {group["name"]})
                    replace = (
                        int(record.get("latest_version", 0)) > int(current.get("latest_version", 0))
                        or record.get("updated", "") > current.get("updated", "")
                    )
                    if replace:
                        record["query_hits"] = merged_hits
                        staged[record["arxiv_id"]] = record
                    else:
                        current["query_hits"] = merged_hits

                window_accepted, window_inserted, window_updated = _flush_staged_records(
                    connection=connection,
                    staged=staged,
                    config=config,
                    manual_seed_ids=manual_seed_ids,
                )
                accepted_count += window_accepted
                inserted_count += window_inserted
                updated_count += window_updated
                if run_mode == "bootstrap":
                    storage.set_state(connection, "bootstrap_progress_date", window_end)
                connection.commit()

        final_accepted, final_inserted, final_updated = _flush_staged_records(
            connection=connection,
            staged=staged,
            config=config,
            manual_seed_ids=manual_seed_ids,
        )
        accepted_count += final_accepted
        inserted_count += final_inserted
        updated_count += final_updated

        export_count = storage.export_jsonl(connection, settings.JSONL_PATH)
        storage.set_state(connection, "last_harvest_date", to_date)
        storage.set_state(connection, "last_run_mode", run_mode)
        storage.set_state(connection, "last_completed_at", storage.now_utc())
        storage.set_state(connection, "last_jsonl_export", storage.now_utc())
        if run_mode == "bootstrap":
            storage.set_state(connection, "bootstrap_progress_date", to_date)
        storage.finish_run(
            connection,
            run_id=run_id,
            fetched_count=fetched_count,
            accepted_count=accepted_count,
            inserted_count=inserted_count,
            updated_count=updated_count,
            status="completed",
            notes=f"exported_jsonl_count={export_count}",
        )
        connection.commit()
        write_state_file(
            connection,
            {
                "last_exported_count": export_count,
                "last_run_id": run_id,
                "effective_from_date": effective_from_date,
            },
        )
        return {
            "run_id": run_id,
            "fetched_count": fetched_count,
            "accepted_count": accepted_count,
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "exported_count": export_count,
            "from_date": effective_from_date,
            "to_date": to_date,
        }
    except Exception as exc:
        storage.finish_run(
            connection,
            run_id=run_id,
            fetched_count=fetched_count,
            accepted_count=accepted_count,
            inserted_count=inserted_count,
            updated_count=updated_count,
            status="failed",
            notes=str(exc),
        )
        connection.commit()
        raise
    finally:
        connection.close()


def export_current_snapshot() -> Dict[str, Any]:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        export_count = storage.export_jsonl(connection, settings.JSONL_PATH)
        storage.set_state(connection, "last_jsonl_export", storage.now_utc())
        connection.commit()
        write_state_file(connection, {"last_exported_count": export_count})
        return {"exported_count": export_count, "jsonl_path": str(settings.JSONL_PATH)}
    finally:
        connection.close()


def report() -> Dict[str, Any]:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        counts = storage.report_counts(connection)
        counts["last_harvest_date"] = storage.get_state(connection, "last_harvest_date")
        counts["last_completed_at"] = storage.get_state(connection, "last_completed_at")
        counts["bootstrap_progress_date"] = storage.get_state(connection, "bootstrap_progress_date")
        counts["latest_published_date"] = storage.get_latest_published_date(connection)
        return counts
    finally:
        connection.close()


def list_papers(limit: int = 50, label: Optional[str] = None) -> Dict[str, Any]:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        papers = storage.list_papers(connection, limit=limit, label=label)
        return {"count": len(papers), "papers": papers}
    finally:
        connection.close()


def search_papers(
    query_text: Optional[str] = None,
    year: Optional[int] = None,
    label: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        papers = storage.search_papers(
            connection,
            query_text=query_text,
            year=year,
            label=label,
            limit=limit,
        )
        return {"count": len(papers), "papers": papers}
    finally:
        connection.close()


def backfill_before_progress(
    backfill_start_date: str,
    backfill_end_date: Optional[str] = None,
    page_size: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
    max_results_per_query: Optional[int] = None,
    window_days: int = 180,
) -> Dict[str, Any]:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        checkpoint = storage.get_state(connection, "bootstrap_progress_date")
    finally:
        connection.close()

    if not checkpoint:
        raise SystemExit("No bootstrap progress found. Run bootstrap first.")

    checkpoint_day = datetime.strptime(checkpoint, "%Y-%m-%d").date()
    default_backfill_end = (checkpoint_day - timedelta(days=1)).isoformat()
    backfill_end = backfill_end_date or default_backfill_end
    if backfill_end > default_backfill_end:
        raise SystemExit(
            f"Backfill end date {backfill_end} exceeds the allowable pre-progress cutoff {default_backfill_end}."
        )
    if backfill_start_date > backfill_end:
        raise SystemExit(
            f"Backfill start date {backfill_start_date} is after the available pre-progress range ending {backfill_end}."
        )

    return run_harvest(
        run_mode="backfill",
        from_date=backfill_start_date,
        to_date=backfill_end,
        page_size=page_size,
        sleep_seconds=sleep_seconds,
        max_results_per_query=max_results_per_query,
        window_days=window_days,
        resume=False,
    )


def _flush_staged_records(
    connection,
    staged: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    manual_seed_ids: Set[str],
) -> Tuple[int, int, int]:
    accepted_count = 0
    inserted_count = 0
    updated_count = 0
    accepted_labels = set(config["accepted_labels"])

    for arxiv_id in list(staged.keys()):
        record = staged.pop(arxiv_id)
        scored = scoring.score_record(record, config=config, manual_seed_ids=manual_seed_ids)
        if scored["label"] not in accepted_labels:
            continue
        accepted_count += 1
        result = storage.upsert_paper(connection, scored)
        if result == "inserted":
            inserted_count += 1
        else:
            updated_count += 1

    return accepted_count, inserted_count, updated_count
