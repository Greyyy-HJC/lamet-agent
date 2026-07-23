#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lamet_kb import pipeline, settings, storage  # noqa: E402


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def optional_year(value: str) -> int:
    parsed = int(value)
    if parsed < 1900 or parsed > 2100:
        raise argparse.ArgumentTypeError("year must be in a reasonable range")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    config = settings.load_json(settings.CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Harvest and maintain a LaMET-focused arXiv metadata knowledge base.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Run the initial metadata crawl.")
    bootstrap.add_argument("--start-date", default=config["bootstrap_start_date"], help="Inclusive ISO date, default from config.")
    bootstrap.add_argument("--end-date", default=date.today().isoformat(), help="Inclusive ISO date, default today.")
    bootstrap.add_argument("--page-size", type=positive_int, default=config["default_page_size"], help="API page size.")
    bootstrap.add_argument("--sleep-seconds", type=float, default=config["default_sleep_seconds"], help="Delay between API pages.")
    bootstrap.add_argument("--window-days", type=positive_int, default=180, help="Split the date range into windows of this many days.")
    bootstrap.add_argument("--max-results-per-query", type=positive_int, default=None, help="Optional cap per query group for test runs.")
    bootstrap.add_argument("--no-resume", action="store_true", help="Ignore saved bootstrap progress and start from the requested start date.")

    update = subparsers.add_parser("update", help="Re-crawl from the last harvested date to today.")
    update.add_argument("--end-date", default=date.today().isoformat(), help="Inclusive ISO date, default today.")
    update.add_argument("--backfill-days", type=nonnegative_int, default=2, help="Re-fetch this many days before the saved cutoff.")
    update.add_argument("--page-size", type=positive_int, default=config["default_page_size"], help="API page size.")
    update.add_argument("--sleep-seconds", type=float, default=config["default_sleep_seconds"], help="Delay between API pages.")
    update.add_argument("--window-days", type=positive_int, default=180, help="Split the date range into windows of this many days.")
    update.add_argument("--max-results-per-query", type=positive_int, default=None, help="Optional cap per query group for test runs.")

    subparsers.add_parser("export", help="Export accepted records from SQLite to JSONL.")
    backfill = subparsers.add_parser("backfill", help="Backfill an earlier date range before the current bootstrap progress without resetting it.")
    backfill.add_argument("--start-date", required=True, help="Inclusive ISO date for the earlier backfill start.")
    backfill.add_argument("--end-date", default=None, help="Inclusive ISO date for the earlier backfill end. Defaults to the day before bootstrap progress.")
    backfill.add_argument("--page-size", type=positive_int, default=config["default_page_size"], help="API page size.")
    backfill.add_argument("--sleep-seconds", type=float, default=config["default_sleep_seconds"], help="Delay between API pages.")
    backfill.add_argument("--window-days", type=positive_int, default=180, help="Split the date range into windows of this many days.")
    backfill.add_argument("--max-results-per-query", type=positive_int, default=None, help="Optional cap per query group for test runs.")
    list_cmd = subparsers.add_parser("list", help="List harvested papers from SQLite.")
    list_cmd.add_argument("--limit", type=positive_int, default=50, help="Maximum number of papers to show.")
    list_cmd.add_argument("--label", choices=["core", "secondary"], default=None, help="Optional label filter.")
    search_cmd = subparsers.add_parser("search", help="Search harvested papers by keyword, year, or label.")
    search_cmd.add_argument("--query", default=None, help="Case-insensitive keyword search over title and abstract.")
    search_cmd.add_argument("--year", type=optional_year, default=None, help="Optional publication year filter.")
    search_cmd.add_argument("--label", choices=["core", "secondary"], default=None, help="Optional label filter.")
    search_cmd.add_argument("--limit", type=positive_int, default=50, help="Maximum number of papers to show.")
    subparsers.add_parser("report", help="Show a local summary.")
    return parser


def run_bootstrap(args: argparse.Namespace) -> int:
    result = pipeline.run_harvest(
        run_mode="bootstrap",
        from_date=args.start_date,
        to_date=args.end_date,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
        max_results_per_query=args.max_results_per_query,
        window_days=args.window_days,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_update(args: argparse.Namespace) -> int:
    settings.ensure_data_dir()
    connection = storage.connect(settings.DB_PATH)
    try:
        last_harvest_date = storage.get_state(connection, "last_harvest_date")
    finally:
        connection.close()

    if not last_harvest_date:
        raise SystemExit("No previous harvest date found. Run bootstrap first.")

    config = settings.load_json(settings.CONFIG_PATH)
    from_date = pipeline.compute_update_start(
        last_harvest_date=last_harvest_date,
        bootstrap_start_date=config["bootstrap_start_date"],
        backfill_days=args.backfill_days,
    )
    result = pipeline.run_harvest(
        run_mode="update",
        from_date=from_date,
        to_date=args.end_date,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
        max_results_per_query=args.max_results_per_query,
        window_days=args.window_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_export() -> int:
    result = pipeline.export_current_snapshot()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_backfill(args: argparse.Namespace) -> int:
    result = pipeline.backfill_before_progress(
        backfill_start_date=args.start_date,
        backfill_end_date=args.end_date,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
        max_results_per_query=args.max_results_per_query,
        window_days=args.window_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_list(args: argparse.Namespace) -> int:
    result = pipeline.list_papers(limit=args.limit, label=args.label)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_search(args: argparse.Namespace) -> int:
    result = pipeline.search_papers(
        query_text=args.query,
        year=args.year,
        label=args.label,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_report() -> int:
    result = pipeline.report()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "bootstrap":
        return run_bootstrap(args)
    if args.command == "update":
        return run_update(args)
    if args.command == "export":
        return run_export()
    if args.command == "backfill":
        return run_backfill(args)
    if args.command == "list":
        return run_list(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "report":
        return run_report()
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
