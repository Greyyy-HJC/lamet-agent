# LaMET arXiv Knowledge Base

This project builds a lightweight local knowledge base for LaMET-related arXiv
papers without downloading the full arXiv corpus. It stores metadata in SQLite,
exports a JSONL snapshot, and supports repeatable incremental updates.

## What it does

- harvests arXiv metadata with topic-focused query groups
- scores each paper as `core`, `secondary`, or `irrelevant`
- stores accepted papers in SQLite
- exports the latest accepted set to JSONL
- records harvest runs and the final harvested date
- provides an update script that crawls from the final harvested date to today

## Project layout

- `config/relevance_config.json`: LaMET keywords, query groups, thresholds
- `config/manual_seeds.json`: optional manual seed arXiv IDs
- `docs/lamet_rubric.md`: strong/secondary relevance rubric
- `docs/schema.md`: SQLite and JSONL schema
- `lamet_kb/`: Python package for API, scoring, storage, pipeline
- `scripts/harvest_lamet.py`: main CLI
- `scripts/update_since_last_harvest.py`: incremental re-crawl wrapper
- `data/`: generated database, exports, and state

## Typical usage

Initial crawl up to a fixed date:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py bootstrap --end-date 2026-07-22
```

Re-crawl from the final harvested date to the current date:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/update_since_last_harvest.py
```

If the arXiv API rate-limits your IP, re-run with a slower pace:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py bootstrap --end-date 2026-07-22 --page-size 10 --sleep-seconds 10 --window-days 30
```

Export JSONL again from the current SQLite database:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py export
```

Show a summary:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py report
```

List harvested papers:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py list --limit 50
```

Search the local library by keyword, year, or label:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py search --query matching --label core --limit 30
python3 scripts/harvest_lamet.py search --query lamet --year 2017 --limit 30
```

Backfill an earlier gap without resetting current progress:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py backfill --start-date 2010-01-01 --end-date 2014-04-25 --page-size 3 --sleep-seconds 15 --window-days 10
```

After broadening the relevance rules, re-harvest a historical span so newly
eligible papers can be added:

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
python3 scripts/harvest_lamet.py bootstrap --start-date 2010-01-01 --end-date 2026-07-22 --page-size 3 --sleep-seconds 15 --window-days 10 --no-resume
```

## Notes

- The implementation uses only the Python standard library.
- The default bootstrap start date is `2013-01-01`, which matches the practical
  emergence window of LaMET-related work. You can override it with
  `--start-date`.
- The crawler is intentionally conservative with page size and pacing because
  the arXiv API can return `503` or timeout under heavier sustained access.
- Incremental updates re-fetch a small backfill window by default to reduce the
  risk of missing records near the previous cutoff date.
- If you broaden `config/relevance_config.json`, existing accepted records stay
  intact, but older papers that were previously filtered out will only appear
  after you re-harvest the relevant historical date range.
