# Data Schema

## SQLite database

Path: `data/lamet_arxiv.sqlite3`

### `papers`

- `arxiv_id`: versionless arXiv identifier, primary key
- `latest_version`: latest harvested version number
- `title`
- `summary`
- `authors_json`
- `primary_category`
- `categories_json`
- `published`
- `updated`
- `comment`
- `journal_ref`
- `doi`
- `abs_url`
- `pdf_url`
- `score`
- `label`
- `confidence`
- `reasons_json`
- `query_hits_json`
- `raw_record_json`
- `created_at`
- `updated_at`

### `harvest_runs`

- `id`
- `run_mode`
- `started_at`
- `finished_at`
- `from_date`
- `to_date`
- `queries_json`
- `fetched_count`
- `accepted_count`
- `inserted_count`
- `updated_count`
- `status`
- `notes`

### `state`

Key-value store for:

- `last_harvest_date`
- `last_run_mode`
- `last_completed_at`
- `last_jsonl_export`

## JSONL export

Path: `data/papers.jsonl`

One JSON object per accepted paper. The export mirrors the paper table but keeps
JSON fields as native arrays or objects rather than serialized SQLite strings.
