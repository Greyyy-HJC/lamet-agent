# Review

## Basic Procedure

Generate one LLM-written scientific review from the completed stage reports,
NetCDF summaries, and SVG artifact paths. Call `write_review` once; it collects
the evidence package, asks the configured backend/model to write the full
review.md or review_CN.md, and stores that file as store['output']. When
`stages.review.defaults.literature` is true, the evidence package also includes
up to `literature_max_papers` background-only entries from the local LaMET paper
library (default 4). The configured report language is generated directly. Do not
call stage-specific report tools again.

## Stage Skill

The review stage is an LLM-written synthesis pass. It asks the configured
backend/model to write the full review from stage reports, NetCDF summaries, and
SVG artifact paths. When `stages.review.defaults.literature` is true, the stage
also injects background-only LaMET literature context from the local SQLite
library, limited by `literature_max_papers` (default 4). The requested report
language is generated directly. SVG paths are provenance only, and figure
statements must be grounded in report text and NetCDF summaries.

## Available Tools

- `write_review`: Collect stage reports, NetCDF summaries, and SVG paths, then ask the configured LLM to write review.md or review_CN.md.
