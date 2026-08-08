"""Prompt text for the review stage."""

STAGE_INSTRUCTION = """
Generate one LLM-written scientific review from the completed stage reports,
NetCDF summaries, and SVG artifact paths. Call write_review once; it collects
the evidence package, asks the configured backend/model to write the full
review.md or review_CN.md, and stores that file as store['output']. When
`stages.review.defaults.literature` is true, the evidence package also includes
up to `literature_max_papers` background-only entries from the local LaMET paper
library (default 4). The configured report language is generated directly. Do not
call stage-specific report tools again.
""".strip()
