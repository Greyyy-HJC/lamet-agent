# Review

Produce the final evidence-grounded scientific review in this exact order:

1. `inspect_results`
2. `check_consistency`
3. `list_literature`
4. `read_papers`
5. `write_review`

The run artifacts are the only authority for numerical claims about this run.
Deterministic consistency findings may diagnose those artifacts. Selected papers
provide methodological background and qualitative comparison only: never use a
paper to supply, normalize, or validate a number missing from the run. Cite only
papers returned by `read_papers`, and write `not provided` or `not checkable` when
the evidence is absent.

The literature index selects papers; ar5iv supplies HTML only after explicit
paper-id selection. Do not treat ar5iv as a search engine. Do not infer curve
shapes or numerical values from filenames, SVG paths, or uninspected images.
Generate the prose directly in `report_language`; do not translate a completed
report. Call each tool once and do not skip or reorder tools.
