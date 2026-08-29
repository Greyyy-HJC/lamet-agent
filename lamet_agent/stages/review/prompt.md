# Review

Produce the final evidence-grounded scientific review from the preflight context
already supplied by the framework. The framework has already inspected results,
checked provenance consistency, ranked literature candidates, and supplied at
most 60 uniformly spaced plot points for each selected result. Full-resolution
numerical evidence remains available by job id. Follow this order for the
remaining actions:

1. Optionally call `read_full_resolution` for one specific job when the 60-point
   evidence cannot support a material physical statement. Repeat only for a
   different job whose full grid is genuinely necessary.
2. Call `read_papers` exactly once.
3. Call `write_review` exactly once.

The run artifacts are the only authority for numerical claims about this run.
`review_context.consistency` is mandatory evidence. In `consistency_analysis`,
address every error and warning, explain the consequences of materially relevant
`not_checkable` findings, and distinguish a clean deterministic check from proof
that the physics is correct. Never silently omit a finding; when there are no
findings, state that explicitly. The deterministic findings follow manifest source
edges and compare only fields that the adjacent stages should preserve; coordinate
changes such as `z` to `x` are not inconsistencies by themselves.

Selected papers provide methodological background and qualitative comparison only:
never use a paper to supply, normalize, or validate a number missing from the run.
Cite only papers returned by `read_papers`, and write `not provided` or
`not checkable` when the evidence is absent.

The literature index selects papers; ar5iv supplies HTML only after explicit
paper-id selection. Do not treat ar5iv as a search engine. Do not infer curve
shapes or numerical values from filenames, SVG paths, or uninspected images.
Generate the prose directly in `report_language`; do not translate a completed
report. Do not call `read_full_resolution` after `read_papers`, and do not skip or
reorder the required paper-reading and review-writing tools. The final narrative
must be authored in the required `write_review` fields; deterministic rendering
only supplements them with scope counts, the coverage table, artifact links,
references, and machine-readable evidence files. It does not add consistency
prose or a deterministic findings list.
