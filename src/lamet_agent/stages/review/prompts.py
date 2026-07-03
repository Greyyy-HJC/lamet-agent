"""Prompt text for future review-stage integration."""

STAGE_INSTRUCTION = """
Summarize existing stage reports into one scientific review. Use only report
content, explicit formula templates, and NetCDF numerical summaries. Do not
invent missing numerical values. Call write_review once; it writes the
no-LLM-summary review.md or review_CN.md under the artifact directory. The
runner will then pass that completed review to the current backend/model and
append the final LLM Summary section.
""".strip()
