"""Prompt text for extrapolation stage."""

STAGE_PROMPT = (
    "For ordinary extrapolation jobs call run_extrapolation once. "
    "For jobs with operation=systematics_budget call run_systematics_budget once. "
    "The runner binds perturbative-matching inputs from the lightcone role."
)
