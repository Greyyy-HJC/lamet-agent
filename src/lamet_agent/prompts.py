"""Prompt templates for minimal agent execution."""

from __future__ import annotations

SYSTEM_PROMPT = """
You are a LaMET analysis agent.
Output one JSON action per stage. Be explicit and deterministic.
If required inputs are missing, ask for user input.
""".strip()

STAGE_PROMPTS: dict[str, str] = {
    "correlator_analysis": (
        "Analyze correlator data and propose extraction strategy for observables."
    ),
    "renormalization": "Apply user-selected renormalization setup deterministically.",
    "fourier_transform": "Run asymptotic extension and Fourier transform conventions.",
    "perturbative_matching": "Apply perturbative matching kernel and propagate errors.",
    "extrapolation": "Perform continuum/chiral/volume extrapolation strategy comparison.",
}

ACTION_OUTPUT_HINT = (
    'Return JSON with keys: "action", "reason", optional "tool_name", optional "args".'
)
