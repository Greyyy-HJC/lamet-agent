"""Core shared utilities for stage routing and prompting."""

from .prompting import ACTION_OUTPUT_HINT, SYSTEM_PROMPT, build_stage_prompt
from .stages import DEFAULT_STAGES, select_stage_sequence

__all__ = [
    "ACTION_OUTPUT_HINT",
    "SYSTEM_PROMPT",
    "DEFAULT_STAGES",
    "build_stage_prompt",
    "select_stage_sequence",
]
