"""Core shared utilities for stage routing and prompting."""

from .prompting import (
    ACTION_OUTPUT_HINT,
    SYSTEM_PROMPT,
    build_stage_prompt,
    build_stage_static_prompt,
    format_tool_observation,
)
from .stages import DEFAULT_STAGES, select_stage_sequence
from .tools import resolve_plot_save_path

__all__ = [
    "ACTION_OUTPUT_HINT",
    "SYSTEM_PROMPT",
    "DEFAULT_STAGES",
    "build_stage_prompt",
    "build_stage_static_prompt",
    "format_tool_observation",
    "resolve_plot_save_path",
    "select_stage_sequence",
]
