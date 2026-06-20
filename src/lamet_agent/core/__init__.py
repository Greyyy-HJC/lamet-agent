"""Core shared utilities for stage routing and prompting."""

from .prompting import (
    ACTION_OUTPUT_HINT,
    SYSTEM_PROMPT,
    build_stage_static_prompt,
    format_tool_observation,
)
from .tools import resolve_plot_save_path

__all__ = [
    "ACTION_OUTPUT_HINT",
    "SYSTEM_PROMPT",
    "build_stage_static_prompt",
    "format_tool_observation",
    "resolve_plot_save_path",
]
