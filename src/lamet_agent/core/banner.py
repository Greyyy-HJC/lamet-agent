"""ASCII startup banner for non-verbose CLI runs.

Purpose:
- print a GRID-style LaMET Agent banner at run start
- format compact stage/job progress headers

Example usage:
- from lamet_agent.core.banner import BANNER, format_job_header
- print(BANNER)
- print(format_job_header("correlator_analysis", "ca_ds_pdf"))
"""

from __future__ import annotations

Glyph = tuple[str, str, str, str, str]

_L: Glyph = ("L    ", "L    ", "L    ", "L    ", "LLLL ")
_A: Glyph = (" AAA ", "A   A", "AAAAA", "A   A", "A   A")
_M: Glyph = ("M   M", "MM MM", "M M M", "M   M", "M   M")
_E: Glyph = ("EEEEE", "E    ", "EEEE ", "E    ", "EEEEE")
_T: Glyph = ("TTTTT", "  T  ", "  T  ", "  T  ", "  T  ")
_G: Glyph = (" GGG ", "G    ", "G  GG", "G   G", " GGG ")
_N: Glyph = ("N   N", "NN  N", "N N N", "N  NN", "N   N")


def _compose_word(letters: tuple[Glyph, ...]) -> tuple[str, str, str, str, str]:
    """Lay out fixed-width glyphs on a shared column grid."""
    return tuple(" ".join(glyph[row] for glyph in letters) for row in range(5))


_LAMET_LINES = _compose_word((_L, _A, _M, _E, _T))
_AGENT_LINES = _compose_word((_A, _G, _E, _N, _T))


def _grid_border(total_width: int) -> str:
    """Return a GRID-style ``|--|--|...`` border with the given total width."""
    segments = total_width // 3
    border = "|--" * segments
    if len(border) < total_width:
        border += "-" * (total_width - len(border) - 1)
        border += "|"
    elif len(border) > total_width:
        border = border[: total_width - 1] + "|"
    else:
        border += "|"
    return border


def _frame_banner(*content_groups: tuple[str, ...]) -> str:
    """Wrap letter rows in a tight GRID-style box."""
    content_lines = [line for group in content_groups for line in group]
    inner_width = max(len(line) for line in content_lines)
    total_width = inner_width + 2
    border = _grid_border(total_width)
    framed = [border, border, border]
    framed.extend(f"|{line.ljust(inner_width)}|" for line in content_lines)
    framed.extend([border, border, border])
    return "\n".join(framed)


BANNER = _frame_banner(_LAMET_LINES, ("",), _AGENT_LINES)


def format_job_header(stage: str, job_id: str) -> str:
    """Return a one-line stage/job progress header."""
    return f"Stage: {stage}  |  Job: {job_id}"
