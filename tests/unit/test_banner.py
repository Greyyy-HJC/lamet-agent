"""Unit tests for the GRID-style LaMET Agent startup banner."""

from __future__ import annotations

from lamet_agent.banner import BANNER, _AGENT_LINES, _LAMET_LINES, _LIGHT_CONE


def test_banner_contains_lamet_agent() -> None:
    assert "LLLLLLL" in BANNER
    assert "AAAAAAA" in BANNER
    assert "GGGGG" in BANNER
    assert "|--" in BANNER


def test_banner_has_light_cone() -> None:
    assert "\\/" in BANNER
    assert "/\\" in BANNER
    assert ".--." in BANNER
    assert "'--'" in BANNER
    assert ".'    '." in BANNER
    assert "*" not in BANNER
    assert "o>>>" not in BANNER


def test_light_cone_fits_the_letter_stack() -> None:
    letter_rows = len(_LAMET_LINES) + len(_AGENT_LINES)
    stack_rows = letter_rows + 1
    assert len(_LIGHT_CONE) <= stack_rows
    assert len(_LIGHT_CONE) >= 8


def test_banner_lines_share_a_width() -> None:
    lines = BANNER.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert len(lines) == 17
