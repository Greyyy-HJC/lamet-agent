"""GRID-style LaMET Agent startup banner.

A light-cone icon sits to the left of the LAMET / AGENT lettering: two cones
joined at the tips, with elliptical bases. The icon and two rows of block
letters are vertically centered independently.
"""

from __future__ import annotations

# ░ ▒ ▓ █ ▄ ▀
Glyph = tuple[str, str, str, str, str, str, str, str]

_L: Glyph = (
    "██          ",
    "██▒▒        ",
    "██▒▒        ",
    "██▒▒        ",
    "██▒▒        ",
    "██▒▒        ",
    "██████████  ",
    "  ▒▒▒▒▒▒▒▒▒▒",
)
_a: Glyph = (
    "            ",
    "            ",
    "  ██████    ",
    "    ▒▒▒▒██  ",
    "  ████████▒▒",
    "██  ▒▒▒▒██▒▒",
    "  ████████▒▒",
    "    ▒▒▒▒▒▒▒▒",
)
_M: Glyph = (
    "██      ██  ",
    "████  ████▒▒",
    "██▒▒██  ██▒▒",
    "██▒▒  ▒▒██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "  ▒▒      ▒▒",
)
_E: Glyph = (
    "██████████  ",
    "██▒▒▒▒▒▒▒▒▒▒",
    "██████      ",
    "██▒▒▒▒▒▒    ",
    "██▒▒        ",
    "██▒▒        ",
    "██████████  ",
    "  ▒▒▒▒▒▒▒▒▒▒",
)
_T: Glyph = (
    "██████████  ",
    "  ▒▒██▒▒▒▒▒▒",
    "    ██▒▒    ",
    "    ██▒▒    ",
    "    ██▒▒    ",
    "    ██▒▒    ",
    "    ██▒▒    ",
    "      ▒▒    ",
)
_A: Glyph = (
    "  ██████    ",
    "██  ▒▒▒▒██  ",
    "██████████▒▒",
    "██▒▒▒▒▒▒██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "  ▒▒      ▒▒",
)
_G: Glyph = (
    "  ████████  ",
    "██  ▒▒▒▒▒▒▒▒",
    "██▒▒  ████  ",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "  ██████  ▒▒",
    "    ▒▒▒▒▒▒  ",
)
_N: Glyph = (
    "██      ██  ",
    "████    ██▒▒",
    "██▒▒██  ██▒▒",
    "██▒▒  ████▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "██▒▒    ██▒▒",
    "  ▒▒      ▒▒",
)

# Future and past light cones joined at the origin, with elliptical bases.
_LIGHT_CONE = (
    R"             ^ t           ",
    R"      _,-----|-----._      ",
    R"   ,-'       |       `-.   ",
    R"  |          |          |  ",
    R"  '.__       |       __,'  ",
    R"    `.`--._______,--','    ",
    R"      `.     |     ,'_____ ",
    R"      / `.   |   _0      / ",
    R"     /    `. | _#       /  ",
    R"    /       `_#      Sxyz  ",
    R"   /       _#|`.      /    ",
    R"  /      _#  |  `.   /     ",
    R" /______0___________/      ",
    R"     ,'      |      `.     ",
    R"   ,'        |        `.   ",
    R"  |          |          |  ",
    R"  '._        |        _,'  ",
    R"     `-.___________,-'     ",
    R"             |             ",
)

# ░ ▒ ▓ █ ▄ ▀
_LIGHT_CONE = (
    R"             ^ t             ",
    R"      ▄▄▄▄▄▄▄│▄▄▄▄▄▄▄        ",
    R"   ▄▀▀       │       ▀▀▄     ",
    R"  █          │          █    ",
    R"   █▄▄       │       ▄▄█     ",
    R"    ▀▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▀      ",
    R"      ▀▄     ¦     ▄▀        ",
    R"     ╔══▀▄   ¦   ▄▀══╗       ",
    R"    ╔╝    ▀▄ ¦ ▄▀    ╚╗      ",
    R"   ╔╝       ▀▄▀       Sxyz   ",
    R"  ╔╝       ▄▀¦▀▄       ╚╗    ",
    R" ╔╝      ▄▀  ¦  ▀▄      ╚╗   ",
    R" ╚═══════════════════════╝   ",
    R"     ▄▀      ¦      ▀▄       ",
    R"   ▄▀        ¦        ▀▄     ",
    R"  █          ¦          █    ",
    R"   ▀▄▄       ¦       ▄▄▀     ",
    R"      ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀        ",
    R"             │               ",
)

_LIGHT_CONE_QUARKS = (
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                 ▄▀          ",
    R"               ▄▀            ",
    R"             ▄▀              ",
    R"           ▄▀                ",
    R"         ▄▀                  ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
    R"                             ",
)


def _compose_word(letters: tuple[Glyph, ...]) -> tuple[str, ...]:
    """Lay out fixed-width glyphs on a shared column grid."""
    return tuple(" ".join(glyph[row] for glyph in letters) for row in range(len(letters[0])))


_LAMET_LINES = _compose_word((_L, _a, _M, _E, _T))
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


def _vcenter(icon: tuple[str, ...], height: int) -> tuple[str, ...]:
    """Pad an icon with blank rows so it sits in the middle of ``height`` rows."""
    width = max(len(line) for line in icon)
    rows = [line.ljust(width) for line in icon]
    extra = height - len(rows)
    if extra < 0:
        raise ValueError("icon is taller than the letter stack")
    top = (extra + 1) // 2
    return tuple([" " * width] * top + rows + [" " * width] * (extra - top))


class _LayeredBanner(str):
    base_positions: frozenset[tuple[int, int]]
    overlay_positions: frozenset[tuple[int, int]]

    def __new__(
        cls,
        text: str,
        base_positions: frozenset[tuple[int, int]],
        overlay_positions: frozenset[tuple[int, int]],
    ) -> _LayeredBanner:
        banner = super().__new__(cls, text)
        banner.base_positions = base_positions
        banner.overlay_positions = overlay_positions
        return banner


def _frame_banner(
    icon: tuple[str, ...],
    lamet: tuple[str, ...],
    agent: tuple[str, ...],
    overlay: tuple[str, ...] | None = None,
) -> _LayeredBanner:
    """Place the light-cone icon on the left of the letter stack without a frame."""
    text = (*lamet, "", *agent)
    height = max(len(icon), len(text))
    icon_lines = _vcenter(icon, height)
    overlay_lines = (
        _vcenter(overlay, height)
        if overlay is not None
        else tuple(" " * len(icon_lines[0]) for _ in icon_lines)
    )
    text_lines = _vcenter(text, height)
    icon_width = max(max(len(line) for line in icon_lines), max(len(line) for line in overlay_lines))
    icon_lines = tuple(line.ljust(icon_width) for line in icon_lines)
    overlay_lines = tuple(line.ljust(icon_width) for line in overlay_lines)
    text_width = max(len(line) for line in text_lines)
    inner_width = icon_width + 1 + text_width
    framed = []
    base_positions = set()
    overlay_positions = set()
    for row, (icon_line, overlay_line, text_line) in enumerate(zip(icon_lines, overlay_lines, text_lines, strict=True)):
        merged_icon = []
        for column, (base_character, overlay_character) in enumerate(zip(icon_line, overlay_line, strict=True)):
            if base_character != " ":
                base_positions.add((row, column))
            if overlay_character != " ":
                overlay_positions.add((row, column))
                merged_icon.append(overlay_character)
            else:
                merged_icon.append(base_character)
        body = f"{''.join(merged_icon)} {text_line.ljust(text_width)}"
        framed.append(body.ljust(inner_width))
    return _LayeredBanner("\n".join(framed), frozenset(base_positions), frozenset(overlay_positions))


BANNER = _frame_banner(_LIGHT_CONE, _LAMET_LINES, _AGENT_LINES, overlay=_LIGHT_CONE_QUARKS)
