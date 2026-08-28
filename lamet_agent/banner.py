"""GRID-style LaMET Agent startup banner.

A light-cone icon sits to the left of the LAMET / AGENT lettering: two cones
joined at the tips, with elliptical bases. The icon and two rows of block
letters are vertically centered independently.
"""

from __future__ import annotations

Glyph = tuple[str, str, str, str, str, str, str, str]

_L: Glyph = (
    "L        ",
    "L        ",
    "L        ",
    "L        ",
    "L        ",
    "L        ",
    "L        ",
    "LLLLLLLLL",
)
_A: Glyph = (
    "    A    ",
    "   A A   ",
    "  A   A  ",
    " A     A ",
    "AAAAAAAAA",
    "A       A",
    "A       A",
    "A       A",
)
_a: Glyph = (
    "         ",
    " aaaaaaa ",
    "a       a",
    "        a",
    " aaaaaaaa",
    "a       a",
    "a      aa",
    " aaaaaa a",
)
_M: Glyph = (
    "M       M",
    "MM     MM",
    "M M   M M",
    "M  M M  M",
    "M   M   M",
    "M       M",
    "M       M",
    "M       M",
)
_E: Glyph = (
    "EEEEEEEEE",
    "E        ",
    "E        ",
    "EEEEEEEE ",
    "E        ",
    "E        ",
    "E        ",
    "EEEEEEEEE",
)
_T: Glyph = (
    "TTTTTTTTT",
    "    T    ",
    "    T    ",
    "    T    ",
    "    T    ",
    "    T    ",
    "    T    ",
    "    T    ",
)
_G: Glyph = (
    "  GGGGGG ",
    " G      G",
    "G        ",
    "G        ",
    "G    GGGG",
    "G       G",
    " G      G",
    "  GGGGGG ",
)
_N: Glyph = (
    "N       N",
    "NN      N",
    "N N     N",
    "N  N    N",
    "N    N  N",
    "N     N N",
    "N      NN",
    "N       N",
)

# Future and past light cones joined at the origin, with elliptical bases.
_LIGHT_CONE = (
    R"           | t         ",
    R"     __---^^^---__     ",
    R"    (__    |    __)    ",
    R"     \:`--___--`:/     ",
    R"      \::::|::::O      ",
    R"      _\:::|:::#______ ",
    R"     /  \::|::#       |",
    R"    /    \:|:#        |",
    R"   /      \|#         |",
    R"__/________X_________/_",
    R" /        #|\       / x",
    R"|        #.|.\     /   ",
    R"|       #..|..\   /    ",
    R"|______#___|___\_/     ",
    R"      O::::|::::\      ",
    R"     /:::::|:::::\     ",
    R"    (__::::|::::__)    ",
    R"       `--___--`       ",
    R"           |           ",
)


def _compose_word(letters: tuple[Glyph, ...]) -> tuple[str, ...]:
    """Lay out fixed-width glyphs on a shared column grid."""
    height = len(letters[0])
    if any(len(glyph) != height for glyph in letters):
        raise ValueError("all banner glyphs must have the same height")
    return tuple(" ".join(glyph[row] for glyph in letters) for row in range(height))


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


def _frame_banner(icon: tuple[str, ...], lamet: tuple[str, ...], agent: tuple[str, ...]) -> str:
    """Place the light-cone icon on the left of the letter stack in a GRID box."""
    text = (*lamet, "", *agent)
    height = max(len(icon), len(text))
    icon_lines = _vcenter(icon, height)
    text_lines = _vcenter(text, height)
    icon_width = max(len(line) for line in icon_lines)
    text_width = max(len(line) for line in text_lines)
    inner_width = icon_width + 1 + text_width
    total_width = inner_width + 2
    remainder = total_width % 3
    if remainder != 1:
        total_width += (1 - remainder) % 3
        inner_width = total_width - 2
    border = _grid_border(total_width)
    framed = [border, border, border]
    for icon_line, text_line in zip(icon_lines, text_lines, strict=True):
        body = f"{icon_line.ljust(icon_width)} {text_line.ljust(text_width)}"
        framed.append(f"|{body.ljust(inner_width)}|")
    framed.extend([border, border, border])
    return "\n".join(framed)


BANNER = _frame_banner(_LIGHT_CONE, _LAMET_LINES, _AGENT_LINES)
