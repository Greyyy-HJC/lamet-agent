"""Markdown reporting helpers for the perturbative-matching stage.

Mirrors ``stages/fourier/reporting.py``: it turns the matching-stage result and
artifacts into an English report plus a Chinese companion. The matching stage is
simpler than the Fourier stage (no scheme scan / model averaging), so the report
focuses on the chosen kernel, the matching convolution, and a small set of
"is this a sane perturbative correction" diagnostics.
"""

from __future__ import annotations

import gzip
import html
import inspect
import io
import os
import re
import ssl
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent import kernels
from lamet_agent.core.llm import PROVIDERS, provider_config, request_llm_text
from lamet_agent.core.reporting import (
    format_report_list as _fmt_list,
    format_report_value as _fmt,
    markdown_artifact_paths,
    resolve_report_target as _report_target,
)


# Logical operator -> human text, keyed by the ``<operator>`` field of a
# ``CG_<operator>_quark_PDF_<scheme>_NLO`` kernel_id.
OPERATOR_TEXT = {
    "gt": ("unpolarized $\\gamma^t$ quark PDF", "非极化 $\\gamma^t$ 夸克 PDF"),
    "gtg5": ("helicity $\\gamma^t\\gamma_5$ quark PDF", "螺旋度 $\\gamma^t\\gamma_5$ 夸克 PDF"),
    "gz": ("unpolarized $\\gamma^z$ quark PDF", "非极化 $\\gamma^z$ 夸克 PDF"),
    "gzg5": ("helicity $\\gamma^z\\gamma_5$ quark PDF", "螺旋度 $\\gamma^z\\gamma_5$ 夸克 PDF"),
    "gtgpg5": (
        "transversity $\\gamma^t\\gamma_\\perp\\gamma_5$ quark PDF",
        "横向 $\\gamma^t\\gamma_\\perp\\gamma_5$ 夸克 PDF",
    ),
}

# Scheme -> human text. The paper and equation numbers are NOT listed here: they are
# tagged on each kernel in kernels.py (@kernel_reference) and read back by
# _kernel_reference below, so kernels from different papers each cite their own.
SCHEME_TEXT = {
    "msbar": "MSbar",
    "ratio": "ratio",
    "hybrid": "hybrid",
}

MATCHING_ARTIFACT_DESCRIPTIONS = {
    "lightcone_artifact": ("Matched light-cone PDF samples (EnsembleData NetCDF)", "匹配后的光锥 PDF 样本（EnsembleData NetCDF）"),
    "matched_plot": ("PDF plot comparing quasi and light-cone PDFs", "quasi 与光锥 PDF 对比 PDF 图"),
    "matched_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的 SVG 对比图"),
}

MATCHING_ARTIFACT_ORDER = ("lightcone_artifact", "matched_plot", "matched_plot_image")


DISTRIBUTION_TOKENS = ("quark_PDF", "gluon_PDF", "DA", "qDA", "gDA")

DA_TOKENS = frozenset({"DA", "qDA", "gDA"})


def is_da_kernel(kernel_id: str) -> bool:
    """True for a distribution-amplitude kernel, whose factorization has a different shape.

    A DA kernel's density is a genuine two-variable ``V(x, y)`` carrying its own poles and
    integrated with a plain ``dy``; a PDF kernel's is a coefficient of ``ksi = x/y`` alone,
    integrated with ``dy/|y|``. The two therefore diverge differently at the endpoints, so
    callers that treat them alike would misstate whichever kernel they were not written for.
    """
    return any(part in DA_TOKENS for part in str(kernel_id).split("_"))


def _parse_kernel_id(kernel_id: str) -> tuple[str, str]:
    """Split a ``<gauge>_<operator>_<distribution>_<scheme>_<order>`` id into (operator, scheme).

    The distribution token (quark_PDF/gluon_PDF for the quark/gluon PDF, DA for the meson
    distribution amplitude) separates the operator from the scheme, and the order (NLO)
    trails it. A token may itself span several ``_`` segments, so match on joined segments
    rather than on a single one. The leading token is the gauge construction (CG or GI) and
    is not returned -- ``_settings_table`` reads it off the id directly. Falls back to
    ('', '') for any id that does not follow the convention so the report degrades
    gracefully instead of raising.
    """
    parts = str(kernel_id).split("_")
    # <gauge>, <op...>, quark_PDF|DA, <scheme>, <order>
    for idx in range(2, len(parts)):
        for token in DISTRIBUTION_TOKENS:
            width = len(token.split("_"))
            if idx + width < len(parts) and "_".join(parts[idx : idx + width]) == token:
                return "_".join(parts[1:idx]), parts[idx + width]
    return "", ""


def _kernel_reference(kernel_id: str) -> tuple[str, str]:
    """Return the ``(arxiv_id, equations)`` tagged on the kernel the manifest selected.

    The manifest names the kernel, the registry name is the function name in kernels.py,
    and the function carries its own provenance (``@kernel_reference``) -- so the paper
    follows the kernel, with no table here to keep in sync and no default paper baked in.
    An unknown or untagged kernel_id yields ``("", "")``: the report then cites nothing
    and the formula is derived from the code alone, rather than pointing at some other
    paper's equations. Every registered kernel is tagged (a test enforces it).
    """
    fn = getattr(kernels, str(kernel_id), None)
    return getattr(fn, "arxiv_id", "") or "", getattr(fn, "equations", "") or ""


def _format_grid(x_grid: np.ndarray, *, language: str) -> str:
    if x_grid.size == 0:
        return "未记录" if language == "zh" else "not recorded"
    if x_grid.size == 1:
        return f"one point at $x={_fmt(x_grid[0])}$"
    diffs = np.diff(x_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        if language == "zh":
            return f"从 $x={_fmt(x_grid[0])}$ 到 $x={_fmt(x_grid[-1])}$，每隔 $\\Delta x={_fmt(diffs[0])}$ 取一个点，共 {x_grid.size} 个点"
        return f"from $x={_fmt(x_grid[0])}$ to $x={_fmt(x_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {x_grid.size} points"
    if language == "zh":
        return f"非均匀网格，共 {x_grid.size} 个点；预览 `{_fmt_list(x_grid)}`"
    return f"nonuniform grid with {x_grid.size} points; preview `{_fmt_list(x_grid)}`"


def _trapz_norm(x_grid: np.ndarray, values: np.ndarray) -> float:
    """Integral of ``values`` over the x grid (the PDF norm sum rule check)."""
    if x_grid.size < 2 or values.size != x_grid.size:
        return float("nan")
    order = np.argsort(x_grid)
    # np.trapezoid is the NumPy 2.x name; fall back to np.trapz on older NumPy.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values[order], x_grid[order]))


def _settings_table(data: dict[str, Any], *, language: str) -> list[str]:
    kernel_id = str(data.get("kernel_id", "not recorded"))
    operator, scheme = _parse_kernel_id(kernel_id)
    op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
    scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")
    # The `CG` prefix of the kernel_id marks the Coulomb-gauge (no Wilson line)
    # construction; anything else is the conventional gauge-invariant one.
    is_coulomb = kernel_id.upper().startswith("CG")
    gauge_en = "Coulomb gauge ($\\partial_i A_i=0$, no Wilson line)" if is_coulomb else "gauge-invariant (straight Wilson line)"
    gauge_zh = "库伦规范（Coulomb gauge，$\\partial_i A_i=0$，无 Wilson 线）" if is_coulomb else "规范不变（gauge-invariant，含直 Wilson 线）"
    # The paper is whatever the selected kernel declares in kernels.py -- the manifest
    # picks the kernel_id, and the citation follows it.
    arxiv_id, equations = _kernel_reference(kernel_id)
    reference_en = f"arXiv:{arxiv_id} {equations}".strip() if arxiv_id else "not declared by the kernel"
    reference_zh = f"arXiv:{arxiv_id} {equations}".strip() if arxiv_id else "该匹配核未标注出处"
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    # The quasi grid is only worth its own row when matching did not simply keep it:
    # normally it is the light-cone grid, and repeating it would be noise.
    quasi_x_grid = np.asarray(data.get("quasi_x_grid", []), dtype=float)
    separate_quasi_grid = quasi_x_grid.size > 0 and (
        quasi_x_grid.size != x_grid.size or not np.allclose(quasi_x_grid, x_grid)
    )
    zspz = data.get("zspz")
    pz_value = data.get("momentum_gev")
    try:
        pz_text = f"$P_z={_fmt(float(pz_value))}$ GeV"
    except (TypeError, ValueError):
        pz_text = str(pz_value or "not recorded")

    if language == "zh":
        rows = [
            ("矩阵元/算符", f"`{kernel_id}`（{op_zh}）"),
            ("匹配核出处", reference_zh),
            ("规范约定", gauge_zh),
            ("匹配方案", f"`{scheme}`（{scheme_en}）"),
            ("夸克/胶子分量", f"`{data.get('component', 'not recorded')}`"),
            ("强子动量", pz_text),
            ("重整化标度", f"$\\mu={_fmt(data.get('mu'))}$ GeV"),
        ]
        if zspz is not None:
            rows.append(("Wilson 线标度", f"$z_sP_z={_fmt(zspz)}$"))
        rows.extend(
            [
                ("重采样模式", f"`{data.get('resample', 'not recorded')}`，共 {data.get('n_sample', 'n/a')} 个样本"),
                ("x 网格（光锥输出）", _format_grid(x_grid, language="zh")),
            ]
        )
        if separate_quasi_grid:
            rows.append(("x 网格（quasi 输入）", _format_grid(quasi_x_grid, language="zh") + "；与输出网格不同，quasi 数据经线性插值"))
        rows.append(("quasi-PDF 来源", f"`{data.get('source', 'not recorded')}`"))
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Operator / kernel", f"`{kernel_id}` ({op_en})"),
            ("Kernel reference", reference_en),
            ("Gauge convention", gauge_en),
            ("Matching scheme", f"`{scheme}` ({scheme_en})"),
            ("Quark/gluon component", f"`{data.get('component', 'not recorded')}`"),
            ("Hadron momentum", pz_text),
            ("Renormalization scale", f"$\\mu={_fmt(data.get('mu'))}$ GeV"),
        ]
        if zspz is not None:
            rows.append(("Wilson-line scale", f"$z_sP_z={_fmt(zspz)}$"))
        rows.extend(
            [
                ("Resampling mode", f"`{data.get('resample', 'not recorded')}` with {data.get('n_sample', 'n/a')} samples"),
                ("x grid (light-cone output)", _format_grid(x_grid, language="en")),
            ]
        )
        if separate_quasi_grid:
            rows.append(("x grid (quasi input)", _format_grid(quasi_x_grid, language="en") + "; differs from the output grid, so the quasi data was linearly interpolated"))
        rows.append(("Quasi-PDF source", f"`{data.get('source', 'not recorded')}`"))
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _field_definitions(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "| 条目 | 含义 |",
            "|---|---|",
            "| Operator / kernel | 选定的匹配核 `CG_<算符>_PDF_<方案>`；算符决定 Dirac 结构（gt、gtg5），方案决定有限项。 |",
            "| Matching scheme | `msbar` / `ratio` / `hybrid`，由 kernel_id 后缀选定；hybrid 还需要 Wilson 线长度 $z_s$。 |",
            "| Hadron momentum | $P_z$，必须与傅立叶阶段一致，进入核的 $\\log(4y^2P_z^2/\\mu^2)$ 项。 |",
            "| Renormalization scale | MSbar 重整化标度 $\\mu$（GeV），默认 2.0。 |",
            "| Resampling mode | quasi-PDF 携带的重采样轴（bootstrap/jackknife）；匹配逐样本进行以保留关联结构。 |",
        ]
    return [
        "| Entry | Meaning |",
        "|---|---|",
        "| Operator / kernel | The selected matching kernel `CG_<operator>_quark_PDF_<scheme>_NLO`; the operator sets the Dirac structure (gt, gtg5), `quark_PDF` marks it as a quark kernel, the scheme sets the finite terms, and NLO is the perturbative order. |",
        "| Matching scheme | `msbar` / `ratio` / `hybrid`, chosen by the kernel_id suffix; hybrid also needs the Wilson-line length $z_s$. |",
        "| Hadron momentum | $P_z$, which must match the Fourier stage and enters the kernel's $\\log(4y^2P_z^2/\\mu^2)$ terms. |",
        "| Renormalization scale | MSbar renormalization scale $\\mu$ in GeV (default 2.0). |",
        "| Resampling mode | The resampling axis carried by the quasi-PDF (bootstrap/jackknife); matching is done sample by sample to preserve the correlation structure. |",
    ]


# --- LLM-derived kernel formula --------------------------------------------
# The explicit matching coefficient is NOT stored as a hand-written formula. At
# report time the model reads the exact ``kernels.py`` code that produced the
# number (the source of truth) and writes the closed form. Everything needed for
# the call lives in this file so only ``reporting.py`` carries the change; the LLM
# config is read from the environment because the report cannot receive it from
# the agent.

# Provider configs (base_url / default_model / key_env) are reused from
# ``core.llm.PROVIDERS`` so this module stays in sync with the rest of the agent.

# No paper is named in this module. The manifest picks a kernel_id, the kernel carries
# its own @kernel_reference (arXiv id + equations), and _kernel_reference reads it back
# -- so adding a kernel from a new paper needs no change here. See kernels.py.

# Generating a formula is a network round-trip; memoize so the per-job and the
# stage-level report reuse one call per (operator, scheme, language). The value
# is ``(markdown, paper_used)`` so the provenance note knows whether the paper
# text actually made it into the prompt.
_FORMULA_CACHE: dict[tuple[str, str, str], tuple[str, bool]] = {}
# Paper text fetched once per source (local path or arXiv id).
_PAPER_CACHE: dict[str, str | None] = {}


@dataclass(frozen=True)
class FormulaLlm:
    """The LLM the report uses to write the kernel's closed form.

    Passed in explicitly, exactly like the review stage's tool arguments: the run's
    ``--backend`` and (for ``api``) the provider/key/model the CLI already resolved are
    handed down as parameters. Reading them back out of the environment would mean the
    report could silently use a different model, or a different key, from the run itself.
    """

    backend: str = "api"
    provider: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    base_url: str | None = None

    def resolved(self) -> tuple[str, str | None, str | None, str | None, str | None]:
        """Validate and fill provider defaults, returning what request_llm_text needs."""
        if self.backend == "codex":
            return "codex", None, None, None, None
        if self.backend != "api":
            raise RuntimeError(
                f"The matching report's formula section needs an LLM, but this run used "
                f"backend={self.backend!r}. Run with --backend api (plus --model "
                f"provider/model_id) or --backend codex."
            )
        if not self.provider:
            raise RuntimeError(
                "The matching report's formula section needs --model provider/model_id "
                f"(one of {sorted(PROVIDERS)})."
            )
        config = provider_config(self.provider)
        if config is None:
            raise RuntimeError(
                f"Unknown provider {self.provider!r}; use one of {sorted(PROVIDERS)}."
            )
        if not self.api_key:
            raise RuntimeError(
                f"The matching report's formula section needs an API key for "
                f"provider={self.provider!r} (--api-key-file, or {config['key_env']})."
            )
        return (
            "api",
            self.provider,
            self.api_key,
            self.model_name or config["default_model"],
            self.base_url or config["base_url"],
        )


def _kernel_source(kernel_id: str) -> str:
    """Return the implemented kernel + coefficient functions as LLM ground truth."""
    pieces: list[str] = []
    # The registry name is the function name in kernels.py, so resolve it directly
    # instead of rebuilding it from the parsed operator/scheme.
    builder = getattr(kernels, str(kernel_id), None)
    if builder is not None:
        pieces.append(inspect.getsource(builder))
    for name in (
        "C_ratio", "C_ratio_perp", "C_msbar", "C_msbar_gz", "C_hybrid",
        "C_ratio_gi", "C_hybrid_gi", "_atan_piece", "build_matching_matrix",
    ):
        fn = getattr(kernels, name, None)
        if fn is not None:
            pieces.append(inspect.getsource(fn))
    if not pieces:
        return inspect.getsource(kernels)
    return "\n\n".join(pieces)


def _strip_html(raw: str) -> str:
    """Crude HTML -> text so an arXiv HTML page is usable as LLM context."""
    no_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", no_scripts)
    return re.sub(r"[ \t\f\v]+", " ", html.unescape(text))


def _fetch_arxiv_source(arxiv_id: str) -> str | None:
    """Download the arXiv LaTeX e-print source and return its ``.tex`` text.

    The ``e-print`` endpoint returns a gzipped tar of the LaTeX source (sometimes a
    single gzipped ``.tex``). Extracting the ``.tex`` files gives the LLM the raw
    ``\\begin{equation}`` math -- the plus-prescription notation survives intact,
    unlike the HTML mirrors which mangle the formulas. Best-effort: any failure
    returns ``None``. The largest ``.tex`` (usually the main manuscript, where the
    matching coefficients live) is placed first so it survives truncation.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lamet-agent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except (TimeoutError, urllib.error.URLError, ssl.SSLError, ValueError):
        return None

    texts: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                texts.append(handle.read().decode("utf-8", errors="replace"))
    except (tarfile.TarError, OSError, EOFError):
        # Not a tar: try a single gzipped member, else treat the bytes as plain text.
        try:
            texts.append(gzip.decompress(raw).decode("utf-8", errors="replace"))
        except (OSError, EOFError):
            try:
                texts.append(raw.decode("utf-8", errors="replace"))
            except UnicodeDecodeError:
                return None

    texts = [t for t in texts if t.strip()]
    if not texts:
        return None
    texts.sort(key=len, reverse=True)
    return "\n\n".join(texts)


def _local_paper_path(arxiv_id: str) -> str | None:
    """Path to a local copy of *this* paper, from ``LAMET_FORMULA_PAPER_PATH_<arxiv_id>``.

    The variable is per paper (dots in the id become underscores, e.g.
    ``LAMET_FORMULA_PAPER_PATH_2412_20461``) precisely because one run can match several
    jobs with kernels from different papers -- a single global path would silently feed
    the wrong paper to every one of them.
    """
    return os.environ.get(f"LAMET_FORMULA_PAPER_PATH_{arxiv_id.replace('.', '_')}")


def _fetch_paper_text(paper_arxiv_id: str, *, max_chars: int = 80_000) -> str | None:
    """Return the paper text (local copy preferred, else arXiv LaTeX source), or None.

    ``paper_arxiv_id`` comes from the kernel the manifest selected (its
    ``@kernel_reference`` tag), so each kernel fetches its own paper -- nothing here
    knows or assumes a particular one. A local copy wins when
    ``LAMET_FORMULA_PAPER_PATH_<arxiv_id>`` points at a ``.txt``/``.md``/``.tex``/HTML
    file; otherwise the arXiv LaTeX e-print source is fetched so the LLM reads the real
    equations (the HTML mirrors are a last-resort fallback -- their math is mangled).
    The fetch is best-effort: any failure, or an untagged kernel, returns ``None`` and
    the formula is then generated from the kernel code alone.
    """
    if not paper_arxiv_id:
        return None  # untagged kernel: no paper to fetch, and none to invent
    arxiv_id = paper_arxiv_id
    local = _local_paper_path(arxiv_id)
    cache_key = local or f"arxiv:{arxiv_id}"
    if cache_key in _PAPER_CACHE:
        return _PAPER_CACHE[cache_key]

    text: str | None = None
    if local:
        candidate = Path(local).expanduser()
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8", errors="replace")
            text = raw if candidate.suffix.lower() in {".txt", ".md", ".tex"} else _strip_html(raw)
    if text is None:
        # Preferred: the LaTeX e-print source (clean math). Fall back to HTML only
        # if the source is unreachable.
        text = _fetch_arxiv_source(arxiv_id)
    if text is None:
        for url in (
            f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
            f"https://ar5iv.org/abs/{arxiv_id}",
            f"https://arxiv.org/abs/{arxiv_id}",
        ):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "lamet-agent/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = _strip_html(resp.read().decode("utf-8", errors="replace"))
                break
            except (TimeoutError, urllib.error.URLError, ssl.SSLError, ValueError):
                continue

    if text is not None:
        text = text.strip()[:max_chars] or None
    _PAPER_CACHE[cache_key] = text
    return text


def _formula_prompt(
    operator: str,
    scheme: str,
    language: str,
    *,
    source: str,
    paper_text: str | None,
    paper_arxiv_id: str,
    equations: str,
) -> str:
    lang_line = (
        "Write the prose in Simplified Chinese." if language == "zh" else "Write the prose in English."
    )
    paper_block = (
        f"LaTeX source of the paper (arXiv:{paper_arxiv_id}). It is the authority for the "
        "NOTATION: copy its symbols and, in particular, its exact plus-prescription convention "
        "for the matching coefficient verbatim.\n\"\"\"\n" + paper_text + "\n\"\"\"\n\n"
        if paper_text
        else "No paper text was available; rely on the code below as the source of truth and use "
        "the paper's $[\\,g(\\xi)\\,]^{D}_{+(1)}$ plus-prescription convention (subtraction point "
        "$\\xi=1$, domain $D$ in the superscript).\n\n"
    )
    # The kernel is tagged with the exact equations it transcribes, so point the model
    # at them instead of making it search the paper for the right coefficient.
    equation_line = (
        f"The kernel implements {equations} of that paper -- document that coefficient.\n\n"
        if equations
        else ""
    )
    return (
        "You are documenting one stage of a LaMET lattice-QCD analysis. Produce a Markdown "
        f"fragment giving the explicit matching coefficient for the `{operator}` operator "
        f"in the `{scheme}` scheme, exactly as the paper presents it.\n\n"
        f"{equation_line}"
        f"{paper_block}"
        "The number in the report was produced by this exact Python code -- it is the single "
        "source of truth for WHICH terms are present. Read it together with the paper and write "
        "the closed-form coefficient it implements: the splitting function, the logs, the "
        "arctan/arctanh branch, and any scheme-specific finite correction. If the paper and the "
        "code disagree on a term, follow the code; but for NOTATION always follow the paper.\n"
        f"```python\n{source}\n```\n\n"
        "Requirements:\n"
        "- Use $...$ for inline math and $$...$$ for display equations (KaTeX/MathJax).\n"
        "- Define notation once: $\\xi=x/y$ and $L=\\ln(4y^2P_z^2/\\mu^2)$.\n"
        "- The coefficient has a plus-prescription at $\\xi=1$ (the code restores it by making "
        "each $y$-column integrate to zero). Reproduce it using the paper's EXACT "
        "plus-prescription notation, copying the bracket structure verbatim from the LaTeX "
        "above: the paper writes $[\\,g(\\xi)\\,]^{D}_{+(x_0)}$ where the subscript $+(x_0)$ marks "
        "the subtraction point ($x_0=1$, i.e. $+(1)$) and the superscript $D$ marks the domain. "
        "Keep that subscript/superscript placement precisely -- do NOT move the $(1)$ into the "
        "superscript or drop the domain. The paper splits the coefficient into more than one "
        "plus-bracket over different domains (e.g. $[0,1]$ and $(-\\infty,\\infty)$): reproduce "
        "exactly that split, and include the paper's definition of $[g]^{D}_{+(x_0)}$ plus any "
        "$\\delta(1-\\xi)$ term.\n"
        "- State the explicit regular coefficient and any scheme-specific correction.\n"
        "- Be concise (a few sentences plus the equations); no headings, no preamble like "
        "'Here is'. Output only the Markdown fragment.\n"
        f"- {lang_line}"
    )


def _llm_kernel_formula(kernel_id: str, *, language: str, llm: FormulaLlm) -> tuple[str, bool]:
    """Generate the explicit kernel coefficient with an LLM, returning ``(md, paper_used)``.

    The model reads the LaTeX of the paper the kernel is tagged with (when reachable)
    together with the exact ``kernels.py`` code that produced the number, and writes
    the closed form using the paper's own plus-prescription notation. The code is
    authoritative for which terms are present; the paper is authoritative for the
    notation. The report stores no hand-written formula; this raises on any LLM
    failure (formula generation is required, no offline fallback). The boolean
    reports whether the paper text actually reached the prompt.
    """
    operator, scheme = _parse_kernel_id(kernel_id)
    paper_arxiv_id, equations = _kernel_reference(kernel_id)
    cache_key = (kernel_id, scheme, language)
    if cache_key in _FORMULA_CACHE:
        return _FORMULA_CACHE[cache_key]

    backend, provider, api_key, model_name, base_url = llm.resolved()
    source = _kernel_source(kernel_id)
    paper_text = _fetch_paper_text(paper_arxiv_id)
    prompt = _formula_prompt(
        operator,
        scheme,
        language,
        source=source,
        paper_text=paper_text,
        paper_arxiv_id=paper_arxiv_id,
        equations=equations,
    )
    # Reuse the shared LLM client (retries + error handling live in core.llm) instead
    # of a second hand-rolled HTTP call. The backend follows the run's --backend; the
    # api-only fields are empty strings under codex and ignored by that backend.
    text = request_llm_text(
        backend=backend,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        messages=[{"role": "user", "content": prompt}],
    ).strip()
    if not text:
        raise RuntimeError("LLM returned an empty matching formula.")
    result = (text, paper_text is not None)
    _FORMULA_CACHE[cache_key] = result
    return result


def _matching_formula_text(data: dict[str, Any], *, language: str, llm: FormulaLlm) -> str:
    kernel_id = str(data.get("kernel_id", ""))
    # Provenance follows the manifest's kernel: whichever kernel_id was selected, its
    # @kernel_reference in kernels.py names the paper and equations cited here.
    paper_arxiv_id, equations = _kernel_reference(kernel_id)
    if paper_arxiv_id:
        reference = f"arXiv:{paper_arxiv_id} {equations}".strip()
    else:
        reference = "匹配核未标注出处" if language == "zh" else "The kernel declares no paper reference"
    formula = (
        r"f(x,\mu)=\int\frac{dy}{|y|}\,C^{-1}\!\left(\frac{x}{y},\frac{\mu}{yP_z}\right)"
        r"\tilde f\!\left(y,P_z\right),"
    )
    discrete = r"f_i=\sum_j K_{ij}\,\tilde f_j,\qquad K=\text{(nx, ny) matching matrix}."
    # The explicit coefficient is generated at report time by an LLM that reads
    # the source paper together with the kernels.py code which produced the number
    # (no formula is hardcoded). A short note records the provenance so a reader
    # knows it was machine-derived and from which sources.
    generated, paper_used = _llm_kernel_formula(kernel_id, language=language, llm=llm)
    if language == "zh":
        source_zh = f"文章 arXiv:{paper_arxiv_id} 与 `kernels.py` 实现" if paper_used else "`kernels.py` 实现"
        note = f"（以下解析形式由模型阅读{source_zh}后生成）\n\n"
    else:
        source_en = (
            f"arXiv:{paper_arxiv_id} together with the `kernels.py` implementation"
            if paper_used
            else "the `kernels.py` implementation"
        )
        note = f"(the explicit form below was generated by the model from {source_en})\n\n"
    explicit = note + generated
    if language == "zh":
        return (
            f"{reference}。光锥 PDF 由 quasi-PDF 经匹配核反卷积得到：\n\n"
            f"$$\n{formula}\n$$\n\n"
            "离散化后即矩阵乘法（本阶段对每个重采样样本独立施加，再重建统计量）：\n\n"
            f"$$\n{discrete}\n$$\n\n"
            "其中 LO 部分为单位阵，匹配修正的解析形式为：\n\n"
            f"{explicit}"
        )
    return (
        f"{reference}. The light-cone PDF is obtained from the quasi-PDF by inverting the matching kernel:\n\n"
        f"$$\n{formula}\n$$\n\n"
        "After discretization this is a matrix product (applied to every resampling sample independently, then the statistics are rebuilt):\n\n"
        f"$$\n{discrete}\n$$\n\n"
        "Here the LO part is the identity, and the explicit matching correction is:\n\n"
        f"{explicit}"
    )


def _scheme_explanation(data: dict[str, Any], *, language: str) -> list[str]:
    kernel_id = str(data.get("kernel_id", ""))
    _operator, scheme = _parse_kernel_id(kernel_id)
    if language == "zh":
        notes = {
            "msbar": "MSbar 方案在裸 ratio 系数上加上有限的 MSbar 转换项（Eq. 2.14）。",
            "ratio": "ratio 方案直接使用裸的正则系数 $C_r$（Eq. 2.16），不含额外有限项。",
            "hybrid": "hybrid 方案在 ratio 系数上加上 Wilson 线的正弦积分修正，依赖 $z_sP_z$（Eq. 2.19-2.20）。",
        }
        body = notes.get(scheme, "未识别的匹配方案，仅记录所选 kernel_id。")
        return ["## 匹配方案", body]
    notes = {
        "msbar": "The MSbar scheme adds a finite MSbar conversion on top of the bare ratio coefficient (Eq. 2.14).",
        "ratio": "The ratio scheme uses the bare regular coefficient $C_r$ directly (Eq. 2.16) with no extra finite terms.",
        "hybrid": "The hybrid scheme adds a Wilson-line sine-integral correction to the ratio coefficient and depends on $z_sP_z$ (Eqs. 2.19-2.20).",
    }
    body = notes.get(scheme, "Unrecognized matching scheme; only the selected kernel_id is recorded.")
    return ["## Matching Scheme", body]


def _diagnostics(data: dict[str, Any], *, language: str) -> list[str]:
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    quasi_mean = np.asarray(data.get("quasi_mean", []), dtype=float)
    lc_mean = np.asarray(data.get("lightcone_mean", []), dtype=float)
    lines: list[str] = []

    if x_grid.size >= 2 and quasi_mean.size == x_grid.size and lc_mean.size == x_grid.size:
        quasi_norm = _trapz_norm(x_grid, quasi_mean)
        lc_norm = _trapz_norm(x_grid, lc_mean)
        rel = abs(lc_norm - quasi_norm) / abs(quasi_norm) if quasi_norm not in (0.0, float("nan")) else float("nan")
        if language == "zh":
            lines.extend(
                [
                    f"- quasi-PDF 归一 $\\int f\\,dx={_fmt(quasi_norm)}$；光锥 PDF 归一 $\\int f\\,dx={_fmt(lc_norm)}$。",
                    f"- 归一相对变化 {_fmt(100 * rel)}%。NLO 匹配是微扰修正，应当接近守恒。",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Quasi-PDF norm $\\int f\\,dx={_fmt(quasi_norm)}$; light-cone norm $\\int f\\,dx={_fmt(lc_norm)}$.",
                    f"- Relative norm change {_fmt(100 * rel)}%. NLO matching is a perturbative correction and should nearly preserve the norm.",
                ]
            )
    else:
        lines.append("- 无可用的匹配诊断。" if language == "zh" else "- Matching diagnostics were not available.")
    return lines


def _figure_block(artifacts: dict[str, Any], *, language: str) -> list[str]:
    heading = "## 图像与可视化评估" if language == "zh" else "## Figures and Visual Assessment"
    label = "quasi 与光锥 PDF 对比图" if language == "zh" else "Quasi vs light-cone comparison"
    image_value = artifacts.get("matched_plot_image")
    pdf_value = artifacts.get("matched_plot")
    lines = [heading, "", f"### {label}"]
    if image_value:
        lines.append(f"![{label}]({image_value})")
        if pdf_value:
            lines.append("")
            lines.append(f"[{label}（PDF 矢量图）]({pdf_value})" if language == "zh" else f"[{label} (PDF, vector)]({pdf_value})")
    elif pdf_value:
        lines.append(f"[{label} (PDF)]({pdf_value})" if language == "en" else f"[{label}（PDF）]({pdf_value})")
    else:
        lines.append("未生成。" if language == "zh" else "Not available.")
    return lines


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |"
    lines = [header, "|---|---|"]
    for key in MATCHING_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = MATCHING_ARTIFACT_DESCRIPTIONS[key][1 if language == "zh" else 0]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def build_matching_report_markdown(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    language: str = "en",
    llm: FormulaLlm,
) -> str:
    artifacts = artifacts or {}
    kernel_id = str(result.get("kernel_id", "not recorded"))
    operator, scheme = _parse_kernel_id(kernel_id)
    op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
    scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")

    if language == "zh":
        lines = [
            "# 微扰匹配分析报告",
            "",
            "## 摘要",
            f"本报告总结将 `{kernel_id}`（{op_zh}）quasi-PDF 经 `{scheme_en}` 方案 NLO 匹配核转换为光锥 PDF 的过程。",
            "",
            "## 分析设置",
            *_settings_table(result, language="zh"),
            "",
            "### 条目解释",
            *_field_definitions(language="zh"),
            "",
            "## 匹配公式",
            _matching_formula_text(result, language="zh", llm=llm),
            "",
            *_scheme_explanation(result, language="zh"),
            "",
            "## 诊断与一致性检查",
            *_diagnostics(result, language="zh"),
            "",
            *_figure_block(artifacts, language="zh"),
            "",
            "## 输出文件",
            *_outputs_table(artifacts, language="zh"),
        ]
    else:
        lines = [
            "# Perturbative Matching Analysis Report",
            "",
            "## Abstract",
            f"This report summarizes converting the `{kernel_id}` ({op_en}) quasi-PDF into the light-cone PDF using the `{scheme_en}`-scheme NLO matching kernel.",
            "",
            "## Analysis Setup",
            *_settings_table(result, language="en"),
            "",
            "### Field Definitions",
            *_field_definitions(language="en"),
            "",
            "## Matching Formula",
            _matching_formula_text(result, language="en", llm=llm),
            "",
            *_scheme_explanation(result, language="en"),
            "",
            "## Diagnostics and Consistency Checks",
            *_diagnostics(result, language="en"),
            "",
            *_figure_block(artifacts, language="en"),
            "",
            "## Output Artifacts",
            *_outputs_table(artifacts, language="en"),
        ]
    return "\n".join(lines) + "\n"


def write_matching_report(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None,
    path: str | Path,
    report_language: str = "en",
    llm: FormulaLlm,
) -> dict[str, Path]:
    """Write one matching report and return its path."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = markdown_artifact_paths(
        artifacts,
        base_dir=target.parent,
        path_keys=MATCHING_ARTIFACT_ORDER,
    )
    target.write_text(
        build_matching_report_markdown(result=result, artifacts=report_artifacts, language=language, llm=llm),
        encoding="utf-8",
    )
    return {"report": target}


def write_matching_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
    llm: FormulaLlm,
) -> dict[str, Path]:
    """Write one report summarizing all matching jobs in a stage."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = jobs[0]["result"]
    for language, target in ((language, target),):
        kernel_id = str(first.get("kernel_id", "not recorded"))
        operator, scheme = _parse_kernel_id(kernel_id)
        op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
        scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")
        lines = [
            "# Perturbative Matching Stage Report" if language == "en" else "# 微扰匹配阶段报告",
            "",
            f"This report summarizes all perturbative-matching jobs for `{kernel_id}` ({op_en}) using the `{scheme_en}` scheme."
            if language == "en"
            else f"本报告汇总 `{kernel_id}`（{op_zh}）在 `{scheme_en}` 方案下的所有动量匹配。",
            "",
            "## Job Summary" if language == "en" else "## Job 汇总",
            "| job | kernel | $P_z$ | output | plot |"
            if language == "en"
            else "| job | kernel | $P_z$ | 输出 | 图像 |",
            "|---|---|---:|---|---|",
        ]
        for item in jobs:
            result = item["result"]
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=MATCHING_ARTIFACT_ORDER,
            )
            lines.append(
                f"| `{item['job_id']}` | {result.get('kernel_id', 'n/a')} | "
                f"{_fmt(result.get('momentum_gev'))} | "
                f"{artifacts.get('lightcone_artifact', 'n/a')} | "
                f"{artifacts.get('matched_plot', 'n/a')} |"
            )
        setting_data = {**first, "momentum_gev": "see per-momentum table" if language == "en" else "见下方动量表"}
        lines.extend(
            [
                "",
                "## Analysis Setup" if language == "en" else "## 分析设置",
                *_settings_table(setting_data, language=language),
                "",
                "### Field Definitions" if language == "en" else "### 条目解释",
                *_field_definitions(language=language),
                "",
                "## Matching Formula" if language == "en" else "## 匹配公式",
                _matching_formula_text(first, language=language, llm=llm),
                "",
                *_scheme_explanation(first, language=language),
                "",
                "## Diagnostics and Consistency Checks" if language == "en" else "## 诊断与一致性检查",
                "| job | $P_z$ | quasi norm | matched norm | norm change |"
                if language == "en"
                else "| job | $P_z$ | quasi 归一 | 匹配后归一 | 归一变化 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in jobs:
            result = item["result"]
            x_grid = np.asarray(result.get("x_grid", []), dtype=float)
            quasi_mean = np.asarray(result.get("quasi_mean", []), dtype=float)
            lc_mean = np.asarray(result.get("lightcone_mean", []), dtype=float)
            if x_grid.size >= 2 and quasi_mean.size == x_grid.size and lc_mean.size == x_grid.size:
                quasi_norm = _trapz_norm(x_grid, quasi_mean)
                lc_norm = _trapz_norm(x_grid, lc_mean)
                rel = abs(lc_norm - quasi_norm) / abs(quasi_norm) if quasi_norm != 0.0 else float("nan")
                lines.append(
                    f"| `{item['job_id']}` | {_fmt(result.get('momentum_gev'))} | {_fmt(quasi_norm)} | "
                    f"{_fmt(lc_norm)} | {_fmt(100 * rel)}% |"
                )
            else:
                lines.append(f"| `{item['job_id']}` | {_fmt(result.get('momentum_gev'))} | n/a | n/a | n/a |")
        lines.extend(
            [
                "",
                "The table compares the quasi-PDF and matched light-cone PDF norm for each momentum. Moderate norm changes are expected from the NLO kernel; a very large norm change usually indicates an x-grid or momentum-convention issue."
                if language == "en"
                else "上表逐动量比较 quasi-PDF 与匹配后光锥 PDF 的归一。NLO 匹配会带来有限修正；若归一变化很大，通常需要检查 x 网格或动量约定。",
                "",
                "## Figures and Visual Assessment" if language == "en" else "## 图像与可视化评估",
            ]
        )
        for item in jobs:
            result = item["result"]
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=MATCHING_ARTIFACT_ORDER,
            )
            image = artifacts.get("matched_plot_image")
            plot = artifacts.get("matched_plot")
            label = "Quasi vs light-cone comparison" if language == "en" else "quasi 与光锥 PDF 对比图"
            lines.extend(["", f"### `{item['job_id']}`: $P_z={_fmt(result.get('momentum_gev'))}$ GeV"])
            if image:
                lines.append(f"![{label}]({image})")
                if plot:
                    lines.append("")
                    lines.append(
                        f"[{label} (PDF, vector)]({plot})"
                        if language == "en"
                        else f"[{label}（PDF 矢量图）]({plot})"
                    )
            elif plot:
                lines.append(
                    f"[{label} (PDF)]({plot})"
                    if language == "en"
                    else f"[{label}（PDF）]({plot})"
                )
            else:
                lines.append("未生成。" if language == "zh" else "Not available.")
        lines.extend(
            [
                "",
                "## Output Artifacts" if language == "en" else "## 输出文件",
                "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |",
                "|---|---|",
            ]
        )
        for item in jobs:
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=MATCHING_ARTIFACT_ORDER,
            )
            for key in MATCHING_ARTIFACT_ORDER:
                value = artifacts.get(key)
                if value:
                    desc = MATCHING_ARTIFACT_DESCRIPTIONS[key]
                    lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc[1 if language == 'zh' else 0]} |")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": target}
