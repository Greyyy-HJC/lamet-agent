"""Markdown reporting helpers for the correlator-analysis stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


CORRELATOR_ARTIFACT_DESCRIPTIONS = {
    "bare_artifact": ("Bare matrix element samples (EnsembleData NetCDF)", "裸矩阵元样本（EnsembleData NetCDF）"),
    "summary_plot": ("PDF plot of the bare matrix element versus Wilson-line length", "裸矩阵元随 Wilson 线长度变化的 PDF 图"),
    "summary_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的裸矩阵元 SVG 图"),
    "tuning_log": ("Window tuning and sample-average fit-quality log", "窗口选择和样本平均拟合质量日志"),
    "sample_log": ("Per-sample and per-z fit-quality log", "逐样本、逐 z 拟合质量日志"),
}

CORRELATOR_ARTIFACT_ORDER = ("bare_artifact", "summary_plot", "summary_plot_image", "tuning_log", "sample_log")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not set"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def _fmt_list(values: Any, *, max_items: int = 8, digits: int = 4) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    items = [_fmt(item, digits=digits) for item in flat[:max_items]]
    suffix = ", ..." if flat.size > max_items else ""
    return "[" + ", ".join(items) + suffix + "]"


def _cn_report_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_CN{path.suffix or '.md'}")


def _report_target(path: Path, report_language: str) -> tuple[Path, str]:
    language = report_language.lower()
    if language == "en":
        return path, "en"
    if language == "ch":
        return _cn_report_path(path), "zh"
    raise ValueError("report_language must be 'en' or 'ch'")


def _md_path(value: Any, *, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return os.path.relpath(path, base_dir)
    return str(value)


def _markdown_artifacts(artifacts: dict[str, Any] | None, *, base_dir: Path) -> dict[str, Any]:
    output = dict(artifacts or {})
    for key in CORRELATOR_ARTIFACT_ORDER:
        if key in output:
            output[key] = _md_path(output[key], base_dir=base_dir)
    for key in ("sample0_pt2_plots", "sample0_fit_plots"):
        if key in output:
            output[key] = [_md_path(path, base_dir=base_dir) for path in (output.get(key) or []) if path]
    return output


def _job_settings_table(result: dict[str, Any], *, language: str) -> list[str]:
    if language == "zh":
        rows = [
            ("拟合形式", f"`{result.get('fitting_form', 'not recorded')}`"),
            ("拟合对象", f"`{result.get('fit_scope', 'not recorded')}`"),
            ("拟合策略", f"`{result.get('fit_strategy', 'not recorded')}`"),
            ("拟合模式", f"`{result.get('fit_mode', 'not recorded')}`"),
            ("Model average", f"`{result.get('model_average', 'not recorded')}`"),
            ("选择规则", f"`{result.get('selection_rule', 'not recorded')}`"),
            ("重采样", f"`{result.get('resample_mode', 'not recorded')}`，共 {result.get('n_samples', 'n/a')} 个样本"),
            ("z 网格", _fmt_list(result.get("z_values", []))),
            ("调参 z", _fmt(result.get("tune_z"))),
            ("correlator_rescale", _fmt(result.get("correlator_rescale"))),
        ]
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Fitting form", f"`{result.get('fitting_form', 'not recorded')}`"),
            ("Fit scope", f"`{result.get('fit_scope', 'not recorded')}`"),
            ("Fit strategy", f"`{result.get('fit_strategy', 'not recorded')}`"),
            ("Fit mode", f"`{result.get('fit_mode', 'not recorded')}`"),
            ("Model average", f"`{result.get('model_average', 'not recorded')}`"),
            ("Selection rule", f"`{result.get('selection_rule', 'not recorded')}`"),
            ("Resampling", f"`{result.get('resample_mode', 'not recorded')}` with {result.get('n_samples', 'n/a')} samples"),
            ("z grid", _fmt_list(result.get("z_values", []))),
            ("Tuning z", _fmt(result.get("tune_z"))),
            ("correlator_rescale", _fmt(result.get("correlator_rescale"))),
        ]
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _window_text(result: dict[str, Any], *, language: str) -> list[str]:
    specs = result.get("shared_window_specs")
    if not specs:
        return ["No shared window metadata was recorded." if language == "en" else "未记录共享窗口信息。"]
    if not isinstance(specs, list):
        specs = [specs]
    lines = [
        "| scope | strategy | nstate | pt2 window | pt3 window | n_data | n_params |"
        if language == "en"
        else "| 拟合对象 | 策略 | nstate | 2pt 窗口 | 3pt 窗口 | n_data | n_params |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        pt2_window = spec.get("pt2_window", f"[{spec.get('tmin', 'n/a')},{spec.get('tmax', 'n/a')})")
        pt3_window = spec.get(
            "pt3_window",
            f"tsep={_fmt_list(spec.get('tsep_ls', []))}, tau_cut={spec.get('tau_cut', 'n/a')}",
        )
        lines.append(
            f"| `{spec.get('fit_scope', spec.get('scope', 'n/a'))}` | "
            f"`{spec.get('fit_strategy', spec.get('strategy', 'n/a'))}` | "
            f"{spec.get('nstate', 'n/a')} | "
            f"{pt2_window} | "
            f"{pt3_window} | "
            f"{spec.get('n_data', 'n/a')} | "
            f"{spec.get('n_params', 'n/a')} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _z_fit_table(result: dict[str, Any], *, language: str) -> list[str]:
    z_fits = result.get("z_fits") or []
    lines = [
        "| z | Q | chi2/dof | logGBF | failed samples | Re sys | Im sys |"
        if language == "en"
        else "| z | Q | chi2/dof | logGBF | 失败样本 | Re sys | Im sys |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in z_fits[:20]:
        if not isinstance(fit, dict):
            continue
        window = fit.get("window") if isinstance(fit.get("window"), dict) else {}
        lines.append(
            f"| {_fmt(fit.get('z'))} | {_fmt(fit.get('Q', window.get('Q')))} | "
            f"{_fmt(fit.get('chi2_dof', fit.get('chi2/DOF', window.get('chi2_dof'))))} | "
            f"{_fmt(fit.get('logGBF', window.get('logGBF')))} | {fit.get('n_failed_samples', 0)} | "
            f"{_fmt(fit.get('real_sys_sdev'))} | {_fmt(fit.get('imag_sys_sdev'))} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| Artifact | Description |" if language == "en" else "| 文件 | 说明 |"
    lines = [header, "|---|---|"]
    for key in CORRELATOR_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = CORRELATOR_ARTIFACT_DESCRIPTIONS[key][0 if language == "en" else 1]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def _diagnostic_plots(artifacts: dict[str, Any], *, language: str) -> list[str]:
    plots = list(artifacts.get("sample0_fit_plots", [])) + list(artifacts.get("sample0_pt2_plots", []))
    plots = [plot for plot in plots if plot and str(plot).endswith(".svg")]
    if not plots:
        return ["No sample-0 diagnostic SVGs were recorded." if language == "en" else "未记录 sample-0 诊断 SVG。"]
    lines = ["Sample-0 diagnostic SVGs:" if language == "en" else "Sample-0 诊断 SVG："]
    for start in range(0, len(plots), 4):
        row = plots[start : start + 4]
        lines.append("<table><tr>")
        for plot in row:
            lines.append(f'<td><img src="{plot}" alt="{Path(str(plot)).stem}" width="230"><br><code>{Path(str(plot)).stem}</code></td>')
        lines.append("</tr></table>")
    return lines


def build_correlator_stage_report_markdown(
    *,
    jobs: list[dict[str, Any]],
    base_dir: Path,
    language: str = "en",
) -> str:
    """Build one Markdown report for all correlator-analysis jobs."""
    title = "# Correlator Analysis Stage Report" if language == "en" else "# Correlator Analysis 阶段报告"
    intro = (
        "This report summarizes correlator fits that extract bare matrix elements from 2pt/3pt data."
        if language == "en"
        else "本报告汇总从 2pt/3pt 关联函数中提取裸矩阵元的拟合结果。"
    )
    lines = [
        title,
        "",
        intro,
        "",
        "## Job Summary" if language == "en" else "## Job 汇总",
        "| job | fit scope | strategy | output | plot |" if language == "en" else "| job | 拟合对象 | 策略 | 输出 | 图像 |",
        "|---|---|---|---|---|",
    ]
    markdown_jobs = []
    for item in jobs:
        result = item.get("result", {})
        artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=base_dir)
        markdown_jobs.append((item, result, artifacts))
        lines.append(
            f"| `{item['job_id']}` | `{result.get('fit_scope', 'n/a')}` | "
            f"`{result.get('fit_strategy', 'n/a')}` | "
            f"{artifacts.get('bare_artifact', 'n/a')} | {artifacts.get('summary_plot', 'n/a')} |"
        )

    for item, result, artifacts in markdown_jobs:
        lines.extend(
            [
                "",
                f"## `{item['job_id']}`",
                "",
                "### Fit Setup" if language == "en" else "### 拟合设置",
                *_job_settings_table(result, language=language),
                "",
                "### Shared Windows" if language == "en" else "### 共享窗口",
                *_window_text(result, language=language),
                "",
                "### Per-z Fit Summary" if language == "en" else "### 逐 z 拟合摘要",
                *_z_fit_table(result, language=language),
                "",
                "### fit_logs" if language == "en" else "### fit_logs",
                (
                    "The `fit_logs` directory contains split logs: the tuning log records window selection and sample-average fit quality, while the sample log records per-sample and per-z fit quality, including failures."
                    if language == "en"
                    else "`fit_logs` 目录包含拆分日志：tuning log 记录窗口选择和样本平均拟合质量；sample log 记录逐样本、逐 z 的拟合质量以及失败信息。"
                ),
                "",
                *_outputs_table(artifacts, language=language),
                "",
                "### Diagnostic SVGs" if language == "en" else "### 诊断 SVG",
                *_diagnostic_plots(artifacts, language=language),
                "",
                "### Summary Figure" if language == "en" else "### 总览图",
                (
                    f"![Bare matrix element summary]({artifacts.get('summary_plot_image')})"
                    if artifacts.get("summary_plot_image")
                    else ("Not available." if language == "en" else "未生成。")
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_correlator_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write one report summarizing all correlator-analysis jobs."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_correlator_stage_report_markdown(jobs=jobs, base_dir=target.parent, language=language),
        encoding="utf-8",
    )
    return {"report": target}
