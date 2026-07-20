"""Reporting helpers for extrapolation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gvar
import numpy as np
import xarray as xr

from lamet_agent.core.reporting import markdown_artifact_paths, resolve_report_target


def write_extrapolation_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write a compact stage-level extrapolation report."""
    target, language = resolve_report_target(Path(path), report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    zh = language == "zh"
    lines = ["# 外推阶段报告" if zh else "# Extrapolation Report", ""]
    for record in jobs:
        result = record.get("result", {})
        raw_artifacts = record.get("artifacts", {})
        artifacts = markdown_artifact_paths(
            raw_artifacts,
            base_dir=target.parent,
            path_keys=(
                "extrapolated_artifact",
                "fit_info_artifact",
                "extrapolated_plot",
                "extrapolated_plot_image",
                "adep_plot",
                "adep_plot_image",
                "pdep_plot",
                "pdep_plot_image",
            ),
        )
        formula = (
            rf"$h(x,p_z,a)=h(x,\infty,0)+\sum_{{i={result.get('lowest_lattice_spacing_order')}}}^2 c_{{a,i}}a^i"
            rf"+\sum_{{j=1}}^{{{int(result.get('highest_momentum_order', 2)) // 2}}}\frac{{c_{{p,j}}}}{{p_z^{{2j}}}}$"
        )
        pdep_text = ", ".join(f"{float(value):.2f}" for value in result.get("pdep_gev", [])) or "not set"
        a_order = result.get("lowest_lattice_spacing_order")
        p_order = result.get("highest_momentum_order")
        fit_rows = []
        artifact_path = raw_artifacts.get("extrapolated_artifact")
        if artifact_path:
            with xr.open_dataset(artifact_path) as dataset:
                x = np.asarray(dataset.coords.get("x", []), dtype=float)
                indices = [index for index in (0, len(x) // 2, len(x) - 1) if 0 <= index < len(x)]
                indices = list(dict.fromkeys(indices))
                for name in dataset.data_vars:
                    if not (name.startswith("c_a_") or name.startswith("c_p_")):
                        continue
                    label = rf"$c_{{a,{name.removeprefix('c_a_')}}}$" if name.startswith("c_a_") else rf"$c_{{p,{name.removeprefix('c_p_')}}}$"
                    values = np.asarray(dataset[name].values, dtype=float)
                    for index in indices:
                        samples = values[:, index]
                        fit_rows.append((label, float(x[index]), str(gvar.gvar(float(np.mean(samples)), float(np.std(samples, ddof=1))))))
        if zh:
            lines.extend(
                [
                    f"## {record.get('job_id')}",
                    "",
                    "本报告汇总 perturbative matching 输出的光锥分布，并对格距和动量依赖进行联合或单变量外推。",
                    "",
                    "## 外推形式",
                    "",
                    formula,
                    "",
                    "## Job 汇总",
                    "| job | 模式 | 输入数 | 参数数 | $\\chi^2/\\mathrm{dof}$ | 输出 |",
                    "|---|---|---:|---:|---:|---|",
                    f"| `{record.get('job_id')}` | {result.get('mode')} | {result.get('n_inputs')} | {result.get('n_parameters')} | {float(result.get('chi2_dof', 0.0)):.3g} | {Path(str(artifacts.get('extrapolated_artifact'))).name if artifacts.get('extrapolated_artifact') else 'n/a'} |",
                    "",
                    "## 分析设置",
                    "| 条目 | 数值或设置 | 解释 |",
                    "|---|---|---|",
                    f"| `lowest_lattice_spacing_order` | {a_order} | 格距修正项的最低幂次；代码使用 $a^i$，并从该值拟合到 $i=2$。 |",
                    f"| `highest_momentum_order` | {p_order} | 有限动量修正的最高阶；代码使用 $1/p_z^{{2j}}$，最高到该输入阶数。 |",
                    f"| `pdep_gev` | {pdep_text} | 额外动量依赖图中指定的 $p_z$（GeV）取值；缺省时不生成 `extrapolate_pdep` 图。 |",
                    "",
                    "## 拟合模型参数表",
                    "| 参数 | $x$ | 拟合结果 |",
                    "|---|---|---|",
                    *(f"| {name} | {x_value:.4g} | {value} |" for name, x_value, value in fit_rows),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"## {record.get('job_id')}",
                    "",
                    "This report summarizes the light-cone distributions from perturbative matching and extrapolates their lattice-spacing and momentum dependence.",
                    "",
                    "## Extrapolation Form",
                    "",
                    formula,
                    "",
                    "## Job Summary",
                    "| job | mode | inputs | parameters | $\\chi^2/\\mathrm{dof}$ | output |",
                    "|---|---|---:|---:|---:|---|",
                    f"| `{record.get('job_id')}` | {result.get('mode')} | {result.get('n_inputs')} | {result.get('n_parameters')} | {float(result.get('chi2_dof', 0.0)):.3g} | {Path(str(artifacts.get('extrapolated_artifact'))).name if artifacts.get('extrapolated_artifact') else 'n/a'} |",
                    "",
                    "## Analysis Settings",
                    "| Item | Value or setting | Explanation |",
                    "|---|---|---|",
                    f"| `lowest_lattice_spacing_order` | {a_order} | Lowest power for lattice-spacing corrections; the code fits $a^i$ terms from this value through $i=2$. |",
                    f"| `highest_momentum_order` | {p_order} | Highest finite-momentum order; the code fits $1/p_z^{{2j}}$ terms up to this input order. |",
                    f"| `pdep_gev` | {pdep_text} | Requested $p_z$ values in GeV for the extra momentum-dependence figure; if unset, `extrapolate_pdep` is not generated. |",
                    "",
                    "## Fit Model Parameter Table",
                    "| Parameter | $x$ | Fit result |",
                    "|---|---|---|",
                    *(f"| {name} | {x_value:.4g} | {value} |" for name, x_value, value in fit_rows),
                    "",
                ]
            )
        if result.get("warning"):
            lines.append(("警告：" if zh else "Warning: ") + str(result["warning"]))
            lines.append("")
        if result.get("use_lattice_spacing_dependence") and not result.get("use_momentum_dependence"):
            lines.append("由于输入为多系综单动量，本阶段只能生成格距依赖图。" if zh else "The inputs contain multiple ensembles at a single momentum, so only the lattice-spacing-dependence figure can be generated.")
            lines.append("")
        if result.get("use_momentum_dependence") and not result.get("use_lattice_spacing_dependence"):
            lines.append("由于输入为单系综多动量，本阶段只能生成动量依赖图；若未设置 `pdep_gev`，则不生成该图。" if zh else "The inputs contain one ensemble at multiple momenta, so only the momentum-dependence figure can be generated; it is omitted unless `pdep_gev` is set.")
            lines.append("")
        title = "外推结果" if zh else "Extrapolated Result"
        if artifacts.get("extrapolated_plot_image"):
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['extrapolated_plot_image']})", ""])
        if artifacts.get("adep_plot_image"):
            title = "格距依赖图" if zh else "Lattice-Spacing Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['adep_plot_image']})", ""])
        if artifacts.get("pdep_plot_image"):
            title = "动量依赖图" if zh else "Momentum Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['pdep_plot_image']})", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return {"report": target}
