"""Review-stage utilities built from existing stage reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from lamet_agent.core.llm import request_llm_text
from lamet_agent.manifest import AnalysisManifest

STAGE_REPORTS = {
    "correlator_analysis": ("ca_report.md", "ca_report_CN.md"),
    "renormalization": ("renorm_report.md", "renorm_report_CN.md"),
    "fourier_transform": ("ft_report.md", "ft_report_CN.md"),
    "perturbative_matching": ("matching_report.md", "matching_report_CN.md"),
    "extrapolation": ("extrapolation_report.md", "extrapolation_report_CN.md"),
}


def write_review_from_manifest(
    manifest: AnalysisManifest,
    *,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ask the configured LLM to write the final review from reports and NetCDF summaries."""
    artifacts_dir = Path(output_dir) if output_dir is not None else manifest.artifacts_directory
    review_dir = artifacts_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    language = "zh" if report_language.lower() == "ch" else "en"
    target = review_dir / ("review_CN.md" if language == "zh" else "review.md")
    materials = []
    stages = [stage for stage in STAGE_REPORTS if (artifacts_dir / stage).is_dir() or stage in manifest.metadata.stages]
    for stage in stages:
        en_name, zh_name = STAGE_REPORTS[stage]
        stage_dir = artifacts_dir / stage
        report_path = stage_dir / (zh_name if language == "zh" else en_name)
        if not report_path.exists() and language == "zh":
            report_path = stage_dir / en_name
        item: dict[str, Any] = {
            "stage": stage,
            "artifact_stage_dir": str(stage_dir),
            "report": str(report_path),
            "report_text": "",
            "netcdf": [],
            "svg": [],
        }
        if report_path.exists():
            item["report_text"] = report_path.read_text(encoding="utf-8")
        for path in sorted(stage_dir.glob("*.nc")):
            if path.name.endswith("_fit_info.nc"):
                continue
            with xr.open_dataset(path) as ds:
                name = next(iter(ds.data_vars))
                values = np.asarray(ds[name].values)
                if values.dtype.fields and {"r", "i"}.issubset(values.dtype.fields):
                    values = values["r"] + 1j * values["i"]
                mean = np.nanmean(values, axis=0) if values.ndim > 1 else values
                summary: dict[str, Any] = {
                    "file": path.name,
                    "variable": name,
                    "dims": dict(ds.sizes),
                    "coords": {
                        key: [float(np.nanmin(ds[key].values)), float(np.nanmax(ds[key].values)), int(len(ds[key].values))]
                        for key in ds.coords
                        if key in {"z", "x"}
                    },
                    "max_abs_mean": float(np.nanmax(np.abs(mean))),
                    "real_mean_range": [float(np.nanmin(np.real(mean))), float(np.nanmax(np.real(mean)))],
                }
                if np.iscomplexobj(mean):
                    summary["imag_mean_range"] = [float(np.nanmin(np.imag(mean))), float(np.nanmax(np.imag(mean)))]
            item["netcdf"].append(summary)
        item["svg"] = [
            {
                "markdown_path": os.path.relpath(path, review_dir),
                "stage_subpath": os.path.relpath(path, stage_dir),
                "absolute_path": str(path),
            }
            for path in sorted(stage_dir.rglob("*.svg"))[:80]
        ]
        materials.append(item)
    if language == "zh":
        system = "你是 lattice QCD 和 LaMET 专家。只根据用户提供的 stage reports、NetCDF 摘要、SVG 文件清单和 manifest 写详细科学综述，不编造未给出的数值；当设置或输出不符合真实 LaMET 场景时，必须给出可执行的 manifest 修改建议。"
        user = (
            "请直接生成完整的 `review_CN.md` 正文。请按 Stage materials 给出的顺序写；这些 stage 来自 `root_directory/artifacts_directory/<stage>` 中实际存在的 stage 子目录以及 manifest 中声明的 stage；例如 correlator_analysis 的诊断图也会从 `correlator_analysis/fit_logs` 子目录收集。"
            "每个有材料的 stage 写一个二级标题章节，并包含 `Summary`、`Key figure`、`Diagnostics and manifest changes` 三个小节。"
            "`Summary` 要详细总结该 stage 的输入、输出、NetCDF 数值范围、物理含义、与上下游 stage 的关系。"
            "`Key figure` 中请你从该 stage 的 `svg` 列表里选择一张最能代表该 stage 质量或物理结果的 SVG，用 Markdown 图片语法嵌入；必须原样复制该图条目里的 `markdown_path`，写成 `![说明](markdown_path)`，不要自己拼路径、不要只写文件名、不要使用 `absolute_path` 作为 Markdown 链接。图下用一段详细文字解释为什么选这张图、它应如何辅助判断该 stage；如果没有 SVG，明确说明未生成可嵌入 SVG。"
            "`Diagnostics and manifest changes` 要判断该 stage 是否自洽，尤其检查是否符合真实 LaMET 分析场景；如果不理想，给出具体、可执行的 `.json` manifest 修改建议。"
            "修改建议必须引用真实 manifest 路径和值，例如 `stages.<stage>.defaults.<key>`、`stages.<stage>.jobs[].params.<key>`、`inputs.kernels[].kernel_parameters.<key>`，并说明建议值或取值范围以及理由。"
            "优先讨论这些可调参数：correlator 的 `pt2_windows`、`pt3_tau_cuts`、`nstate`、`fit_scope`、`fit_strategy`、`prior_width`、`svdcut`；renormalization 的 `scheme_parameters.zs_fm`、`m0_gev`、`delta_m_gev`；fourier 的 `scheme_scan.zmin_values`、`zmax_values`、`z_ext_max`、`smooth`、`order`、`posterior_prior_error_scale`、`y_grid`；matching 的 `kernel_id`、`mu`、`pz_gev`、`zs_fm`。"
            "不要建议改 lamet-agent 代码。你不能查看 SVG 图像本身；SVG 清单只代表已生成图像的路径和 provenance，"
            "不得从 SVG 像素、path 几何、文件名臆测数值或曲线形状。图像相关判断只能来自 report 文本和 NetCDF 摘要。"
            "缺失 report、NetCDF 或 SVG 时要明确说明缺失，不能补数值。输出必须是 Markdown，语言必须是中文。\n\n"
            f"Manifest JSON:\n```json\n{json.dumps(manifest.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n```\n\n"
            f"Stage materials:\n```json\n{json.dumps(materials, ensure_ascii=False, indent=2)}\n```"
        )
    else:
        system = "You are a lattice-QCD and LaMET expert. Write a detailed scientific review using only the supplied stage reports, NetCDF summaries, SVG file lists, and manifest. Do not invent unreported numbers; when settings or outputs do not match a realistic LaMET analysis scenario, give executable manifest-level recommendations."
        user = (
            "Generate the complete `review.md` body directly. Follow the order in Stage materials; these stages come from stage subdirectories under `root_directory/artifacts_directory/<stage>` plus stages declared in the manifest. For example, correlator diagnostics are also collected from the `correlator_analysis/fit_logs` subdirectory. "
            "Return normal Markdown only; do not wrap the whole answer in a fenced code block. "
            "Write one level-2 section for each stage with available material, and include `Summary`, `Key figure`, and `Diagnostics and manifest changes` subsections. "
            "`Summary` must describe the stage inputs, outputs, NetCDF numerical ranges, physical meaning, and relationship to upstream and downstream stages in detail. "
            "`Key figure` must choose one SVG from that stage's `svg` list and embed it with Markdown image syntax. You must copy the chosen entry's `markdown_path` exactly as `![description](markdown_path)`; do not invent paths, do not use only the basename, and do not use `absolute_path` as the Markdown link. Then give a detailed explanation below the figure stating why it was selected and how it helps assess the stage; if no SVG exists, say that no embeddable SVG was generated. "
            "`Diagnostics and manifest changes` must judge whether the stage is self-consistent and whether it matches a realistic LaMET analysis scenario; if it is not ideal, give concrete, executable `.json` manifest changes. "
            "Recommendations must cite real manifest paths and values such as `stages.<stage>.defaults.<key>`, `stages.<stage>.jobs[].params.<key>`, or `inputs.kernels[].kernel_parameters.<key>`, and state suggested values or ranges with reasons. "
            "Prioritize these tunable parameters: for correlator, `pt2_windows`, `pt3_tau_cuts`, `nstate`, `fit_scope`, `fit_strategy`, `prior_width`, `svdcut`; for renormalization, `scheme_parameters.zs_fm`, `m0_gev`, `delta_m_gev`; for Fourier, `scheme_scan.zmin_values`, `zmax_values`, `z_ext_max`, `smooth`, `order`, `posterior_prior_error_scale`, `y_grid`; for matching, `kernel_id`, `mu`, `pz_gev`, `zs_fm`. "
            "Do not recommend changing lamet-agent source code. You cannot inspect SVG images; the SVG list only records figure paths and provenance. "
            "Do not infer numerical values or curve shapes from SVG pixels, path geometry, or filenames. Figure-related statements must come from report text and NetCDF summaries. "
            "State missing reports, NetCDF files, or SVG figures explicitly and do not fill in missing numbers. Output Markdown in English.\n\n"
            f"Manifest JSON:\n```json\n{json.dumps(manifest.model_dump(mode='json'), indent=2)}\n```\n\n"
            f"Stage materials:\n```json\n{json.dumps(materials, indent=2)}\n```"
        )
    review = request_llm_text(
        backend=backend,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=api_key,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    review = review.strip()
    if review.startswith("```"):
        lines = review.splitlines()
        if lines and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            review = "\n".join(lines[1:-1]).strip()
    target.write_text(review + "\n", encoding="utf-8")
    return {"review": str(target), "artifact": str(target), "n_stages": len(materials)}


def write_review(store: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Tool wrapper: write review from ``store['manifest']``."""
    result = write_review_from_manifest(store["manifest"], **kwargs)
    store["output"] = result["review"]
    return result


STAGE_TOOLS = {"write_review": write_review}
