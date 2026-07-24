"""Review-stage utilities built from existing stage reports."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from lamet_agent.core.llm import request_llm_text
from lamet_agent.core.reporting import translate_markdown_report
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.manifest_params import merge_stage_params

STAGE_REPORTS = {
    "correlator_analysis": "ca_report.md",
    "renormalization": "renorm_report.md",
    "fourier_transform": "ft_report.md",
    "perturbative_matching": "matching_report.md",
    "extrapolation": "extrapolation_report.md",
}


def _effective_params(manifest: AnalysisManifest, stage: str, job: Any) -> dict[str, Any]:
    return merge_stage_params(manifest.stages[stage].defaults, job.params)


def _zs_path(manifest: AnalysisManifest, stage: str, job: Any) -> str:
    jobs = manifest.stages[stage].jobs
    index = next(index for index, candidate in enumerate(jobs) if candidate.id == job.id)
    if "zs_fm" in job.params:
        return f"stages.{stage}.jobs[{index}].params.zs_fm"
    return f"stages.{stage}.defaults.zs_fm"


def hybrid_zs_consistency_checks(manifest: AnalysisManifest) -> list[dict[str, Any]]:
    """Compare hybrid matching and renormalization ``zs_fm`` along manifest DAG chains."""
    matching_stage = manifest.stages.get("perturbative_matching")
    if matching_stage is None:
        return []

    from lamet_agent.stages.matching.functions import is_hybrid_kernel, resolve_kernel_id

    jobs_by_id = {
        job.id: (stage, job)
        for stage, config in manifest.stages.items()
        for job in config.jobs
    }
    checks: list[dict[str, Any]] = []
    for matching_index, matching_job in enumerate(matching_stage.jobs):
        matching_params = _effective_params(manifest, "perturbative_matching", matching_job)
        kernel_id = matching_params.get("kernel_id")
        if kernel_id is None:
            matching_kernels = [
                item for item in manifest.kernels if item.stage == "perturbative_matching"
            ]
            if len(matching_kernels) == 1:
                kernel_id = matching_kernels[0].kernel_id
        declaration = next((item for item in manifest.kernels if item.kernel_id == kernel_id), None)
        is_hybrid = False
        if declaration is not None:
            try:
                is_hybrid = is_hybrid_kernel(resolve_kernel_id(declaration.kernel_id, declaration.scheme))
            except ValueError:
                is_hybrid = declaration.scheme == "hybrid_ratio" and "hybrid" in declaration.kernel_id.lower()

        base: dict[str, Any] = {
            "matching_job": matching_job.id,
            "matching_job_path": f"stages.perturbative_matching.jobs[{matching_index}]",
            "renormalization_job": None,
            "matching_zs_fm": matching_params.get("zs_fm"),
            "renormalization_zs_fm": None,
            "matching_zs_path": (
                _zs_path(manifest, "perturbative_matching", matching_job)
                if "zs_fm" in matching_params
                else None
            ),
            "renormalization_zs_path": None,
        }
        if not is_hybrid:
            checks.append({**base, "status": "not_applicable", "reason": "matching kernel is not hybrid"})
            continue

        quasi_ref = matching_job.inputs.get("quasi")
        fourier_entry = jobs_by_id.get(quasi_ref) if isinstance(quasi_ref, str) else None
        if fourier_entry is None or fourier_entry[0] != "fourier_transform":
            checks.append(
                {
                    **base,
                    "status": "unverifiable",
                    "reason": "matching quasi input does not resolve to an in-manifest Fourier job",
                }
            )
            continue
        fourier_job = fourier_entry[1]
        renorm_ref = fourier_job.inputs.get("input")
        renorm_entry = jobs_by_id.get(renorm_ref) if isinstance(renorm_ref, str) else None
        if renorm_entry is None or renorm_entry[0] != "renormalization":
            checks.append(
                {
                    **base,
                    "status": "unverifiable",
                    "reason": "Fourier input does not resolve to an in-manifest renormalization job",
                }
            )
            continue

        renorm_job = renorm_entry[1]
        renorm_params = _effective_params(manifest, "renormalization", renorm_job)
        compared = {
            **base,
            "renormalization_job": renorm_job.id,
            "renormalization_zs_fm": renorm_params.get("zs_fm"),
            "renormalization_zs_path": (
                _zs_path(manifest, "renormalization", renorm_job)
                if "zs_fm" in renorm_params
                else None
            ),
        }
        if renorm_params.get("scheme") != "hybrid_ratio":
            checks.append(
                {**compared, "status": "not_applicable", "reason": "upstream renormalization is not hybrid_ratio"}
            )
            continue
        try:
            matching_zs = float(matching_params["zs_fm"])
            renorm_zs = float(renorm_params["zs_fm"])
        except (KeyError, TypeError, ValueError):
            checks.append(
                {**compared, "status": "unverifiable", "reason": "one or both jobs lack a numeric zs_fm"}
            )
            continue
        status = "consistent" if math.isclose(matching_zs, renorm_zs, rel_tol=0.0, abs_tol=1e-12) else "mismatch"
        checks.append(
            {
                **compared,
                "status": status,
                "reason": "zs_fm values agree" if status == "consistent" else "zs_fm values differ",
                "recommended_path": f"stages.perturbative_matching.jobs[{matching_index}].params.zs_fm",
            }
        )
    return checks


def _format_manifest_consistency(checks: list[dict[str, Any]]) -> str:
    lines = [
        "## Manifest Parameter Consistency",
        "",
        "This check follows `matching.quasi → fourier.input → renormalization job`; findings are review-only and do not block execution.",
        "",
        "| Matching job | Renormalization job | Matching `zs_fm` | Renormalization `zs_fm` | Status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    if not checks:
        lines.append("| — | — | — | — | not applicable |")
    for check in checks:
        renorm_job = f"`{check['renormalization_job']}`" if check.get("renormalization_job") else "—"
        matching_zs = check.get("matching_zs_fm")
        renorm_zs = check.get("renormalization_zs_fm")
        lines.append(
            f"| `{check['matching_job']}` | "
            f"{renorm_job} | "
            f"{matching_zs if matching_zs is not None else '—'} | "
            f"{renorm_zs if renorm_zs is not None else '—'} | "
            f"`{check['status']}` |"
        )
    mismatches = [check for check in checks if check["status"] == "mismatch"]
    unverifiable = [check for check in checks if check["status"] == "unverifiable"]
    if mismatches:
        lines.extend(["", "### Required changes"])
        for check in mismatches:
            lines.append(
                f"- `{check['matching_job']}` differs from upstream `{check['renormalization_job']}`: set "
                f"`{check['recommended_path']}` to `{check['renormalization_zs_fm']}`."
            )
    if unverifiable:
        lines.extend(["", "### Not verifiable"])
        for check in unverifiable:
            reason = check["reason"]
            lines.append(f"- `{check['matching_job']}`: {reason}.")
    return "\n".join(lines)


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
    language = "ch" if report_language.lower() == "ch" else "en"
    english_target = review_dir / "review.md"
    target = review_dir / ("review_CN.md" if language == "ch" else "review.md")
    consistency_checks = hybrid_zs_consistency_checks(manifest)
    review_params = merge_stage_params(manifest.stages["review"].defaults, {})
    use_literature = bool(review_params.get("literature", False))
    materials = []
    stages = [stage for stage in STAGE_REPORTS if (artifacts_dir / stage).is_dir() or stage in manifest.metadata.stages]
    for stage in stages:
        en_name = STAGE_REPORTS[stage]
        stage_dir = artifacts_dir / stage
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
    literature_context = []
    kb_path = Path(__file__).resolve().parents[4] / "lamet-papers" / "data" / "lamet_arxiv.sqlite3"
    if use_literature and kb_path.exists():
        manifest_json = manifest.model_dump(mode="json")
        metadata = manifest_json.get("metadata", {})
        correlators = manifest_json.get("inputs", {}).get("correlators", [])
        anchor_terms: list[tuple[str, int, str]] = []
        seen_terms: set[str] = set()
        observable = str(metadata.get("target_observable", "")).strip().lower()
        parton = str(metadata.get("parton", "")).strip().lower()
        if observable == "pdf":
            for term, weight in [("pdf", 6), ("quasi-pdf", 6), ("quasi pdf", 6), ("parton distribution", 4)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "target_observable=pdf"))
                    seen_terms.add(term)
        elif observable == "gpd":
            for term, weight in [("gpd", 6), ("generalized parton distribution", 5), ("pseudo-distribution", 3)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "target_observable=gpd"))
                    seen_terms.add(term)
        elif observable == "tmd":
            for term, weight in [("tmd", 6), ("tmdpdf", 5), ("transverse momentum dependent", 5), ("collins-soper", 4)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "target_observable=tmd"))
                    seen_terms.add(term)
        elif observable == "da":
            for term, weight in [("distribution amplitude", 6), ("light-cone distribution amplitude", 5), ("lcda", 5)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "target_observable=da"))
                    seen_terms.add(term)
        if parton == "quark":
            for term, weight in [("quark", 5), ("isovector", 3), ("valence", 3)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "parton=quark"))
                    seen_terms.add(term)
        elif parton == "gluon":
            for term, weight in [("gluon", 5), ("glue", 3)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, "parton=gluon"))
                    seen_terms.add(term)
        elif parton:
            if parton not in seen_terms:
                anchor_terms.append((parton, 5, f"parton={parton}"))
                seen_terms.add(parton)
        hadrons = sorted({str(item.get("hadron", "")).strip().lower() for item in correlators if item.get("hadron")})
        for hadron in hadrons:
            aliases = [hadron]
            if hadron == "proton":
                aliases.append("nucleon")
            for term in aliases:
                if term and term not in seen_terms:
                    anchor_terms.append((term, 5, f"hadron={hadron}"))
                    seen_terms.add(term)
        gfixes = sorted({str(item.get("gfix", "")).strip().lower() for item in correlators if item.get("gfix")})
        for gfix in gfixes:
            if gfix == "cg":
                for term in ["coulomb gauge", "coulomb"]:
                    if term not in seen_terms:
                        anchor_terms.append((term, 4, "gfix=CG"))
                        seen_terms.add(term)
            elif gfix in {"landau", "lg"}:
                if "landau gauge" not in seen_terms:
                    anchor_terms.append(("landau gauge", 4, f"gfix={gfix.upper()}"))
                    seen_terms.add("landau gauge")
        schemes = set()
        kernel_ids = set()
        momentum_values = set()
        momentum_labels = set()
        mu_values = set()
        for item in correlators:
            for momentum in item.get("momentum", []) or []:
                token = str(momentum).strip().upper()
                if token:
                    momentum_labels.add(token)
                    if any(f"P{axis}{digit}" in token for axis in "XYZ" for digit in "123456789"):
                        momentum_labels.add(f"boosted:{token}")
        for kernel in manifest_json.get("inputs", {}).get("kernels", []):
            if kernel.get("scheme"):
                schemes.add(str(kernel["scheme"]).strip().lower())
            if kernel.get("kernel_id"):
                kernel_ids.add(str(kernel["kernel_id"]).strip().lower())
        for stage_config in manifest_json.get("stages", {}).values():
            for scope in [stage_config.get("defaults", {})] + [job.get("params", {}) for job in stage_config.get("jobs", [])]:
                if scope.get("scheme"):
                    schemes.add(str(scope["scheme"]).strip().lower())
                if scope.get("kernel_id"):
                    kernel_ids.add(str(scope["kernel_id"]).strip().lower())
                for key in ["momentum_gev", "initial_momentum_gev", "final_momentum_gev"]:
                    if scope.get(key) is not None:
                        momentum_values.add(float(scope[key]))
                if scope.get("mu") is not None:
                    mu_values.add(float(scope["mu"]))
        for scheme in sorted(schemes):
            if scheme == "hybrid_ratio":
                for term, weight in [("hybrid ratio", 4), ("hybrid-ratio scheme", 4), ("hybrid renormalization", 3)]:
                    if term not in seen_terms:
                        anchor_terms.append((term, weight, "scheme=hybrid_ratio"))
                        seen_terms.add(term)
            elif scheme == "ratio":
                for term, weight in [("ratio scheme", 3), ("matching factor", 2)]:
                    if term not in seen_terms:
                        anchor_terms.append((term, weight, "scheme=ratio"))
                        seen_terms.add(term)
            elif "ri/mom" in scheme or "rimom" in scheme:
                for term in ["ri/mom", "ri mom"]:
                    if term not in seen_terms:
                        anchor_terms.append((term, 4, "scheme=ri/mom"))
                        seen_terms.add(term)
            elif "self" in scheme:
                for term in ["self-renormalized", "self-renormalization", "zmsbar"]:
                    if term not in seen_terms:
                        anchor_terms.append((term, 4, f"scheme={scheme}"))
                        seen_terms.add(term)
        for kernel_id in sorted(kernel_ids):
            if "nnlo" in kernel_id:
                for term in ["nnlo", "two-loop matching"]:
                    if term not in seen_terms:
                        anchor_terms.append((term, 3, "matching=NNLO"))
                        seen_terms.add(term)
            elif "nlo" in kernel_id:
                for term in ["nlo", "one-loop matching"]:
                    if term not in seen_terms:
                        anchor_terms.append((term, 3, "matching=NLO"))
                        seen_terms.add(term)
        if momentum_values:
            momentum_label = f"momentum_gev={max(momentum_values):.2f}"
            for term, weight in [("large momentum", 3), ("boosted", 3), ("boosted correlations", 3)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, momentum_label))
                    seen_terms.add(term)
        elif any(label.startswith("boosted:") for label in momentum_labels):
            momentum_label = next(label for label in sorted(momentum_labels) if label.startswith("boosted:")).split(":", 1)[1]
            for term, weight in [("large momentum", 3), ("boosted", 3), ("boosted correlations", 3)]:
                if term not in seen_terms:
                    anchor_terms.append((term, weight, f"momentum={momentum_label}"))
                    seen_terms.add(term)
        if mu_values:
            mu_label = f"mu_gev={max(mu_values):.2f}"
            for term in ["matching scale", "renormalization scale", "msbar"]:
                if term not in seen_terms:
                    anchor_terms.append((term, 2, mu_label))
                    seen_terms.add(term)
        lattice_spacings = sorted(
            {str(item.get("lattice_spacing_fm")) for item in correlators if item.get("lattice_spacing_fm") is not None}
        )
        if lattice_spacings:
            lattice_label = f"lattice_spacing_fm={lattice_spacings[0]}"
            for term in ["lattice spacing", "continuum limit"]:
                if term not in seen_terms:
                    anchor_terms.append((term, 2, lattice_label))
                    seen_terms.add(term)
        volumes = sorted({str(item.get("volume")) for item in correlators if item.get("volume")})
        if volumes and "finite volume" not in seen_terms:
            anchor_terms.append(("finite volume", 1, f"volume={volumes[0]}"))
            seen_terms.add("finite volume")
        for term in ["lamet", "large momentum effective theory", "matching", "renormalization"]:
            if term not in seen_terms:
                anchor_terms.append((term, 1, "lamet"))
                seen_terms.add(term)
        search_terms = [term for term, _weight, _label in anchor_terms][:12] or [
            "lamet",
            "quasi-pdf",
            "matching",
            "renormalization",
        ]
        query = (
            "SELECT arxiv_id, title, summary, published, label, score, abs_url FROM papers "
            "WHERE label IN ('core', 'secondary') AND ("
            + " OR ".join("(lower(title) LIKE ? OR lower(summary) LIKE ?)" for _ in search_terms)
            + ") ORDER BY published DESC LIMIT 80"
        )
        params: list[str] = []
        for term in search_terms:
            pattern = f"%{term}%"
            params.extend([pattern, pattern])
        with sqlite3.connect(kb_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        ranked_rows = []
        for row in rows:
            title_lower = row["title"].lower()
            summary_lower = row["summary"].lower()
            matched_topics = []
            matched_fields = set()
            match_score = 0
            for term, weight, label in anchor_terms:
                in_title = term in title_lower
                in_summary = term in summary_lower
                if not in_title and not in_summary:
                    continue
                match_score += weight * (3 if in_title else 1)
                if label not in matched_topics:
                    matched_topics.append(label)
                matched_fields.add(label.split("=", 1)[0])
            if not matched_topics:
                continue
            if "target_observable" in matched_fields and "parton" in matched_fields:
                match_score += 12
            if "hadron" in matched_fields:
                match_score += 8
            if "scheme" in matched_fields or "matching" in matched_fields:
                match_score += 8
            if "gfix" in matched_fields:
                match_score += 4
            if "momentum_gev" in matched_fields:
                match_score += 3
            if "lattice_spacing_fm" in matched_fields or "volume" in matched_fields:
                match_score += 2
            match_score += 2 if row["label"] == "core" else 0
            ranked_rows.append((match_score, int(row["score"]), row["published"], row, matched_topics[:6]))
        ranked_rows.sort(reverse=True)
        for _, _, _, row, matched_topics in ranked_rows[:8]:
            summary = " ".join(row["summary"].split())
            if len(summary) > 420:
                summary = summary[:417] + "..."
            literature_context.append(
                {
                    "arxiv_id": row["arxiv_id"],
                    "published": row["published"][:10],
                    "label": row["label"],
                    "title": row["title"],
                    "matched_topics": matched_topics,
                    "relevance": f"Strongest overlap with manifest anchors: {', '.join(matched_topics)}.",
                    "summary": summary,
                    "abs_url": row["abs_url"],
                }
            )
    lamet_review_rules_en = (
        "You are an expert AI specialized in LaMET lattice numerical analysis. Your task is to generate a fact-grounded Review from the supplied five-step analysis reports and provide Recommended Manifest Changes.\n"
        + "Domain background: LaMET extracts light-cone PDFs/TMDs/DAs/GPDs from lattice QCD through Fourier reconstruction, perturbative matching, and momentum extrapolation of large-momentum quasi distributions. The standard flow is: Step 1 correlator_analysis usually fits two- and three-point correlators to obtain ground-state spectra, overlap factors, and bare matrix elements h(z,Pz), with diagnostics from fit quality, excited-state gaps, relative overlap errors, and signal-to-noise at z=0 and maximal z; if the manifest or report shows `fit_scope=\"qda_ratio\"`, this step is a nonlocal 2pt z-ratio analysis and must be described as extracting bare matrix elements from nonlocal two-point correlator ratios, without forcing 3pt/2pt ratio, overlap, source-sink separation, tau, or current-insertion diagnostics. Step 2 renormalization removes UV divergences and gives h_R(z), with diagnostics from renormalization constants, error amplification, and window dependence; Step 3 fourier_transform reconstructs quasi distributions from h_R(z), with diagnostics from zmin/zmax, oscillations, x/y-space errors, and the zeroth moment; Step 4 perturbative_matching applies LaMET kernels to obtain light-cone distributions, with diagnostics from positivity, first moment and deviation from 1, intermediate-x errors, and matching order; Step 5 extrapolation performs infinite-momentum or continuum extrapolation, with diagnostics from model reasonableness, fit quality, and stability.\n"
        + "Data extraction rules: read or calculate only from the reports and NetCDF summaries; write 'not provided' when absent. For standard 3pt_ratio Step1, extract fit quality, excited-state gaps, overlap relative errors, and signal-to-noise at z=0 and maximal z. For `fit_scope=\"qda_ratio\"`, extract only nonlocal 2pt z-ratio fit quality, 2pt fit windows, the ordinary local 2pt denominator, z-dependent signal-to-noise, and long-Wilson-line noise behavior; do not treat missing 3pt/overlap/tsep/tau diagnostics as a problem. Step2 extract renormalization constants with errors and statistical-error amplification. Step3 extract zmin/zmax, quasi-distribution error bars, and zeroth moment. Step4 extract matched q(x) error bars, first moment, and deviation from 1. Step5 extract extrapolation model, fit quality, and final total uncertainty. For each stage, write one coherent physics summary of the operation, key result, and quality.\n"
        + (
            "Literature context rules: retrieved literature context is background only. It may be used for standard methodology, qualitative comparison, common diagnostics, and typical systematic effects, but never as evidence of this run's numerical results. All numerical claims about this run must come only from the supplied manifest, stage reports, NetCDF summaries, and deterministic checks. Do not use literature abstracts to supply, infer, normalize, or validate any number for this run. If literature and run materials differ, trust the run materials. When a run diagnostic is missing, write 'not provided' rather than filling the gap from literature. Manifest-change recommendations must be triggered by run evidence, not only by literature expectations.\n"
            + "When literature is enabled and relevant context exists for a stage, append one short paragraph at the end of that stage's Diagnostics subsection as a literature-based context paragraph. This paragraph must stay inside Diagnostics rather than becoming a new heading. It may cite the most relevant retrieved paper(s) as qualitative background to explain the stage purpose, why the observed issue matters physically, whether the observed signal quality, statistical noise, or systematic-error pattern is qualitatively reasonable for this kind of LaMET analysis, and which systematic effects are commonly discussed for that stage. Prefer papers whose `matched_topics` overlap most directly with the manifest observable, parton/hadron channel, gauge fixing, ensemble scale, momentum, renormalization scheme, and matching method; use weaker semantic neighbors only as secondary context. Keep it qualitative and concise, and do not use literature as numerical evidence for this run.\n"
            if use_literature
            else ""
        )
        + "Physical Summary writing rules: each stage's Physical Summary must read like publication-level prose that can be inserted directly into a paper's Results or Analysis section. Use third-person passive voice or 'we' as the subject, and describe the completed analysis rather than an ongoing process. Do not use meta-language such as 'according to the report', 'the Step 1 indicators show', or 'here we see'. Each paragraph must contain 3-5 compact professional sentences, optionally beginning with a short bold label such as **Correlator analysis.**, **Renormalization.**, **Fourier transform.**, **Perturbative matching.**, or **Extrapolation.** The paragraph must naturally include the physical purpose, key methods or settings, core numerical results with statistical/systematic precision, and a short physics interpretation of quality. All physics quantities must use `$...$`, for example `$\\chi^2/\\mathrm{dof}$`, `$h(z,P_z)$`, and `$\\langle x \\rangle$`. If the indicators are good, use scholarly language such as 'demonstrates good convergence', 'is well under control', or 'agrees with expectations'; if thresholds are approached or exceeded, state the issue faithfully with language such as 'shows a mild tension', 'indicates potential systematic effects', or 'may require further investigation'.\n"
        + "Diagnostics writing rules: Diagnostics must not repeat the Summary and must not judge physics reliability only from `$\\chi^2/\\mathrm{dof}$`, logGBF, or job success. It must separate three kinds of statements: (1) numerical facts directly supported by reports/NetCDF summaries; (2) analysis issues that can be addressed through manifest changes; and (3) raw-data or external LQCD limitations that cannot be fixed by lamet-agent tuning and would require new measurements or improved external simulation conditions. If quasi distributions oscillate outside the physical region, positivity/normalization is unreasonable, error bands are large, different momenta or lattice spacings are inconsistent, renormalization produces an anomalously enlarged dynamic range, or long-distance matrix elements are noise dominated, Diagnostics must state that successful numerical execution does not imply a physically reliable result. External explanations may include the intrinsic noisiness of gluon operators, insufficient three-point statistics, limited source-sink separation or excited-state control, degraded overlap at large momentum, exponential signal-to-noise loss for long Wilson lines, lattice-spacing/volume/finite-momentum systematics, autocorrelations or too few effectively independent configurations, and weak signal from the original 2pt/3pt construction or projection. For `fit_scope=\"qda_ratio\"`, external explanations should instead focus on nonlocal 2pt z-ratio statistics, long-distance nonlocal two-point signal-to-noise, pt2 windows, the ordinary local 2pt denominator, autocorrelations, sample size, and Wilson-line length; do not require three-point statistics, source-sink separations, tau coverage, current insertion, or overlap diagnostics. These explanations must be phrased as physics interpretations consistent with the observed anomalies, not as proven facts; when the corresponding diagnostic is absent, state that confirmation requires checking the raw correlators, statistics, or independent LQCD inputs.\n"
        + "Recommendation triggers: recommend manifest changes only when triggered; otherwise state that the current setting is reasonable and no change is justified. For standard 3pt_ratio Step1, poor fit quality, very large overlap errors, or h(z) signal-to-noise < 3 triggers fit-window, nstate, or statistics recommendations. For `fit_scope=\"qda_ratio\"`, poor nonlocal 2pt z-ratio fit quality or long-z signal-to-noise < 3 should prioritize `pt2_windows`, `nstate`/`nstate_values`, `fit_strategy`, `prior_width`, `svdcut`, and should not recommend `pt3_tau_cuts`. Step2 error amplification > 2.0 or significant window dependence triggers renormalization-window or scheme recommendations. Step3 strong quasi-distribution oscillations with zmax signal-to-noise < 3, or zeroth-moment deviation > 10%, triggers larger zmax or improved transform-method recommendations. Step4 first moment differs from 1 by more than 3 sigma, or q(x) shows clear unphysical values, triggers larger Pz or higher-order matching recommendations. Step5 poor extrapolation fit quality or leave-one-momentum-out changes above 1 sigma triggers adding intermediate momenta or reassessing the model. Explicitly flag any other significant anomaly.\n"
        + "Each Recommended Manifest Changes item must contain `parameter`, `current_value`, `recommended_change`, `evidence`, and `expected_effect`; use `related_parameter` when the exact manifest key is uncertain. Never invent unreported numbers or phenomena. Do not use vague words such as 'maybe' or 'possibly'. If evidence is insufficient, state that the indicators are normal and there is no clear basis for a change.\n"
    )
    system = lamet_review_rules_en + "Write a detailed scientific review using only the supplied stage reports, NetCDF summaries, SVG file lists, and manifest. Do not invent unreported numbers; when settings or outputs do not match a realistic LaMET analysis scenario, give executable manifest-level recommendations."
    user = (
        "Generate the complete `review.md` body directly. Follow the order in Stage materials; these stages come from stage subdirectories under `root_directory/artifacts_directory/<stage>` plus stages declared in the manifest. For example, correlator diagnostics are also collected from the `correlator_analysis/fit_logs` subdirectory. "
        + "Return normal Markdown only; do not wrap the whole answer in a fenced code block. "
        + "Write one level-2 section for each stage with available material, and include `Physical Summary`, `Key figure`, `Diagnostics`, and `Recommended Manifest Changes` subsections. "
        + "`Physical Summary` must follow the publication-style prose rules in the system prompt rather than report-like listing, and may only use numerical values supplied by the reports and NetCDF summaries. "
        + "`Key figure` must choose one SVG from that stage's `svg` list; if the list contains an ensemble overview figure such as `ca_<ensemble>_*.svg`, `rn_<ensemble>_*.svg`, `ft_<ensemble>_xdep.svg`, or `mt_<ensemble>.svg`, choose that overview figure first, otherwise follow the usual single-job figure selection rule. Embed it with Markdown image syntax. You must copy the chosen entry's `markdown_path` exactly as `![description](markdown_path)`; do not invent paths, do not use only the basename, and do not use `absolute_path` as the Markdown link. The `markdown_path` usually has the form of a path from the review directory to a sibling stage directory, for example `../correlator_analysis/xxx.svg`; preserve that exact relative path string when embedding the image. Then give a detailed explanation below the figure stating why it was selected and how it helps assess the stage; if no SVG exists, say that no embeddable SVG was generated. "
        + "`Diagnostics` must judge whether the stage is self-consistent and whether it matches a realistic LaMET analysis scenario; it must follow the Diagnostics rules in the system prompt and explicitly distinguish successful execution, manifest-tunable analysis issues, and raw-data or external LQCD limitations that lamet-agent tuning cannot fix. `Recommended Manifest Changes` must use the required field format above; if no trigger is met, state that the current setting is reasonable and no change is justified. "
        + "Recommendations must cite real manifest paths and values such as `stages.<stage>.defaults.<key>`, `stages.<stage>.jobs[].params.<key>`, or `inputs.kernels[].kernel_parameters.<key>`, and state suggested values or ranges with reasons. "
        + "Prioritize these tunable parameters: for correlator, `pt2_windows`, `nstate`, `fit_scope`, `fit_strategy`, `prior_width`, `svdcut`, and discuss `pt3_tau_cuts` only for three-point fit scopes; for renormalization, `zs_fm`, `scheme_parameters.m0_gev`, `scheme_parameters.delta_m_gev`; for Fourier, `scheme_scan.zmin_values`, `zmax_values`, `z_ext_max`, `smooth`, `order`, `posterior_prior_error_scale`, `y_grid`; for matching, `kernel_id`, `mu`, `momentum_gev`. "
        + "If `zs_fm` has already been described in the renormalization section, do not repeat the same `zs_fm` discussion in the matching section; discuss it under matching only when the manifest consistency checks show a renormalization/matching mismatch or when there is an independent matching-specific `zs_fm` issue. "
        + "Do not recommend changing lamet-agent source code. You cannot inspect SVG images; the SVG list only records figure paths and provenance. "
        + "Do not infer numerical values or curve shapes from SVG pixels, path geometry, or filenames. Figure-related statements must come from report text and NetCDF summaries. "
        + (
            "Use literature context sparingly and only as qualitative background when it directly helps explain the physical role of a stage, a common systematic effect, why a diagnostic matters, or whether the observed signal/noise/systematic pattern is qualitatively consistent with related LaMET literature. The paragraph may mention the most relevant retrieved paper(s), but must not turn literature into a separate evidence chain or into numerical validation for this run. If a stage has relevant literature context, place it as one additional paragraph immediately below the main Diagnostics paragraph for that same stage, not as a separate section and not under Recommended Manifest Changes.\n"
            if use_literature
            else ""
        )
        + "State missing reports, NetCDF files, or SVG figures explicitly and do not fill in missing numbers. Output Markdown in English.\n\n"
        + f"Manifest JSON:\n```json\n{json.dumps(manifest.model_dump(mode='json'), indent=2)}\n```\n\n"
        + f"Stage materials:\n```json\n{json.dumps(materials, indent=2)}\n```\n\n"
        + (
            f"Relevant literature context (background only):\n```json\n{json.dumps(literature_context, indent=2)}\n```\n\n"
            if use_literature
            else ""
        )
        + f"Deterministic manifest consistency checks:\n```json\n{json.dumps(consistency_checks, indent=2)}\n```"
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
    output = review + "\n\n" + _format_manifest_consistency(consistency_checks) + "\n"
    for stage in STAGE_REPORTS:
        output = output.replace(f"]({stage}/", f"](../{stage}/").replace(f"](./{stage}/", f"](../{stage}/")
    english_target.write_text(output, encoding="utf-8")
    if language == "ch":
        output = translate_markdown_report(
            output,
            backend=backend,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
    target.write_text(output, encoding="utf-8")
    return {"review": str(target), "artifact": str(target), "n_stages": len(materials)}


def write_review(store: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Tool wrapper: write review from ``store['manifest']``."""
    result = write_review_from_manifest(store["manifest"], **kwargs)
    store["output"] = result["review"]
    return result


STAGE_TOOLS = {"write_review": write_review}
