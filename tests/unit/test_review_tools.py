"""Tests for deterministic review-stage manifest consistency checks."""

import sqlite3
from pathlib import Path

from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.review.functions import (
    _resolve_literature_db_path,
    hybrid_zs_consistency_checks,
    write_review_from_manifest,
)


def _manifest(*, matching_zs: float = 0.2, renorm_zs: float = 0.2) -> AnalysisManifest:
    return AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "review",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 1984,
                "stages": ["renormalization", "fourier_transform", "perturbative_matching", "review"],
            },
            "inputs": {
                "correlators": [
                    {
                        "correlator_id": "c2",
                        "correlator_type": "2pt",
                        "data_path": "c2.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "CG",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "volume": "S16T32",
                        "momentum": ["PX0PY0PZ5"],
                        "lattice_spacing_fm": 0.1,
                    }
                ],
                "artifacts": [
                    {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
                    {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
                ],
                "kernels": [
                    {
                        "stage": "perturbative_matching",
                        "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
                        "kernel_path": "kernels.py",
                        "kernel_parameters": {},
                    }
                ],
            },
            "stages": {
                "renormalization": {
                    "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": renorm_zs},
                    "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
                },
                "fourier_transform": {
                    "defaults": {},
                    "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
                },
                "perturbative_matching": {
                    "defaults": {
                        "scheme": "hybrid",
                        "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
                        "zs_fm": 9.9,
                    },
                    "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}, "params": {"zs_fm": matching_zs}}],
                },
                "review": {"defaults": {}, "jobs": [{"id": "review_job"}]},
            },
        }
    )


def test_literature_db_fallback_does_not_assume_package_depth(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    db_path = repository / "papers" / "data" / "lamet_arxiv.sqlite3"
    db_path.parent.mkdir(parents=True)
    db_path.touch()
    module_path = repository / "nested" / "lamet_agent" / "stages" / "review" / "functions.py"
    monkeypatch.setattr("lamet_agent.stages.review.functions.__file__", str(module_path))
    manifest = _manifest()
    manifest._root_directory = tmp_path / "external-run-root"

    assert _resolve_literature_db_path(manifest) == db_path


def test_hybrid_zs_consistency_follows_dag_and_job_overrides() -> None:
    consistent = hybrid_zs_consistency_checks(_manifest())
    mismatch = hybrid_zs_consistency_checks(_manifest(matching_zs=0.3))

    assert consistent[0]["status"] == "consistent"
    assert consistent[0]["matching_zs_path"] == "stages.perturbative_matching.jobs[0].params.zs_fm"
    assert consistent[0]["renormalization_zs_path"] == "stages.renormalization.defaults.zs_fm"
    assert mismatch[0]["status"] == "mismatch"
    assert mismatch[0]["recommended_path"] == "stages.perturbative_matching.jobs[0].params.zs_fm"


def test_hybrid_zs_consistency_marks_external_partial_chain_unverifiable() -> None:
    manifest = _manifest()
    manifest.inputs.artifacts.append(
        manifest.inputs.artifacts[0].model_copy(
            update={"id": "external_rn", "stage": "renormalization", "path": "external.nc"}
        )
    )
    manifest.stages["fourier_transform"].jobs[0].inputs["input"] = "external_rn"

    checks = hybrid_zs_consistency_checks(manifest)

    assert checks[0]["status"] == "unverifiable"


def test_hybrid_zs_consistency_handles_independent_chains_and_nonhybrid_matching() -> None:
    manifest = _manifest()
    renorm_job = manifest.stages["renormalization"].jobs[0]
    fourier_job = manifest.stages["fourier_transform"].jobs[0]
    matching_job = manifest.stages["perturbative_matching"].jobs[0]
    manifest.stages["renormalization"].jobs.append(
        renorm_job.model_copy(update={"id": "rn_two", "params": {"zs_fm": 0.4}})
    )
    manifest.stages["fourier_transform"].jobs.append(
        fourier_job.model_copy(update={"id": "ft_two", "inputs": {"input": "rn_two"}})
    )
    manifest.stages["perturbative_matching"].jobs.append(
        matching_job.model_copy(update={"id": "mt_two", "inputs": {"quasi": "ft_two"}, "params": {"zs_fm": 0.4}})
    )

    checks = hybrid_zs_consistency_checks(manifest)

    assert [check["status"] for check in checks] == ["consistent", "consistent"]
    manifest.inputs.kernels[0].kernel_id = "CG_gt_quark_PDF_ratio_NLO"
    manifest.stages["perturbative_matching"].defaults["scheme"] = "ratio"
    manifest.stages["perturbative_matching"].defaults["kernel_id"] = "CG_gt_quark_PDF_ratio_NLO"
    assert all(check["status"] == "not_applicable" for check in hybrid_zs_consistency_checks(manifest))


def test_review_appends_deterministic_consistency_sections(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "lamet_agent.stages.review.functions.request_llm_text",
        lambda **kwargs: "# LLM Review",
    )
    manifest = _manifest(matching_zs=0.3)
    manifest._artifacts_directory = tmp_path / "artifacts"

    english = write_review_from_manifest(manifest)
    chinese = write_review_from_manifest(manifest, report_language="ch")

    english_path = Path(english["review"])
    chinese_path = Path(chinese["review"])
    english_text = english_path.read_text(encoding="utf-8")
    chinese_text = chinese_path.read_text(encoding="utf-8")
    assert english_path == tmp_path / "artifacts" / "6_review" / "review.md"
    assert chinese_path == tmp_path / "artifacts" / "6_review" / "review_CN.md"
    assert "## Manifest Parameter Consistency" in english_text
    assert "`mismatch`" in english_text
    assert "stages.perturbative_matching.jobs[0].params.zs_fm" in english_text
    assert "## Manifest 参数一致性" in chinese_text
    assert "`mismatch`" in chinese_text
    assert not (tmp_path / "ch" / "6_review" / "review.md").exists()


def test_review_rewrites_stage_svg_links_relative_to_review_dir(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "![key](correlator_analysis/ca_HISQa060_X_re.svg)"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    manifest = _manifest()
    manifest._artifacts_directory = tmp_path / "artifacts"
    stage_dir = tmp_path / "artifacts" / "1_correlator_analysis"
    stage_dir.mkdir(parents=True)
    for index in range(13):
        (stage_dir / f"ca_{index:02d}.svg").write_text("<svg/>", encoding="utf-8")

    result = write_review_from_manifest(manifest)
    text = Path(result["review"]).read_text(encoding="utf-8")

    assert "](../1_correlator_analysis/ca_HISQa060_X_re.svg)" in text
    assert prompts[0].count('"markdown_path"') == 12
    assert '"markdown_path": "../1_correlator_analysis/ca_00.svg"' in prompts[0]
    assert '"absolute_path"' not in prompts[0]
    assert '"stage_subpath"' not in prompts[0]


def test_review_prompt_avoids_repeating_matching_zs_fm(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    english_manifest = _manifest()
    english_manifest._artifacts_directory = tmp_path / "en"
    chinese_manifest = _manifest()
    chinese_manifest._artifacts_directory = tmp_path / "ch"
    write_review_from_manifest(english_manifest)
    write_review_from_manifest(chinese_manifest, report_language="ch")

    assert len(prompts) == 2
    assert "do not repeat the same `zs_fm` discussion in the matching section" in prompts[0]
    assert "do not repeat the same `zs_fm` discussion in the matching section" in prompts[1]
    assert "Authoritative stage parameter contracts" in prompts[0]
    assert "never recommend writing runner-derived kinematics such as momentum_gev" in prompts[0]
    assert "Any repository change that adds, removes, renames, or changes a manifest parameter" in prompts[0]
    assert "Output Markdown in Simplified Chinese" in prompts[1]


def test_review_prompt_omits_literature_context_when_disabled(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    manifest = _manifest()
    manifest._artifacts_directory = tmp_path
    write_review_from_manifest(manifest)

    assert "Relevant literature context (background only)" not in prompts[0]
    assert "Literature context rules:" not in prompts[0]


def test_review_prompt_includes_literature_context_when_enabled(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    manifest = _manifest()
    manifest.stages["review"].defaults["literature"] = True
    manifest._artifacts_directory = tmp_path
    write_review_from_manifest(manifest)

    assert "Relevant literature context (background only)" in prompts[0]
    assert "Literature context rules:" in prompts[0]
    assert "append one short paragraph at the end of that stage's Diagnostics subsection" in prompts[0]
    assert "may cite the most relevant retrieved paper(s) as qualitative background" in prompts[0]
    assert "for example `../1_correlator_analysis/xxx.svg`" in prompts[0]
    assert "must not turn literature into a separate evidence chain or into numerical validation for this run" in prompts[0]
    assert "Prefer papers whose `matched_topics` overlap most directly" in prompts[0]


def test_review_literature_ranking_prefers_manifest_anchor_matches(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    db_path = tmp_path / "papers" / "data" / "lamet_arxiv.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE papers ("
            "arxiv_id TEXT, title TEXT, summary TEXT, published TEXT, label TEXT, score INTEGER, abs_url TEXT)"
        )
        conn.executemany(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "exact-1",
                    "Pion Quark PDF in the Coulomb Gauge with Hybrid-Ratio Matching",
                    "A LaMET study of boosted pion quasi-PDFs with one-loop matching and continuum-limit checks.",
                    "2026-01-01T00:00:00Z",
                    "core",
                    95,
                    "https://arxiv.org/abs/exact-1",
                ),
                (
                    "generic-1",
                    "Large Momentum Effective Theory Overview",
                    "A generic review of LaMET methodology and renormalization ideas.",
                    "2026-01-02T00:00:00Z",
                    "core",
                    99,
                    "https://arxiv.org/abs/generic-1",
                ),
            ],
        )

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    manifest = _manifest()
    manifest._root_directory = tmp_path
    manifest.stages["review"].defaults["literature"] = True
    manifest.stages["review"].defaults["literature_max_papers"] = 1
    manifest._artifacts_directory = tmp_path
    write_review_from_manifest(manifest)

    prompt = prompts[0]
    assert '"matched_topics": [' in prompt
    assert "Pion Quark PDF in the Coulomb Gauge with Hybrid-Ratio Matching" in prompt
    assert "Large Momentum Effective Theory Overview" not in prompt
