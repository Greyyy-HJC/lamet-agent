"""Tests for deterministic review-stage manifest consistency checks."""

from pathlib import Path

from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.review.functions import hybrid_zs_consistency_checks, write_review_from_manifest


def _manifest(*, matching_zs: float = 0.2, renorm_zs: float = 0.2) -> AnalysisManifest:
    return AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "review",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["renormalization", "fourier_transform", "perturbative_matching", "review"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
                    {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
                ],
                "kernels": [
                    {
                        "stage": "perturbative_matching",
                        "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
                        "kernel_path": "kernels.py",
                        "scheme": "hybrid_ratio",
                        "kernel_parameters": {},
                    }
                ],
            },
            "stages": {
                "renormalization": {
                    "defaults": {"scheme": "hybrid_ratio", "zs_fm": renorm_zs},
                    "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
                },
                "fourier_transform": {
                    "defaults": {},
                    "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
                },
                "perturbative_matching": {
                    "defaults": {"kernel_id": "CG_gt_quark_PDF_hybrid_NLO", "zs_fm": 9.9},
                    "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}, "params": {"zs_fm": matching_zs}}],
                },
                "review": {"defaults": {}, "jobs": [{"id": "review_job"}]},
            },
        }
    )


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
    manifest.inputs.kernels[0].scheme = "ratio"
    manifest.stages["perturbative_matching"].defaults["kernel_id"] = "CG_gt_quark_PDF_ratio_NLO"
    assert all(check["status"] == "not_applicable" for check in hybrid_zs_consistency_checks(manifest))


def test_review_appends_deterministic_consistency_sections(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "lamet_agent.stages.review.functions.request_llm_text",
        lambda **kwargs: "# LLM Review",
    )
    monkeypatch.setattr(
        "lamet_agent.stages.review.functions.translate_markdown_report",
        lambda markdown, **kwargs: markdown,
    )
    manifest = _manifest(matching_zs=0.3)
    manifest._artifacts_directory = tmp_path / "artifacts"

    english = write_review_from_manifest(manifest, output_dir=tmp_path / "en")
    chinese = write_review_from_manifest(manifest, report_language="ch", output_dir=tmp_path / "ch")

    english_text = Path(english["review"]).read_text(encoding="utf-8")
    chinese_text = Path(chinese["review"]).read_text(encoding="utf-8")
    assert "## Manifest Parameter Consistency" in english_text
    assert "`mismatch`" in english_text
    assert "stages.perturbative_matching.jobs[0].params.zs_fm" in english_text
    assert "## Manifest Parameter Consistency" in chinese_text
    assert "`mismatch`" in chinese_text


def test_review_rewrites_stage_svg_links_relative_to_review_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "lamet_agent.stages.review.functions.request_llm_text",
        lambda **kwargs: "![key](correlator_analysis/ca_HISQa060_X_re.svg)",
    )
    manifest = _manifest()
    manifest._artifacts_directory = tmp_path / "artifacts"

    result = write_review_from_manifest(manifest, output_dir=tmp_path / "artifacts")
    text = Path(result["review"]).read_text(encoding="utf-8")

    assert "](../correlator_analysis/ca_HISQa060_X_re.svg)" in text


def test_review_prompt_avoids_repeating_matching_zs_fm(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    monkeypatch.setattr("lamet_agent.stages.review.functions.translate_markdown_report", lambda markdown, **kwargs: markdown)

    write_review_from_manifest(_manifest(), output_dir=tmp_path / "en")
    write_review_from_manifest(_manifest(), report_language="ch", output_dir=tmp_path / "ch")

    assert "do not repeat the same `zs_fm` discussion in the matching section" in prompts[0]
    assert "do not repeat the same `zs_fm` discussion in the matching section" in prompts[1]


def test_review_prompt_omits_literature_context_when_disabled(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    monkeypatch.setattr("lamet_agent.stages.review.functions.translate_markdown_report", lambda markdown, **kwargs: markdown)

    write_review_from_manifest(_manifest(), output_dir=tmp_path / "en")

    assert "Relevant literature context (background only)" not in prompts[0]
    assert "Literature context rules:" not in prompts[0]


def test_review_prompt_includes_literature_context_when_enabled(tmp_path: Path, monkeypatch) -> None:
    prompts = []

    def fake_request_llm_text(**kwargs):
        prompts.append("\n".join(message["content"] for message in kwargs["messages"]))
        return "# LLM Review"

    monkeypatch.setattr("lamet_agent.stages.review.functions.request_llm_text", fake_request_llm_text)
    monkeypatch.setattr("lamet_agent.stages.review.functions.translate_markdown_report", lambda markdown, **kwargs: markdown)

    manifest = _manifest()
    manifest.stages["review"].defaults["literature"] = True
    write_review_from_manifest(manifest, output_dir=tmp_path / "en")

    assert "Relevant literature context (background only)" in prompts[0]
    assert "Literature context rules:" in prompts[0]
