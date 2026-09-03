"""Focused tests for the evidence-grounded Review stage."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.review._check_consistency import run as check_consistency
from lamet_agent.stages.review._inspect_results import run as inspect_results
from lamet_agent.stages.review._list_literature import run as list_literature
from lamet_agent.stages.review.tools.read_papers import run as read_papers
from lamet_agent.stages.review.tools.write_review import run as write_review


def test_review_collects_preceding_stage_reports_only(tmp_path: Path) -> None:
    artifact_base = tmp_path / "runs"
    source_directory = artifact_base / "01_correlator_analysis" / "ca"
    source_directory.mkdir(parents=True)
    (source_directory.parent / "report.md").write_text("# Correlator stage\n", encoding="utf-8")
    terminal = {
        "stage_id": "correlator_analysis",
        "job_id": "ca",
        "result": "matrix_element",
        "decisions": {},
        "diagnostics": {},
        "artifacts": [],
    }
    variation_directory = source_directory.parent / "ca__fit_high"
    variation_directory.mkdir()
    missing_directory = source_directory.parent / "ca__no_job_report"
    missing_directory.mkdir()
    review_directory = artifact_base / "02_review" / "review"
    review_directory.mkdir(parents=True)
    manifest = {
        "metadata": {"run_id": "review", "target_observable": "pdf", "parton": "quark"},
        "stages": {
            "correlator_analysis": {
                "jobs": [
                    {"id": "ca", "inputs": {}},
                    {"id": "ca__fit_high", "inputs": {}},
                    {"id": "ca__no_job_report", "inputs": {}},
                ]
            },
            "review": {"jobs": [{"id": "review", "inputs": {"results": ["ca"]}}]},
        },
    }
    ensemble = EnsembleInfo("HISQ", "HISQa060_X", 0.06, 0.06, 48, 64, 0.3)
    data = EnsembleData(
        ensemble,
        "jackknife",
        [np.ones(2), np.ones(2)],
        ["z"],
        {"z": [0.0, 1.0]},
        attrs={"hadron": "pion", "parton": "quark", "target_observable": "pdf"},
    )
    context = ToolContext(
        manifest,
        tmp_path / "manifest.json",
        "review",
        "review",
        {},
        {"results": [data]},
        {"results": [terminal]},
        {},
        review_directory,
        np.random.default_rng(1),
        runtime_records={
            "ca": {
                "summary": terminal,
                "review_summary": {"job_id": "ca", "result": "matrix_element"},
                "output": data,
                "artifact_directory": source_directory,
            }
        },
    )

    inspect_results(context)

    bundle = json.loads((review_directory / "review_bundle.json").read_text(encoding="utf-8"))
    assert bundle["stage_reports"][0]["text"] == "# Correlator stage\n"
    assert "job_reports" not in bundle
    assert bundle["review_summaries"]["ca"]["result"] == "matrix_element"
    assert bundle["stage_reports"][0]["path"] == "../../01_correlator_analysis/report.md"
    assert bundle["results"][0]["job_id"] == "ca"
    assert bundle["results"][0]["ensemble"] == ensemble._asdict()


def test_review_builtin_index_is_ranked_from_run_topics(tmp_path: Path) -> None:
    manifest = {
        "metadata": {"run_id": "review", "target_observable": "pdf", "parton": "quark"},
        "stages": {
            "fourier_transform": {"jobs": [{"id": "ft", "inputs": {}, "sector": "valence"}]},
            "review": {"jobs": [{"id": "review", "inputs": {"results": []}}]},
        },
    }
    context = ToolContext(
        manifest,
        tmp_path / "manifest.json",
        "review",
        "review",
        {"catalog": "builtin", "max_papers": 4},
        {},
        {},
        {
            "consistency": {"findings": []},
            "result_summary": [
                {
                    "attrs": {
                        "hadron": "pion",
                        "parton": "quark",
                        "target_observable": "pdf",
                        "polarization": "unpolarized",
                        "gfix": "GI",
                    }
                }
            ],
        },
        tmp_path,
        np.random.default_rng(1),
    )

    observation = list_literature(context)

    assert 1 <= len(observation["candidates"]) <= 12
    assert all(candidate["score"] > 0 for candidate in observation["candidates"])
    assert all("target_observable=pdf" in candidate["matched_topics"] for candidate in observation["candidates"][:4])
    assert not {"2310.10579", "2407.03516", "2412.19988"} & {candidate["id"] for candidate in observation["candidates"]}


def test_review_literature_filters_structured_metadata_and_accepts_explicit_tags(tmp_path: Path) -> None:
    common = {
        "relevance": "core",
        "review_topics": ["target_observable=pdf", "parton=quark", "hadron=pion"],
        "lattice_setup": {"uses_lattice_data": True},
    }
    records = [
        {
            **common,
            "arxiv_id": "2401.00001",
            "title": "Pion PDF",
            "tags": {
                "observables": ["pdf"],
                "partons": ["quark"],
                "hadrons": ["pion"],
                "polarizations": ["unpolarized"],
                "kinematic_dependence": ["collinear"],
                "stages": ["fourier_transform"],
            },
        },
        {
            **common,
            "arxiv_id": "2401.00002",
            "title": "Pion GPD",
            "tags": {
                "observables": ["pdf", "gpd"],
                "partons": ["quark"],
                "hadrons": ["pion"],
                "polarizations": ["unpolarized"],
                "kinematic_dependence": ["off_forward"],
                "stages": ["fourier_transform"],
            },
        },
        {
            **common,
            "arxiv_id": "2401.00003",
            "title": "Pion TMD",
            "tags": {
                "observables": ["pdf"],
                "partons": ["quark"],
                "hadrons": ["pion"],
                "polarizations": ["unpolarized"],
                "kinematic_dependence": ["tmd"],
                "stages": ["fourier_transform"],
            },
        },
    ]
    catalog_path = tmp_path / "arxiv.json"
    catalog_path.write_text(json.dumps({"papers": records}), encoding="utf-8")
    manifest = {
        "metadata": {"run_id": "review", "target_observable": "pdf", "parton": "quark"},
        "stages": {"review": {"jobs": [{"id": "review", "inputs": {"results": []}}]}},
    }
    context = ToolContext(
        manifest,
        tmp_path / "manifest.json",
        "review",
        "review",
        {"catalog": str(catalog_path), "max_papers": 4},
        {},
        {},
        {
            "consistency": {"findings": []},
            "result_summary": [{"attrs": {"hadron": "pion", "polarization": "unpolarized"}}],
        },
        tmp_path,
        np.random.default_rng(1),
    )

    observation = list_literature(context, stages=["fourier_transform"])

    assert [candidate["id"] for candidate in observation["candidates"]] == ["2401.00001"]
    assert observation["candidates"][0]["exact_metadata_matches"] == [
        "observables",
        "partons",
        "hadrons",
        "polarizations",
    ]


def test_review_reads_ar5iv_and_falls_back_to_index(tmp_path: Path, monkeypatch) -> None:
    record = {
        "id": "2401.00001",
        "title": "A selected paper",
        "authors": ["A. Author"],
        "year": "2024",
        "summary": "Indexed summary.",
        "source": "https://arxiv.org/abs/2401.00001",
        "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2401.00001",
        "source_file": None,
        "text_path": None,
        "evidence": ["LaMET analysis"],
        "matched_topics": ["target_observable=pdf"],
    }
    online_directory = tmp_path / "online"
    online_directory.mkdir()
    online = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "online",
        {"max_papers": 1},
        {},
        {},
        {"literature_candidates": [record], "literature_catalog_directory": str(tmp_path)},
        online_directory,
        np.random.default_rng(1),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"<html><body><h1>Paper</h1><p>Useful result.</p></body></html>"),
    )

    read_papers(online, paper_ids=[record["id"]])

    assert online.state["selected_papers"][0]["retrieval"] == "ar5iv"
    assert "Useful result." in (online_directory / "literature/2401.00001.txt").read_text(encoding="utf-8")

    offline_directory = tmp_path / "offline"
    offline_directory.mkdir()
    offline = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "offline",
        {"max_papers": 1},
        {},
        {},
        {"literature_candidates": [record], "literature_catalog_directory": str(tmp_path)},
        offline_directory,
        np.random.default_rng(1),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    read_papers(offline, paper_ids=[record["id"]])

    assert offline.state["selected_papers"][0]["retrieval"] == "index_fallback"
    assert not offline.state["selected_papers"][0]["full_text_available"]
    assert "Indexed summary." in (offline_directory / "literature/2401.00001.txt").read_text(encoding="utf-8")


def test_review_consistency_follows_distribution_edges(tmp_path: Path) -> None:
    context = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "review",
        {"checks": ["identity", "grids"]},
        {},
        {},
        {
            "result_summary": [
                {
                    "job_id": "ft",
                    "stage_id": "fourier_transform",
                    "dims": ["x"],
                    "coords": {"x": [0.0, 1.0]},
                    "attrs": {"parton": "quark", "polarization": "unpolarized"},
                },
                {
                    "job_id": "mt",
                    "stage_id": "perturbative_matching",
                    "dims": ["x"],
                    "coords": {"x": [0.0, 1.0]},
                    "attrs": {"parton": "quark", "polarization": "unpolarized"},
                },
            ],
            "review_bundle": {
                "jobs": {
                    "ft": {"stage_id": "fourier_transform", "inputs": {}},
                    "mt": {"stage_id": "perturbative_matching", "inputs": {"quasi": "ft"}},
                }
            },
        },
        tmp_path,
        np.random.default_rng(1),
    )

    check_consistency(context)

    assert not [finding for finding in context.state["consistency"]["findings"] if finding["status"] != "not_checkable"]


def test_review_consistency_ignores_denominator_momentum_but_checks_ensemble(tmp_path: Path) -> None:
    ensemble = EnsembleInfo("HISQ", "HISQa060_X", 0.06, 0.06, 48, 64, 0.3)._asdict()
    context = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "review",
        {"checks": ["kinematics"]},
        {},
        {},
        {
            "result_summary": [
                {
                    "job_id": "ca_p0",
                    "stage_id": "correlator_analysis",
                    "ensemble": ensemble,
                    "attrs": {"momentum_gev": 0.0},
                },
                {
                    "job_id": "ca_p4",
                    "stage_id": "correlator_analysis",
                    "ensemble": ensemble,
                    "attrs": {"momentum_gev": 1.72},
                },
                {
                    "job_id": "rn_p4",
                    "stage_id": "renormalization",
                    "ensemble": ensemble,
                    "attrs": {"momentum_gev": 1.72},
                },
            ],
            "review_bundle": {
                "jobs": {
                    "ca_p0": {"stage_id": "correlator_analysis", "inputs": {}},
                    "ca_p4": {"stage_id": "correlator_analysis", "inputs": {}},
                    "rn_p4": {
                        "stage_id": "renormalization",
                        "inputs": {"target": "ca_p4", "denominator": "ca_p0"},
                    },
                }
            },
        },
        tmp_path,
        np.random.default_rng(1),
    )

    check_consistency(context)

    assert context.state["consistency"]["findings"] == []


def test_review_renderer_uses_chinese_headings_and_selected_references(tmp_path: Path) -> None:
    for name in ("review_bundle.json", "consistency.json", "literature_selection.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "literature").mkdir()
    (tmp_path / "literature/2401.00001.txt").write_text("Paper text.\n", encoding="utf-8")
    context = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "review",
        {"report_language": "ch"},
        {},
        {},
        {
            "review_bundle": {
                "run_id": "run",
                "results": [{}],
                "stage_reports": [
                    {"stage_id": "fourier_transform", "path": "../../03_fourier_transform/report.md", "available": True}
                ],
                "review_summaries": {},
            },
            "consistency": {
                "findings": [
                    {
                        "status": "warning",
                        "group": "units",
                        "source_job": "source",
                        "consumer_job": "consumer",
                        "field": "coord_unit",
                        "message": "deterministic finding text",
                    }
                ],
                "counts": {"error": 0, "warning": 1, "info": 0, "not_checkable": 0},
            },
            "selected_papers": [
                {
                    "id": "2401.00001",
                    "title": "Selected paper",
                    "authors": ["A. Author"],
                    "year": "2024",
                    "source": "https://arxiv.org/abs/2401.00001",
                    "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2401.00001",
                    "retrieval": "ar5iv",
                }
            ],
            "literature_artifacts": ["literature/2401.00001.txt"],
        },
        tmp_path,
        np.random.default_rng(1),
    )

    write_review(
        context,
        title="综述",
        scope_and_provenance="结果范围及来源清楚。",
        workflow_summary="工作流完成。",
        data_and_parameter_coverage="数据与参数覆盖已检查。",
        consistency_analysis="LLM 对结构化一致性警告进行了物理解读。",
        physical_analysis="物理结果。",
        systematics_and_limitations="系统误差。",
        literature_comparison="文献比较。",
        conclusion="结论。",
    )

    report = (tmp_path / "review.md").read_text(encoding="utf-8")
    assert "## 范围与溯源" in report
    assert "LLM 对结构化一致性警告进行了物理解读。" in report
    assert "deterministic finding text" not in report
    assert "## 参考文献" in report
    assert "arXiv:2401.00001" in report
    assert "../../03_fourier_transform/report.md" in report
