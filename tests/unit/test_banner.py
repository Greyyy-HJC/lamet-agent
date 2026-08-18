"""Unit tests for CLI startup banner and quiet progress headers."""

from __future__ import annotations

import io

from lamet_agent.agent import run_agent
from lamet_agent.core.banner import BANNER, format_job_header
from lamet_agent.core.trace import AgentTrace
from lamet_agent.manifest import AnalysisManifest


def test_banner_contains_lamet_agent() -> None:
    assert BANNER
    assert "LLLL" in BANNER
    assert "AAAA" in BANNER or " AAA " in BANNER
    assert "GGG" in BANNER
    assert "|--" in BANNER


def test_format_job_header() -> None:
    assert format_job_header("correlator_analysis", "ca_ds_pdf") == (
        "Stage: correlator_analysis  |  Job: ca_ds_pdf"
    )


def test_agent_trace_quiet_ui_emits_banner_and_job_header() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=False, quiet_ui=True, emit=buffer.write)
    trace.run_banner(
        run_id="demo",
        backend="mock",
        stages=["correlator_analysis", "renormalization"],
    )
    trace.job_begin("correlator_analysis", "ca", input_issues=["missing field"])
    text = buffer.getvalue()
    assert "LLLL" in text
    assert "Run: demo  backend=mock" in text
    assert "Stages: correlator_analysis, renormalization" in text
    assert "Stage: correlator_analysis  |  Job: ca" in text
    assert "Input issues: ['missing field']" in text
    assert "Cycle 1" not in text
    assert "[Model output]" not in text


def test_agent_trace_verbose_does_not_emit_quiet_ui() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=True, quiet_ui=False, emit=buffer.write)
    trace.run_banner(run_id="demo", backend="mock", stages=["correlator_analysis"])
    trace.job_begin("correlator_analysis", "ca")
    text = buffer.getvalue()
    assert "LLLL" not in text
    assert "Stage: correlator_analysis  |  Job: ca" not in text


def _correlator_manifest() -> AnalysisManifest:
    return AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 1984,
                "stages": ["correlator_analysis"],
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
                        "source_operator": "g5", "sink_operator": "g5", "volume": "S16T32",
                        "momentum": ["PX0PY0PZ0"],
                        "lattice_spacing_fm": 0.1,


                    },
                    {
                        "correlator_id": "c3",
                        "correlator_type": "3pt",
                        "data_path": "c3.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "CG",
                        "source_operator": "g5", "sink_operator": "g5", "volume": "S16T32",
                        "momentum": ["PX0PY0PZ0"],
                        "lattice_spacing_fm": 0.1,


                        "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                        "bT": [0],
                        "bz": [0],
                        "tsep": [8],
                    },
                ],
                "artifacts": [],
                "kernels": [],
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {},
                    "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}],
                }
            },
        }
    )


def test_run_agent_quiet_ui_prints_banner_and_jobs(capsys) -> None:
    run_agent(_correlator_manifest(), backend="mock", verbose=False)
    out = capsys.readouterr().out
    assert "LLLL" in out
    assert "Run: demo  backend=mock" in out
    assert "Stage: correlator_analysis  |  Job: ca" in out
    assert "Cycle 1" not in out
    assert "[Model output]" not in out


def test_run_agent_verbose_does_not_print_banner(capsys) -> None:
    run_agent(_correlator_manifest(), backend="mock", verbose=True)
    out = capsys.readouterr().out
    assert "LLLL" not in out
    assert "Stage: correlator_analysis  |  Job: ca" not in out
    assert "Agent run: demo" in out
    assert "Cycle 1" in out
