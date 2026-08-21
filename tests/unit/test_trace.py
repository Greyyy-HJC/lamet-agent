"""Unit tests for agent trace formatting."""

from __future__ import annotations

import io

from lamet_agent.agent import run_agent
from lamet_agent.core.trace import AgentTrace
from lamet_agent.manifest import AnalysisManifest


class _InputSession:
    def begin_stage(self, static_user: str) -> None:
        pass

    def decide(self, *, last_observation: dict | None) -> dict:
        return {
            "action": "request_user_input",
            "reason": "Test-only pause.",
            "args": {"prompt": "Continue?"},
        }


def test_agent_trace_emits_cycle_sections() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=True, emit=buffer.write)
    trace.run_begin(run_id="demo", backend="cli", stages=["correlator_analysis"])
    trace.stage_begin("correlator_analysis")
    trace.stage_context("static stage context")
    trace.cycle_begin(1)
    trace.llm_call_begin(backend="cli")
    trace.llm_call_end()
    trace.model_output(
        {
            "action": "call_tool",
            "reason": "inspect scale first",
            "tool_name": "inspect_correlator_scale",
            "args": {"pt2_path": "fake.h5"},
        }
    )
    trace.observation({"tool_name": "inspect_correlator_scale", "result": {"Lt": 24}})
    trace.cycle_begin(2)
    trace.prompt_delta({"tool_name": "inspect_correlator_scale", "result": {"Lt": 24}})
    text = buffer.getvalue()
    assert "Cycle 1" in text
    assert "[Stage context]" in text
    assert "static stage context" in text
    assert "[Prompt to LLM]" not in text
    assert "Reason: inspect scale first" in text
    assert "inspect_correlator_scale" in text
    assert "[Observation]" in text
    assert "[Observation forwarded to LLM: inspect_correlator_scale]" in text
    assert text.count('"Lt": 24') == 1


def test_agent_trace_prints_request_user_input_questions() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=True, emit=buffer.write)
    trace.model_output(
        {
            "action": "request_user_input",
            "reason": "missing fields",
            "questions": ["fourier job quasi_y_ls is required"],
        }
    )
    text = buffer.getvalue()
    assert "Action: request_user_input" in text
    assert "fourier job quasi_y_ls is required" in text


def test_quiet_trace_shows_llm_and_tool_progress() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=False, quiet_ui=True, emit=buffer.write)
    trace.llm_call_begin(backend="api", model_spec="deepseek/deepseek-v4-flash")
    trace.llm_call_end()
    trace.tool_call_begin("run_lanczos_analysis")
    trace.tool_call_end("run_lanczos_analysis", succeeded=True)
    trace.report_begin(
        "perturbative_matching",
        detail="cited-paper lookup and report-only LLM formula",
    )
    trace.report_end("perturbative_matching", succeeded=True)
    text = buffer.getvalue()
    assert "Calling LLM (deepseek/deepseek-v4-flash)..." in text
    assert "LLM response received." in text
    assert "Running tool: run_lanczos_analysis..." in text
    assert "Tool completed: run_lanczos_analysis." in text
    assert "Writing stage report: perturbative_matching" in text
    assert "Stage report completed: perturbative_matching." in text


def test_run_agent_verbose_prints_trace(capsys) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
                "parton": "quark", "resample_mode": "jk", "sample_error_mode": "covariance", "random_seed": 1984, "stages": ["correlator_analysis"],
            },
            "inputs": {"correlators": [
                {"correlator_id": "c2", "correlator_type": "2pt", "data_path": "c2.h5", "ensemble": "E",
                 "hadron": "pion", "gfix": "CG", "source_operator": "g5", "sink_operator": "g5", "volume": "S16T32", "momentum": ["PX0PY0PZ0"],
                 "lattice_spacing_fm": 0.1,  },
                {"correlator_id": "c3", "correlator_type": "3pt", "data_path": "c3.h5", "ensemble": "E",
                 "hadron": "pion", "gfix": "CG", "source_operator": "g5", "sink_operator": "g5", "volume": "S16T32", "momentum": ["PX0PY0PZ0"],
                 "lattice_spacing_fm": 0.1,
                 "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",   "bT": [0], "bz": [0], "tsep": [8]}
            ], "artifacts": [], "kernels": []},
            "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
        }
    )
    run_agent(manifest, backend="test", verbose=True, session=_InputSession())
    out = capsys.readouterr().out
    assert "Agent run: demo" in out
    assert "Cycle 1" in out
    assert "[Stage context]" in out
    assert "[Model output]" in out
