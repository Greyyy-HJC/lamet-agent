"""Unit tests for agent trace formatting."""

from __future__ import annotations

import io

from lamet_agent.agent import run_agent
from lamet_agent.core.trace import AgentTrace
from lamet_agent.manifest import AnalysisManifest


def test_agent_trace_emits_cycle_sections() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=True, emit=buffer.write)
    trace.run_begin(run_id="demo", model="mock", stages=["correlator_analysis"])
    trace.stage_begin("correlator_analysis")
    trace.stage_context("static stage context")
    trace.cycle_begin(1)
    trace.llm_call_begin(model="mock")
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
            "questions": ["fourier job k_grid is required"],
        }
    )
    text = buffer.getvalue()
    assert "Action: request_user_input" in text
    assert "fourier job k_grid is required" in text


def test_run_agent_verbose_prints_trace(capsys) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
                "parton": "quark", "resample_mode": "jk", "stages": ["correlator_analysis"],
            },
            "inputs": {"correlators": [
                {"correlator_id": "c2", "kind": "2pt", "data_path": "c2.h5", "ensemble": "E",
                 "hadron": "pion", "gfix": "CG", "source_sink": "SS", "momentum": "P",
                 "a_fm": 0.1, "pz_gev": 0, "src_gamma": "5", "sink_gamma": "5"},
                {"correlator_id": "c3", "kind": "3pt", "data_path": "c3.h5", "ensemble": "E",
                 "hadron": "pion", "gfix": "CG", "source_sink": "SS", "momentum": "P",
                 "a_fm": 0.1, "pz_gev": 0, "src_gamma": "5", "sink_gamma": "5",
                 "current_gamma": "T", "z_direction": "X", "eta": "eta0", "bt": [0], "bz": [0], "tsep": 8}
            ], "artifacts": [], "kernels": []},
            "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
        }
    )
    run_agent(manifest, model="mock", verbose=True)
    out = capsys.readouterr().out
    assert "Agent run: demo" in out
    assert "Cycle 1" in out
    assert "[Stage context]" in out
    assert "[Model output]" in out
