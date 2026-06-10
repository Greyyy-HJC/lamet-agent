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
    assert "[Observation for LLM]" in text


def test_agent_trace_prints_request_user_input_questions() -> None:
    buffer = io.StringIO()
    trace = AgentTrace(enabled=True, emit=buffer.write)
    trace.model_output(
        {
            "action": "request_user_input",
            "reason": "missing fields",
            "questions": ["metadata.fourier.k_grid is required"],
        }
    )
    text = buffer.getvalue()
    assert "Action: request_user_input" in text
    assert "metadata.fourier.k_grid is required" in text


def test_run_agent_verbose_prints_trace(capsys) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "goal": "full_lamet_pipeline",
            "correlators": [
                {
                    "dataset_id": "c2",
                    "kind": "2pt",
                    "path": "fake/c2.h5",
                    "format": "hdf5",
                }
            ],
            "kernels": [
                {
                    "kernel_id": "k1",
                    "function": "lamet_agent.kernels:identity_kernel",
                }
            ],
        }
    )
    run_agent(manifest, model="mock", stages=["correlator_analysis"], verbose=True)
    out = capsys.readouterr().out
    assert "Agent run: demo" in out
    assert "Cycle 1" in out
    assert "[Stage context]" in out
    assert "[Model output]" in out
