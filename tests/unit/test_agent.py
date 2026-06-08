from __future__ import annotations

import json
import urllib.error
from pathlib import Path

from lamet_agent.agent import run_agent
from lamet_agent.core import llm
from lamet_agent.manifest import AnalysisManifest


def _demo_manifest() -> AnalysisManifest:
    return AnalysisManifest.model_validate(
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


def test_run_agent_default_pipeline_stops_for_missing_fourier_input() -> None:
    result = run_agent(_demo_manifest(), model="mock")
    assert result["status"] == "waiting_for_user_input"
    assert result["completed_stages"] == ["correlator_analysis", "renormalization"]
    assert "fourier_transform" in result["pending_user_input"]
    assert result["actions"][0]["action"]["action"] == "call_tool"


def test_run_agent_replays_external_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        json.dumps({"action": "finish", "reason": "done"}) + "\n",
        encoding="utf-8",
    )

    result = run_agent(
        _demo_manifest(),
        model="external",
        actions_path=transcript,
        stages=["correlator_analysis"],
    )

    assert result["status"] == "completed"
    assert result["completed_stages"] == ["correlator_analysis"]
    assert result["actions"][0]["action"]["reason"] == "done"


def test_run_agent_requests_user_input_for_incomplete_fourier_metadata() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "fourier_demo",
            "metadata": {"fourier_input": "matrix_element.npz"},
        }
    )

    result = run_agent(manifest, model="mock", stages=["fourier_transform"])

    assert result["status"] == "waiting_for_user_input"
    assert result["completed_stages"] == []
    action = result["actions"][0]["action"]
    assert action["action"] == "request_user_input"
    assert "Missing metadata.fourier.observable/order" in "\n".join(action["questions"])
    assert result["pending_user_input"]["fourier_transform"] == action["questions"]
    assert result["stage_results"]["fourier_transform"] == []


def test_deepseek_request_retries_transient_url_error(monkeypatch) -> None:
    calls = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"action\":\"finish\",\"reason\":\"done\"}"}}]}).encode()

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError("temporary ssl eof")
        return _Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    action = llm._post_chat_completion(
        messages=[{"role": "user", "content": "finish"}],
        api_key="test-key",
        deepseek_model="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    assert calls["count"] == 2
    assert action["action"] == "finish"
