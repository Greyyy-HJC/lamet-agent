from __future__ import annotations

import json
from pathlib import Path

from lamet_agent.agent import run_agent
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


def test_run_agent_executes_default_stages_with_mock_model() -> None:
    result = run_agent(_demo_manifest(), model="mock")
    assert result["status"] == "completed"
    assert len(result["completed_stages"]) == 5
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
