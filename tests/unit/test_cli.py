"""Unit tests for CLI helpers."""

from __future__ import annotations

from lamet_agent.__main__ import _cli_run_summary, _resolve_llm_config


def test_cli_run_summary_omits_actions_and_stage_results() -> None:
    full = {
        "run_id": "demo",
        "status": "completed",
        "backend": "mock",
        "stages": ["correlator_analysis"],
        "completed_stages": ["correlator_analysis"],
        "input_issues": {},
        "pending_user_input": {},
        "summary": '{"action_count": 3}',
        "manifest": "m.json",
        "correlators": ["c2"],
        "kernels": ["k1"],
        "actions": [{"stage": "correlator_analysis", "action": {}}],
        "stage_results": {"correlator_analysis": []},
    }
    compact = _cli_run_summary(full)
    assert "actions" not in compact
    assert "stage_results" not in compact
    assert "input_issues" not in compact
    assert compact["run_id"] == "demo"
    assert compact["manifest"] == "m.json"
    assert compact["pending_user_input"] == {}


def test_resolve_llm_config_passes_codex_model_name(tmp_path) -> None:
    provider, model_name, api_key, base_url = _resolve_llm_config(
        backend="codex",
        model="test-codex-model",
        api_key_file=tmp_path / "missing-api-key",
        base_url=None,
    )

    assert provider is None
    assert model_name == "test-codex-model"
    assert api_key is None
    assert base_url is None
