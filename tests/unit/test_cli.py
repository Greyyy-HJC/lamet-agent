"""Unit tests for CLI helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lamet_agent.__main__ import _cli_run_summary, _resolve_llm_config, app


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


def test_run_validation_failure_falls_back_to_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "draft.json"
    calls: list[tuple[object, dict]] = []

    def fail_validation(_manifest):
        raise ValueError("missing metadata.stages")

    def fake_plan(manifest_path, **kwargs):
        calls.append((manifest_path, kwargs))
        return None

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", fail_validation)
    monkeypatch.setattr("lamet_agent.__main__.run_interactive_plan", fake_plan)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda *_args, **_kwargs: pytest.fail("run_agent must not run after validation failure"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--backend", "mock"])

    assert result.exit_code == 0, result.output
    assert "| RUN VALIDATION FAILED" in result.output
    assert "| Falling back to interactive PLAN mode." in result.output
    assert "| No workflow stages will run during this command." in result.output
    assert "| Accepting the plan only writes quick/full manifests." in result.output
    assert "Validation error:\nmissing metadata.stages" in result.output
    assert len(calls) == 1
    manifest_path, kwargs = calls[0]
    assert manifest_path == manifest
    assert kwargs["backend"] == "mock"
    assert kwargs["provider"] is None
    assert kwargs["model_name"] is None
    assert kwargs["api_key"] is None
    assert kwargs["base_url"] is None


def test_run_validation_fallback_forwards_codex_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "draft.json"
    calls: list[dict] = []

    monkeypatch.setattr(
        "lamet_agent.__main__.validate_manifest_file",
        lambda _manifest: (_ for _ in ()).throw(ValueError("invalid draft")),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda _manifest, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda *_args, **_kwargs: pytest.fail("run_agent must not run after validation failure"),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(manifest),
            "--backend",
            "codex",
            "--model",
            "test-codex-model",
            "--base-url",
            "https://example.invalid/v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["backend"] == "codex"
    assert calls[0]["provider"] is None
    assert calls[0]["model_name"] == "test-codex-model"
    assert calls[0]["base_url"] == "https://example.invalid/v1"


def test_run_validation_failure_with_external_backend_does_not_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "draft.json"

    monkeypatch.setattr(
        "lamet_agent.__main__.validate_manifest_file",
        lambda _manifest: (_ for _ in ()).throw(ValueError("invalid external manifest")),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("external backend must not enter plan mode"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--backend", "external"])

    assert result.exit_code != 0
    assert "invalid external manifest" in result.output
    assert "falling back" not in result.output


def test_run_valid_manifest_does_not_plan(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])
    run_calls: list[object] = []

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("valid manifest must not enter plan mode"),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **_kwargs: run_calls.append(value) or {"run_id": "demo", "status": "completed"},
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--backend", "mock"])

    assert result.exit_code == 0, result.output
    assert run_calls == [parsed]
    assert '"status": "completed"' in result.output
