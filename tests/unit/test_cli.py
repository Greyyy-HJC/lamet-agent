"""Unit tests for CLI helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lamet_agent.__main__ import _cli_run_summary, _format_cli_error, _resolve_llm_config, app


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


def test_format_cli_error_strips_pydantic_docs_url() -> None:
    from pydantic import ValidationError

    from lamet_agent.manifest import AnalysisManifest

    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": ".",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {
            "correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]},
            "review": {"defaults": {}, "jobs": [{"id": "review"}]},
        },
    }
    with pytest.raises(ValidationError) as exc_info:
        AnalysisManifest.model_validate(payload)

    formatted = _format_cli_error(exc_info.value)
    assert "unused stages" in formatted
    assert "pydantic.dev" not in formatted
    assert "For further information visit" not in formatted
    assert "input_value" not in formatted


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


def _write_matching_manifest(path, *, denser_lc: bool = False, unused_review: bool = False) -> None:
    matching_defaults = {"scheme": "ratio"}
    if denser_lc:
        matching_defaults["lc_x_ls"] = {"start": -1.0, "stop": 2.0, "num": 300}
    else:
        matching_defaults["quasi_y_ls"] = {"start": -2.0, "stop": 2.0, "num": 400}
        matching_defaults["lc_x_ls"] = {"start": 0.0, "stop": 1.0, "num": 80}
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(path.parent),
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "random_seed": 1984,
            "stages": ["fourier_transform", "perturbative_matching"],
        },
        "inputs": {
            "correlators": [],
            "artifacts": [{"id": "rn", "stage": "renormalization", "path": "rn.nc"}],
            "kernels": [
                {
                    "stage": "perturbative_matching",
                    "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
                    "kernel_path": "kernels.py",
                    "kernel_parameters": {},
                }
            ],
        },
        "stages": {
            "fourier_transform": {
                "defaults": {"order": ["LA"], "y_grid": {"start": -2.0, "stop": 2.0, "num": 100}},
                "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
            },
            "perturbative_matching": {
                "defaults": matching_defaults,
                "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}}],
            },
        },
    }
    if unused_review:
        payload["stages"]["review"] = {
            "defaults": {"literature": False, "literature_max_papers": 4},
            "jobs": [{"id": "review"}],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_prints_matching_grid_warning_and_fails(tmp_path) -> None:
    manifest = tmp_path / "matching.json"
    _write_matching_manifest(manifest, denser_lc=True)

    result = CliRunner().invoke(app, ["validate", str(manifest)])

    assert result.exit_code != 0
    assert "WARNING: MATCHING GRID DENSITY" in result.output
    assert "oscillate" in result.output
    assert '"status": "invalid"' in result.output
    assert "Matching job 'mt'" in result.output


def test_run_prints_matching_grid_warning_without_planning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "matching.json"
    _write_matching_manifest(manifest, denser_lc=True)
    parsed_holder: list[object] = []

    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("matching-grid warning must not enter plan mode"),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **_kwargs: parsed_holder.append(value) or {"run_id": "demo", "status": "completed"},
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--backend", "mock"])

    assert result.exit_code == 0, result.output
    assert "WARNING: MATCHING GRID DENSITY" in result.output
    assert parsed_holder


def test_validate_rejects_unused_stage_configuration(tmp_path) -> None:
    manifest = tmp_path / "unused.json"
    _write_matching_manifest(manifest, unused_review=True)

    result = CliRunner().invoke(app, ["validate", str(manifest)])

    assert result.exit_code != 0
    assert "unused stages" in result.output
    assert "pydantic.dev" not in result.output
    assert "For further information visit" not in result.output


def test_run_unused_stage_falls_back_to_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "unused.json"
    _write_matching_manifest(manifest, unused_review=True)
    calls: list[object] = []

    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda manifest_path, **kwargs: calls.append((manifest_path, kwargs)),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda *_args, **_kwargs: pytest.fail("run_agent must not run after unused-stage validation failure"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--backend", "mock"])

    assert result.exit_code == 0, result.output
    assert "| RUN VALIDATION FAILED" in result.output
    assert "unused stages" in result.output
    assert "pydantic.dev" not in result.output
    assert "For further information visit" not in result.output
    assert len(calls) == 1
    assert calls[0][0] == manifest
