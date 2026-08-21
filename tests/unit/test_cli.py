"""Unit tests for CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lamet_agent.__main__ import _cli_run_summary, _format_cli_error, _resolve_llm_config, app
from lamet_agent.core import llm


@pytest.fixture(autouse=True)
def _stub_available_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm,
        "list_available_models",
        lambda **_kwargs: [
            "gpt-5.6-luna",
            "gpt-4o-mini",
            "deepseek-v4-flash",
            "local-model",
        ],
    )


def test_describe_stage_prints_authoritative_human_reference() -> None:
    result = CliRunner().invoke(app, ["describe-stage", "renormalization"])

    assert result.exit_code == 0, result.output
    assert "renormalization\n===============\n" in result.output
    assert "- scheme [required" in result.output
    assert "Choice behavior:" in result.output
    assert "Cross-parameter and context rules" in result.output


def test_describe_stage_rejects_unknown_stage() -> None:
    result = CliRunner().invoke(app, ["describe-stage", "unknown"])

    assert result.exit_code != 0
    assert "unknown stage 'unknown'" in result.output


def test_cli_run_summary_omits_actions_and_stage_results() -> None:
    full = {
        "run_id": "demo",
        "status": "completed",
        "provider": "codex",
        "provider_type": "cli",
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


def test_resolve_llm_config_passes_codex_model_name() -> None:
    provider_type, provider, model_name, api_key, base_url, key_source = _resolve_llm_config(
        provider="codex",
        model="test-codex-model",
        api_key_file=None,
    )

    assert provider_type == "cli"
    assert provider == "codex"
    assert model_name == "test-codex-model"
    assert api_key is None
    assert base_url is None
    assert key_source is None


def test_resolve_llm_config_api_reads_key_file(tmp_path) -> None:
    key_file = tmp_path / "openai.key"
    key_file.write_text("sk-from-file\n", encoding="utf-8")

    provider_type, provider, model_name, api_key, base_url, key_source = _resolve_llm_config(
        provider="openai",
        model="gpt-4o-mini",
        api_key_file=key_file,
    )

    assert provider_type == "api"
    assert provider == "openai"
    assert model_name == "gpt-4o-mini"
    assert api_key == "sk-from-file"
    assert base_url == "https://api.openai.com/v1/"
    assert key_source == f"file:{key_file}"


def test_resolve_llm_config_api_file_does_not_fall_back_to_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "openai.key"
    key_file.write_text("sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    _provider_type, _provider, _model_name, api_key, _base_url, key_source = _resolve_llm_config(
        provider="openai",
        model="gpt-4o-mini",
        api_key_file=key_file,
    )

    assert api_key == "sk-from-file"
    assert key_source == f"file:{key_file}"


def test_resolve_llm_config_api_missing_file_does_not_use_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    with pytest.raises(typer.BadParameter, match="does not exist"):
        _resolve_llm_config(
            provider="openai",
            model="gpt-4o-mini",
            api_key_file=tmp_path / "missing.key",
        )


def test_resolve_llm_config_api_rejects_empty_file(tmp_path) -> None:
    key_file = tmp_path / "empty.key"
    key_file.write_text("  \n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="is empty"):
        _resolve_llm_config(
            provider="deepseek",
            model="deepseek-v4-flash",
            api_key_file=key_file,
        )


def test_resolve_llm_config_api_uses_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider_type, provider, model_name, api_key, base_url, key_source = _resolve_llm_config(
        provider="deepseek",
        model=None,
        api_key_file=None,
    )

    assert provider_type == "api"
    assert provider == "deepseek"
    assert model_name == "deepseek-v4-flash"
    assert api_key == "sk-from-env"
    assert base_url == "https://api.deepseek.com/"
    assert key_source == "env:DEEPSEEK_API_KEY"


def test_resolve_llm_config_api_requires_matching_provider_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    with pytest.raises(typer.BadParameter, match="DEEPSEEK_API_KEY"):
        _resolve_llm_config(
            provider="deepseek",
            model=None,
            api_key_file=None,
        )


def test_resolve_llm_config_api_requires_key_file_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(typer.BadParameter, match="OPENAI_API_KEY"):
        _resolve_llm_config(
            provider="openai",
            model=None,
            api_key_file=None,
        )


def test_resolve_llm_config_custom_url_requires_model() -> None:
    with pytest.raises(typer.BadParameter, match="requires --model"):
        _resolve_llm_config(
            provider="https://llm.example.test/v1",
            model=None,
            api_key_file=None,
        )


def test_resolve_llm_config_custom_url_requires_key_file(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="custom.*requires --api-key-file"):
        _resolve_llm_config(
            provider="https://llm.example.test/v1",
            model="local-model",
            api_key_file=None,
        )

    key_file = tmp_path / "custom.key"
    key_file.write_text("custom-key\n", encoding="utf-8")
    provider_type, provider, model, api_key, base_url, key_source = (
        _resolve_llm_config(
            provider="https://llm.example.test/v1",
            model="local-model",
            api_key_file=key_file,
        )
    )
    assert provider_type == "api"
    assert provider == base_url == "https://llm.example.test/v1"
    assert model == "local-model"
    assert api_key == "custom-key"
    assert key_source == f"file:{key_file}"


def test_resolve_llm_config_infers_single_loopback_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "local.key"
    key_file.write_text("local-key\n", encoding="utf-8")
    monkeypatch.setattr(
        llm,
        "list_available_models",
        lambda **_kwargs: ["registry/complex-local-model:q4_k_m"],
    )

    provider_type, provider, model, api_key, base_url, _key_source = (
        _resolve_llm_config(
            provider="http://localhost:11434/v1",
            model=None,
            api_key_file=key_file,
        )
    )

    assert provider_type == "api"
    assert provider == base_url == "http://localhost:11434/v1"
    assert model == "registry/complex-local-model:q4_k_m"
    assert api_key == "local-key"


def test_resolve_llm_config_rejects_unavailable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "api.key"
    key_file.write_text("api-key\n", encoding="utf-8")
    monkeypatch.setattr(
        llm,
        "list_available_models",
        lambda **_kwargs: ["model-a", "model-b"],
    )

    with pytest.raises(typer.BadParameter, match="model-a, model-b"):
        _resolve_llm_config(
            provider="https://llm.example.test/v1",
            model="missing-model",
            api_key_file=key_file,
        )


def _cli_combined_output(result) -> str:
    try:
        stderr = result.stderr or ""
    except ValueError:
        stderr = ""
    return f"{result.output}{stderr}"


def test_run_api_prints_env_key_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])
    captured: dict = {}

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("valid manifest must not enter plan mode"),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **kwargs: captured.update(kwargs)
        or {"run_id": "demo", "status": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        ["run", str(manifest), "--provider", "openai"],
    )

    text = _cli_combined_output(result)
    assert result.exit_code == 0, result.output
    assert "LLM PROVIDER" in text
    assert "type=api" in text
    assert "provider=openai" in text
    assert "model=gpt-5.6-luna" in text
    assert "base_url=https://api.openai.com/v1/" in text
    assert "api_key=env:OPENAI_API_KEY" in text
    assert "sk-test" not in text
    assert captured["api_key"] == "sk-test"
    assert captured["provider"] == "openai"
    assert captured["backend"] == "api"
    assert captured["model_name"] == "gpt-5.6-luna"
    assert "\n\n" in text[text.index("LLM PROVIDER"):]


def test_run_api_prints_file_key_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])
    key_file = tmp_path / "api.key"
    key_file.write_text("sk-file-key\n", encoding="utf-8")

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **_kwargs: {"run_id": "demo", "status": "completed"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(manifest),
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-key-file",
            str(key_file),
        ],
    )

    text = _cli_combined_output(result)
    assert result.exit_code == 0, result.output
    assert "LLM PROVIDER" in text
    assert "type=api" in text
    assert f"api_key=file:{key_file}" in text
    assert "sk-file-key" not in text
    assert "sk-should-not-be-used" not in text
    assert "\n\n" in text[text.index("LLM PROVIDER"):]


def test_run_codex_prints_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])
    captured: dict = {}

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **kwargs: captured.update(kwargs)
        or {"run_id": "demo", "status": "completed"},
    )

    result = CliRunner().invoke(
        app,
        ["run", str(manifest), "--provider", "codex", "--model", "test-codex-model"],
    )

    text = _cli_combined_output(result)
    assert result.exit_code == 0, result.output
    assert "LLM PROVIDER" in text
    assert "type=cli" in text
    assert "provider=codex" in text
    assert "model=test-codex-model" in text
    assert "auth=Codex login" in text
    assert "api_key=" not in text
    assert captured["model_name"] == "test-codex-model"
    assert captured["backend"] == "cli"
    assert captured["provider"] == "codex"
    assert "\n\n" in text[text.index("LLM PROVIDER"):]


def test_run_codex_prints_sdk_default_when_model_omitted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **_kwargs: {"run_id": "demo", "status": "completed"},
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "codex"])

    text = _cli_combined_output(result)
    assert result.exit_code == 0, result.output
    assert "type=cli" in text
    assert "model=SDK default" in text


def test_run_api_missing_key_fails_before_agent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda *_args, **_kwargs: pytest.fail("run_agent must not run without an API key"),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        ["run", str(manifest), "--provider", "openai"],
    )

    text = _cli_combined_output(result)
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in text
    assert "--api-key-file" in text


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
            "sample_error_mode": "covariance",
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

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "codex"])

    assert result.exit_code == 0, result.output
    assert "| RUN VALIDATION FAILED" in result.output
    assert "| Falling back to interactive PLAN mode." in result.output
    assert "| No workflow stages will run during this command." in result.output
    assert "| Accepting the plan only writes quick/full manifests." in result.output
    assert "Validation error:\nmissing metadata.stages" in result.output
    assert len(calls) == 1
    manifest_path, kwargs = calls[0]
    assert manifest_path == manifest
    assert kwargs["backend"] == "cli"
    assert kwargs["provider"] == "codex"
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
            "--provider",
            "codex",
            "--model",
            "test-codex-model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "LLM PROVIDER" in _cli_combined_output(result)
    assert "type=cli" in _cli_combined_output(result)
    assert calls[0]["backend"] == "cli"
    assert calls[0]["provider"] == "codex"
    assert calls[0]["model_name"] == "test-codex-model"
    assert calls[0]["base_url"] is None


def test_run_validation_failure_with_unknown_provider_does_not_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "draft.json"

    monkeypatch.setattr(
        "lamet_agent.__main__.validate_manifest_file",
        lambda _manifest: (_ for _ in ()).throw(ValueError("invalid external manifest")),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("unknown provider must not enter plan mode"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "unknown"])

    assert result.exit_code != 0
    assert "Unknown provider" in result.output


def test_run_valid_manifest_does_not_plan(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "valid.json"
    parsed = SimpleNamespace(correlators=[], kernels=[])
    run_calls: list[object] = []

    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_paths", lambda _manifest: None)
    monkeypatch.setattr("lamet_agent.__main__.validate_manifest_file", lambda _manifest: parsed)
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("valid manifest must not enter plan mode"),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda value, **_kwargs: run_calls.append(value) or {"run_id": "demo", "status": "completed"},
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "codex"])

    assert result.exit_code == 0, result.output
    assert run_calls == [parsed]
    assert '"status": "completed"' in result.output
    assert "LLM PROVIDER" in result.output


def _write_matching_manifest(path, *, out_of_range_lc: bool = False, unused_review: bool = False) -> None:
    project_root = Path(__file__).resolve().parents[2]
    artifact_path = path.parent / "rn.bin"
    artifact_path.write_bytes(b"artifact")
    matching_defaults = {
        "scheme": "ratio",
        "mu": 2.0,
        "lc_x_ls": {"start": -3.0, "stop": 3.0} if out_of_range_lc else {"start": 0.0, "stop": 1.0},
    }
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(project_root),
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform", "perturbative_matching"],
        },
        "inputs": {
            "correlators": [],
            "artifacts": [{"id": "rn", "stage": "renormalization", "path": str(artifact_path)}],
            "kernels": [
                {
                    "stage": "perturbative_matching",
                    "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
                    "kernel_path": "lamet_agent/kernels.py",
                    "kernel_parameters": {},
                }
            ],
        },
        "stages": {
            "fourier_transform": {
                "defaults": {
                    "gfix": "GI", "order": ["LA"], "sector": "valence", "Lambda0_gev": 0.0,
                    "posterior_prior_error_scale": 3.0,
                    "scheme_scan": {"zmin_fm": [0.1], "zmax_fm": [0.8], "zmax_ext_fm": 1.2, "smooth": "linear", "model_average": False},
                    "quasi_y_ls": {"start": -2.0, "stop": 2.0, "num": 100},
                },
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


def test_validate_rejects_missing_input_path(tmp_path) -> None:
    manifest = tmp_path / "missing-artifact.json"
    _write_matching_manifest(manifest)
    artifact_path = tmp_path / "rn.bin"
    artifact_path.unlink()

    result = CliRunner().invoke(app, ["validate", str(manifest)])

    assert result.exit_code != 0
    assert "inputs.artifacts[0].path does not exist" in result.output
    assert str(artifact_path) in result.output


def test_run_path_failure_enters_path_repair_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "missing-artifact.json"
    _write_matching_manifest(manifest)
    (tmp_path / "rn.bin").unlink()
    calls: list[tuple[object, dict]] = []

    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda manifest_path, **kwargs: calls.append((manifest_path, kwargs)),
    )
    monkeypatch.setattr(
        "lamet_agent.__main__.run_agent",
        lambda *_args, **_kwargs: pytest.fail("run_agent must not run after path validation failure"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "codex"])

    assert result.exit_code == 0, result.output
    assert "inputs.artifacts[0].path does not exist" in result.output
    assert len(calls) == 1
    assert calls[0][0] == manifest
    assert calls[0][1]["path_repair_project_root"] == Path(__file__).resolve().parents[2]


def test_run_path_failure_with_unknown_provider_does_not_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "missing-artifact.json"
    _write_matching_manifest(manifest)
    (tmp_path / "rn.bin").unlink()
    monkeypatch.setattr(
        "lamet_agent.__main__.run_interactive_plan",
        lambda *_args, **_kwargs: pytest.fail("unknown provider must not enter path repair"),
    )

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "unknown"])

    assert result.exit_code != 0
    assert "Unknown provider" in result.output


def test_validate_reports_matching_lc_window_outside_fourier_grid(tmp_path) -> None:
    manifest = tmp_path / "matching.json"
    _write_matching_manifest(manifest, out_of_range_lc=True)

    result = CliRunner().invoke(app, ["validate", str(manifest)])

    assert result.exit_code != 0
    assert '"code": "matching.lc_x_ls.window"' in result.output
    assert "extends beyond" in result.output
    assert '"status": "invalid"' in result.output


def test_validate_reports_structured_fourier_physics_issue(tmp_path) -> None:
    manifest = tmp_path / "matching.json"
    _write_matching_manifest(manifest)

    result = CliRunner().invoke(app, ["validate", str(manifest)])

    assert result.exit_code != 0
    assert '"code": "fourier.kinematics.momentum_required"' in result.output
    assert '"path": "stages.fourier_transform.jobs.ft.inputs"' in result.output
    assert "Converting coordinate separation to Ioffe time" in result.output
    assert "Declare discrete momentum, volume, and lattice_spacing_fm" in result.output


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

    result = CliRunner().invoke(app, ["run", str(manifest), "--provider", "codex"])

    assert result.exit_code == 0, result.output
    assert "| RUN VALIDATION FAILED" in result.output
    assert "unused stages" in result.output
    assert "pydantic.dev" not in result.output
    assert "For further information visit" not in result.output
    assert len(calls) == 1
    assert calls[0][0] == manifest
