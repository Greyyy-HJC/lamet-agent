from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from lamet_agent.agent import AgentState, _hydrate_external_artifact_inputs, _run_job, run_agent
from lamet_agent import agent as agent_module
from lamet_agent.core import llm
from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.core.tools import resolve_stage_tools
from lamet_agent.core.trace import AgentTrace
from lamet_agent.manifest import AnalysisManifest, validate_manifest_file
from lamet_agent.manifest_params import merge_stage_params


def _demo_manifest() -> AnalysisManifest:
    return AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
                "parton": "quark", "resample_mode": "jk", "random_seed": 1984, "stages": ["correlator_analysis"],
            },
            "inputs": {"correlators": [], "artifacts": [], "kernels": []},
            "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]}},
        }
    )


def test_correlator_sample0_plot_collection_includes_qda_artifacts() -> None:
    result = {
        "z_fits": [
            {
                "z": 2,
                "sample0_plot_paths": {
                    "qda_ratio_re_pdf": "fit_logs/qda_bz2_re.pdf",
                    "qda_ratio_re_svg": "fit_logs/qda_bz2_re.svg",
                },
            }
        ]
    }
    assert agent_module._correlator_sample0_plots(result) == [
        "fit_logs/qda_bz2_re.pdf",
        "fit_logs/qda_bz2_re.svg",
    ]


def test_run_agent_uses_manifest_stage_order(tmp_path: Path, monkeypatch) -> None:
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "action": "call_tool",
                        "tool_name": "mark_done",
                        "args": {},
                        "reason": "produce output",
                    }
                ),
                json.dumps({"action": "finish", "reason": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def mark_done(store, **kwargs):
        store["output"] = "ok"
        return {"ok": True}

    monkeypatch.setattr(
        "lamet_agent.agent.resolve_stage_tools",
        lambda stage: {"mark_done": mark_done},
    )
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])
    result = run_agent(_demo_manifest(), backend="external", actions_path=transcript)

    assert result["status"] == "completed"
    assert result["completed_stages"] == ["correlator_analysis"]
    assert result["actions"][-1]["action"]["reason"] == "done"


def test_run_agent_stops_on_request_user_input(tmp_path: Path, monkeypatch) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["correlator_analysis", "review"],
            },
            "inputs": {"correlators": [], "artifacts": [], "kernels": []},
            "stages": {
                "correlator_analysis": {
                    "defaults": {},
                    "jobs": [{"id": "ca_first"}, {"id": "ca_second"}],
                },
                "review": {"defaults": {}, "jobs": [{"id": "review_job"}]},
            },
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "action": "request_user_input",
                "reason": "Need a narrower tune_z grid.",
                "args": {
                    "prompt": "Which tune_z_values should I retry with?",
                    "question_id": "tune_z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", lambda stage: {})
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    result = run_agent(manifest, backend="external", actions_path=transcript)

    assert result["status"] == "waiting_for_user_input"
    assert result["pending_user_input"] == {
        "correlator_analysis": {"ca_first": ["Which tune_z_values should I retry with?"]}
    }
    assert "ca_second" not in result["stage_results"].get("correlator_analysis", {})
    assert "review" not in result["stage_results"]
    assert result["completed_stages"] == []


def test_run_agent_raises_when_job_finishes_without_output(tmp_path: Path, monkeypatch) -> None:
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        json.dumps({"action": "finish", "reason": "done without output"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", lambda stage: {})
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    with pytest.raises(ValueError, match="finished without store\\['output'\\]"):
        run_agent(_demo_manifest(), backend="external", actions_path=transcript)


def test_run_agent_reports_explicit_codex_model(monkeypatch) -> None:
    def fake_codex_decide(
        messages: list[dict[str, str]],
        *,
        model_name: str | None = None,
    ) -> dict:
        assert model_name == "test-codex-model"
        return {"action": "finish", "reason": "done"}

    monkeypatch.setattr(llm, "_codex_decide", fake_codex_decide)
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    result = run_agent(
        _demo_manifest(),
        backend="codex",
        model_name="test-codex-model",
    )

    assert result["model"] == "test-codex-model"


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
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    assert calls["count"] == 2
    assert action["action"] == "finish"


def test_provider_json_parse_error_gets_repair_retry(monkeypatch) -> None:
    calls = {"count": 0}
    bodies: list[dict] = []

    class _Response:
        def __init__(self, content: str) -> None:
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": self._content}}]}).encode()

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        bodies.append(json.loads(request.data.decode("utf-8")))
        if calls["count"] == 1:
            return _Response('{"action":"finish" "reason":"missing comma"}')
        return _Response('{"action":"finish","reason":"done"}')

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    action = llm._post_chat_completion(
        messages=[{"role": "user", "content": "finish"}],
        api_key="test-key",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    assert calls["count"] == 2
    assert action == {"action": "finish", "reason": "done"}
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert "not valid JSON" in bodies[1]["messages"][-1]["content"]


def test_provider_config_exposes_deepseek_and_openai() -> None:
    assert llm.provider_config("deepseek")["base_url"] == "https://api.deepseek.com"
    openai = llm.provider_config("openai")
    assert openai["base_url"] == "https://api.openai.com/v1"
    assert openai["default_model"] == "gpt-4o-mini"
    assert openai["key_env"] == "OPENAI_API_KEY"
    assert llm.provider_config("mock") is None


def test_openai_request_targets_openai_endpoint_and_model(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "{\"action\":\"finish\",\"reason\":\"done\"}"}}]}
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.headers.get("Authorization")
        return _Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    action = llm._request_llm_action(
        backend="api",
        messages=[{"role": "user", "content": "go"}],
        api_key="sk-test",
        provider="openai",
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["auth"] == "Bearer sk-test"
    assert action["action"] == "finish"


def test_parse_api_model_accepts_provider_and_model_id() -> None:
    assert llm.parse_api_model("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")
    assert llm.parse_api_model("openai/gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_parse_api_model_provider_shorthand_uses_default_model() -> None:
    assert llm.parse_api_model("openai") == ("openai", "gpt-4o-mini")
    assert llm.parse_api_model("deepseek") == ("deepseek", "deepseek-chat")


def test_parse_api_model_rejects_unknown_provider() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown API provider"):
        llm.parse_api_model("unknown/foo")


def test_make_llm_session_unknown_backend_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown LLM backend"):
        llm.make_llm_session("deeepseek", None)


def test_make_llm_session_api_requires_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="openai"):
        llm.make_llm_session("api", None, api_key=None, provider="openai")
    session = llm.make_llm_session("api", None, api_key="sk-test", provider="openai")
    assert hasattr(session, "decide")


def test_make_llm_session_codex_uses_codex_decide(monkeypatch) -> None:
    captured: list[tuple[list[dict[str, str]], str | None]] = []

    def fake_codex_decide(
        messages: list[dict[str, str]],
        *,
        model_name: str | None = None,
    ) -> dict:
        captured.append((messages, model_name))
        return {"action": "finish", "reason": "done"}

    monkeypatch.setattr(llm, "_codex_decide", fake_codex_decide)

    session = llm.make_llm_session("codex", model_name="test-codex-model")
    session.begin_stage("stage prompt")
    action = session.decide(last_observation={"tool_name": "inspect", "result": {"ok": True}})

    assert action == {"action": "finish", "reason": "done"}
    messages, model_name = captured[0]
    assert model_name == "test-codex-model"
    assert messages[0]["role"] == "system"
    assert "LaMET analysis agent" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "stage prompt"}
    assert messages[2]["role"] == "user"
    assert "Tool result" in messages[2]["content"]


def test_codex_decide_does_not_pass_strict_output_schema(monkeypatch) -> None:
    captured: dict = {}

    class _Sandbox:
        read_only = "read-only"

    class _Thread:
        def run(self, task_input, **kwargs):
            captured["task_input"] = task_input
            captured["run_kwargs"] = kwargs
            return types.SimpleNamespace(final_response='{"action":"finish","reason":"done"}')

    class _Codex:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thread_start(self, **kwargs):
            captured["thread_start_kwargs"] = kwargs
            return _Thread()

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        types.SimpleNamespace(Codex=_Codex, Sandbox=_Sandbox),
    )

    action = llm._codex_decide(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "stage prompt"},
        ],
        model_name="test-codex-model",
    )

    assert action == {"action": "finish", "reason": "done"}
    assert captured["thread_start_kwargs"]["developer_instructions"] == "system instructions"
    assert captured["thread_start_kwargs"]["sandbox"] == _Sandbox.read_only
    assert captured["thread_start_kwargs"]["ephemeral"] is True
    assert captured["thread_start_kwargs"]["model"] == "test-codex-model"
    assert captured["run_kwargs"] == {"sandbox": _Sandbox.read_only}
    assert "stage prompt" in captured["task_input"]


def test_request_llm_text_passes_codex_model_to_thread(monkeypatch) -> None:
    captured: dict = {}

    class _Sandbox:
        read_only = "read-only"

    class _Thread:
        def run(self, task_input, **kwargs):
            return types.SimpleNamespace(final_response="plan response")

    class _Codex:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def thread_start(self, **kwargs):
            captured.update(kwargs)
            return _Thread()

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        types.SimpleNamespace(Codex=_Codex, Sandbox=_Sandbox),
    )

    response = llm.request_llm_text(
        backend="codex",
        messages=[
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "planning prompt"},
        ],
        model_name="test-codex-model",
    )

    assert response == "plan response"
    assert captured["model"] == "test-codex-model"


def test_run_agent_registers_job_output_for_downstream_role(tmp_path: Path, monkeypatch) -> None:
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"action": "call_tool", "tool_name": "set_value", "args": {}, "reason": "set"}),
                json.dumps({"action": "finish", "reason": "first done"}),
                json.dumps({"action": "call_tool", "tool_name": "read_value", "args": {}, "reason": "read"}),
                json.dumps({"action": "finish", "reason": "second done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def set_value(store):
        store["output"] = "ok"
        return {"out": "output"}

    def read_value(store):
        store["output"] = store["input"]
        return {"value": store["input"]}

    def fake_tools(stage):
        if stage == "correlator_analysis":
            return {"set_value": set_value}
        if stage == "renormalization":
            return {"read_value": read_value}
        return {}

    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", fake_tools)
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    manifest = AnalysisManifest.model_validate({
        "metadata": {
            "run_id": "dag", "root_directory": ".", "target_observable": "pdf",
            "parton": "quark", "resample_mode": "jk", "random_seed": 1984,
            "stages": ["correlator_analysis", "renormalization"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {
            "correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]},
            "renormalization": {"defaults": {}, "jobs": [{"id": "rn", "inputs": {"input": "ca"}}]},
        },
    })

    result = run_agent(
        manifest,
        backend="external",
        actions_path=transcript,
    )

    assert result["status"] == "completed"
    assert result["stage_results"]["renormalization"]["rn"][0]["result"] == {"value": "ok"}


def test_self_renorm_apply_job_rejects_fit_tool_then_recovers(tmp_path: Path, monkeypatch) -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    manifest._artifacts_directory = tmp_path
    job = manifest.stages["renormalization"].jobs[1]
    fit_called = False
    apply_kwargs: dict[str, object] = {}

    def forbidden_fit(store, **kwargs):
        nonlocal fit_called
        fit_called = True
        raise AssertionError("fit tool must not run for an apply job")

    def apply(store, **kwargs):
        apply_kwargs.update(kwargs)
        store["output"] = "renormalized"
        return {"artifact": str(tmp_path / "rn.nc")}

    def diagnostics(store, **kwargs):
        return {"plots": {"zmsbar_compare": str(tmp_path / "diag.pdf")}}

    def plot(store, **kwargs):
        return {"plot": str(tmp_path / "rn.pdf")}

    stage_tools = {
        "fit_self_renormalization_factor": forbidden_fit,
        "apply_self_renormalization": apply,
        "plot_self_renormalization_diagnostics": diagnostics,
        "plot_renormalized_matrix_element": plot,
    }
    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", lambda stage: stage_tools)

    class ScriptedSession:
        def __init__(self):
            self.actions = iter([
                {"action": "call_tool", "tool_name": "fit_self_renormalization_factor", "args": {}},
                {"action": "request_user_input", "reason": "not needed"},
                {
                    "action": "call_tool",
                    "tool_name": "apply_self_renormalization",
                    "args": {
                        "target": "bare_a06m130_pz6",
                        "zR": "rn_zr_fit",
                        "kernel_id": None,
                        "order": None,
                        "Nf": None,
                        "save_path": None,
                    },
                },
                {"action": "finish", "reason": "too early"},
                {"action": "call_tool", "tool_name": "plot_self_renormalization_diagnostics", "args": {}},
                {"action": "call_tool", "tool_name": "plot_renormalized_matrix_element", "args": {}},
                {"action": "finish", "reason": "done"},
            ])

        def begin_stage(self, prompt):
            assert "hybrid_self_renormalization" in prompt
            assert '"apply_self_renormalization"' in prompt
            assert '"fit_self_renormalization_factor"' not in prompt.split("Stage instruction:", 1)[0]

        def decide(self, *, last_observation=None):
            return next(self.actions)

    observations = _run_job(
        "renormalization",
        job,
        manifest,
        AgentState(run_id=manifest.run_id),
        ScriptedSession(),
        input_issues=[],
        max_tool_steps=7,
        backend="external",
        model_spec=None,
        trace=AgentTrace(enabled=False),
        store={"target": "bare", "zR": "factor"},
        report_language="en",
    )

    assert fit_called is False
    assert apply_kwargs["target"] == "target"
    assert apply_kwargs["zR"] == "zR"
    assert apply_kwargs["save_path"] is not None
    assert "order" not in apply_kwargs
    assert "Nf" not in apply_kwargs
    assert observations[0]["tool_name"] == "fit_self_renormalization_factor"
    assert "must call required tool 'apply_self_renormalization' next" in observations[0]["error"]
    assert "cannot request_user_input" in observations[1]["error"]
    assert observations[2]["result"]["artifact"].endswith("rn.nc")
    assert "cannot finish" in observations[3]["error"]
    assert observations[4]["result"]["plots"]["zmsbar_compare"].endswith("diag.pdf")
    assert observations[5]["result"]["plot"].endswith("rn.pdf")


def test_self_renorm_job_fails_when_required_tool_never_succeeds(tmp_path: Path, monkeypatch) -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    manifest._artifacts_directory = tmp_path
    job = manifest.stages["renormalization"].jobs[1]

    def failing_apply(store, **kwargs):
        raise ValueError("synthetic apply failure")

    monkeypatch.setattr(
        "lamet_agent.agent.resolve_stage_tools",
        lambda stage: {"apply_self_renormalization": failing_apply},
    )

    class RetryingSession:
        def begin_stage(self, prompt):
            pass

        def decide(self, *, last_observation=None):
            return {"action": "call_tool", "tool_name": "apply_self_renormalization", "args": {}}

    with pytest.raises(
        ValueError,
        match=r"renormalization/rn_a06m130_pz6.*apply_self_renormalization.*synthetic apply failure",
    ):
        _run_job(
            "renormalization",
            job,
            manifest,
            AgentState(run_id=manifest.run_id),
            RetryingSession(),
            input_issues=[],
            max_tool_steps=2,
            backend="external",
            model_spec=None,
            trace=AgentTrace(enabled=False),
            store={"target": "bare", "zR": "factor"},
            report_language="en",
        )


def _write_renorm_nc(path: Path) -> None:
    coord = np.arange(0.0, 5.0)
    base_re = np.exp(-0.45 * coord)
    base_im = 0.1 * np.exp(-0.45 * coord)
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[
            base_re + 1j * base_im,
            1.01 * base_re + 0.98j * base_im,
            0.99 * base_re + 1.02j * base_im,
        ],
        dims=("z",),
        coords={"z": coord.tolist()},
        attrs={
            "momentum": "PX5PY0PZ0",
            "volume": "S48T64",
            "lattice_spacing_fm": "0.0574",
            "hadron": "pion",
            "gfix": "CG",
            "bz_direction": "X",
        },
        name="renormalized_matrix_element",
    )
    data.to_netcdf(path)


def _ensemble_info(name: str) -> EnsembleInfo:
    return EnsembleInfo("test", name, 0.06, 0.06, 48, 96, 0.14)


def _write_matrix_nc(path: Path, *, ensemble: str, momentum_gev: float, attrs: dict[str, str] | None = None) -> None:
    metadata = {"ensemble": ensemble, "momentum_gev": str(momentum_gev), "fitting_form": "Breit"}
    metadata.update(attrs or {})
    EnsembleData(
        ensemble=_ensemble_info(ensemble),
        resample="jackknife",
        values=[
            np.array([1.0 + 0.2j, 0.8 + 0.1j]),
            np.array([1.1 + 0.3j, 0.9 + 0.2j]),
            np.array([0.9 + 0.1j, 0.7 + 0.0j]),
        ],
        dims=("z",),
        coords={"z": [0, 1]},
        attrs=metadata,
        name="matrix_element",
    ).to_netcdf(path)


def test_matrix_overlay_single_job_is_skipped(tmp_path: Path) -> None:
    nc_path = tmp_path / "ca_p1.nc"
    _write_matrix_nc(nc_path, ensemble="ens", momentum_gev=1.0)

    artifacts = agent_module._write_matrix_overlay_artifacts(
        [{"job_id": "ca_p1", "artifacts": {"bare_artifact": str(nc_path)}, "result": {}}],
        tmp_path,
        artifact_key="bare_artifact",
        prefix="ca",
        title_suffix="bare matrix elements",
        y_label="Bare",
    )

    assert artifacts == {}
    assert not (tmp_path / "ca_ens_re.pdf").exists()


def test_matrix_overlay_generates_re_im_with_x_shift(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / "ca_p1.nc", tmp_path / "ca_p2.nc"]
    _write_matrix_nc(paths[0], ensemble="ens", momentum_gev=1.0)
    _write_matrix_nc(
        paths[1],
        ensemble="ens",
        momentum_gev=2.0,
        attrs={"fitting_form": "NonBreit", "initial_momentum_gev": "2.0", "final_momentum_gev": "1.0"},
    )
    x_values: list[np.ndarray] = []
    labels: list[str] = []
    real_default_plot = agent_module.default_plot

    def capture_plot():
        fig, ax = real_default_plot()
        real_errorbar = ax.errorbar

        def capture_errorbar(x, *args, **kwargs):
            x_values.append(np.asarray(x, dtype=float))
            labels.append(str(kwargs.get("label")))
            return real_errorbar(x, *args, **kwargs)

        ax.errorbar = capture_errorbar
        return fig, ax

    monkeypatch.setattr(agent_module, "default_plot", capture_plot)

    artifacts = agent_module._write_matrix_overlay_artifacts(
        [
            {"job_id": "ca_p1", "artifacts": {"bare_artifact": str(paths[0])}, "result": {"ensemble": "ens"}},
            {"job_id": "ca_p2", "artifacts": {"bare_artifact": str(paths[1])}, "result": {"ensemble": "ens"}},
        ],
        tmp_path,
        artifact_key="bare_artifact",
        prefix="ca",
        title_suffix="bare matrix elements",
        y_label="Bare",
    )

    assert (tmp_path / "ca_ens_re.pdf").exists()
    assert (tmp_path / "ca_ens_im.svg").exists()
    assert any(key.startswith("matrix_overlay_re_") for key in artifacts)
    assert np.allclose(x_values[0], [-0.03, 0.97])
    assert np.allclose(x_values[1], [0.03, 1.03])
    assert np.allclose(x_values[2], [-0.03, 0.97])
    assert np.allclose(x_values[3], [0.03, 1.03])
    assert any("Q^2=1.00" in label and "\\xi=0.33" in label for label in labels)


def _write_fourier_nc(path: Path, *, ensemble: str, pz: float) -> None:
    EnsembleData(
        ensemble=_ensemble_info(ensemble),
        resample="jackknife",
        values=[
            np.array([0.2 + 0.1j, 0.4 + 0.0j, 0.2 - 0.1j]),
            np.array([0.22 + 0.12j, 0.38 - 0.01j, 0.21 - 0.09j]),
        ],
        dims=("x",),
        coords={"x": [0.0, 0.5, 1.0]},
        attrs={"ensemble": ensemble, "momentum_gev": str(pz)},
        name="fourier_transform",
    ).to_netcdf(path)


def test_fourier_overlay_uses_original_grid_and_same_color_for_re_im(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / "ft_p1.nc", tmp_path / "ft_p2.nc"]
    _write_fourier_nc(paths[0], ensemble="ens", pz=1.0)
    _write_fourier_nc(paths[1], ensemble="ens", pz=2.0)
    plotted: list[tuple[np.ndarray, str, str]] = []
    from matplotlib.axes import Axes

    real_plot = Axes.plot

    def capture_line(self, x, y, *args, **kwargs):
        plotted.append((np.asarray(x, dtype=float), str(kwargs.get("color")), str(kwargs.get("label"))))
        return real_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_line)

    artifacts = agent_module._write_fourier_overlay_artifacts(
        [
            {
                "job_id": "ft_p1",
                "artifacts": {"fourier_artifact": str(paths[0])},
                "result": {"ensemble": "HISQa060_X", "momentum_gev": 1.0},
            },
            {
                "job_id": "ft_p2",
                "artifacts": {"fourier_artifact": str(paths[1])},
                "result": {"ensemble": "HISQa060_X", "momentum_gev": 2.0},
            },
        ],
        tmp_path,
    )

    assert (tmp_path / "ft_HISQa060_X_xdep.pdf").exists()
    assert any(key.startswith("fourier_overlay_image_") for key in artifacts)
    assert all(np.allclose(x, [0.0, 0.5, 1.0]) for x, _, _ in plotted)
    assert plotted[0][1] == plotted[1][1]
    assert plotted[2][1] == plotted[3][1]


def test_matching_overlay_labels_quasi_and_lightcone(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / "mt_p1.nc", tmp_path / "mt_p2.nc"]
    for path in paths:
        EnsembleData(
            ensemble=_ensemble_info("ens"),
            resample="jackknife",
            values=[np.array([0.2, 0.4, 0.2]), np.array([0.22, 0.38, 0.21])],
            dims=("x",),
            coords={"x": [0.0, 0.5, 1.0]},
            attrs={"ensemble": "ens"},
            name="lightcone",
        ).to_netcdf(path)
    labels: list[str] = []
    titles: list[str] = []
    xlims: list[tuple[float, float]] = []
    ylims: list[tuple[float, float]] = []
    real_default_plot = agent_module.default_plot

    def capture_plot():
        fig, ax = real_default_plot()
        real_plot = ax.plot
        real_set_title = ax.set_title
        real_set_xlim = ax.set_xlim
        real_set_ylim = ax.set_ylim

        def capture_line(x, y, *args, **kwargs):
            labels.append(str(kwargs.get("label")))
            return real_plot(x, y, *args, **kwargs)

        def capture_title(label, *args, **kwargs):
            titles.append(str(label))
            return real_set_title(label, *args, **kwargs)

        def capture_xlim(left=None, right=None, *args, **kwargs):
            if np.isscalar(left) and np.isscalar(right):
                xlims.append((float(left), float(right)))
            return real_set_xlim(left, right, *args, **kwargs)

        def capture_ylim(bottom=None, top=None, *args, **kwargs):
            if np.isscalar(bottom) and np.isscalar(top):
                ylims.append((float(bottom), float(top)))
            return real_set_ylim(bottom, top, *args, **kwargs)

        ax.plot = capture_line
        ax.set_title = capture_title
        ax.set_xlim = capture_xlim
        ax.set_ylim = capture_ylim
        return fig, ax

    monkeypatch.setattr(agent_module, "default_plot", capture_plot)

    artifacts = agent_module._write_matching_overlay_artifacts(
        [
            {
                "job_id": "mt_p1",
                "artifacts": {"lightcone_artifact": str(paths[0])},
                "result": {
                    "momentum_gev": 1.0,
                    "ensemble": "HISQa060_X",
                    "x_grid": [0.0, 0.5, 1.0],
                    "quasi_mean": [0.1, 0.2, 0.1],
                    "quasi_sdev": [0.01, 0.01, 0.01],
                    "matching_plot_xlim": [-0.01, 1.01],
                    "matching_plot_ylim": [-0.2, 1.3],
                },
            },
            {
                "job_id": "mt_p2",
                "artifacts": {"lightcone_artifact": str(paths[1])},
                "result": {
                    "momentum_gev": 2.0,
                    "ensemble": "HISQa060_X",
                    "x_grid": [0.0, 0.5, 1.0],
                    "quasi_mean": [0.2, 0.3, 0.2],
                    "quasi_sdev": [0.01, 0.01, 0.01],
                    "matching_plot_xlim": [-0.01, 1.01],
                    "matching_plot_ylim": [-0.4, 1.7],
                },
            },
        ],
        tmp_path,
    )

    assert (tmp_path / "mt_HISQa060_X.pdf").exists()
    assert any(key.startswith("matching_overlay_image_") for key in artifacts)
    assert any("quasi" in label for label in labels)
    assert any("light-cone" in label for label in labels)
    assert titles == ["HISQa060_X"]
    assert xlims == [(-0.01, 1.01)]
    assert ylims == [(-0.4, 1.7)]


def test_hydrate_external_artifact_inputs_loads_fourier_input(tmp_path: Path) -> None:
    nc_path = tmp_path / "rn_p5.nc"
    _write_renorm_nc(nc_path)
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "partial",
                "root_directory": str(tmp_path),
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 1984,
                "stages": ["fourier_transform"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {
                        "id": "rn_p5",
                        "stage": "renormalization",
                        "path": str(nc_path),
                    }
                ],
                "kernels": [],
            },
            "stages": {
                "fourier_transform": {
                    "defaults": {
                        "order": "NLA",
                        "part": "re",
                        "coord_unit": "lattice",
                        "y_grid": {"start": -1.0, "stop": 1.0, "num": 3},
                    },
                        "jobs": [{"id": "ft_p5", "inputs": {"input": "rn_p5"}}],
                },
            },
        }
    )
    artifact = manifest.inputs.artifacts[0]
    store = {"input": artifact}
    job = manifest.stages["fourier_transform"].jobs[0]
    effective_params = merge_stage_params(manifest.stages["fourier_transform"].defaults, job.params)

    _hydrate_external_artifact_inputs(
        "fourier_transform",
        job,
        manifest,
        store,
        effective_params=effective_params,
        artifacts_dir=tmp_path / "artifacts" / "fourier_transform",
    )

    assert isinstance(store["input"], EnsembleData)
    assert "matrix_element_data" in store
    assert store["matrix_element_data"].dims == ["z"]
    assert store["input"].attrs["momentum"] == "PX5PY0PZ0"
    assert store["input"].attrs["hadron"] == "pion"


def test_run_agent_hydrates_partial_fourier_artifact_before_tools(tmp_path: Path, monkeypatch) -> None:
    nc_path = tmp_path / "rn_p5.nc"
    _write_renorm_nc(nc_path)
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "partial",
                "root_directory": str(tmp_path),
                "artifacts_directory": "artifacts",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 1984,
                "stages": ["fourier_transform"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {
                        "id": "rn_p5",
                        "stage": "renormalization",
                        "path": str(nc_path),
                    }
                ],
                "kernels": [],
            },
            "stages": {
                "fourier_transform": {
                    "defaults": {
                        "order": "NLA",
                        "part": "re",
                        "coord_unit": "lattice",
                        "y_grid": {"start": -1.0, "stop": 1.0, "num": 3},
                    },
                        "jobs": [{"id": "ft_p5", "inputs": {"input": "rn_p5"}}],
                },
            },
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()

    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"action": "call_tool", "tool_name": "run_fourier_transform", "args": {}, "reason": "ft"}),
                json.dumps({"action": "finish", "reason": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    observed: dict[str, object] = {}

    def fake_run_fourier_transform(store, **kwargs):
        observed["input_type"] = type(store["input"]).__name__
        observed["has_matrix_element_data"] = "matrix_element_data" in store
        store["output"] = store["matrix_element_data"]
        return {"artifact": str(tmp_path / "ft_p5.nc")}

    real_tools = resolve_stage_tools

    def fake_resolve(stage):
        tools = real_tools(stage)
        if stage == "fourier_transform":
            tools = dict(tools)
            tools["run_fourier_transform"] = fake_run_fourier_transform
        return tools

    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", fake_resolve)
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    result = run_agent(manifest, backend="external", actions_path=transcript)

    assert result["status"] == "completed"
    assert observed["input_type"] == "EnsembleData"
    assert observed["has_matrix_element_data"] is True
    assert result["actions"][0]["action"]["tool_name"] == "run_fourier_transform"
    assert "load_renormalized_matrix_element_samples" not in {
        action["action"].get("tool_name") for action in result["actions"]
    }


def test_run_agent_writes_fourier_stage_report_after_jobs(tmp_path: Path, monkeypatch) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["fourier_transform"],
            },
            "inputs": {
                "artifacts": [
                        {
                            "id": "rn_p4",
                            "stage": "renormalization",
                            "path": str(tmp_path / "rn_p4.nc"),
                            "kind": "renormalized_matrix_element",
                            "format": "nc",
                    }
                ],
                "correlators": [],
                "kernels": [],
            },
            "stages": {
                "fourier_transform": {
                    "defaults": {},
                    "jobs": [{"id": "ft_p4", "inputs": {"input": "rn_p4"}}],
                },
            },
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"action": "call_tool", "tool_name": "run_fourier_transform", "args": {}, "reason": "ft"}),
                json.dumps({"action": "finish", "reason": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run_fourier_transform(store, **kwargs):
        store["fourier_result"] = {
            "observable": "pion_quark_quasi_pdf",
            "method": "GI",
            "order": "LA",
            "part": "re",
            "resample_mode": "jackknife",
            "coord_unit": "fm",
            "fit_coord_unit": "fm",
            "momentum_gev": 1.72,
            "Lambda0_gev": 0.0,
            "posterior_prior_error_scale": 3.0,
            "output_scale": 2.0,
            "y_grid": [-0.5, 0.0, 0.5],
            "scheme_labels": ["LA_prior_3"],
            "fit_model_labels": ["LA_prior_3"],
            "fit_model_mean_weights": [1.0],
            "fit_model_chi2_dof": [0.8],
            "fit_model_q": [0.9],
            "fit_model_logGBF": [12.0],
            "fit_failures": [0],
            "selected_range_label": "zmin_1_zmax_4",
            "selected_fit_range": [1.0, 4.0],
            "scheme_results": [
                {
                    "label": "LA_prior_3",
                    "fit_range": [1.0, 4.0],
                    "z_ext_max": 5.0,
                    "smooth": "linear",
                }
            ],
            "artifact": str(tmp_path / "fourier_result.nc"),
            "fit_info_artifact": str(tmp_path / "fourier_fit_info.nc"),
        }
        store["fourier_summary"] = {"out": "fourier_summary"}
        store["fourier_plot"] = {"plot": str(tmp_path / "fourier_xdep.pdf")}
        store["fourier_extension_plot"] = {
            "plot_re": str(tmp_path / "fourier_re.pdf"),
            "plot_im": str(tmp_path / "fourier_im.pdf"),
        }
        store["output"] = EnsembleData(
            ensemble=None,
            resample="jackknife",
            values=[np.array([0.1, 0.2, 0.1])],
            dims=("x",),
            coords={"x": [-0.5, 0.0, 0.5]},
        )
        return {"artifact": store["fourier_result"]["artifact"]}

    real_tools = resolve_stage_tools

    def fake_resolve(stage):
        tools = real_tools(stage)
        if stage == "fourier_transform":
            tools = dict(tools)
            tools["run_fourier_transform"] = fake_run_fourier_transform
        return tools

    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", fake_resolve)
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])
    monkeypatch.setattr("lamet_agent.agent._hydrate_external_artifact_inputs", lambda *args, **kwargs: None)

    result = run_agent(manifest, backend="external", actions_path=transcript)

    report_path = Path(result["stage_reports"]["fourier_transform"]["report"])
    assert report_path.exists()
    assert "report_cn" not in result["stage_reports"]["fourier_transform"]
    assert not report_path.with_name("ft_report_CN.md").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "`ft_p4`" in report_text
    assert "Lambda0_gev" in report_text


def test_run_agent_writes_correlator_stage_report_after_jobs(tmp_path: Path, monkeypatch) -> None:
    manifest = AnalysisManifest.model_validate(
        {
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
            "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca_p4"}]}},
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"action": "call_tool", "tool_name": "fit_bare_matrix_grid", "args": {}, "reason": "fit"}),
                json.dumps({"action": "finish", "reason": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_fit_bare_matrix_grid(store, **kwargs):
        store["output"] = EnsembleData(
            ensemble=None,
            resample="jackknife",
            values=[np.array([0.3 + 0.1j, 0.2 + 0.05j])],
            dims=("z",),
            coords={"z": [0, 1]},
        )
        return {
            "artifact": str(tmp_path / "ca_p4.nc"),
            "plot_pdf": str(tmp_path / "ca_p4.pdf"),
            "fit_strategy": "joint",
            "fit_scope": "3pt_ratio+FH",
            "fit_mode": "bare_matrix",
            "fitting_form": "Breit",
            "model_average": False,
            "selection_rule": "best_Q",
            "shared_window_specs": [{"fit_scope": "3pt_ratio", "fit_strategy": "joint", "nstate": 2}],
            "tuning_log_path": str(tmp_path / "fit_logs" / "ca_p4_tuning.log"),
            "sample_log_path": str(tmp_path / "fit_logs" / "ca_p4_samples.log"),
            "z_values": [0, 1],
            "tune_z": 0,
            "z_fits": [
                {
                    "z": 0,
                    "Q": 0.8,
                    "chi2_dof": 0.9,
                    "logGBF": 1.2,
                    "n_failed_samples": 0,
                    "real_sys_sdev": 0.01,
                    "imag_sys_sdev": 0.02,
                    "sample0_plot_paths": {"ratio_re_pdf": str(tmp_path / "fit_logs" / "ca_p4_z0_sample0.pdf")},
                }
            ],
            "sample0_pt2_plot_paths": {"meff_pdf": str(tmp_path / "fit_logs" / "ca_p4_meff.pdf")},
            "n_samples": 1,
            "resample_mode": "jackknife",
        }

    monkeypatch.setattr("lamet_agent.agent.resolve_stage_tools", lambda stage: {"fit_bare_matrix_grid": fake_fit_bare_matrix_grid})
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])
    monkeypatch.setattr("lamet_agent.stages.correlator.reporting.translate_markdown_report", lambda markdown, **kwargs: "# translated correlator report\n\nca_p4.nc")

    result = run_agent(manifest, backend="external", actions_path=transcript, report_language="ch")

    report_path = Path(result["stage_reports"]["correlator_analysis"]["report"])
    assert report_path.exists()
    assert report_path.name == "ca_report_CN.md"
    assert "report_cn" not in result["stage_reports"]["correlator_analysis"]
    assert report_path.with_name("ca_report.md").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# translated correlator report" in report_text
    assert "ca_p4.nc" in report_text
    assert ".png" not in report_text


def test_run_agent_writes_renorm_stage_report_after_jobs(tmp_path: Path, monkeypatch) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["renormalization"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {"id": "target", "stage": "correlator_analysis", "path": str(tmp_path / "target.nc")},
                    {"id": "denom", "stage": "correlator_analysis", "path": str(tmp_path / "denom.nc")},
                ],
                "kernels": [],
            },
            "stages": {
                "renormalization": {
                    "defaults": {"scheme": "hybrid_ratio", "zs_fm": 0.3},
                    "jobs": [{"id": "rn_p4", "inputs": {"target": "target", "denominator": "denom"}}],
                },
            },
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"action": "call_tool", "tool_name": "apply_ratio_scheme_renormalization", "args": {}, "reason": "renorm"}),
                json.dumps({"action": "call_tool", "tool_name": "plot_renormalized_matrix_element", "args": {}, "reason": "plot"}),
                json.dumps({"action": "finish", "reason": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_apply_ratio_scheme_renormalization(store, **kwargs):
        store["output"] = EnsembleData(
            ensemble=None,
            resample="jackknife",
            values=[np.array([1.0 + 0.0j, 0.9 + 0.1j])],
            dims=("z",),
            coords={"z": [0, 1]},
        )
        store["matrix_element_data"] = store["output"]
        store["matrix_element_netcdf"] = str(tmp_path / "rn_p4.nc")
        return {
            "artifact": str(tmp_path / "rn_p4.nc"),
            "n_z": 2,
            "n_sample": 1,
            "zs_fm": 0.3,
            "zs_lattice": 5.2,
            "zs_grid": 5.0,
            "delta_m_gev": 0.1,
            "m0_gev": 0.2,
        }

    def fake_plot_renormalized_matrix_element(store, **kwargs):
        return {"plot": str(tmp_path / "rn_p4.pdf")}

    def fake_load_bare_matrix_element_grid(store, **kwargs):
        store["matrix_element_data"] = EnsembleData(
            ensemble=None,
            resample="jackknife",
            values=[np.array([1.0 + 0.0j, 0.9 + 0.1j])],
            dims=("z",),
            coords={"z": [0, 1]},
        )
        return {"data": "matrix_element_data"}

    monkeypatch.setattr(
        "lamet_agent.agent.resolve_stage_tools",
        lambda stage: {
            "load_bare_matrix_element_grid": fake_load_bare_matrix_element_grid,
            "apply_ratio_scheme_renormalization": fake_apply_ratio_scheme_renormalization,
            "plot_renormalized_matrix_element": fake_plot_renormalized_matrix_element,
        },
    )
    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])

    result = run_agent(manifest, backend="external", actions_path=transcript)

    report_path = Path(result["stage_reports"]["renormalization"]["report"])
    assert report_path.exists()
    assert "report_cn" not in result["stage_reports"]["renormalization"]
    assert not report_path.with_name("renorm_report_CN.md").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "`rn_p4`" in report_text
    assert "hybrid-ratio" in report_text
    assert "rn_p4.nc" in report_text
    assert "rn_p4.pdf" in report_text
    assert ".png" not in report_text


def test_run_job_applies_renormalization_normalization_to_store(tmp_path: Path) -> None:
    from lamet_agent.agent import AgentState, _run_job
    from lamet_agent.core.llm import LlmSession
    from lamet_agent.core.trace import AgentTrace
    from lamet_agent.manifest import StageJob

    samples = np.asarray([[2 + 0j, 4 + 0j], [3 + 0j, 6 + 0j]], dtype=complex)
    target = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[samples[0], samples[1]],
        dims=("z",),
        coords={"z": [0.0, 1.0]},
        attrs={"lattice_spacing_fm": "0.1"},
        name="target",
    )
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "demo",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["renormalization"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {"id": "ca", "stage": "correlator_analysis", "path": str(tmp_path / "ca.nc")},
                ],
                "kernels": [],
            },
            "stages": {
                "renormalization": {
                    "defaults": {
                        "normalization": True,
                        "scheme": "hybrid_ratio",
                        "zs_fm": 0.3,
                    },
                    "jobs": [{"id": "rn", "inputs": {"target": "ca", "denominator": "ca"}}],
                },
            },
        }
    )
    manifest._root_directory = tmp_path.resolve()
    manifest._artifacts_directory = (tmp_path / "artifacts").resolve()
    job = manifest.stages["renormalization"].jobs[0]
    store = {"target": target, "denominator": target}

    class _FinishSession(LlmSession):
        def __init__(self) -> None:
            self.actions = iter(
                [
                    {"action": "call_tool", "tool_name": "apply_ratio_scheme_renormalization", "args": {}},
                    {"action": "call_tool", "tool_name": "plot_renormalized_matrix_element", "args": {}},
                    {"action": "finish", "reason": "done"},
                ]
            )

        def begin_stage(self, prompt: str) -> None:
            return None

        def decide(self, *, last_observation=None):
            return next(self.actions)

    _run_job(
        "renormalization",
        job,
        manifest,
        AgentState(run_id="demo"),
        _FinishSession(),
        input_issues=[],
        max_tool_steps=3,
        backend="external",
        model_spec=None,
        trace=AgentTrace(enabled=False),
        store=store,
        report_language="en",
    )

    assert store["target"].attrs.get("normalized_at_z0") == "true"
    assert np.allclose(store["target"].values[:, 0], 1.0)
