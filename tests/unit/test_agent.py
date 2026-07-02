from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import numpy as np

from lamet_agent.agent import _hydrate_external_artifact_inputs, run_agent
from lamet_agent.core import llm
from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import resolve_stage_tools
from lamet_agent.manifest import AnalysisManifest


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


def test_run_agent_uses_manifest_stage_order(tmp_path: Path, monkeypatch) -> None:
    transcript = tmp_path / "actions.jsonl"
    transcript.write_text(
        json.dumps({"action": "finish", "reason": "done"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("lamet_agent.agent.validate_stage_inputs", lambda stage, manifest, job: [])
    result = run_agent(_demo_manifest(), backend="external", actions_path=transcript)

    assert result["status"] == "completed"
    assert result["completed_stages"] == ["correlator_analysis"]
    assert result["actions"][0]["action"]["reason"] == "done"


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
    captured: list[list[dict[str, str]]] = []

    def fake_codex_decide(messages: list[dict[str, str]]) -> dict:
        captured.append(messages)
        return {"action": "finish", "reason": "done"}

    monkeypatch.setattr(llm, "_codex_decide", fake_codex_decide)

    session = llm.make_llm_session("codex")
    session.begin_stage("stage prompt")
    action = session.decide(last_observation={"tool_name": "inspect", "result": {"ok": True}})

    assert action == {"action": "finish", "reason": "done"}
    assert captured[0][0]["role"] == "system"
    assert "LaMET analysis agent" in captured[0][0]["content"]
    assert captured[0][1] == {"role": "user", "content": "stage prompt"}
    assert captured[0][2]["role"] == "user"
    assert "Tool result" in captured[0][2]["content"]


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
        ]
    )

    assert action == {"action": "finish", "reason": "done"}
    assert captured["thread_start_kwargs"]["developer_instructions"] == "system instructions"
    assert captured["thread_start_kwargs"]["sandbox"] == _Sandbox.read_only
    assert captured["thread_start_kwargs"]["ephemeral"] is True
    assert captured["run_kwargs"] == {"sandbox": _Sandbox.read_only}
    assert "stage prompt" in captured["task_input"]


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
        name="renormalized_matrix_element",
    )
    data.to_netcdf(path)


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
                        "a_fm": 0.0574,
                        "pz_gev": 2.15,
                        "hadron": "pion",
                        "gfix": "CG",
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
                    "jobs": [{"id": "ft_p5", "inputs": {"input": "rn_p5"}, "params": {"pz_gev": 2.15}}],
                },
            },
        }
    )
    artifact = manifest.inputs.artifacts[0]
    store = {"input": artifact}
    job = manifest.stages["fourier_transform"].jobs[0]
    effective_params = {**manifest.stages["fourier_transform"].defaults, **job.params}

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
                        "a_fm": 0.0574,
                        "pz_gev": 2.15,
                        "hadron": "pion",
                        "gfix": "CG",
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
                    "jobs": [{"id": "ft_p5", "inputs": {"input": "rn_p5"}, "params": {"pz_gev": 2.15}}],
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
            "pz_gev": 1.72,
            "Lambda0": 0.1,
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
        store["fourier_plot"] = {"plot": str(tmp_path / "fourier_result.pdf")}
        store["fourier_extension_plot"] = {
            "plot_re": str(tmp_path / "fourier_extension_re.pdf"),
            "plot_im": str(tmp_path / "fourier_extension_im.pdf"),
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
    assert "`ft_p4`" in report_path.read_text(encoding="utf-8")


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
            "fit_scope": "ratio+FH",
            "fit_mode": "bare_matrix",
            "fitting_form": "Breit",
            "model_average": False,
            "selection_rule": "best_Q",
            "shared_window_specs": [{"fit_scope": "ratio", "fit_strategy": "joint", "nstate": 2}],
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

    result = run_agent(manifest, backend="external", actions_path=transcript, report_language="ch")

    report_path = Path(result["stage_reports"]["correlator_analysis"]["report"])
    assert report_path.exists()
    assert report_path.name == "ca_report_CN.md"
    assert "report_cn" not in result["stage_reports"]["correlator_analysis"]
    assert not report_path.with_name("ca_report.md").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Correlator Analysis 阶段报告" in report_text
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
                    "defaults": {"scheme": "hybrid_ratio", "scheme_parameters": {"zs_fm": 0.3}},
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

    monkeypatch.setattr(
        "lamet_agent.agent.resolve_stage_tools",
        lambda stage: {
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
