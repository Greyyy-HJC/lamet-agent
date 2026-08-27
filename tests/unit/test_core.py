"""Focused checks for the independent ``lamet_agent_neo`` architecture."""

from __future__ import annotations

import copy
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Literal, TypedDict

import numpy as np
import pytest

from lamet_agent.agent import (
    LlmSession,
    ToolContext,
    _discover_tools,
    _resolve_runtime_null_hooks,
    _write_transcript_header,
    create_session,
)
from lamet_agent.contract import (
    CheckContext,
    Depends,
    Issue,
    List,
    Provides,
    Recommends,
    Suggests,
    Value,
    _apply_recommended_defaults,
    _unresolved_null_hooks,
    evaluate_checks,
    evaluate_rules,
    stage_job_rules,
)
from lamet_agent.__main__ import _build_parser
from lamet_agent.llm import Message, _AssistantResponse, _ToolCall, create_backend
from lamet_agent.manifest import Manifest, _load_stage_contract, load_manifest
from lamet_agent.structured import annotation_schema


def _valid_metadata(tmp_path: Path, **overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "run_id": "toy",
        "root_directory": str(tmp_path),
        "artifacts_directory": "runs",
        "random_seed": 1,
        "workers": 1,
        "target_observable": "pdf",
        "resample_mode": "jackknife",
        "sample_error_mode": "covariance",
        "bin_size": 1,
    }
    metadata.update(overrides)
    return metadata


class _ScriptedBackend:
    identity = "scripted:test"

    def __init__(self, responses: list[_AssistantResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[dict[str, object]], str]] = []
        self.response_schemas: list[object] = []

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, object]],
        prompt_digest: str,
        response_schema=None,
    ) -> _AssistantResponse:
        self.calls.append((list(messages), list(tools), prompt_digest))
        self.response_schemas.append(response_schema)
        if not self._responses:
            raise RuntimeError("scripted backend has no response for this turn")
        return self._responses.pop(0)


class _RecommendedInterval(TypedDict):
    start: int
    stop: int


class _PlateauAssessment(TypedDict):
    stable_start: int


def test_llm_session_appends_structured_recommendations_to_job_history(tmp_path: Path) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse('{"value":1}', structured={"value": 1}),
            _AssistantResponse('{"value":2}', structured={"value": 2}),
        ]
    )
    session = LlmSession(backend, tmp_path / "llm.md", history=[Message("system", "fixed stage prefix")])
    session.add_context("fit_data", {"mean": [1.0], "sdev": [0.1]})
    schema = {
        "name": "integer_recommendation",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }

    session.complete(label="first recommendation", user_message="first question", response_schema=schema)
    session.complete(label="second recommendation", user_message="second question", response_schema=schema)

    assert [message.role for message in backend.calls[0][0]] == ["system", "user"]
    assert [message.role for message in backend.calls[1][0]] == ["system", "user", "assistant", "user"]
    first_payload = json.loads(backend.calls[0][0][-1].content)
    assert first_payload["context"] == [{"key": "fit_data", "content": {"mean": [1.0], "sdev": [0.1]}}]
    assert first_payload["request"] == "first question"
    assert backend.calls[1][0][-1].content == "second question"
    assert session.calls == 2


class _IntervalSuggestion(TypedDict):
    windows: list[_RecommendedInterval]


def _recommend_interval(_context: ToolContext, session: LlmSession) -> list[_RecommendedInterval]:
    schema, _ = annotation_schema(_IntervalSuggestion)
    response = session.complete(
        label="interval recommendation",
        user_message="Choose one nonempty half-open interval from coordinates 0,1,2,3.",
        response_schema={"name": "interval_recommendation", "schema": schema},
    )
    return list(response.structured["windows"])


def _estimate_interval_without_llm(_context: ToolContext, _session: LlmSession) -> list[_RecommendedInterval]:
    return [{"start": 1, "stop": 3}]


def _estimate_interval_with_two_llm_calls(_context: ToolContext, session: LlmSession) -> list[_RecommendedInterval]:
    assessment_schema, _ = annotation_schema(_PlateauAssessment)
    assessment_response = session.complete(
        label="plateau assessment",
        user_message="Identify the first stable coordinate among 0,1,2,3.",
        response_schema={"name": "plateau_assessment", "schema": assessment_schema},
    )
    interval_schema, _ = annotation_schema(_IntervalSuggestion)
    response = session.complete(
        label="interval recommendation",
        user_message=f"Choose an interval using {dict(assessment_response.structured)} and last coordinate 3.",
        response_schema={"name": "interval_recommendation", "schema": interval_schema},
    )
    return list(response.structured["windows"])


def _null_hook_rules(hook=_recommend_interval):
    return (
        Depends("", "settings", physics="settings are declared"),
        Depends(
            "settings",
            "windows",
            physics="fit windows follow the observed plateau",
            null_hook=hook,
        ),
        List(
            "settings.windows",
            "window",
            physics="at least one fit window is required",
            validator=bool,
        ),
        Depends(
            "settings.windows.window",
            "start",
            physics="each interval has a start",
        ),
        Depends(
            "settings.windows.window",
            "stop",
            physics="each interval has a stop",
        ),
        Value("settings.windows.window.start", int, physics="starts are integers"),
        Value("settings.windows.window.stop", int, physics="stops are integers"),
    )


def test_neo_plotting_owns_the_figure_and_clears_it_after_saving(tmp_path: Path) -> None:
    import gvar
    from matplotlib import rcParams

    from lamet_agent.plotting import (
        COLOR_CYCLE,
        band,
        bar,
        configure_plot,
        errorband,
        errorline,
        hband,
        hline,
        line,
        save_figure,
        start_plot,
        vband,
        vline,
    )

    assert start_plot() is None
    values = np.asarray([gvar.gvar(1.0, 0.1), gvar.gvar(1.5, 0.2)], dtype=object)
    assert errorband([0.0, 1.0], values, color=COLOR_CYCLE[1], label="result") is None
    assert errorline([0.0, 1.0], values, color=COLOR_CYCLE[0], marker="s", label="points") is None
    assert line([0.0, 1.0], [0.8, 1.2], color="0.3", marker="o", label="line") is None
    assert band([0.0, 1.0], [0.7, 1.0], [0.9, 1.4], color="0.8", label="band") is None
    assert vband(0.2, 0.4, color="0.7", label="vband") is None
    assert hband(0.9, 1.1, color="0.6", label="hband") is None
    assert bar([0.25, 0.75], [0.2, 0.3], width=0.1, color="0.5", label="bar") is None
    with pytest.raises(ValueError, match="unsupported marker"):
        errorline([0.0, 1.0], values, marker="r--")
    with pytest.raises(ValueError, match="unsupported marker"):
        line([0.0, 1.0], [1.0, 1.2], marker="r--")
    with pytest.raises(TypeError, match="gvar"):
        errorline([0.0, 1.0], np.asarray([[1.0, 2.0], [1.1, 2.1]]))
    assert hline(0.0, color=COLOR_CYCLE[2], linestyle="dashed") is None
    assert vline(0.5, color=COLOR_CYCLE[3], linestyle=":") is None
    with pytest.raises(ValueError, match="unsupported line style"):
        hline(0.0, linestyle="custom")
    assert configure_plot(xlabel="x", ylabel="y", legend=True) is None
    pdf_path = tmp_path / "result.pdf"
    svg_path = tmp_path / "result.svg"
    assert save_figure(pdf_path, svg_path) is None

    assert pdf_path.is_file()
    assert svg_path.is_file()
    assert rcParams["font.family"] == ["serif"]
    assert rcParams["mathtext.fontset"] == "stix"
    with pytest.raises(RuntimeError, match="no current plot"):
        save_figure(tmp_path / "again.pdf")
    assert start_plot() is None
    errorband([0.0, 1.0], values)
    errorline([0.0, 1.0], values)
    hline(0.0)
    vline(0.5)
    hline(0.2, color=COLOR_CYCLE[4])
    vline(0.7, color=COLOR_CYCLE[5])
    cycle_path = tmp_path / "cycle.svg"
    save_figure(cycle_path)
    cycle_svg = cycle_path.read_text(encoding="utf-8").lower()
    assert all(color.lower() in cycle_svg for color in COLOR_CYCLE)


def test_plotting_shared_labels_and_formatters() -> None:
    from lamet_agent.plotting import (
        COLOR_CYCLE,
        QUASI_DISTRIBUTION_LABELS,
        X_LABEL,
        Z_OVER_A_LABEL,
        momentum_label,
        quasi_distribution_label,
        series_color,
    )

    assert X_LABEL == r"$x$"
    assert Z_OVER_A_LABEL == r"$z~/~a$"
    assert quasi_distribution_label("re") == QUASI_DISTRIBUTION_LABELS["real"]
    assert quasi_distribution_label("imag") == QUASI_DISTRIBUTION_LABELS["imag"]
    assert momentum_label(np.float64(1.72)) == r"$P_z=1.72\,\mathrm{GeV}$"
    assert momentum_label(None, default="job") == "job"
    assert series_color(len(COLOR_CYCLE)) == COLOR_CYCLE[0]


def test_neo_core_exports_are_minimal() -> None:
    from lamet_agent import agent, contract, data, llm, manifest, parallel, plotting
    from lamet_agent.parallel import lanczos

    assert agent.__all__ == ["ToolContext", "create_session"]
    assert contract.__all__ == [
        "Depends",
        "Provides",
        "Recommends",
        "Suggests",
        "List",
        "Value",
        "Source",
        "Issue",
        "CheckContext",
        "evaluate_rules",
        "evaluate_checks",
        "stage_job_rules",
    ]
    assert not hasattr(contract, "Contains")
    assert data.__all__ == ["EnsembleInfo", "EnsembleData"]
    assert llm.__all__ == ["Message", "LlmBackend", "create_backend"]
    assert manifest.__all__ == ["Job", "Manifest", "load_manifest"]
    assert plotting.__all__ == [
        "COLOR_CYCLE",
        "X_LABEL",
        "Z_OVER_A_LABEL",
        "Z_FM_LABEL",
        "INVERSE_LATTICE_SPACING_LABEL",
        "BARE_MATRIX_ELEMENT_LABEL",
        "SELF_RENORMALIZATION_FACTOR_LABEL",
        "RENORMALIZED_MATRIX_ELEMENT_LABELS",
        "QUASI_DISTRIBUTION_LABELS",
        "series_color",
        "continuous_color",
        "lattice_spacing_label",
        "momentum_label",
        "quasi_distribution_label",
        "start_plot",
        "configure_plot",
        "line",
        "band",
        "vband",
        "hband",
        "bar",
        "errorline",
        "errorband",
        "histogram",
        "hline",
        "vline",
        "save_figure",
    ]
    assert not hasattr(plotting, "mean_sdev")
    assert parallel.__all__ == ["FitNumericalError", "nonlinear_fit", "fourier_transform"]
    assert lanczos.__all__ == ["prepare_lanczos_data", "analyze_prepared_lanczos"]


def test_neo_cli_uses_provider_and_optional_model() -> None:
    args = _build_parser().parse_args(["run", "manifest.json", "--provider", "codex"])
    assert args.provider == "codex"
    assert args.model is None
    assert not hasattr(args, "backend")


def test_neo_manifest_loader_accepts_jsonc_comments(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonc"
    path.write_text(
        """{
  // A URL-like string is content, not a comment.
  "metadata": {"run_id": "https://example.test/*run*/"},
  /* Stage blocks may be documented inline. */
  "stages": {}
}
""",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    assert manifest.path == path.resolve()
    assert manifest.document == {
        "metadata": {"run_id": "https://example.test/*run*/"},
        "stages": {},
    }


def test_neo_manifest_loader_rejects_unterminated_jsonc_comment(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonc"
    path.write_text('{"metadata": {} /* unfinished', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError, match="Unterminated block comment"):
        load_manifest(path)


def test_neo_provider_selection_has_one_public_backend_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, **kwargs: _ModelsResponse(["gpt-5.6-luna", "gpt-test"])
    )
    assert create_backend("openai").identity.endswith(":gpt-5.6-luna")
    assert create_backend("openai", "gpt-test").identity.endswith(":gpt-test")
    assert create_backend("codex").identity == "codex:default"
    with pytest.raises(ValueError, match="requires a model"):
        create_backend("https://llm.example.test/v1")


class _ModelsResponse:
    def __init__(self, model_ids: list[str]) -> None:
        self._payload = json.dumps({"data": [{"id": model_id} for model_id in model_ids]}).encode()

    def __enter__(self) -> _ModelsResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _ChatResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> _ChatResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_neo_api_model_is_checked_against_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kwargs: _ModelsResponse(["gpt-a", "gpt-b"]))
    assert create_backend("openai", "gpt-a").identity.endswith(":gpt-a")
    with pytest.raises(ValueError, match="available models: gpt-a, gpt-b"):
        create_backend("openai", "missing")


def test_neo_backend_factory_owns_api_key_file_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kwargs: _ModelsResponse(["gpt-a"]))
    key_file = tmp_path / "provider.key"
    key_file.write_text("key-from-file\n", encoding="utf-8")
    assert create_backend("openai", "gpt-a", api_key_file=key_file).identity.endswith(":gpt-a")
    with pytest.raises(ValueError, match="does not exist or is not a file"):
        create_backend("openai", "gpt-a", api_key_file=tmp_path / "missing.key")
    key_file.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        create_backend("openai", "gpt-a", api_key_file=key_file)
    with pytest.raises(ValueError, match="only valid for API providers"):
        create_backend("codex", api_key_file=key_file)


def test_neo_local_api_infers_its_only_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "provider.key"
    key_file.write_text("key\n", encoding="utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda request: _ModelsResponse(["local-model"]))
    assert create_backend("http://localhost:11434/v1", api_key_file=key_file).identity.endswith(":local-model")


def test_neo_local_api_rejects_ambiguous_model_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "provider.key"
    key_file.write_text("key\n", encoding="utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda request: _ModelsResponse(["local-a", "local-b"]))
    with pytest.raises(ValueError, match="exposes multiple models"):
        create_backend("http://127.0.0.1:8000/v1", api_key_file=key_file)


def test_unified_backend_preserves_multiple_ordered_tool_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "fit both candidates",
                    "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "fit", "arguments": '{"window":1}'}},
                        {"id": "call-2", "type": "function", "function": {"name": "fit", "arguments": '{"window":2}'}},
                    ],
                }
            }
        ]
    }
    key_file = tmp_path / "provider.key"
    key_file.write_text("key\n", encoding="utf-8")
    responses = iter([_ModelsResponse(["model"]), _ChatResponse(payload)])
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kwargs: next(responses))
    response = create_backend("https://example.test/v1", "model", key_file).complete(
        messages=[Message("user", "run")], tools=[], prompt_digest="digest"
    )
    assert [(call.id, call.name, call.arguments["window"]) for call in response.calls] == [
        ("call-1", "fit", 1),
        ("call-2", "fit", 2),
    ]


def test_api_backend_retries_only_malformed_tool_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = {
        "choices": [
            {
                "message": {
                    "content": "bad",
                    "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "fit", "arguments": "{bad"}}],
                }
            }
        ]
    }
    valid = {
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [
                        {"id": "good", "type": "function", "function": {"name": "fit", "arguments": '{"window":3}'}}
                    ],
                }
            }
        ]
    }
    key_file = tmp_path / "provider.key"
    key_file.write_text("key\n", encoding="utf-8")
    responses = iter(
        [_ModelsResponse(["model"]), _ChatResponse(malformed), _ChatResponse(malformed), _ChatResponse(valid)]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kwargs: next(responses))

    response = create_backend("https://example.test/v1", "model", key_file).complete(
        messages=[Message("user", "run")], tools=[], prompt_digest="digest"
    )

    assert response.calls[0].arguments == {"window": 3}


def test_contract_traverses_virtual_list_items() -> None:
    rules = (
        Depends("", "items", physics="items are declared"),
        List("items", "item", physics="items are a list"),
        Depends("items.item", "value", physics="each item has a value"),
        Value("items.item.value", int, physics="values are integers"),
    )
    issues = evaluate_rules({"items": [{"value": 1}, {"value": "bad"}]}, rules)
    assert [issue.path for issue in issues] == ["items[1].value"]


def test_depends_null_hook_defers_missing_and_null_but_not_empty_lists() -> None:
    rules = _null_hook_rules()

    assert evaluate_rules({"settings": {}}, rules) == []
    assert evaluate_rules({"settings": {"windows": None}}, rules) == []
    assert [(issue.path, issue.message) for issue in evaluate_rules({"settings": {"windows": []}}, rules)] == [
        ("settings.windows", "failed its intrinsic value check")
    ]
    assert evaluate_rules({"settings": {"windows": [{"start": 1, "stop": 3}]}}, rules) == []


def test_runtime_null_hook_uses_a_typed_response_and_updates_params(
    tmp_path: Path,
) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "selected from the plateau",
                structured={"windows": [{"start": 1, "stop": 3}]},
            )
        ]
    )
    params = {"settings": {"windows": None}}
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "demo",
        "job",
        params,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    contract = SimpleNamespace(PARAM_RULES=_null_hook_rules(), CHECKS=())
    transcript = tmp_path / "llm_transcript.md"
    _write_transcript_header(transcript)

    _resolve_runtime_null_hooks(
        context=context,
        contract=contract,
        session=LlmSession(backend, transcript),
    )

    assert params["settings"]["windows"] == [{"start": 1, "stop": 3}]
    assert context.state["null_hook_provenance"]["settings.windows"] == {
        "backend": "scripted:test",
        "hook": "_recommend_interval",
        "llm_requests": 1,
        "value": [{"start": 1, "stop": 3}],
    }
    schema = backend.response_schemas[0]["schema"]["properties"]["windows"]["items"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["start", "stop"]
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "interval recommendation, request 1: sent to LLM" in transcript_text
    assert "interval recommendation, request 1: received from LLM" in transcript_text


def test_invalid_runtime_null_hook_value_is_rolled_back(tmp_path: Path) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "no usable interval",
                structured={"windows": []},
            )
        ]
    )
    params = {"settings": {"windows": None}}
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "demo",
        "job",
        params,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    transcript = tmp_path / "llm_transcript.md"
    _write_transcript_header(transcript)

    with pytest.raises(ValueError, match="invalid null-hook value"):
        _resolve_runtime_null_hooks(
            context=context,
            contract=SimpleNamespace(PARAM_RULES=_null_hook_rules(), CHECKS=()),
            session=LlmSession(backend, transcript),
        )

    assert params == {"settings": {"windows": None}}


def test_recommends_fills_a_static_default_before_normal_validation(
    tmp_path: Path,
) -> None:
    rules = (
        Depends("", "settings", physics="settings are declared"),
        Recommends(
            "settings",
            "mode",
            physics="safe mode is the fallback",
            default="safe",
        ),
        Value("settings.mode", Literal["safe", "fast"], physics="mode is controlled"),
    )
    assert evaluate_rules({"settings": {}}, rules) == []
    assert evaluate_rules({"settings": {"mode": None}}, rules) == []
    assert evaluate_rules({"settings": []}, rules)[0].message == "expected an object"
    assert evaluate_rules({"settings": {"mode": "invalid"}}, rules)[0].path == ("settings.mode")
    params = {"settings": {"mode": None}}
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "demo",
        "job",
        params,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    transcript = tmp_path / "llm_transcript.md"
    _write_transcript_header(transcript)

    _resolve_runtime_null_hooks(
        context=context,
        contract=SimpleNamespace(PARAM_RULES=rules, CHECKS=()),
        session=LlmSession(_ScriptedBackend([]), transcript),
    )

    assert params == {"settings": {"mode": "safe"}}
    assert context.state["recommended_defaults"] == {"settings.mode": "safe"}


def test_null_hook_may_estimate_without_calling_the_llm(tmp_path: Path) -> None:
    params = {"settings": {}}
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "demo",
        "job",
        params,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    backend = _ScriptedBackend([])
    transcript = tmp_path / "llm_transcript.md"
    _write_transcript_header(transcript)

    _resolve_runtime_null_hooks(
        context=context,
        contract=SimpleNamespace(PARAM_RULES=_null_hook_rules(_estimate_interval_without_llm), CHECKS=()),
        session=LlmSession(backend, transcript),
    )

    assert params["settings"]["windows"] == [{"start": 1, "stop": 3}]
    assert backend.calls == []
    assert context.state["null_hook_provenance"]["settings.windows"]["llm_requests"] == 0


def test_null_hook_may_make_multiple_typed_llm_requests(tmp_path: Path) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "plateau assessment",
                structured={"stable_start": 1},
            ),
            _AssistantResponse(
                "final interval",
                structured={"windows": [{"start": 1, "stop": 3}]},
            ),
        ]
    )
    params = {"settings": {"windows": None}}
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "demo",
        "job",
        params,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    transcript = tmp_path / "llm_transcript.md"
    _write_transcript_header(transcript)

    _resolve_runtime_null_hooks(
        context=context,
        contract=SimpleNamespace(
            PARAM_RULES=_null_hook_rules(_estimate_interval_with_two_llm_calls),
            CHECKS=(),
        ),
        session=LlmSession(backend, transcript),
    )

    assert len(backend.calls) == 2
    assert "stable_start" in backend.calls[1][0][-1].content
    assert context.state["null_hook_provenance"]["settings.windows"]["llm_requests"] == 2


def test_depends_owns_structured_mapping_type_once() -> None:
    rules = (
        Depends("", "settings", physics="settings are declared"),
        Depends("settings", "left", physics="settings own left"),
        Depends("settings", "right", physics="settings own right"),
        Value("settings.left", int, physics="left is an integer"),
        Value("settings.right", int, physics="right is an integer"),
    )

    issues = evaluate_rules({"settings": []}, rules)

    assert [(issue.path, issue.message) for issue in issues] == [("settings", "expected an object")]


def test_shipped_contracts_do_not_repeat_depends_mapping_types() -> None:
    for stage_id in (
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
        "review",
    ):
        stage_contract = _load_stage_contract(stage_id)
        for rules in (stage_contract.PARAM_RULES, stage_contract.INPUT_RULES):
            structured_paths = {rule.parent for rule in rules if isinstance(rule, Depends)}
            redundant = [
                rule.path
                for rule in rules
                if isinstance(rule, Value) and rule.expected is dict and rule.path in structured_paths
            ]
            assert redundant == []


def test_contract_list_owns_type_and_intrinsic_validation() -> None:
    validated: list[list[object]] = []

    def nonempty(values: list[object]) -> bool:
        validated.append(values)
        return bool(values)

    rules = (
        Depends("", "items", physics="items are declared"),
        List("items", "item", physics="items must be nonempty", validator=nonempty),
        Value("items.item", int, physics="items are integers"),
    )

    assert evaluate_rules({"items": [1]}, rules) == []
    assert [(issue.path, issue.message) for issue in evaluate_rules({"items": []}, rules)] == [
        ("items", "failed its intrinsic value check")
    ]
    assert [(issue.path, issue.message) for issue in evaluate_rules({"items": "bad"}, rules)] == [
        ("items", "expected a list")
    ]
    assert validated == [[1], []]


def test_contract_value_uses_literal_as_its_choice_source() -> None:
    string_rule = Value("mode", Literal["first", "second"], physics="mode is controlled")
    depends = Depends("", "mode", physics="mode is declared")
    assert evaluate_rules({"mode": "first"}, (depends, string_rule)) == []
    issues = evaluate_rules({"mode": "third"}, (depends, string_rule))
    assert [(issue.path, issue.message) for issue in issues] == [("mode", "must be one of 'first', 'second'")]

    integer_rule = Value("sign", Literal[-1, 1], physics="sign is controlled")
    sign_depends = Depends("", "sign", physics="sign is declared")
    assert evaluate_rules({"sign": 1}, (sign_depends, integer_rule)) == []
    assert evaluate_rules({"sign": True}, (sign_depends, integer_rule))[0].message == "must be one of -1, 1"
    assert not hasattr(string_rule, "choices")


def test_contract_provides_activates_virtual_dependency_branches() -> None:
    rules = (
        Depends("lsqfit", "window", physics="The fit window is required."),
        Value("lsqfit.window", int, physics="The fit window is an integer."),
        Provides(
            "",
            "lsqfit",
            "analysis_method",
            physics="Least-squares fitting owns its parameter object.",
        ),
        Depends("", "analysis_method", physics="Select one analysis method."),
        Value(
            "analysis_method",
            Literal["lsqfit", "lanczos"],
            physics="The analysis method owns its choices.",
        ),
        Provides(
            "",
            "lanczos",
            "analysis_method",
            physics="Lanczos owns its parameter object.",
        ),
        Depends("lanczos", "iterations", physics="Lanczos iterations are required."),
        Value("lanczos.iterations", int, physics="Lanczos iterations are integers."),
    )

    assert evaluate_rules({"analysis_method": "lsqfit", "window": 4}, rules) == []
    assert evaluate_rules({"analysis_method": "lanczos", "iterations": 3}, rules) == []
    assert [(issue.path, issue.message) for issue in evaluate_rules({"analysis_method": "unknown"}, rules)] == [
        (
            "analysis_method",
            "must be one of 'lsqfit', 'lanczos'",
        )
    ]
    assert [
        (issue.path, issue.message, issue.physics) for issue in evaluate_rules({"analysis_method": "lsqfit"}, rules)
    ] == [
        (
            "window",
            "is required",
            "The fit window is required.",
        )
    ]
    mixed = {
        "analysis_method": "lsqfit",
        "window": 4,
        "iterations": 3,
    }
    assert evaluate_rules(mixed, rules) == []
    assert mixed == {"analysis_method": "lsqfit", "window": 4}
    invalid_inactive = {
        "analysis_method": "lsqfit",
        "window": 4,
        "iterations": "three",
        "typo": 1,
    }
    assert [(issue.path, issue.message) for issue in evaluate_rules(invalid_inactive, rules)] == [
        ("typo", "unknown key 'typo'")
    ]
    assert "iterations" not in invalid_inactive


def test_contract_only_applies_defaults_and_hooks_in_the_selected_provider() -> None:
    rules = (
        Depends("", "analysis_method", physics="Select one analysis method."),
        Provides("", "lsqfit", "analysis_method", physics="Fit settings."),
        Provides("", "lanczos", "analysis_method", physics="Lanczos settings."),
        Recommends(
            "lsqfit",
            "mode",
            physics="The fit mode has a stable fallback.",
            default="safe",
        ),
        Depends(
            "lanczos",
            "windows",
            physics="Lanczos windows may be estimated.",
            null_hook=_estimate_interval_without_llm,
        ),
    )
    params = {"analysis_method": "lsqfit"}

    assert _apply_recommended_defaults(params, rules) == {"mode": "safe"}
    assert params == {"analysis_method": "lsqfit", "mode": "safe"}
    assert _unresolved_null_hooks(params, rules) == ()

    lanczos = {"analysis_method": "lanczos"}
    unresolved = _unresolved_null_hooks(lanczos, rules)
    assert [rule.path for rule in unresolved] == ["windows"]


def test_absolute_provider_selector_does_not_own_global_choices() -> None:
    rules = (
        Provides("", "da", "$.metadata.target_observable", physics="DA settings."),
        Depends("da", "phase_transfer_da", physics="DA phase is explicit."),
        Value("da.phase_transfer_da", bool, physics="DA phase is boolean."),
        Provides("", "pdf", "$.metadata.target_observable", physics="PDF settings."),
        Recommends("pdf", "sector", physics="PDF sector has a default.", default="valence"),
    )
    root = {"metadata": {"target_observable": "da"}}
    params = {"phase_transfer_da": True}
    assert evaluate_rules(params, rules, root_document=root) == []
    assert params == {"phase_transfer_da": True}

    root["metadata"]["target_observable"] = "pdf"
    params = {}
    assert evaluate_rules(params, rules, root_document=root) == []
    assert params == {"sector": "valence"}

    root["metadata"]["target_observable"] = "gpd"
    assert evaluate_rules({}, rules, root_document=root) == []


def test_nested_provider_uses_the_same_explicit_selector_path_as_value() -> None:
    rules = (
        Depends("", "strategy", physics="Strategy is explicit."),
        Value("strategy", Literal["external"], physics="Strategy is controlled."),
        Provides("", "external", "strategy", physics="External branch."),
        Depends("external", "scheme", physics="Scheme is explicit."),
        Value("external.scheme", Literal["ratio", "hybrid"], physics="Scheme is controlled."),
        Provides("external", "hybrid", "external.scheme", physics="Hybrid branch."),
        Depends("external.hybrid", "switch", physics="Hybrid switch is explicit."),
        Value("external.hybrid.switch", float, physics="Hybrid switch is numeric."),
    )
    hybrid = {"strategy": "external", "scheme": "hybrid", "switch": 0.2}
    assert evaluate_rules(hybrid, rules) == []
    ratio = {"strategy": "external", "scheme": "ratio", "switch": 0.2}
    assert evaluate_rules(ratio, rules) == []
    assert ratio == {"strategy": "external", "scheme": "ratio"}


def test_job_provider_values_override_stage_defaults() -> None:
    rules = stage_job_rules(
        (
            Depends("", "analysis_method", physics="Method is explicit."),
            Value("analysis_method", Literal["lsqfit"], physics="Method is controlled."),
            Provides("", "lsqfit", "analysis_method", physics="Fit branch."),
            Depends("lsqfit", "window", physics="Window is explicit."),
            Value("lsqfit.window", int, physics="Window is integer."),
        ),
        (),
    )
    document = {
        "defaults": {"analysis_method": "lsqfit", "window": 2},
        "jobs": [{"id": "fit", "window": 7}],
    }
    assert evaluate_rules(document, rules) == []
    assert document["jobs"][0]["window"] == 7


def test_virtual_provider_projects_active_values_without_hiding_typos() -> None:
    rules = stage_job_rules(
        (
            Depends("", "operation", physics="Operation is explicit."),
            Value("operation", Literal["fit", "budget"], physics="Operation is controlled."),
            Provides("", "fit", "operation", physics="Fit branch."),
            Depends("fit", "fit_only", physics="Fit parameter is explicit."),
            Value("fit.fit_only", int, physics="Fit parameter is integer."),
            Provides("", "budget", "operation", physics="Budget branch."),
            Depends("budget", "budget_only", physics="Budget parameter is explicit."),
            Value("budget.budget_only", int, physics="Budget parameter is integer."),
        ),
        (),
    )
    document = {
        "defaults": {"operation": "fit", "fit_only": 1},
        "jobs": [{"id": "budget", "operation": "budget", "fit_only": 9, "budget_only": 2}],
    }
    assert evaluate_rules(document, rules) == []
    assert document["jobs"] == [{"id": "budget", "operation": "budget", "budget_only": 2, "inputs": {}}]

    typo = {
        "defaults": {"operation": "fit", "fit_only": 1, "fit_typo": 3},
        "jobs": [{"id": "fit"}],
    }
    assert [(issue.path, issue.message) for issue in evaluate_rules(typo, rules)] == [
        ("jobs[0].fit_typo", "unknown key 'fit_typo'")
    ]


def test_suggests_fills_list_items_before_bfs_validation() -> None:
    rules = (
        Depends("", "jobs", physics="jobs are declared"),
        List("jobs", "job", physics="jobs preserve order"),
        Suggests("", "defaults", "jobs.job", physics="defaults fill jobs"),
        Depends("jobs.job", "mode", physics="mode is required"),
        Depends("jobs.job", "nested", physics="nested settings are required"),
        Depends("jobs.job.nested", "left", physics="left is required"),
        Depends("jobs.job.nested", "right", physics="right is required"),
        Value("jobs.job.mode", str, physics="mode is text"),
        Value("jobs.job.nested.left", int, physics="left is integer"),
        Value("jobs.job.nested.right", int, physics="right is integer"),
    )
    document = {
        "defaults": {"mode": "safe", "nested": {"left": 1, "right": 2}},
        "jobs": [{"nested": {"left": 1, "right": 7}}, {}],
    }

    assert evaluate_rules(document, rules) == []
    assert document["jobs"] == [
        {"mode": "safe", "nested": {"left": 1, "right": 7}},
        {"mode": "safe", "nested": {"left": 1, "right": 2}},
    ]

    partial = {
        "defaults": {"mode": "safe", "nested": {"left": 1, "right": 2}},
        "jobs": [{"nested": {"right": 7}}],
    }
    assert [(issue.path, issue.message) for issue in evaluate_rules(partial, rules)] == [
        ("jobs[0].nested.left", "is required")
    ]


def test_suggests_missing_source_is_empty_and_bad_types_are_reported() -> None:
    rules = (
        Depends("", "jobs", physics="jobs are declared"),
        List("jobs", "job", physics="jobs preserve order"),
        Suggests("", "defaults", "jobs.job", physics="defaults fill jobs"),
        Depends("jobs.job", "mode", physics="mode is required"),
        Value("jobs.job.mode", str, physics="mode is text"),
    )
    missing = {"jobs": [{"mode": "local"}]}
    assert evaluate_rules(missing, rules) == []
    assert "defaults" not in missing

    bad_source = {"defaults": 1.5, "jobs": [{"mode": "local"}]}
    assert [(issue.path, issue.message) for issue in evaluate_rules(bad_source, rules)] == [
        ("defaults", "expected an object")
    ]
    bad_target = {"jobs": [1.5]}
    assert [(issue.path, issue.message) for issue in evaluate_rules(bad_target, rules)] == [
        ("jobs[0]", "expected an object"),
    ]


@pytest.mark.parametrize(
    "rules, message",
    [
        (
            (
                Suggests("", "left", "jobs.job", physics="first"),
                Suggests("", "right", "jobs.job", physics="second"),
            ),
            "duplicate Suggests target",
        ),
        (
            (
                Suggests("", "left", "jobs", physics="parent"),
                Suggests("", "right", "jobs.job", physics="child"),
            ),
            "overlapping Suggests targets",
        ),
        (
            (
                Suggests("", "right", "left", physics="left"),
                Suggests("", "left", "right", physics="right"),
            ),
            "cyclic Suggests dependency",
        ),
    ],
)
def test_suggests_rejects_ambiguous_injection_graphs(rules, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_rules({}, rules)


def test_manifest_skips_stage_relationship_checks_after_rule_failure(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def relationship_check(context):
        calls.append(context)
        return None

    contract = SimpleNamespace(
        PARAM_RULES=(
            Depends("", "mode", physics="mode is required"),
            Value("mode", Literal["valid"], physics="mode is controlled"),
        ),
        INPUT_RULES=(),
        CHECKS=(relationship_check,),
    )
    contract.JOB_RULES = stage_job_rules(contract.PARAM_RULES, contract.INPUT_RULES)
    monkeypatch.setattr("lamet_agent.manifest._load_stage_contract", lambda *args: contract)
    document = {
        "metadata": {
            "run_id": "invalid-stage-value",
            "root_directory": str(tmp_path),
            "artifacts_directory": "runs",
            "random_seed": 1,
            "workers": 1,
            "target_observable": "pdf",
            "resample_mode": "jackknife",
            "sample_error_mode": "covariance",
            "bin_size": 1,
        },
        "stages": {
            "demo": {
                "defaults": {"mode": "invalid"},
                "jobs": [{"id": "job", "inputs": {}}],
            }
        },
    }

    issues = Manifest(tmp_path / "manifest.json", document).validate()

    assert any(issue.path.endswith("jobs[0].mode") and "must be one of" in issue.message for issue in issues)
    assert calls == []


def test_fourier_scan_intrinsic_values_are_owned_by_rules() -> None:
    contract = _load_stage_contract("fourier_transform")
    scheme_scan = {
        "order": ["NLA"],
        "sector": "unsupported",
        "Lambda0_gev": 0.1,
        "posterior_prior_error_scale": [1.0],
        "model_average": False,
        "max_schemes": 1,
    }

    issues = evaluate_rules({"scheme_scan": scheme_scan}, contract.PARAM_RULES, complete=False)

    assert [(issue.path, issue.message) for issue in issues] == [
        ("scheme_scan.sector", "must be one of 'valence', 'singlet', 'full'")
    ]


def test_all_shipped_tools_have_provider_schemas() -> None:
    workflow_stages = {
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
    }
    for stage_id in (
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
        "review",
    ):
        tools = _discover_tools(stage_id)
        assert bool(tools) is (stage_id not in workflow_stages)
        assert all(tool.schema["function"]["name"] == tool.name for tool in tools)


def test_no_argument_tool_ignores_provider_empty_object_placeholder(tmp_path: Path) -> None:
    from lamet_agent.agent import _invoke
    from lamet_agent.data import EnsembleData

    tool = next(item for item in _discover_tools("review") if item.name == "inspect_results")
    target = EnsembleData(
        None,
        "bootstrap",
        [np.ones(2), np.ones(2)],
        ["z"],
        {"z": [0.0, 0.1]},
        attrs={"coord_unit": "fm"},
    )
    context = ToolContext(
        {},
        tmp_path / "manifest.json",
        "review",
        "rn",
        {},
        {"results": [target]},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    observation = _invoke(tool, context, {"{}": {}})
    assert observation["ignored_arguments"] == ["{}"]
    assert "result_summary" in context.state

    argument_tool = next(item for item in _discover_tools("review") if item.name == "list_literature")
    with pytest.raises(ValueError, match="unknown arguments"):
        _invoke(argument_tool, context, {"{}": {}})


def test_all_shipped_stage_contracts_load_without_tool_imports() -> None:
    for stage_id in (
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
        "review",
    ):
        contract = _load_stage_contract(stage_id)
        assert hasattr(contract, "PARAM_RULES")
        assert hasattr(contract, "INPUT_RULES")
        assert hasattr(contract, "CHECKS")


def test_shipped_contracts_do_not_use_optional_depends() -> None:
    assert "required" not in Depends.__dataclass_fields__


@pytest.mark.parametrize(
    "name,value",
    (
        ("resample_mode", "jackknife"),
        ("sample_error_mode", "median"),
        ("bootstrap_samples", 100),
        ("bin_size", 1),
    ),
)
def test_correlator_contract_rejects_global_sampling_controls(name: str, value: object) -> None:
    contract = _load_stage_contract("correlator_analysis")
    issues = evaluate_rules({name: value}, contract.PARAM_RULES, complete=False)
    assert [(issue.path, issue.message) for issue in issues] == [(name, f"unknown key {name!r}")]


def test_manifest_enforces_global_sampling_relationships(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path, resample_mode="bootstrap")
    manifest = Manifest(tmp_path / "manifest.json", {"metadata": metadata, "stages": {}})
    assert any(issue.path == "metadata.samples" and "required" in issue.message for issue in manifest.validate())

    metadata["samples"] = 100
    assert not [issue for issue in manifest.validate() if issue.path.startswith("metadata.")]

    manifest.document["metadata"]["resample_mode"] = "jackknife"
    assert not [issue for issue in manifest.validate() if issue.path.startswith("metadata.")]

    manifest.document["metadata"].pop("samples")
    manifest.document["metadata"]["sample_error_mode"] = "median"
    assert any(
        issue.path == "metadata.sample_error_mode" and "require" in issue.message for issue in manifest.validate()
    )


def test_manifest_rejects_legacy_sampling_abbreviations(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path, resample_mode="bs", bs_samples=100)
    issues = Manifest(tmp_path / "manifest.json", {"metadata": metadata, "stages": {}}).validate()
    assert any(issue.path == "metadata.resample_mode" and "must be one of" in issue.message for issue in issues)
    assert any(issue.path == "metadata.bs_samples" and "unknown key" in issue.message for issue in issues)


def test_correlator_manifest_accepts_missing_hook_windows() -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "examples" / "pion_pdf_gi_manifest_neo.json")
    defaults = manifest.document["stages"]["correlator_analysis"]["defaults"]
    defaults.pop("pt2_windows")

    assert manifest.validate() == []


def test_fourier_manifest_accepts_missing_recommended_tail_ranges() -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "examples" / "pion_pdf_gi_manifest_neo.json")
    defaults = manifest.document["stages"]["fourier_transform"]["defaults"]
    defaults.pop("zmin_fm")
    defaults.pop("zmax_fm")

    assert manifest.validate() == []


@pytest.mark.parametrize("stem", ("pion_da_gi", "kaon_da_gi"))
def test_da_examples_expand_the_reference_systematics_branches(stem: str) -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "examples" / f"{stem}_manifest_neo.json")
    authored_stages = manifest.document["stages"]
    assert len(authored_stages["fourier_transform"]["jobs"]) == 9
    assert len(authored_stages["perturbative_matching"]["jobs"]) == 9
    assert len(authored_stages["extrapolation"]["jobs"]) == 1

    resolved = load_manifest(manifest.path)
    assert resolved.validate() == []
    stages = resolved.document["stages"]
    assert "systematics" not in resolved.document
    assert '"job"' not in json.dumps(resolved.document)
    extrapolation_jobs = [job for job in resolved._resolved_jobs() if job.stage_id == "extrapolation"]
    assert all(
        job.params["priors"] == {"mean": 0.0, "sdev": 3.0}
        for job in extrapolation_jobs
        if job.params["operation"] == "fit"
    )
    budget_job = next(job for job in extrapolation_jobs if job.params["operation"] == "systematics_budget")
    assert "priors" not in budget_job.params
    assert len(stages["fourier_transform"]["jobs"]) == 27
    assert len(stages["perturbative_matching"]["jobs"]) == 45
    assert [job["id"] for job in stages["extrapolation"]["jobs"]] == [
        "extrapolate_all",
        "extrapolate_lambda_low",
        "extrapolate_lambda_high",
        "extrapolate_mu_low",
        "extrapolate_mu_high",
        "extrapolate_a_sym",
        "extrapolate_p_sym",
        "extrapolate_ap_sym",
        "extrapolation_systematics_budget",
    ]
    budget = stages["extrapolation"]["jobs"][-1]
    assert budget["systematics_groups"] == {
        "main": 0,
        "zs": [],
        "lambda_extrapolation": [1, 2],
        "lamet_scale": [3, 4],
        "other_extrapolations": [5, 6, 7],
    }
    fourier = {job["id"]: job for job in stages["fourier_transform"]["jobs"]}
    assert fourier["ft_a06m130_pz6_lambda_low"]["zmin_fm"] == [
        0.4592,
        0.5166,
        0.574,
        0.6314,
    ]
    matching = {job["id"]: job for job in stages["perturbative_matching"]["jobs"]}
    assert matching["mt_a06m130_pz6_lambda_low"]["inputs"]["quasi"] == "ft_a06m130_pz6_lambda_low"
    assert matching["mt_a06m130_pz6_mu_low"]["mu"] == pytest.approx(2.0**0.5)
    assert matching["mt_a06m130_pz6_mu_high"]["mu"] == pytest.approx(2.0 * 2.0**0.5)
    assert len(manifest.document["stages"]["fourier_transform"]["jobs"]) == 9
    assert "zmin_shift" not in json.dumps(manifest.document)

    parsed = load_manifest(manifest.path)
    assert parsed.validate() == []
    assert "systematics" not in parsed.document
    assert parsed.document["stages"]["extrapolation"]["defaults"]["required_terms"] == [
        "a",
        "inv_p2",
        "inv_p4",
        "ap2",
    ]
    assert parsed.document["stages"]["extrapolation"]["jobs"][0]["priors"] == {"mean": 0.0, "sdev": 3.0}


def test_job_source_object_is_rejected_at_the_input_role() -> None:
    path = Path(__file__).parents[2] / "examples" / "pion_pdf_gi_manifest_neo.json"
    manifest = load_manifest(path)
    document = copy.deepcopy(manifest.document)
    document["stages"]["fourier_transform"]["jobs"][0]["inputs"]["input"] = {"job": "rn_p4"}

    issues = Manifest(path, document).validate()

    source_issues = [issue for issue in issues if issue.path.endswith("fourier_transform.jobs[0].inputs.input")]
    assert len(source_issues) == 1
    assert source_issues[0].message == "is not an allowed input source"
    assert all(not issue.path.endswith(".job") for issue in issues)


def test_systematics_compiler_rejects_explicit_variation_jobs() -> None:
    path = Path(__file__).parents[2] / "examples" / "pion_da_gi_manifest_neo.json"
    manifest = load_manifest(path)
    document = copy.deepcopy(manifest.document)
    explicit = copy.deepcopy(document["stages"]["fourier_transform"]["jobs"][0])
    explicit["id"] += "_lambda_low"
    document["stages"]["fourier_transform"]["jobs"].append(explicit)

    issues = Manifest(path, document).validate()

    assert len(issues) == 1
    assert issues[0].path == "systematics"
    assert "explicitly authored variation jobs" in issues[0].message


def test_correlator_contract_keeps_lanczos_and_ground_fit_parameters_exclusive() -> None:
    contract = _load_stage_contract("correlator_analysis")
    lanczos = {
        "analysis_method": "lanczos",
        "component": "both",
        "nstate": [2],
        "correlator_ids": ["c2", "c3"],
        "scope": "3pt_matrix",
        "t0": 4,
        "time_step": 2,
    }
    assert evaluate_rules(lanczos, contract.PARAM_RULES) == []
    context = CheckContext({}, "correlator_analysis", "job", lanczos, {})
    assert evaluate_checks(contract.CHECKS, context) == []

    mixed = {**lanczos, "fit_scope": ["spectrum"]}
    assert evaluate_rules(mixed, contract.PARAM_RULES) == []
    assert "fit_scope" not in mixed

    ground_fit = {
        "analysis_method": "lsqfit",
        "component": "re",
        "nstate": [2],
        "correlator_ids": ["c2"],
        "fit_scope": ["spectrum"],
        "fit_strategy": ["independent"],
        "fitting_form": "Breit",
        "model_average": False,
        "pt2_windows": [{"tmin": 2, "tmax": 8}],
        "svdcut": 1e-6,
        "posterior_prior_error_scale": 1.0,
        "q_min": 0.05,
    }
    assert evaluate_rules(ground_fit, contract.PARAM_RULES) == []
    assert ground_fit["prior_width"] == [1.0]
    assert (
        evaluate_checks(
            contract.CHECKS,
            CheckContext({}, "correlator_analysis", "job", ground_fit, {}),
        )
        == []
    )


def test_each_shipped_stage_contract_reports_incomplete_params_instead_of_crashing(tmp_path: Path) -> None:
    for stage_id in (
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
        "review",
    ):
        manifest = {
            "metadata": {
                "run_id": "incomplete",
                "root_directory": str(tmp_path),
                "artifacts_directory": "runs",
                "random_seed": 1,
                "workers": 1,
            },
            "stages": {stage_id: {"defaults": {}, "jobs": [{"id": "job", "inputs": {}}]}},
        }
        issues = Manifest(tmp_path / "manifest.json", manifest).validate()
        assert issues


def test_neo_correlator_descriptors_use_physical_field_names() -> None:
    examples = Path(__file__).parents[2] / "examples"
    for path in examples.glob("*correlators_neo.json"):
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        ensemble = descriptor["ensemble"]
        assert "m_pi" in ensemble
        assert "m_pi_gev" not in ensemble
        for record in descriptor["correlators"]:
            assert "correlator_type" in record
            assert "kind" not in record
            current = record.get("current")
            if current is not None:
                assert "construction" not in current
                assert "observable" not in current


def test_matching_check_reports_the_exact_parameter_path() -> None:
    contract = _load_stage_contract("perturbative_matching")
    context = CheckContext(
        {},
        "perturbative_matching",
        "job",
        {"kernel_id": "CG_gt_quark_PDF_hybrid_NLO", "scheme": "ratio", "zs_fm": 0.2},
        {"quasi": "earlier"},
    )
    issues = evaluate_checks(contract.CHECKS, context)
    assert [(issue.path, issue.message) for issue in issues] == [
        ("scheme", "must equal 'hybrid' for kernel 'CG_gt_quark_PDF_hybrid_NLO'")
    ]


def test_matching_kernel_parameters_follow_the_selected_signature() -> None:
    contract = _load_stage_contract("perturbative_matching")

    def issues(kernel_id: str, scheme: str, parameters: dict[str, object], **extra: object):
        params = {
            "kernel_id": kernel_id,
            "scheme": scheme,
            "mu": 2.0,
            "lc_x_ls": [0.0, 1.0],
            "kernel_parameters": parameters,
            **extra,
        }
        context = CheckContext({}, "perturbative_matching", "job", params, {"quasi": "earlier"})
        return evaluate_checks(contract.CHECKS, context)

    ratio = "CG_gt_quark_PDF_ratio_NLO"
    rgr = "CG_gt_quark_PDF_hybrid_RGR_re_NLO"
    rgr_parameters = {"kappa": 1, "mu_min_gev": 0.6}
    assert issues(rgr, "hybrid", rgr_parameters, hybrid={"zs_fm": 0.18}) == []
    assert issues(ratio, "ratio", {}, hybrid={"zs_fm": 0.18}) == []
    switched = issues(ratio, "ratio", rgr_parameters)
    assert {issue.path for issue in switched} == {
        "kernel_parameters.kappa",
        "kernel_parameters.mu_min_gev",
    }
    assert any(
        issue.path == "kernel_parameters.rgr_kappa"
        for issue in issues(rgr, "hybrid", {"rgr_kappa": 1.0}, hybrid={"zs_fm": 0.18})
    )
    assert any(
        issue.path == "kernel_parameters.kappa"
        for issue in issues(rgr, "hybrid", {"kappa": True}, hybrid={"zs_fm": 0.18})
    )
    for managed in ("x_out", "x_in", "momentum_gev", "scale_gev", "zs_fm"):
        current = issues(rgr, "hybrid", {managed: 0.18}, hybrid={"zs_fm": 0.18})
        assert any(issue.path == f"kernel_parameters.{managed}" and "stage" in issue.message for issue in current)


def test_matching_kernel_parameter_rules_require_a_dict_and_required_signature_values() -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "examples" / "pion_pdf_cg_manifest_neo.json")
    manifest.document["stages"]["perturbative_matching"]["defaults"]["kernel_parameters"] = []
    assert any(
        issue.path.endswith("kernel_parameters") and "expected dict" in issue.message for issue in manifest.validate()
    )

    contract = _load_stage_contract("perturbative_matching")

    def kernel(x_out, x_in, *, momentum_gev: float, scale_gev: float, cutoff: int, enabled: bool = True):
        return None

    missing = contract._kernel_parameter_issues(kernel, {})
    assert [(issue.path, issue.message) for issue in missing] == [
        ("kernel_parameters.cutoff", "is required by the selected kernel signature")
    ]
    assert contract._kernel_parameter_issues(kernel, {"cutoff": 2, "enabled": False}) == []


def test_matching_check_requires_zs_fm_exactly_for_hybrid_kernels(monkeypatch) -> None:
    contract = _load_stage_contract("perturbative_matching")

    def without_zs(x_out, x_in, *, momentum_gev: float, scale_gev: float):
        return None

    def with_zs(x_out, x_in, *, momentum_gev: float, scale_gev: float, zs_fm: float):
        return None

    context = CheckContext(
        {},
        "perturbative_matching",
        "job",
        {
            "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
            "scheme": "hybrid",
            "kernel_parameters": {},
            "zs_fm": 0.18,
        },
        {"quasi": "earlier"},
    )
    monkeypatch.setattr(contract, "load_kernel", lambda _kernel_id: without_zs)
    issue = contract.check_kernel_parameters(context)
    assert isinstance(issue, Issue) and "must include" in issue.message

    context.params["kernel_id"] = "CG_gt_quark_PDF_ratio_NLO"
    context.params["scheme"] = "ratio"
    monkeypatch.setattr(contract, "load_kernel", lambda _kernel_id: with_zs)
    issue = contract.check_kernel_parameters(context)
    assert isinstance(issue, Issue) and "must omit" in issue.message


def test_renormalization_type_controls_inputs_and_requires_a_kernel() -> None:
    examples = Path(__file__).parents[2] / "examples"
    manifest = load_manifest(examples / "pion_da_gi_manifest_neo.json")
    assert manifest.validate() == []
    jobs = manifest.jobs_by_stage["renormalization"]
    from lamet_agent.stages.renormalization.parameters import effective_params

    fit = effective_params(jobs[0].params)
    apply = effective_params(jobs[1].params)
    contract = _load_stage_contract("renormalization")
    assert evaluate_rules(dict(jobs[0].params), contract.PARAM_RULES) == []
    assert evaluate_rules(dict(jobs[1].params), contract.PARAM_RULES) == []
    assert (fit["type"], fit["kernel_id"], set(jobs[0].inputs)) == (
        "fit",
        "z_msbar_pdf_nlo",
        {"reference"},
    )
    assert (apply["type"], apply["kernel_id"], set(jobs[1].inputs)) == (
        "apply",
        "z_msbar_da_nlo",
        {"target", "zR"},
    )

    wrong_type = load_manifest(examples / "pion_da_gi_manifest_neo.json")
    wrong_type.document["stages"]["renormalization"]["jobs"][0]["type"] = "apply"
    assert any(issue.path.endswith("inputs.target") and "required" in issue.message for issue in wrong_type.validate())

    missing_kernel = load_manifest(examples / "pion_da_gi_manifest_neo.json")
    missing_kernel.document["stages"]["renormalization"]["defaults"].pop("kernel_id")
    assert any(issue.path.endswith("kernel_id") for issue in missing_kernel.validate())

    wrong_signature = load_manifest(examples / "pion_da_gi_manifest_neo.json")
    fit_params = wrong_signature.document["stages"]["renormalization"]["jobs"][0]
    fit_params["kernel_id"] = "GI_gzg5_DA_ratio_NLO"
    assert any(issue.path.endswith("kernel_id") and "z_fm" in issue.message for issue in wrong_signature.validate())

    redundant_type = load_manifest(examples / "pion_pdf_gi_manifest_neo.json")
    redundant_type.document["stages"]["renormalization"]["jobs"][0]["type"] = "apply"
    assert redundant_type.validate() == []
    assert "type" not in redundant_type.jobs_by_stage["renormalization"][0].params


def test_finish_rejects_second_terminal_result(tmp_path: Path) -> None:
    context = ToolContext(
        {}, tmp_path / "manifest.json", "review", "job", {}, {}, {}, {}, tmp_path, np.random.default_rng(1)
    )
    summary = {
        "stage_id": "review",
        "job_id": "job",
        "result": "review",
        "decisions": {},
        "diagnostics": {},
        "artifacts": [],
    }
    context.finish("report", summary)
    try:
        context.finish("again", summary)
    except RuntimeError:
        pass
    else:
        raise AssertionError("finish must reject a second terminal result")


def test_dynamic_summary_converts_numpy_scalars_and_arrays() -> None:
    from lamet_agent.agent import _summarize

    summary = _summarize({"count": np.int64(3), "scale": np.float64(2.5), "values": np.asarray([1, 2], dtype=np.int64)})

    assert summary == {"count": 3, "scale": 2.5, "values": [1, 2]}
    json.dumps(summary)


def test_manifest_has_exact_two_top_level_keys(tmp_path: Path) -> None:
    descriptor = tmp_path / "input.json"
    descriptor.write_text("{}", encoding="utf-8")
    document = {
        "metadata": _valid_metadata(tmp_path, random_seed=7),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(descriptor)}]}}],
            }
        },
    }
    manifest = Manifest(tmp_path / "manifest.json", document)
    assert not manifest.validate()
    manifest.document["extra"] = True
    assert any("extra" in issue.path for issue in manifest.validate())


def test_manifest_rejects_redundant_metadata_stages(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    document = {
        "metadata": {
            "run_id": "toy",
            "root_directory": str(tmp_path),
            "artifacts_directory": "runs",
            "random_seed": 7,
            "workers": 1,
            "target_observable": "pdf",
            "resample_mode": "jackknife",
            "sample_error_mode": "covariance",
            "bin_size": 1,
            "stages": ["review"],
        },
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }

    issues = Manifest(tmp_path / "manifest.json", document).validate()

    assert any(issue.path == "metadata.stages" and "unknown key" in issue.message for issue in issues)


def test_top_level_stage_order_controls_execution_and_artifact_numbering(tmp_path: Path) -> None:
    document = {
        "metadata": {
            "run_id": "ordered",
            "root_directory": str(tmp_path),
            "artifacts_directory": "runs",
            "random_seed": 7,
            "workers": 1,
            "target_observable": "pdf",
            "resample_mode": "jackknife",
            "sample_error_mode": "covariance",
            "bin_size": 1,
        },
        "stages": {
            "second_authored": {"defaults": {}, "jobs": [{"id": "job_b", "inputs": {}}]},
            "first_authored": {"defaults": {}, "jobs": [{"id": "job_a", "inputs": {}}]},
        },
    }

    jobs = Manifest(tmp_path / "manifest.json", document)._resolved_jobs()

    assert [job.stage_id for job in jobs] == ["second_authored", "first_authored"]
    assert [job.artifact_directory.parent.name for job in jobs] == [
        "01_second_authored",
        "02_first_authored",
    ]


def test_manifest_requires_a_positive_run_worker_count(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    document = {
        "metadata": _valid_metadata(tmp_path, random_seed=7),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    document["metadata"].pop("workers")
    manifest = Manifest(tmp_path / "manifest.json", document)
    assert any(issue.path == "metadata.workers" for issue in manifest.validate())
    document["metadata"]["workers"] = 0
    assert any(issue.path == "metadata.workers" for issue in manifest.validate())


def test_scripted_review_run_uses_one_tool_per_turn(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "metadata": _valid_metadata(tmp_path, random_seed=2),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    responses = [
        _AssistantResponse("inspect", _ToolCall("1", "inspect_results", {})),
        _AssistantResponse("check", _ToolCall("2", "check_consistency", {})),
        _AssistantResponse("list", _ToolCall("3", "list_literature", {})),
        _AssistantResponse("read", _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]})),
        _AssistantResponse(
            "write",
            _ToolCall(
                "5",
                "write_review",
                {
                    "title": "Toy review",
                    "analysis": "The scoped outputs are mutually consistent.",
                    "conclusion": "The toy workflow is internally consistent.",
                },
            ),
        ),
    ]
    result = create_session(_ScriptedBackend(responses)).run_manifest(Manifest(tmp_path / "manifest.json", manifest))
    assert result["summaries"]["review_1"]["result"] == "review"
    assert (tmp_path / "runs" / "01_review" / "review_1" / "review.md").is_file()
    transcript = (tmp_path / "runs" / "01_review" / "review_1" / "llm_transcript.md").read_text(encoding="utf-8")
    assert "## Turn 1: sent to LLM" in transcript
    assert '"role": "system"' in transcript
    assert '"tools": [' in transcript
    assert "## Turn 1: received from LLM" in transcript
    assert "## Turn 2: sent to LLM" in transcript
    assert '"role": "tool"' in transcript
    assert "## Turn 5: received from LLM" in transcript
    assert '"name": "write_review"' in transcript
    assert "tool result:" not in transcript
    assert "Run completed" not in transcript
    stdout = capsys.readouterr().out
    assert "Stage: review" in stdout
    assert "Job: review/review_1" in stdout
    assert stdout.count("Calling LLM (scripted:test)") == 5
    assert stdout.count("LLM response received.") == 5
    assert "Running tool: inspect_results..." in stdout
    assert "Tool completed: write_review." in stdout
    assert "Stage review finished." in stdout
    assert "Agent run complete (1 job(s))." in stdout


def test_scripted_review_run_executes_multi_call_responses_sequentially(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "metadata": _valid_metadata(tmp_path, random_seed=2),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "inspect and check",
                tool_calls=(_ToolCall("1", "inspect_results", {}), _ToolCall("2", "check_consistency", {})),
            ),
            _AssistantResponse(
                "list and read",
                tool_calls=(
                    _ToolCall("3", "list_literature", {}),
                    _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]}),
                ),
            ),
            _AssistantResponse(
                "write",
                _ToolCall(
                    "5",
                    "write_review",
                    {
                        "title": "Toy review",
                        "analysis": "The scoped outputs are mutually consistent.",
                        "conclusion": "The toy workflow is internally consistent.",
                    },
                ),
            ),
        ]
    )
    result = create_session(backend).run_manifest(Manifest(tmp_path / "manifest.json", manifest))
    assert result["summaries"]["review_1"]["result"] == "review"
    assert len(backend.calls) == 3
    second_turn_messages = backend.calls[1][0]
    assert [message.role for message in second_turn_messages[-3:]] == ["assistant", "tool", "tool"]
    assert [call.name for call in second_turn_messages[-3].calls] == ["inspect_results", "check_consistency"]
    stdout = capsys.readouterr().out
    assert stdout.count("Calling LLM (scripted:test)") == 3
    assert stdout.index("Running tool: inspect_results...") < stdout.index("Running tool: check_consistency...")


def test_manifest_run_accepts_a_path_as_the_public_entrypoint(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "metadata": _valid_metadata(tmp_path, random_seed=2),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    responses = [
        _AssistantResponse("inspect", _ToolCall("1", "inspect_results", {})),
        _AssistantResponse("check", _ToolCall("2", "check_consistency", {})),
        _AssistantResponse("list", _ToolCall("3", "list_literature", {})),
        _AssistantResponse("read", _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]})),
        _AssistantResponse(
            "write",
            _ToolCall(
                "5",
                "write_review",
                {
                    "title": "Toy review",
                    "analysis": "The scoped outputs are mutually consistent.",
                    "conclusion": "The toy workflow is internally consistent.",
                },
            ),
        ),
    ]
    result = create_session(_ScriptedBackend(responses)).run_manifest(load_manifest(manifest_path))
    assert result["summaries"]["review_1"]["result"] == "review"


def test_deterministic_stage_workflow_bypasses_the_backend(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.stages.perturbative_matching.workflow as workflow

    calls = []

    def deterministic(context, _ask):
        calls.append(context.job_id)
        context.finish(
            "matched",
            {
                "stage_id": context.stage_id,
                "job_id": context.job_id,
                "result": "matched_distribution",
                "decisions": {},
                "diagnostics": {},
                "artifacts": [],
            },
        )

    monkeypatch.setattr(workflow, "run", deterministic)
    backend = _ScriptedBackend([])
    context = ToolContext(
        {"metadata": {"workers": 1}},
        tmp_path / "manifest.json",
        "perturbative_matching",
        "matching",
        {
            "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
            "scheme": "ratio",
            "mu": 2.0,
            "lc_x_ls": [0.0, 1.0],
            "kernel_parameters": {},
        },
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )

    output, summary = create_session(backend)._run_context(context, [], "", "")

    assert output == "matched"
    assert summary["result"] == "matched_distribution"
    assert calls == ["matching"]
    assert backend.calls == []


def test_correlator_workflow_asks_only_for_typed_fit_parameters(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.stages.correlator_analysis.workflow as workflow
    from lamet_agent.data import EnsembleData

    events = []
    monkeypatch.setattr(workflow, "inspect", lambda _context: events.append("inspect"))

    def fit(_context, *, tune_z_values):
        events.append(("fit", tune_z_values))
        return {"metrics": {"recommended_candidate_id": "matrix_001"}}

    def publish(context, *, candidate_id):
        events.append(("publish", candidate_id))
        context.finish(
            "matrix",
            {
                "stage_id": context.stage_id,
                "job_id": context.job_id,
                "result": "bare_matrix_element",
                "decisions": {"candidate_id": candidate_id},
                "diagnostics": {},
                "artifacts": [],
            },
        )

    monkeypatch.setattr(workflow, "fit_qda", fit)
    monkeypatch.setattr(workflow, "publish", publish)
    backend = _ScriptedBackend([_AssistantResponse("", structured={"tune_z_values": [0.1]})])

    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "correlator",
        {
            "analysis_method": "lsqfit",
            "fit_scope": ["qda_ratio"],
            "component": "re",
            "correlator_ids": ["qda"],
            "pt2_windows": [{"tmin": 2, "tmax": 8}],
        },
        {},
        {},
        {
            "correlators": {
                "qda": EnsembleData(
                    None,
                    "bootstrap",
                    [np.asarray([1.0, 0.8]), np.asarray([1.0, 0.9])],
                    ["z"],
                    {"z": [0.0, 0.1]},
                )
            }
        },
        tmp_path,
        np.random.default_rng(1),
    )

    workflow.run(context, LlmSession(backend, tmp_path / "recommendation.md"))

    assert events == ["inspect", ("fit", [0.1]), ("publish", "matrix_001")]
    assert backend.calls[0][1] == []
    schema = backend.response_schemas[0]["schema"]
    assert schema["required"] == ["tune_z_values"]
    assert schema["properties"]["tune_z_values"]["items"] == {"type": "number"}


def test_recommendation_sends_data_on_first_human_failure_but_not_on_retry(tmp_path: Path) -> None:
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.correlator_analysis.tools.recommend_qda_tune_z.recommendation import recommend

    backend = _ScriptedBackend(
        [
            _AssistantResponse("", structured={"tune_z_values": [0.1]}),
            _AssistantResponse("", structured={"tune_z_values": [0.2]}),
        ]
    )
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "correlator",
        {"component": "re", "correlator_ids": ["qda"]},
        {},
        {},
        {
            "correlators": {
                "qda": EnsembleData(
                    None,
                    "bootstrap",
                    [np.asarray([1.0, 0.8]), np.asarray([1.0, 0.9])],
                    ["z"],
                    {"z": [0.0, 0.1]},
                )
            }
        },
        tmp_path,
        np.random.default_rng(1),
    )
    session = LlmSession(backend, tmp_path / "recommendation.md")
    attempts = {"matrix_001": {"parameters": {"window": [2, 8]}, "Q": 0.01, "chi2_dof": 2.0}}

    assert recommend(context, session, previous_attempts=attempts) == {"tune_z_values": [0.1]}
    assert recommend(context, session, previous_attempts=attempts) == {"tune_z_values": [0.2]}

    first_payload = json.loads(backend.calls[0][0][-1].content)
    first_evidence = first_payload["request"]["evidence"]
    second_evidence = json.loads(backend.calls[1][0][-1].content)["evidence"]
    assert set(first_evidence) == {"fixed_parameters", "previous_attempts"}
    assert set(second_evidence) == {"fixed_parameters", "previous_attempts"}
    assert first_evidence["previous_attempts"] == attempts
    assert first_payload["context"][0]["key"] == "correlator_fit_data"
    assert "correlators" in first_payload["context"][0]["content"]


def test_joint_qda_null_hook_and_tune_z_share_one_recommendation(tmp_path: Path) -> None:
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.correlator_analysis.tools._joint_fit_recommendation import initial, pt2_windows

    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "",
                structured={"pt2_windows": [{"tmin": 2, "tmax": 8}], "tune_z_values": [0.1]},
            )
        ]
    )
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "correlator",
        {
            "component": "re",
            "correlator_ids": ["qda"],
            "fit_scope": ["qda_ratio"],
            "nstate": [1],
        },
        {},
        {},
        {
            "correlators": {
                "qda": EnsembleData(
                    None,
                    "bootstrap",
                    [np.asarray([1.0, 0.8]), np.asarray([1.0, 0.9])],
                    ["z"],
                    {"z": [0.0, 0.1]},
                )
            }
        },
        tmp_path,
        np.random.default_rng(1),
    )
    session = LlmSession(backend, tmp_path / "joint.md")

    assert pt2_windows(context, session) == [{"tmin": 2, "tmax": 8}]
    assert initial(context, session)["tune_z_values"] == [0.1]
    assert session.recommendation_calls == 1


def test_fourier_tail_range_recommendation_reuses_context_and_obeys_job_budget(tmp_path: Path) -> None:
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.fourier_transform.recommendation import initial, revise

    data = EnsembleData(
        None,
        "bootstrap",
        [np.asarray([0.8, 1.0, 0.8], dtype=complex), np.asarray([0.7, 1.0, 0.7], dtype=complex)],
        ["z"],
        {"z": [-0.1, 0.0, 0.1]},
        attrs={"coord_unit": "fm", "momentum_gev": 2.0},
    )
    backend = _ScriptedBackend(
        [
            _AssistantResponse("", structured={"zmin_fm": [0.05], "zmax_fm": [0.1]}),
            _AssistantResponse("", structured={"zmin_fm": [0.04], "zmax_fm": [0.1]}),
        ]
    )
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "fourier_transform",
        "fourier",
        {
            "scheme_scan": {"order": ["LA"], "posterior_prior_error_scale": [3.0]},
            "zmax_ext_fm": 0.2,
        },
        {},
        {},
        {"fourier_input": data, "tail_inspection": {"spacing_fm": 0.1}},
        tmp_path,
        np.random.default_rng(1),
    )
    session = LlmSession(backend, tmp_path / "fourier.md", max_recommendation_calls=2)

    assert initial(context, session) == {"zmin_fm": [0.05], "zmax_fm": [0.1]}
    attempts = {"candidate": {"parameters": {"order": "LA"}, "Q": 0.01, "chi2_dof": 2.0}}
    assert revise(context, session, attempts) == {"zmin_fm": [0.04], "zmax_fm": [0.1]}
    with pytest.raises(RuntimeError, match="limit exceeded"):
        revise(context, session, attempts)

    first_payload = json.loads(backend.calls[0][0][-1].content)
    second_payload = json.loads(backend.calls[1][0][-1].content)
    assert first_payload["context"][0]["key"] == "fourier_tail_fit_data"
    assert "context" not in second_payload
    assert second_payload["evidence"]["previous_attempts"] == attempts


def test_fourier_workflow_allows_user_attempt_plus_two_job_recommendations(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.stages.fourier_transform.workflow as workflow

    monkeypatch.setattr(workflow, "inspect", lambda _context: None)
    qualities = [0.01, 0.02, 0.8]
    attempted_ranges = []

    def attempt(context):
        attempted_ranges.append((list(context.params["zmin_fm"]), list(context.params["zmax_fm"])))
        quality = qualities[len(attempted_ranges) - 1]
        return {
            "range_candidates": [],
            "model_candidates": [{"label": f"candidate_{len(attempted_ranges)}", "Q": quality}],
        }

    revisions = []

    def revise(_context, session, previous_attempts):
        revisions.append(previous_attempts)
        session.recommendation_calls += 1
        value = 0.2 + 0.1 * len(revisions)
        return {"zmin_fm": [value], "zmax_fm": [0.8]}

    published = []
    monkeypatch.setattr(workflow, "attempt", attempt)
    monkeypatch.setattr(workflow, "revise", revise)
    monkeypatch.setattr(workflow, "publish", lambda _context, result: published.append(result))
    context = ToolContext(
        {"metadata": {}},
        tmp_path / "manifest.json",
        "fourier_transform",
        "fourier",
        {"zmin_fm": [0.2], "zmax_fm": [0.8], "scheme_scan": {"q_min": 0.05}},
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    session = LlmSession(_ScriptedBackend([]), tmp_path / "fourier.md", max_recommendation_calls=2)

    workflow.run(context, session)

    assert len(attempted_ranges) == 3
    assert len(revisions) == 2
    assert session.recommendation_calls == 2
    assert published[0]["model_candidates"][0]["Q"] == 0.8


def test_correlator_workflow_recommends_once_more_after_low_quality(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.stages.correlator_analysis.workflow as workflow

    monkeypatch.setattr(workflow, "inspect", lambda _context: None)
    recommendations = []

    def initial(_context, _session):
        recommendations.append(None)
        return {"tune_z_values": [1.0]}

    def revise(_context, _session, previous_attempts):
        recommendations.append(previous_attempts)
        return {"pt2_windows": [{"tmin": 2, "tmax": 8}], "tune_z_values": [2.0]}

    attempts = []

    def fit(context, *, tune_z_values):
        attempts.append(tune_z_values)
        quality = 0.01 if len(attempts) == 1 else 0.8
        context.state["matrix_element_candidates"] = [
            {
                "id": f"matrix_{len(attempts):03d}",
                "fit_strategy": "independent",
                "fit_scope": "qda_ratio",
                "window": {"t_min": 2, "t_max": 8, "tau_min": None},
                "nstate": 1,
                "prior_width": 1.0,
                "min_Q": quality,
                "worst_chi2_dof": 1.0,
                "feasible_at_all_tune_z": True,
                "numerical_failure": False,
                "tune_z_values": tune_z_values,
                "tune_z_diagnostics": {str(tune_z_values[0]): {"Q": quality, "chi2": 8.0, "dof": 8.0, "chi2_dof": 1.0}},
            }
        ]
        return {"metrics": {"recommended_candidate_id": f"matrix_{len(attempts):03d}"}}

    published = []
    monkeypatch.setattr(workflow, "initial", initial)
    monkeypatch.setattr(workflow, "revise", revise)
    monkeypatch.setattr(workflow, "fit_qda", fit)
    monkeypatch.setattr(workflow, "publish", lambda _context, *, candidate_id: published.append(candidate_id))
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "correlator",
        {
            "analysis_method": "lsqfit",
            "fit_scope": ["qda_ratio"],
            "q_min": 0.05,
            "pt2_windows": [{"tmin": 2, "tmax": 8}],
        },
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )

    workflow.run(context, LlmSession(_ScriptedBackend([]), tmp_path / "llm.md"))

    assert attempts == [[1.0], [2.0]]
    assert recommendations[0] is None
    assert recommendations[1]["matrix_001"]["min_Q"] == 0.01
    assert recommendations[1]["matrix_001"]["parameters"]["window"] == {
        "t_min": 2,
        "t_max": 8,
        "tau_min": None,
    }
    assert recommendations[1]["matrix_001"]["by_tune_z"]["1.0"]["Q"] == 0.01
    assert published == ["matrix_002"]


def test_renormalization_workflow_routes_virtual_provider_type(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.stages.renormalization.workflow as workflow

    events = []
    monkeypatch.setattr(workflow, "inspect", lambda _context: events.append("inspect"))
    monkeypatch.setattr(workflow, "fit", lambda _context: events.append("fit"))
    monkeypatch.setattr(workflow, "apply", lambda _context: events.append("apply"))
    context = ToolContext(
        {"metadata": {}},
        tmp_path / "manifest.json",
        "renormalization",
        "fit",
        {
            "strategy": "self_renormalization",
            "type": "fit",
            "scheme": "ratio",
            "kernel_id": "z_msbar_pdf_nlo",
            "mu": 2.0,
            "LambdaQCD_gev": 0.1,
            "d": -0.08,
        },
        {"reference": "source"},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )

    workflow.run(context, lambda **_kwargs: None)

    assert events == ["inspect", "fit"]


def test_agent_fails_immediately_when_the_model_returns_no_tool_call(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "metadata": _valid_metadata(tmp_path, random_seed=2),
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "English",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    session = create_session(_ScriptedBackend([_AssistantResponse("plain answer", None)]))
    with pytest.raises(RuntimeError, match="returned no tool call"):
        session.run_manifest(Manifest(tmp_path / "manifest.json", manifest))
    transcript = (tmp_path / "runs" / "01_review" / "review_1" / "llm_transcript.md").read_text(encoding="utf-8")
    assert "## Turn 1: sent to LLM" in transcript
    assert "## Turn 1: received from LLM" in transcript
    assert "plain answer" in transcript
    assert "Run failed" not in transcript
    assert "returned no tool call" not in transcript


def test_finish_rejects_a_declared_artifact_that_does_not_exist(tmp_path: Path) -> None:
    context = ToolContext(
        {}, tmp_path / "manifest.json", "review", "job", {}, {}, {}, {}, tmp_path, np.random.default_rng(1)
    )
    summary = {
        "stage_id": "review",
        "job_id": "job",
        "result": "review",
        "decisions": {},
        "diagnostics": {},
        "artifacts": ["missing.md"],
    }
    with pytest.raises(FileNotFoundError, match="missing.md"):
        context.finish("report", summary)
