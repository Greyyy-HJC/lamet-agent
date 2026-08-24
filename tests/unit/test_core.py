"""Focused checks for the independent ``lamet_agent_neo`` architecture."""

from __future__ import annotations

import ast
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Literal, TypedDict

import numpy as np
import pytest

from lamet_agent.agent import (
    ToolContext,
    _discover_tools,
    _resolve_runtime_null_hooks,
    _write_transcript_header,
    create_session,
)
from lamet_agent.contract import (
    CheckContext,
    Depends,
    List,
    Provides,
    Recommends,
    Value,
    _apply_recommended_defaults,
    _unresolved_null_hooks,
    evaluate_checks,
    evaluate_rules,
)
from lamet_agent.__main__ import _build_parser
from lamet_agent.llm import Message, _AssistantResponse, _ToolCall, create_backend
from lamet_agent.manifest import Manifest, _load_stage_contract, load_manifest


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

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, object]],
        prompt_digest: str,
    ) -> _AssistantResponse:
        self.calls.append((list(messages), list(tools), prompt_digest))
        if not self._responses:
            raise RuntimeError("scripted backend has no response for this turn")
        return self._responses.pop(0)


class _RecommendedInterval(TypedDict):
    start: int
    stop: int


class _PlateauAssessment(TypedDict):
    stable_start: int


def _recommend_interval(_context: ToolContext, ask) -> list[_RecommendedInterval]:
    return ask(
        instruction="Choose one nonempty half-open interval.",
        evidence={"allowed_coordinates": [0, 1, 2, 3]},
    )


def _estimate_interval_without_llm(
    _context: ToolContext, _ask
) -> list[_RecommendedInterval]:
    return [{"start": 1, "stop": 3}]


def _estimate_interval_with_two_llm_calls(
    _context: ToolContext, ask
) -> list[_RecommendedInterval]:
    assessment = ask(
        instruction="Identify the first stable coordinate.",
        evidence={"coordinates": [0, 1, 2, 3]},
        response_type=_PlateauAssessment,
    )
    return ask(
        instruction="Choose an interval using the preliminary assessment.",
        evidence={"assessment": assessment, "last_coordinate": 3},
    )


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


def test_neo_introduces_no_unreviewed_numeric_literals() -> None:
    """Every executable neo literal must already have legacy provenance."""
    root = Path(__file__).parents[2]

    def numeric_literals(package: Path) -> set[int | float | complex]:
        values: set[int | float | complex] = set()
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            values.update(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float, complex))
                and not isinstance(node.value, bool)
            )
        return values

    legacy = numeric_literals(root / "lamet_agent")
    neo = numeric_literals(root / "lamet_agent_neo")
    approved_neo_semantics = {34.1344746}
    assert neo <= legacy | approved_neo_semantics


def test_neo_plotting_owns_the_figure_and_clears_it_after_saving(tmp_path: Path) -> None:
    import gvar
    from matplotlib import rcParams

    from lamet_agent.plotting import (
        COLOR_CYCLE,
        configure_plot,
        errorband,
        errorbar,
        hband,
        hline,
        line,
        plot,
        save_figure,
        start_plot,
        vband,
        vline,
    )

    assert start_plot() is None
    values = np.asarray([gvar.gvar(1.0, 0.1), gvar.gvar(1.5, 0.2)], dtype=object)
    assert errorband(
        [0.0, 1.0], values, color=COLOR_CYCLE[1], label="result"
    ) is None
    assert errorbar(
        [0.0, 1.0], values, color=COLOR_CYCLE[0], marker="s", label="points"
    ) is None
    assert plot(
        [0.0, 1.0], [0.9, 1.4], color=COLOR_CYCLE[2], marker="^", label="central points"
    ) is None
    assert line(
        [0.0, 1.0], [0.8, 1.3], color=COLOR_CYCLE[3], label="central line"
    ) is None
    with pytest.raises(ValueError, match="unsupported marker"):
        errorbar([0.0, 1.0], values, marker="r--")
    with pytest.raises(ValueError, match="unsupported marker"):
        plot([0.0, 1.0], [0.9, 1.4], marker="r--")
    with pytest.raises(TypeError, match="gvar"):
        errorbar([0.0, 1.0], np.asarray([[1.0, 2.0], [1.1, 2.1]]))
    assert hline(0.0, color=COLOR_CYCLE[2], linestyle="dashed") is None
    assert vline(0.5, color=COLOR_CYCLE[3], linestyle=":") is None
    assert hband(0.8, 1.2, color=COLOR_CYCLE[4]) is None
    assert vband(0.2, 0.4, color=COLOR_CYCLE[5]) is None
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
    errorbar([0.0, 1.0], values)
    plot([0.0, 1.0], [0.9, 1.4])
    line([0.0, 1.0], [0.8, 1.3])
    hline(0.0)
    vline(0.5)
    hband(0.8, 1.2)
    vband(0.2, 0.4)
    cycle_path = tmp_path / "cycle.svg"
    save_figure(cycle_path)
    cycle_svg = cycle_path.read_text(encoding="utf-8").lower()
    assert all(color.lower() in cycle_svg for color in COLOR_CYCLE)


def test_neo_contract_plotting_and_parallel_exports_are_minimal() -> None:
    from lamet_agent import contract, parallel, plotting

    assert contract.__all__ == [
        "Depends",
        "Provides",
        "Recommends",
        "List",
        "Value",
        "Issue",
        "CheckContext",
        "evaluate_rules",
        "evaluate_checks",
    ]
    assert not hasattr(contract, "Contains")
    assert plotting.__all__ == [
        "COLOR_CYCLE",
        "start_plot",
        "configure_plot",
        "errorbar",
        "errorband",
        "plot",
        "line",
        "hline",
        "vline",
        "hband",
        "vband",
        "save_figure",
    ]
    assert not hasattr(plotting, "mean_sdev")
    assert parallel.__all__ == ["FitNumericalError", "nonlinear_fit", "fourier_transform"]


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
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kwargs: _ModelsResponse(["gpt-5.6-luna", "gpt-test"]))
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


def test_neo_backend_factory_owns_api_key_file_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        "choices": [{"message": {"content": "fit both candidates", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "fit", "arguments": "{\"window\":1}"}},
            {"id": "call-2", "type": "function", "function": {"name": "fit", "arguments": "{\"window\":2}"}},
        ]}}]
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
    assert [
        (issue.path, issue.message)
        for issue in evaluate_rules({"settings": {"windows": []}}, rules)
    ] == [("settings.windows", "failed its intrinsic value check")]
    assert evaluate_rules(
        {"settings": {"windows": [{"start": 1, "stop": 3}]}}, rules
    ) == []


def test_runtime_null_hook_uses_a_typed_response_and_updates_params(
    tmp_path: Path,
) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "selected from the plateau",
                _ToolCall(
                    "recommend-1",
                    "return_parameter_estimate",
                    {"value": [{"start": 1, "stop": 3}]},
                ),
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
        backend=backend,
        transcript_path=transcript,
    )

    assert params["settings"]["windows"] == [{"start": 1, "stop": 3}]
    assert context.state["null_hook_provenance"]["settings.windows"] == {
        "backend": "scripted:test",
        "hook": "_recommend_interval",
        "llm_requests": 1,
        "value": [{"start": 1, "stop": 3}],
    }
    schema = backend.calls[0][1][0]["function"]["parameters"]["properties"][
        "value"
    ]
    assert schema["items"]["additionalProperties"] is False
    assert schema["items"]["required"] == ["start", "stop"]
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "Null hook settings.windows, request 1: sent to LLM" in transcript_text
    assert "Null hook settings.windows, request 1: received from LLM" in transcript_text


def test_invalid_runtime_null_hook_value_is_rolled_back(tmp_path: Path) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "no usable interval",
                _ToolCall(
                    "recommend-1",
                    "return_parameter_estimate",
                    {"value": []},
                ),
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
            contract=SimpleNamespace(
                PARAM_RULES=_null_hook_rules(), CHECKS=()
            ),
            backend=backend,
            transcript_path=transcript,
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
    assert evaluate_rules({"settings": {"mode": "invalid"}}, rules)[0].path == (
        "settings.mode"
    )
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
        backend=_ScriptedBackend([]),
        transcript_path=transcript,
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
        contract=SimpleNamespace(
            PARAM_RULES=_null_hook_rules(_estimate_interval_without_llm), CHECKS=()
        ),
        backend=backend,
        transcript_path=transcript,
    )

    assert params["settings"]["windows"] == [{"start": 1, "stop": 3}]
    assert backend.calls == []
    assert context.state["null_hook_provenance"]["settings.windows"][
        "llm_requests"
    ] == 0


def test_null_hook_may_make_multiple_typed_llm_requests(tmp_path: Path) -> None:
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "plateau assessment",
                _ToolCall(
                    "estimate-1",
                    "return_parameter_estimate",
                    {"value": {"stable_start": 1}},
                ),
            ),
            _AssistantResponse(
                "final interval",
                _ToolCall(
                    "estimate-2",
                    "return_parameter_estimate",
                    {"value": [{"start": 1, "stop": 3}]},
                ),
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
        backend=backend,
        transcript_path=transcript,
    )

    assert len(backend.calls) == 2
    second_evidence = json.loads(backend.calls[1][0][1].content)["evidence"]
    assert second_evidence["assessment"] == {"stable_start": 1}
    assert context.state["null_hook_provenance"]["settings.windows"][
        "llm_requests"
    ] == 2


def test_depends_owns_structured_mapping_type_once() -> None:
    rules = (
        Depends("", "settings", physics="settings are declared"),
        Depends("settings", "left", physics="settings own left"),
        Depends("settings", "right", physics="settings own right"),
        Value("settings.left", int, physics="left is an integer"),
        Value("settings.right", int, physics="right is an integer"),
    )

    issues = evaluate_rules({"settings": []}, rules)

    assert [(issue.path, issue.message) for issue in issues] == [
        ("settings", "expected an object")
    ]


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
            structured_paths = {
                rule.parent for rule in rules if isinstance(rule, Depends)
            }
            redundant = [
                rule.path
                for rule in rules
                if isinstance(rule, Value)
                and rule.expected is dict
                and rule.path in structured_paths
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
    assert [(issue.path, issue.message) for issue in issues] == [
        ("mode", "must be one of 'first', 'second'")
    ]

    integer_rule = Value("sign", Literal[-1, 1], physics="sign is controlled")
    sign_depends = Depends("", "sign", physics="sign is declared")
    assert evaluate_rules({"sign": 1}, (sign_depends, integer_rule)) == []
    assert evaluate_rules({"sign": True}, (sign_depends, integer_rule))[0].message == "must be one of -1, 1"
    assert not hasattr(string_rule, "choices")


def test_contract_provides_defines_selector_values_and_real_dependencies() -> None:
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
        Provides(
            "",
            "lanczos",
            "analysis_method",
            physics="Lanczos owns its parameter object.",
        ),
        Depends("lanczos", "iterations", physics="Lanczos iterations are required."),
        Value("lanczos.iterations", int, physics="Lanczos iterations are integers."),
    )

    assert evaluate_rules(
        {"analysis_method": "lsqfit", "lsqfit": {"window": 4}}, rules
    ) == []
    assert evaluate_rules(
        {"analysis_method": "lanczos", "lanczos": {"iterations": 3}}, rules
    ) == []
    assert [(issue.path, issue.message) for issue in evaluate_rules(
        {"analysis_method": "unknown"}, rules
    )] == [
        (
            "analysis_method",
            "must be provided by one of 'lsqfit', 'lanczos'",
        )
    ]
    assert [(issue.path, issue.message, issue.physics) for issue in evaluate_rules(
        {"analysis_method": "lsqfit"}, rules
    )] == [
        (
            "lsqfit",
            "is required when analysis_method='lsqfit'",
            "Least-squares fitting owns its parameter object.",
        )
    ]
    assert [(issue.path, issue.message) for issue in evaluate_rules(
        {
            "analysis_method": "lsqfit",
            "lsqfit": {"window": 4},
            "lanczos": {"iterations": 3},
        },
        rules,
    )] == [("lanczos", "unknown key 'lanczos'")]


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
    params = {"analysis_method": "lsqfit", "lsqfit": {}}

    assert _apply_recommended_defaults(params, rules) == {"lsqfit.mode": "safe"}
    assert params == {
        "analysis_method": "lsqfit",
        "lsqfit": {"mode": "safe"},
    }
    assert _unresolved_null_hooks(params, rules) == ()


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
    monkeypatch.setattr("lamet_agent_neo.manifest._load_stage_contract", lambda *args: contract)
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
                "jobs": [{"id": "job", "inputs": {}, "params": {}}],
            }
        },
    }

    issues = Manifest(tmp_path / "manifest.json", document).validate()

    assert any(issue.path.endswith("params.mode") and "must be one of" in issue.message for issue in issues)
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
        "component": "both",
        "output_scale": 1.0,
        "q_min": 0.05,
    }

    issues = evaluate_rules({"scheme_scan": scheme_scan}, contract.PARAM_RULES, complete=False)

    assert [(issue.path, issue.message) for issue in issues] == [
        ("scheme_scan.sector", "must be one of 'valence', 'singlet', 'full'")
    ]


def test_all_shipped_tools_have_provider_schemas() -> None:
    for stage_id in ("correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"):
        tools = _discover_tools(stage_id)
        assert tools
        assert all(tool.schema["function"]["name"] == tool.name for tool in tools)


def test_no_argument_tool_ignores_provider_empty_object_placeholder(tmp_path: Path) -> None:
    from lamet_agent.agent import _invoke
    from lamet_agent.data import EnsembleData

    tool = next(
        item
        for item in _discover_tools("renormalization")
        if item.name == "inspect_renormalization"
    )
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
        "renormalization",
        "rn",
        {},
        {"target": target},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    observation = _invoke(tool, context, {"{}": {}})
    assert observation["ignored_arguments"] == ["{}"]
    assert "aligned_inputs" in context.state

    argument_tool = next(
        item
        for item in _discover_tools("correlator_analysis")
        if item.name == "inspect_correlators"
    )
    with pytest.raises(ValueError, match="unknown arguments"):
        _invoke(argument_tool, context, {"{}": {}})


def test_all_shipped_stage_contracts_load_without_tool_imports() -> None:
    for stage_id in ("correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"):
        contract = _load_stage_contract(stage_id)
        assert hasattr(contract, "PARAM_RULES")
        assert hasattr(contract, "INPUT_RULES")
        assert hasattr(contract, "CHECKS")


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
    assert [(issue.path, issue.message) for issue in issues] == [
        (name, f"unknown key {name!r}")
    ]


def test_manifest_enforces_global_sampling_relationships(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path, resample_mode="bootstrap")
    manifest = Manifest(tmp_path / "manifest.json", {"metadata": metadata, "stages": {}})
    assert any(issue.path == "metadata.bootstrap_samples" and "required" in issue.message for issue in manifest.validate())

    metadata["bootstrap_samples"] = 100
    assert not [issue for issue in manifest.validate() if issue.path.startswith("metadata.")]

    metadata["resample_mode"] = "jackknife"
    assert any(issue.path == "metadata.bootstrap_samples" and "must be omitted" in issue.message for issue in manifest.validate())

    metadata.pop("bootstrap_samples")
    metadata["sample_error_mode"] = "median"
    assert any(issue.path == "metadata.sample_error_mode" and "require" in issue.message for issue in manifest.validate())


def test_manifest_rejects_legacy_sampling_abbreviations(tmp_path: Path) -> None:
    metadata = _valid_metadata(tmp_path, resample_mode="bs", bs_samples=100)
    issues = Manifest(tmp_path / "manifest.json", {"metadata": metadata, "stages": {}}).validate()
    assert any(issue.path == "metadata.resample_mode" and "must be one of" in issue.message for issue in issues)
    assert any(issue.path == "metadata.bs_samples" and "unknown key" in issue.message for issue in issues)


def test_correlator_manifest_accepts_missing_hook_windows() -> None:
    manifest = load_manifest(
        Path(__file__).parents[2] / "examples" / "pion_pdf_gi_manifest_neo.json"
    )
    lsqfit = manifest.document["stages"]["correlator_analysis"]["defaults"]["lsqfit"]
    lsqfit.pop("pt2_windows")

    assert manifest.validate() == []


@pytest.mark.parametrize(
    "stem",
    ("pion_pdf_cg", "pion_pdf_gi", "pion_da_gi", "kaon_da_gi"),
)
def test_refactor_examples_reuse_legacy_physics_parameter_names(stem: str) -> None:
    examples = Path(__file__).parents[2] / "examples"
    legacy = json.loads((examples / f"{stem}_manifest.json").read_text(encoding="utf-8"))
    neo_manifest = load_manifest(examples / f"{stem}_manifest_neo.json")
    neo = neo_manifest.document

    def key_names(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {name for child in value.values() for name in key_names(child)}
        if isinstance(value, list):
            return {name for child in value for name in key_names(child)}
        return set()

    aligned_names = {
        "run_id", "resample_mode", "bin_size", "component", "nstate",
        "fit_scope", "fit_strategy", "fitting_form", "model_average",
        "posterior_prior_error_scale", "normalization",
        "scheme", "strategy", "mu",
        "target_observable", "parton", "quasi_y_ls", "zmin_fm", "zmax_fm",
        "smooth", "zmax_ext_fm", "order", "sector", "Lambda0_gev",
        "lc_x_ls",
    }
    if stem.startswith("pion_pdf"):
        aligned_names |= {"zs_fm", "m0_gev", "delta_m_gev"}
    else:
        aligned_names |= {"LambdaQCD_gev", "d", "phase_transfer_da", "psi1_flavor_class", "psi2_flavor_class"}
    if stem == "pion_pdf_cg":
        aligned_names |= {"rgr_kappa", "rgr_mu_min_gev"}
    legacy_names = key_names(legacy)
    neo_names = key_names(neo)
    assert aligned_names <= legacy_names
    assert aligned_names <= neo_names
    expected_target = "pdf" if "pdf" in stem else "da"
    metadata = neo["metadata"]
    assert metadata["target_observable"] == expected_target
    assert metadata["resample_mode"] == ("jackknife" if expected_target == "pdf" else "bootstrap")
    assert metadata["bin_size"] > 0
    assert "bootstrap_samples" in metadata if expected_target == "da" else "bootstrap_samples" not in metadata
    correlator_defaults = neo["stages"]["correlator_analysis"]["defaults"]
    assert not {"resample_mode", "sample_error_mode", "bootstrap_samples", "bin_size"} & set(correlator_defaults)
    assert "target_observable" not in neo["stages"]["fourier_transform"]["defaults"]
    correlator_lsqfit = neo["stages"]["correlator_analysis"]["defaults"]["lsqfit"]
    assert not {"allowed_methods", "matrix_fit", "qda_fit"} & set(correlator_lsqfit)
    assert neo_manifest.validate() == []


def test_correlator_contract_keeps_lanczos_and_ground_fit_parameters_exclusive() -> None:
    contract = _load_stage_contract("correlator_analysis")
    lanczos = {
        "observable": "matrix_element",
        "analysis_method": "lanczos",
        "resample_group": "toy",
        "component": "both",
        "nstate": [2],
        "correlator_ids": ["c2", "c3"],
        "lanczos": {"scope": "3pt_matrix"},
    }
    assert evaluate_rules(lanczos, contract.PARAM_RULES) == []
    context = CheckContext({}, "correlator_analysis", "job", lanczos, {})
    assert evaluate_checks(contract.CHECKS, context) == []

    mixed = {**lanczos, "lsqfit": {}}
    issues = evaluate_rules(mixed, contract.PARAM_RULES)
    assert [(issue.path, issue.message) for issue in issues] == [
        ("lsqfit", "unknown key 'lsqfit'")
    ]

    ground_fit = {
        **{key: value for key, value in lanczos.items() if key != "lanczos"},
        "observable": "spectrum",
        "analysis_method": "lsqfit",
        "component": "re",
        "lsqfit": {
            "fit_scope": ["spectrum"],
            "fit_strategy": ["independent"],
            "fitting_form": "Breit",
            "prior_width": [1.0],
            "model_average": False,
            "time_range": {"min": 2, "max": 8},
            "pt2_windows": [{"tmin": 2, "tmax": 8}],
            "svdcut": 1e-6,
            "posterior_prior_error_scale": 1.0,
            "q_min": 0.05,
            "tune_z": 0,
        },
    }
    assert evaluate_rules(ground_fit, contract.PARAM_RULES) == []
    assert evaluate_checks(
        contract.CHECKS,
        CheckContext({}, "correlator_analysis", "job", ground_fit, {}),
    ) == []


def test_each_shipped_stage_contract_reports_incomplete_params_instead_of_crashing(tmp_path: Path) -> None:
    for stage_id in ("correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"):
        manifest = {
            "metadata": {"run_id": "incomplete", "root_directory": str(tmp_path), "artifacts_directory": "runs", "random_seed": 1, "workers": 1},
            "stages": {stage_id: {"defaults": {}, "jobs": [{"id": "job", "inputs": {}, "params": {}}]}},
        }
        issues = Manifest(tmp_path / "manifest.json", manifest).validate()
        assert issues


def test_neo_correlator_descriptors_use_physical_field_names() -> None:
    examples = Path(__file__).parents[2] / "examples"
    for path in examples.glob("*correlators_neo.json"):
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        ensemble = descriptor["ensemble"]
        assert "m_pi_gev" in ensemble
        assert "m_pi" not in ensemble
        for record in descriptor["correlators"]:
            assert "correlator_type" in record
            assert "kind" not in record
            current = record.get("current")
            if current is not None:
                assert "construction" not in current
                assert "observable" not in current


def test_matching_check_reports_the_exact_parameter_path() -> None:
    contract = _load_stage_contract("perturbative_matching")
    context = CheckContext({}, "perturbative_matching", "job", {"kernel_id": "CG_gt_quark_PDF_hybrid_NLO", "scheme": "ratio", "zs_fm": 0.2}, {"quasi": {"job": "earlier"}})
    issues = evaluate_checks(contract.CHECKS, context)
    assert [(issue.path, issue.message) for issue in issues] == [("params.scheme", "must equal 'hybrid' for kernel 'CG_gt_quark_PDF_hybrid_NLO'")]


def test_finish_rejects_second_terminal_result(tmp_path: Path) -> None:
    context = ToolContext({}, tmp_path / "manifest.json", "review", "job", {}, {}, {}, {}, tmp_path, np.random.default_rng(1))
    summary = {"stage_id": "review", "job_id": "job", "result": "review", "decisions": {}, "diagnostics": {}, "artifacts": []}
    context.finish("report", summary)
    try:
        context.finish("again", summary)
    except RuntimeError:
        pass
    else:
        raise AssertionError("finish must reject a second terminal result")


def test_manifest_has_exact_two_top_level_keys(tmp_path: Path) -> None:
    descriptor = tmp_path / "input.json"
    descriptor.write_text("{}", encoding="utf-8")
    document = {
        "metadata": _valid_metadata(tmp_path, random_seed=7),
        "stages": {
            "review": {
                "defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]},
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(descriptor)}]}, "params": {}}],
            }
        },
    }
    manifest = Manifest(tmp_path / "manifest.json", document)
    assert not manifest.validate()
    document["extra"] = True
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
                "defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]},
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}],
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
            "second_authored": {"defaults": {}, "jobs": [{"id": "job_b", "inputs": {}, "params": {}}]},
            "first_authored": {"defaults": {}, "jobs": [{"id": "job_a", "inputs": {}, "params": {}}]},
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
                "defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]},
                "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}],
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
        "stages": {"review": {"defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]}, "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}]}},
    }
    responses = [
        _AssistantResponse("inspect", _ToolCall("1", "inspect_results", {})),
        _AssistantResponse("check", _ToolCall("2", "check_consistency", {})),
        _AssistantResponse("list", _ToolCall("3", "list_literature", {})),
        _AssistantResponse("read", _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]})),
        _AssistantResponse("write", _ToolCall("5", "write_review", {"title": "Toy review", "analysis": "The scoped outputs are mutually consistent.", "conclusion": "The toy workflow is internally consistent."})),
    ]
    result = create_session(_ScriptedBackend(responses)).run_manifest(Manifest(tmp_path / "manifest.json", manifest))
    assert result["summaries"]["review_1"]["result"] == "review"
    assert (tmp_path / "runs" / "01_review" / "review_1" / "review.md").is_file()
    transcript = (tmp_path / "runs" / "01_review" / "review_1" / "llm_transcript.md").read_text(
        encoding="utf-8"
    )
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
        "stages": {"review": {"defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]}, "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}]}},
    }
    backend = _ScriptedBackend([
        _AssistantResponse("inspect and check", tool_calls=(_ToolCall("1", "inspect_results", {}), _ToolCall("2", "check_consistency", {}))),
        _AssistantResponse("list and read", tool_calls=(_ToolCall("3", "list_literature", {}), _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]}))),
        _AssistantResponse("write", _ToolCall("5", "write_review", {"title": "Toy review", "analysis": "The scoped outputs are mutually consistent.", "conclusion": "The toy workflow is internally consistent."})),
    ])
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
        "stages": {"review": {"defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]}, "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}]}},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    responses = [
        _AssistantResponse("inspect", _ToolCall("1", "inspect_results", {})),
        _AssistantResponse("check", _ToolCall("2", "check_consistency", {})),
        _AssistantResponse("list", _ToolCall("3", "list_literature", {})),
        _AssistantResponse("read", _ToolCall("4", "read_papers", {"paper_ids": ["refactor_demo_lamet"]})),
        _AssistantResponse("write", _ToolCall("5", "write_review", {"title": "Toy review", "analysis": "The scoped outputs are mutually consistent.", "conclusion": "The toy workflow is internally consistent."})),
    ]
    result = create_session(_ScriptedBackend(responses)).run_manifest(load_manifest(manifest_path))
    assert result["summaries"]["review_1"]["result"] == "review"


def test_agent_fails_immediately_when_the_model_returns_no_tool_call(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "metadata": _valid_metadata(tmp_path, random_seed=2),
        "stages": {"review": {"defaults": {"catalog": "builtin", "max_papers": 1, "report_language": "English", "checks": ["identity"]}, "jobs": [{"id": "review_1", "inputs": {"results": [{"file": str(source)}]}, "params": {}}]}},
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
    context = ToolContext({}, tmp_path / "manifest.json", "review", "job", {}, {}, {}, {}, tmp_path, np.random.default_rng(1))
    summary = {"stage_id": "review", "job_id": "job", "result": "review", "decisions": {}, "diagnostics": {}, "artifacts": ["missing.md"]}
    with pytest.raises(FileNotFoundError, match="missing.md"):
        context.finish("report", summary)
