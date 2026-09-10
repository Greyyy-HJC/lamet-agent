"""Provider wire schemas preserve application constraints and patch semantics."""

import copy
import json

import pytest
from jsonschema import ValidationError

from lamet_agent.agent import _discover_tools
from lamet_agent.llm import _prepare_cli_schema
from lamet_agent.plan.tools import planning_tool_schemas


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_plan_patch_json_and_optional_fields_round_trip(provider):
    original = planning_tool_schemas()
    saved = copy.deepcopy(original)
    contract = _prepare_cli_schema(original, None, provider=provider)
    value = {"stages": [1, None, {"key": "value"}]}
    patch_value = {"value": json.dumps(value)} if provider == "codex" else json.dumps(value)
    null_value = {"value": "null"} if provider == "codex" else "null"
    removed = {"op": "remove", "path": "obsolete"}
    if provider == "codex":
        removed["value"] = None
    wire = {
        "text": "",
        "tool_calls": [
            {
                "name": "apply_manifest_patch",
                "arguments": {
                    "patches": [
                        {"op": "add", "path": "new", "value": patch_value},
                        {"op": "replace", "path": "nullable", "value": null_value},
                        removed,
                    ]
                },
            }
        ],
    }
    patches = contract.decode(wire)["tool_calls"][0]["arguments"]["patches"]
    assert patches == [
        {"op": "add", "path": "new", "value": value},
        {"op": "replace", "path": "nullable", "value": None},
        {"op": "remove", "path": "obsolete"},
    ]
    assert original == saved


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_plan_optional_defaults_and_local_range_validation(provider):
    contract = _prepare_cli_schema(planning_tool_schemas(), None, provider=provider)
    arguments = {"query": "input"}
    if provider == "codex":
        arguments["max_results"] = None
    wire = {"text": "", "tool_calls": [{"name": "find_paths", "arguments": arguments}]}
    assert contract.decode(wire)["tool_calls"][0]["arguments"] == {"query": "input"}
    arguments["max_results"] = {"value": 101} if provider == "codex" else 101
    with pytest.raises((ValueError, ValidationError)):
        contract.decode(wire)


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_ask_retains_constraints_and_explicit_optional_null(provider):
    original = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "uniqueItems": True,
                "items": {"type": "number", "minimum": 0},
            },
            "note": {"anyOf": [{"type": "string", "minLength": 2}, {"type": "null"}]},
        },
        "required": ["values"],
        "additionalProperties": False,
    }
    contract = _prepare_cli_schema([], {"schema": original}, provider=provider)
    wire = {"values": [1, 2], "note": {"value": None} if provider == "codex" else None}
    assert contract.decode(wire) == {"values": [1, 2], "note": None}
    for values in ([1], [1, 1], [-1, 2], [1, 2, 3, 4]):
        with pytest.raises(ValidationError):
            contract.decode({**wire, "values": values})


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_wire_subset_for_real_tool_schemas(provider):
    def check(schema):
        assert "uniqueItems" not in schema
        if provider == "claude":
            assert not ({"maxItems", "minimum", "maximum", "minLength", "maxLength"} & schema.keys())
            assert schema.get("minItems", 0) in (0, 1)
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            if provider == "codex":
                assert set(schema["required"]) == set(schema["properties"])
        for child in schema.get("properties", {}).values():
            check(child)
        for child in schema.get("anyOf", []):
            check(child)
        if "items" in schema:
            check(schema["items"])

    for tools in (planning_tool_schemas(), [t.schema for t in _discover_tools("review")], []):
        contract = _prepare_cli_schema(tools, None, provider=provider)
        check(contract.schema)
        assert contract.decode({"text": "done", "tool_calls": [] if tools else None}) == {
            "text": "done",
            "tool_calls": [],
        }


def test_open_dictionary_decodes_and_validates_values():
    contract = _prepare_cli_schema(
        [], {"schema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                }
            },
            "required": ["values"],
            "additionalProperties": False,
        }},
        provider="codex",
    )
    assert contract.decode({"values": '{"a": 1}'}) == {"values": {"a": 1}}
    with pytest.raises(ValidationError):
        contract.decode({"values": '{"a": "wrong"}'})
    with pytest.raises(ValueError):
        contract.decode({"values": '```json\n{"a": 1}\n```'})
