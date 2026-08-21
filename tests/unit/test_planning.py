from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from lamet_agent.__main__ import app
from lamet_agent.core.tools import validate_stage_diagnostics
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.planning import (
    PlanAgentState,
    _PlanAgentSession,
    _ask_plan_agent_question,
    _apply_user_answer_to_candidate,
    _expand_pt2_windows,
    _expand_pt3_windows,
    _get_path_value,
    _run_planning_tool,
    _stage_parameter_gaps,
    _next_questions_for_state,
    _manifest_question_id_from_user_input_action,
    _planning_system_prompt,
    apply_manifest_json_patches,
    build_repaired_manifests,
    check_manifest_draft,
    convert_correlator_h5,
    inspect_correlator_h5_files,
    load_relaxed_manifest,
    plan_correlator_h5_conversions,
    run_interactive_plan,
    validate_candidate_payload,
)
from lamet_agent.planning.core import normalize_planning_constraints
from lamet_agent.stages.correlator.functions import _read_2pt, _read_3pt


class _PlanningApiStub:
    """Test-only OpenAI-compatible response stub for the planning loop."""

    def __init__(self) -> None:
        self._state: dict[int, dict[str, object]] = {}

    @staticmethod
    def _latest_observation(messages: list[dict[str, str]]) -> dict:
        try:
            payload = json.loads(messages[-1]["content"])
        except (IndexError, KeyError, json.JSONDecodeError):
            return {}
        observation = payload.get("observation", {})
        return observation if isinstance(observation, dict) else {}

    @staticmethod
    def _latest_user_answer(messages: list[dict[str, str]]) -> object:
        for message in reversed(messages):
            try:
                observation = json.loads(message["content"]).get("observation", {})
            except (KeyError, json.JSONDecodeError):
                continue
            if observation.get("event") == "user_answer":
                return observation.get("value")
        return 1984

    @staticmethod
    def _revision_patches(payload: dict, original: dict, note: str) -> list[dict]:
        text = note.lower()
        if "renormalization" in text:
            stages = payload.get("stages", {})
            metadata = payload.get("metadata", {})
            order = list(metadata.get("stages", []))
            jobs = stages.get("correlator_analysis", {}).get("jobs", [])
            denominator = next(
                (job["id"] for job in jobs if "p0" in str(job.get("id", ""))),
                jobs[0]["id"] if jobs else "ca",
            )
            targets = [
                job["id"]
                for job in jobs
                if job.get("id") != denominator and "p" in str(job.get("id", ""))
            ]
            renorm_jobs = [
                {
                    "id": target.replace("ca_", "rn_", 1)
                    if target.startswith("ca_")
                    else f"rn_{target}",
                    "inputs": {"target": target, "denominator": denominator},
                }
                for target in targets
            ]
            if "renormalization" not in order:
                index = (
                    order.index("correlator_analysis") + 1
                    if "correlator_analysis" in order
                    else len(order)
                )
                order.insert(index, "renormalization")
            return [
                {"op": "replace", "path": "/metadata/stages", "value": order},
                {
                    "op": "add",
                    "path": "/stages/renormalization",
                    "value": {
                        "defaults": {
                            "normalization": False,
                            "scheme": "hybrid",
                            "strategy": "external_denominator",
                            "zs_fm": 0.18,
                            "m0_gev": 0.0,
                            "delta_m_gev": 0.0,
                        },
                        "jobs": renorm_jobs,
                    },
                },
            ]
        if ("fit window" in text or "window" in text) and (
            "search" in text or "scan" in text
        ):
            defaults = payload.get("stages", {}).get("correlator_analysis", {}).get(
                "defaults", {}
            )
            return [
                {
                    "op": "replace",
                    "path": "/stages/correlator_analysis/defaults/pt2_windows",
                    "value": _expand_pt2_windows(defaults.get("pt2_windows")),
                    "note": "LLM expanded the fit-window search.",
                },
                {
                    "op": "replace",
                    "path": "/stages/correlator_analysis/defaults/pt3_windows",
                    "value": _expand_pt3_windows(defaults.get("pt3_windows")),
                    "note": "LLM expanded the fit-window search.",
                },
            ]
        if ("tau" in text or "pt3_windows" in text or "pt3_tau_cuts" in text) and (
            "undo" in text or "revert" in text
        ):
            return [
                {
                    "op": "replace",
                    "path": "/stages/correlator_analysis/defaults/pt3_windows",
                    "value": _get_path_value(
                        original, "stages.correlator_analysis.defaults.pt3_windows"
                    ),
                    "note": "LLM reverted the tau-cut search.",
                }
            ]
        return []

    def __call__(self, *, messages: list[dict[str, str]], **_kwargs: object) -> str:
        key = id(messages)
        state = self._state.setdefault(
            key,
            {
                "phase": "load",
                "seen": 0,
                "payload": {},
                "original": {},
                "revision": "",
            },
        )
        if len(messages) != state["seen"]:
            observation = self._latest_observation(messages)
            if observation.get("tool_name") == "load_manifest":
                manifest = copy.deepcopy(observation.get("manifest", {}))
                state["payload"] = manifest
                state["original"] = copy.deepcopy(manifest)
            candidate = observation.get("candidate_manifest")
            if isinstance(candidate, dict):
                state["payload"] = copy.deepcopy(candidate)
            event = observation.get("event")
            error = str(observation.get("error", ""))
            if event == "user_revision":
                state["revision"] = str(observation.get("text", ""))
                state["phase"] = "revision"
            elif event == "user_answer":
                question_id = str(observation.get("question_id", ""))
                if question_id.startswith("stage_params."):
                    value = str(observation.get("value", "")).strip().lower()
                    state["phase"] = "blocked" if value in {"no", "n", "false", "0"} else "build"
                elif question_id in {"stage.add_remaining", "metadata.required"} or question_id.startswith(
                    ("stage_required.", "stage_optional.")
                ):
                    state["phase"] = "conversions"
                else:
                    state["phase"] = "answer"
            elif event == "question_skipped":
                state["phase"] = "conversions"
            elif "not the full canonical stage flow" in error:
                state["phase"] = "stage_completion"
            elif "still have missing parameters or input roles" in error:
                state["phase"] = "blocked"
            elif "missing parameters or input roles" in error:
                state["phase"] = "parameter_completion"
            state["seen"] = len(messages)

        phase = str(state["phase"])
        if phase == "load":
            state["phase"] = "check"
            action = {"action": "call_tool", "tool_name": "load_manifest", "args": {}, "reason": "Inspect draft."}
        elif phase == "check":
            state["phase"] = "maybe_seed"
            action = {"action": "call_tool", "tool_name": "check_manifest_draft", "args": {}, "reason": "Check draft."}
        elif phase == "maybe_seed":
            state["phase"] = "conversions"
            action = {
                "action": "request_user_input",
                "reason": "metadata required values are missing.",
                "args": {
                    "question_id": "metadata.required",
                    "prompt": "metadata required choices: random_seed, resample_mode, and sample_error_mode.",
                },
            }
        elif phase == "answer":
            state["phase"] = "conversions"
            action = {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {
                    "patches": [
                        {
                            "op": "add",
                            "path": "/metadata/random_seed",
                            "value": int(self._latest_user_answer(messages)),
                        }
                    ]
                },
                "reason": "Apply user answer.",
            }
        elif phase == "conversions":
            state["phase"] = "inspect"
            action = {"action": "call_tool", "tool_name": "plan_correlator_h5_conversions", "args": {}, "reason": "Plan conversions."}
        elif phase == "inspect":
            state["phase"] = "build"
            action = {"action": "call_tool", "tool_name": "inspect_correlator_h5_files", "args": {}, "reason": "Inspect inputs."}
        elif phase == "build":
            state["phase"] = "propose"
            action = {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates."}
        elif phase == "stage_completion":
            action = {
                "action": "request_user_input",
                "reason": "The manifest is not the full canonical stage flow.",
                "args": {
                    "question_id": "stage.add_remaining",
                    "prompt": "Add extra downstream stages?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Add stages."},
                        {"label": "2", "value": "no", "description": "Keep partial."},
                    ],
                },
            }
        elif phase == "parameter_completion":
            action = {
                "action": "request_user_input",
                "reason": "A configured stage is missing required parameters.",
                "args": {
                    "question_id": "stage_params.missing",
                    "prompt": "Add missing parameters?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Add them."},
                        {"label": "2", "value": "no", "description": "Leave unchanged."},
                    ],
                },
            }
        elif phase == "revision":
            note = str(state["revision"])
            state["phase"] = "build"
            action = {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {
                    "patches": self._revision_patches(
                        state["payload"], state["original"], note
                    ),
                    "suppress_full_expansions": [
                        "stages.correlator_analysis.defaults.pt3_windows"
                    ]
                    if "revert" in note.lower() or "undo" in note.lower()
                    else [],
                },
                "reason": "Apply revision.",
            }
        elif phase == "blocked":
            state["phase"] = "done"
            action = {"action": "finish", "reason": "Missing parameters.", "args": {"error": True}}
        else:
            state["phase"] = "done"
            action = {"action": "propose_plan", "reason": "Present candidate.", "args": {"summary": "Test planning summary."}}
        return json.dumps(action)


@pytest.fixture
def planning_api_stub(monkeypatch: pytest.MonkeyPatch) -> _PlanningApiStub:
    stub = _PlanningApiStub()
    monkeypatch.setattr("lamet_agent.planning.request_llm_text", stub)
    return stub


def test_planning_prompt_requires_validation_contract_maintenance() -> None:
    prompt = _planning_system_prompt()

    assert "STAGE_PARAM_CONTRACT in validation.py" in prompt
    assert "adds, removes, renames, or changes a manifest parameter" in prompt


def test_plan_gfix_gap_matches_external_fourier_provenance() -> None:
    full = json.loads(Path("examples/pion_pdf_cg_manifest.json").read_text(encoding="utf-8"))
    assert not any(gap["parameter"] == "gfix" for gap in _stage_parameter_gaps(full))

    payload = {
        "metadata": {
            "run_id": "partial",
            "root_directory": ".",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform"],
        },
        "inputs": {
            "correlators": [],
            "artifacts": [{
                "id": "rn",
                "stage": "renormalization",
                "path": "rn.nc",
                "momentum": "PX5PY0PZ0",
                "volume": "S48T64",
                "lattice_spacing_fm": 0.0574,
                "hadron": "pion",
                "gfix": "CG",
                "polarization": "unpolarized",
            }],
            "kernels": [],
        },
        "stages": {
            "fourier_transform": {
                "defaults": {
                    "Lambda0_gev": 0.0,
                    "order": "LA",
                    "posterior_prior_error_scale": 3.0,
                    "quasi_y_ls": {"start": -1.0, "stop": 1.0, "num": 4},
                    "scheme_scan": {"zmin_fm": [0.1], "zmax_fm": [0.8], "zmax_ext_fm": 1.2},
                    "sector": "valence",
                },
                "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
            }
        },
    }

    gaps = _stage_parameter_gaps(payload)
    assert [gap["code"] for gap in gaps if gap["parameter"] == "gfix"] == ["fourier.gfix.required"]

    payload["stages"]["fourier_transform"]["defaults"]["gfix"] = "GI"
    gaps = _stage_parameter_gaps(payload)
    assert [gap["code"] for gap in gaps if gap["parameter"] == "gfix"] == ["fourier.gfix.provenance"]

    payload["stages"]["fourier_transform"]["defaults"]["gfix"] = "CG"
    assert not any(gap["parameter"] == "gfix" for gap in _stage_parameter_gaps(payload))


def _write_kernel(root: Path) -> None:
    (root / "lamet_agent").mkdir(parents=True)
    (root / "lamet_agent" / "kernels.py").write_text("# test kernel\n", encoding="utf-8")


def _required_correlator_defaults(scope: str = "3pt_ratio") -> dict[str, object]:
    return {
        "component": "both",
        "fit_scope": [scope],
        "fit_strategy": ["joint"],
        "fitting_form": "Breit",
        "model_average": False,
        "nstate": [2],
        "posterior_prior_error_scale": 3.0,
        "q_min": 0.05,
    }


def _minimal_payload(root: Path, data_path: str = "data/c2.h5") -> dict:
    return {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis", "renormalization"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": data_path,
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T3",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                }
            ],
            "artifacts": [],
            "kernels": [
                {
                    "stage": "matching",
                    "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
                    "kernel_path": "lamet_agent/kernels.py",
                    "kernel_parameters": {},
                }
            ],
        },
        "stages": {
            "correlator_analysis": {"defaults": _required_correlator_defaults(), "jobs": [{"id": "ca", "correlator_ids": ["c2"], "params": {"momentum": "PX0PY0PZ0"}}]},
            "renormalization": {
                "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "normalization": False, "zs_fm": 0.2, "m0_gev": 0.0, "delta_m_gev": 0.0},
                "jobs": [{"id": "rn", "inputs": {"target": "ca", "denominator": "ca"}}],
            },
        },
    }


def test_load_relaxed_manifest_accepts_jsonc(tmp_path: Path) -> None:
    path = tmp_path / "draft.jsonc"
    path.write_text(
        """
        {
          // comments are accepted in plan mode
          "metadata": {"run_id": "demo",},
          "inputs": {},
          "stages": {}
        }
        """,
        encoding="utf-8",
    )

    payload, raw = load_relaxed_manifest(path)

    assert payload["metadata"]["run_id"] == "demo"
    assert "// comments" in raw


def test_load_relaxed_manifest_preserves_url_like_strings(tmp_path: Path) -> None:
    path = tmp_path / "draft.jsonc"
    path.write_text(
        """
        {
          "metadata": {
            "run_id": "demo",
            "root_directory": "https://example.invalid/project",
          },
          "inputs": {},
          "stages": {}
        }
        """,
        encoding="utf-8",
    )

    payload, _raw = load_relaxed_manifest(path)

    assert payload["metadata"]["root_directory"] == "https://example.invalid/project"


def test_check_manifest_draft_reports_scheme_mismatch_and_missing_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_kernel(root)
    payload = _minimal_payload(root)
    payload["stages"]["perturbative_matching"] = {"defaults": {"scheme": "ratio"}, "jobs": []}
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    issues = check_manifest_draft(path, payload)

    messages = [issue.message for issue in issues]
    assert any("Correlator data file does not exist" in message for message in messages)
    assert any("differs from renormalization scheme" in message for message in messages)


def test_plan_reports_stage_parameter_gaps_before_building(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform"],
        },
        "inputs": {"correlators": [], "artifacts": [{"id": "rn", "path": "rn.nc", "stage": "renormalization"}], "kernels": []},
        "stages": {"fourier_transform": {"defaults": {}, "jobs": [{"id": "ft", "inputs": {"input": "rn"}}]}},
    }
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    state = PlanAgentState(path, "", payload, payload)
    state.stage_completion_checked = True
    state.stage_required_checked.add("fourier_transform")
    state.stage_optional_checked.add("fourier_transform")

    listed = _run_planning_tool(state, "list_stage_parameter_gaps", {})
    gaps = listed["stage_parameter_gaps"]
    assert any(gap["parameter"] == "order" for gap in gaps)
    assert not any(gap["parameter"] == "coord_unit" for gap in gaps)
    assert any(gap["parameter"] == "quasi_y_ls" for gap in gaps)
    assert any(gap["parameter"] == "momentum_gev" for gap in gaps)
    quasi_gap = next(gap for gap in gaps if gap["parameter"] == "quasi_y_ls")
    assert "momentum-fraction" in quasi_gap["physics"]

    blocked = _run_planning_tool(state, "build_quick_full_candidates", {})
    assert blocked["ok"] is False
    assert "missing parameters" in blocked["error"]
    assert blocked["next_questions"][0]["question_id"] == "stage_params.fourier_transform.shared.Lambda0_gev"
    assert "Physical reason:" in blocked["next_questions"][0]["prompt"]


def test_complete_example_builds_without_planning_questions() -> None:
    manifest_path = Path("examples/pion_pdf_cg_manifest.json")
    payload, _ = load_relaxed_manifest(manifest_path)
    state = PlanAgentState(manifest_path, "", payload, copy.deepcopy(payload))

    gaps = _stage_parameter_gaps(payload, manifest_path)
    loaded = _run_planning_tool(state, "load_manifest", {})
    built = _run_planning_tool(state, "build_quick_full_candidates", {})

    assert gaps == []
    assert loaded["next_questions"] == []
    assert built["ok"] is True


def test_plan_nonbreit_gpd_propagates_exchanged_flow_through_upstream_jobs() -> None:
    correlators = [
        {
            "correlator_id": "c2_p2",
            "correlator_type": "2pt",
            "momentum": "PX0PY0PZ2",
            "volume": "S24T72",
            "lattice_spacing_fm": 0.1,
            "hadron": "pion",
            "gfix": "GI",
        },
        {
            "correlator_id": "c2_p3",
            "correlator_type": "2pt",
            "momentum": "PX0PY0PZ3",
            "volume": "S24T72",
            "lattice_spacing_fm": 0.1,
            "hadron": "pion",
            "gfix": "GI",
        },
        {
            "correlator_id": "c3_2to3",
            "correlator_type": "3pt",
            "momentum": "PX0PY0PZ3",
            "hadron": "pion",
            "current_operator": "gt",
            "polarization": "unpolarized",
        },
        {
            "correlator_id": "c3_3to2",
            "correlator_type": "3pt",
            "momentum": "PX0PY0PZ2",
            "hadron": "pion",
            "current_operator": "gt",
            "polarization": "unpolarized",
        },
    ]
    payload = {
        "metadata": {
            "run_id": "paired-plan",
            "root_directory": ".",
            "target_observable": "gpd",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis", "renormalization", "fourier_transform"],
        },
        "inputs": {"correlators": correlators, "artifacts": [], "kernels": []},
        "stages": {
            "correlator_analysis": {
                "defaults": {},
                "jobs": [
                    {
                        "id": "ca_2to3",
                        "correlator_ids": ["c2_p2", "c2_p3", "c3_2to3"],
                        "params": {
                            "fitting_form": "NonBreit",
                            "initial_momentum": "PX0PY0PZ2",
                            "final_momentum": "PX0PY0PZ3",
                        },
                    },
                    {
                        "id": "ca_3to2",
                        "correlator_ids": ["c2_p2", "c2_p3", "c3_3to2"],
                        "params": {
                            "fitting_form": "NonBreit",
                            "initial_momentum": "PX0PY0PZ3",
                            "final_momentum": "PX0PY0PZ2",
                        },
                    },
                ],
            },
            "renormalization": {
                "defaults": {},
                "jobs": [
                    {"id": "rn_2to3", "inputs": {"target": "ca_2to3"}},
                    {"id": "rn_3to2", "inputs": {"target": "ca_3to2"}},
                ],
            },
            "fourier_transform": {
                "defaults": {},
                "jobs": [
                    {
                        "id": "ft_2to3",
                        "inputs": {"input": "rn_2to3", "hermitian_partner": "rn_3to2"},
                    }
                ],
            },
        },
    }

    gaps = _stage_parameter_gaps(payload)
    assert not any(gap["code"] == "fourier.inputs.observable_contract" for gap in gaps)

    payload["stages"]["correlator_analysis"]["jobs"][1]["params"]["initial_momentum"] = "PX0PY0PZ1"
    gaps = _stage_parameter_gaps(payload)
    assert any(
        gap["code"] == "fourier.inputs.observable_contract"
        and "exchange the initial and final momenta" in gap["message"]
        for gap in gaps
    )


def test_plan_partial_gpd_artifact_checks_partner_kinematics() -> None:
    target = {
        "id": "rn_2to3",
        "stage": "renormalization",
        "path": "rn_2to3.nc",
        "momentum": "PX0PY0PZ2",
        "initial_momentum": "PX0PY0PZ2",
        "final_momentum": "PX0PY0PZ3",
        "volume": "S24T72",
        "lattice_spacing_fm": 0.1,
        "hadron": "pion",
        "gfix": "GI",
        "polarization": "unpolarized",
    }
    partner = {
        **target,
        "id": "rn_3to2",
        "path": "rn_3to2.nc",
        "initial_momentum": "PX0PY0PZ3",
        "final_momentum": "PX0PY0PZ2",
    }
    payload = {
        "metadata": {
            "run_id": "partial-paired-plan",
            "root_directory": ".",
            "target_observable": "gpd",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform"],
        },
        "inputs": {"correlators": [], "artifacts": [target, partner], "kernels": []},
        "stages": {
            "fourier_transform": {
                "defaults": {},
                "jobs": [
                    {
                        "id": "ft_2to3",
                        "inputs": {"input": "rn_2to3", "hermitian_partner": "rn_3to2"},
                    }
                ],
            }
        },
    }

    assert not any(
        gap["code"] == "fourier.inputs.observable_contract"
        for gap in _stage_parameter_gaps(payload)
    )
    payload["inputs"]["artifacts"][1].pop("initial_momentum")
    assert any(
        gap["code"] == "fourier.inputs.observable_contract"
        for gap in _stage_parameter_gaps(payload)
    )


def test_plan_input_answer_preserves_single_job_input_and_never_broadcasts_partner() -> None:
    payload = {
        "metadata": {"target_observable": "gpd", "stages": ["fourier_transform"]},
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {
            "fourier_transform": {
                "defaults": {},
                "jobs": [{"id": "ft_2to3", "inputs": {"input": "rn_2to3"}}],
            }
        },
    }
    state = PlanAgentState(Path("draft.json"), "", payload, copy.deepcopy(payload))
    _apply_user_answer_to_candidate(
        state,
        "stage_required.fourier_transform",
        json.dumps({"hermitian_partner": "rn_3to2"}),
    )
    assert state.candidate_payload["stages"]["fourier_transform"]["jobs"][0]["inputs"] == {
        "input": "rn_2to3",
        "hermitian_partner": "rn_3to2",
    }

    payload["stages"]["fourier_transform"]["jobs"].append(
        {"id": "ft_3to2", "inputs": {"input": "rn_3to2"}}
    )
    state = PlanAgentState(Path("draft.json"), "", payload, copy.deepcopy(payload))
    _apply_user_answer_to_candidate(
        state,
        "stage_required.fourier_transform",
        json.dumps({"hermitian_partner": "rn_3to2"}),
    )
    assert state.candidate_payload["stages"]["fourier_transform"]["jobs"][0]["inputs"] == {
        "input": "rn_2to3"
    }
    assert state.candidate_payload["stages"]["fourier_transform"]["jobs"][1]["inputs"] == {
        "input": "rn_3to2"
    }


@pytest.mark.parametrize("target", ["pdf", "da"])
def test_plan_normalization_removes_gpd_only_fields_from_other_observables(target: str) -> None:
    payload = {
        "metadata": {"target_observable": target},
        "stages": {
            "fourier_transform": {
                "defaults": {"bilocal_anchor": "mid_at_0"},
                "jobs": [
                    {
                        "id": "ft",
                        "inputs": {"input": "rn", "hermitian_partner": "rn_reverse"},
                        "params": {"bilocal_anchor": "barpsi_at_0"},
                    }
                ],
            }
        },
    }

    normalize_planning_constraints(payload)

    stage = payload["stages"]["fourier_transform"]
    assert "bilocal_anchor" not in stage["defaults"]
    assert "bilocal_anchor" not in stage["jobs"][0]["params"]
    assert "hermitian_partner" not in stage["jobs"][0]["inputs"]


def test_fourier_plan_and_validate_report_the_same_quasi_y_ls_rule(tmp_path: Path) -> None:
    payload = {
        "metadata": {
            "run_id": "same-rule",
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
                    "id": "rn",
                    "path": "rn.nc",
                    "stage": "renormalization",
                    "momentum": "PX1PY0PZ0",
                    "volume": "S16T32",
                    "lattice_spacing_fm": 0.1,
                    "hadron": "pion",
                    "polarization": "unpolarized",
                }
            ],
            "kernels": [],
        },
        "stages": {
            "fourier_transform": {
                "defaults": {},
                "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
            }
        },
    }
    manifest = AnalysisManifest.model_validate(payload)
    job = manifest.stages["fourier_transform"].jobs[0]

    diagnostics = validate_stage_diagnostics("fourier_transform", manifest, job)
    gaps = _stage_parameter_gaps(payload, tmp_path / "draft.json")

    diagnostic = next(item for item in diagnostics if item.code == "fourier.quasi_y_ls.required")
    gap = next(item for item in gaps if item["code"] == "fourier.quasi_y_ls.required")
    assert gap["message"] == diagnostic.message
    assert gap["physics"] == diagnostic.physics
    assert gap["suggested_fix"] == diagnostic.suggested_fix


def test_all_stage_planning_gaps_use_the_validation_contract(tmp_path: Path) -> None:
    cases: list[tuple[str, dict]] = []

    correlator = _minimal_payload(tmp_path)
    correlator["metadata"]["stages"] = ["correlator_analysis"]
    correlator["inputs"]["kernels"] = []
    correlator["stages"] = {"correlator_analysis": correlator["stages"]["correlator_analysis"]}
    cases.append(("correlator_analysis", correlator))

    renorm = _minimal_payload(tmp_path)
    renorm["metadata"]["stages"] = ["renormalization"]
    renorm["inputs"]["correlators"] = []
    renorm["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    renorm["inputs"]["kernels"] = []
    renorm["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid", "strategy": "external_denominator"},
            "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
        }
    }
    cases.append(("renormalization", renorm))

    matching = _minimal_payload(tmp_path)
    matching["metadata"]["stages"] = ["perturbative_matching"]
    matching["inputs"]["correlators"] = []
    matching["inputs"]["artifacts"] = [{
        "id": "quasi", "stage": "fourier_transform", "path": "quasi.nc",
        "momentum": "PX2PY0PZ0", "volume": "S16T32", "lattice_spacing_fm": 0.1,
    }]
    matching["inputs"]["kernels"] = [{
        "stage": "perturbative_matching",
        "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
        "kernel_path": "lamet_agent/kernels.py",
    }]
    matching["stages"] = {
        "perturbative_matching": {
            "defaults": {"scheme": "hybrid"},
            "jobs": [{"id": "mt", "inputs": {"quasi": "quasi"}}],
        }
    }
    cases.append(("perturbative_matching", matching))

    extrapolation = _minimal_payload(tmp_path)
    extrapolation["metadata"]["stages"] = ["extrapolation"]
    extrapolation["inputs"]["correlators"] = []
    extrapolation["inputs"]["artifacts"] = []
    extrapolation["inputs"]["kernels"] = []
    extrapolation["stages"] = {
        "extrapolation": {
            "defaults": {},
            "jobs": [{"id": "ex", "inputs": {"lightcone": []}}],
        }
    }
    cases.append(("extrapolation", extrapolation))

    review = _minimal_payload(tmp_path)
    review["metadata"]["stages"] = ["review"]
    review["inputs"]["correlators"] = []
    review["inputs"]["artifacts"] = []
    review["inputs"]["kernels"] = []
    review["stages"] = {
        "review": {"defaults": {"literature_max_papers": 1}, "jobs": [{"id": "review"}]}
    }
    cases.append(("review", review))

    for stage, payload in cases:
        manifest = AnalysisManifest.model_validate(payload)
        if stage == "review":
            payload["stages"][stage]["defaults"]["literature_max_papers"] = 0
            manifest.stages[stage].defaults["literature_max_papers"] = 0
        diagnostics = validate_stage_diagnostics(stage, manifest, manifest.stages[stage].jobs[0])
        gaps = _stage_parameter_gaps(payload, tmp_path / "draft.json")
        assert {item.code for item in diagnostics} == {
            item["code"] for item in gaps if item["stage"] == stage
        }


def test_planning_reports_legacy_kernel_zs_and_flat_parameter_gaps(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_kernel(root)
    payload = _minimal_payload(root)
    payload["inputs"]["kernels"][0]["kernel_parameters"] = {"zs_fm": 0.2}

    issues = check_manifest_draft(tmp_path / "draft.json", payload)
    issue_paths = {issue.manifest_path for issue in issues}
    assert "inputs.kernels[0].kernel_parameters.zs_fm" in issue_paths

    payload["inputs"]["kernels"][0]["kernel_parameters"] = {}
    payload["stages"]["renormalization"]["defaults"].pop("zs_fm")
    gaps = _stage_parameter_gaps(payload)
    assert any(gap["path"] == "stages.renormalization.defaults.zs_fm" for gap in gaps)


def test_planning_accepts_ratio_without_hybrid_parameters(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"]["defaults"] = {"scheme": "ratio", "strategy": "external_denominator", "normalization": False}

    gaps = _stage_parameter_gaps(payload)

    assert not any(gap["stage"] == "renormalization" for gap in gaps)


def test_planning_distinguishes_self_renormalization_fit_jobs(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["inputs"]["kernels"] = [{
        "stage": "renormalization",
        "kernel_id": "ZMSbar_pdf",
        "kernel_path": "lamet_agent/kernels.py",
        "kernel_parameters": {},
    }]
    payload["stages"]["renormalization"] = {
        "defaults": {"scheme": "ratio", "strategy": "self_renormalization", "normalization": False, "mu": 2.0},
        "jobs": [
            {
                "id": "rn_fit",
                "inputs": {"reference": "ca"},
                "params": {"LambdaQCD_gev": 0.1, "d": -0.08183},
            }
        ],
    }

    gaps = _stage_parameter_gaps(payload)

    assert not any(gap["stage"] == "renormalization" for gap in gaps)


def test_plan_load_manifest_reports_combined_metadata_question(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"].pop("random_seed", None)
    payload["metadata"].pop("resample_mode", None)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    loaded = _run_planning_tool(state, "load_manifest", {})

    assert loaded["next_questions"][0]["question_id"] == "metadata.required"


def test_plan_reports_correlator_metadata_question_before_ambiguous_paths(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["inputs"]["correlators"][0].pop("momentum", None)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    loaded = _run_planning_tool(state, "load_manifest", {})

    assert loaded["next_questions"][0]["question_id"] == "inputs.correlators.0.momentum"


def test_run_fallback_plan_repairs_invalid_paths_in_order(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    payload = _minimal_payload(tmp_path / "wrong-root")
    payload["inputs"]["artifacts"] = [
        {"id": "external", "stage": "renormalization", "path": "missing-artifact.bin"}
    ]
    payload["inputs"]["kernels"][0]["kernel_path"] = "missing-kernel.py"
    state = PlanAgentState(
        tmp_path / "draft.json",
        "",
        copy.deepcopy(payload),
        copy.deepcopy(payload),
        path_repair_project_root=project_root,
    )

    question = _next_questions_for_state(state)[0]
    assert question["question_id"] == "metadata.root_directory"
    assert question["choices"][0]["value"] == str(project_root.resolve())

    applied = _apply_user_answer_to_candidate(
        state,
        "metadata.root_directory",
        str(project_root.resolve()),
    )
    assert applied["event"] == "user_answer_applied"
    assert state.candidate_payload["metadata"]["root_directory"] == str(project_root.resolve())
    assert _next_questions_for_state(state)[0]["question_id"] == "inputs.correlators.0.data_path"

    (project_root / "correct-data.h5").write_bytes(b"data")
    _apply_user_answer_to_candidate(
        state,
        "inputs.correlators.0.data_path",
        "correct-data.h5",
    )
    assert _next_questions_for_state(state)[0]["question_id"] == "inputs.artifacts.0.path"

    (project_root / "correct-artifact.bin").write_bytes(b"artifact")
    _apply_user_answer_to_candidate(
        state,
        "inputs.artifacts.0.path",
        "correct-artifact.bin",
    )
    assert _next_questions_for_state(state)[0]["question_id"] == "inputs.kernels.0.kernel_path"

    (project_root / "correct-kernel.py").write_text("# kernel\n", encoding="utf-8")
    _apply_user_answer_to_candidate(
        state,
        "inputs.kernels.0.kernel_path",
        "correct-kernel.py",
    )

    assert _next_questions_for_state(state)[0]["question_id"] == "stage_required.correlator_analysis"
    assert not (project_root / "artifacts").exists()


def test_plan_does_not_write_conversion_control_answers_to_manifest(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    applied = _apply_user_answer_to_candidate(state, "inputs.correlators.0.axis_mapping", "yes")

    assert applied["event"] == "user_answer_not_applied"
    assert "axis_mapping" not in state.candidate_payload["inputs"]["correlators"][0]


def test_plan_normalizes_legacy_matching_kernel_stage(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["perturbative_matching"]
    payload["inputs"]["correlators"] = []
    payload["inputs"]["artifacts"] = [
        {
            "id": "ft",
            "stage": "fourier_transform",
            "path": "ft.nc",
            "momentum": "PX2PY0PZ0",
            "volume": "S16T3",
            "lattice_spacing_fm": 0.1,
        }
    ]
    payload["inputs"]["kernels"][0]["stage"] = "matching"
    payload["stages"] = {
        "perturbative_matching": {
            "defaults": {"mu": 2.0},
            "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}}],
        }
    }

    gaps = _stage_parameter_gaps(payload)
    quick, full, edits = build_repaired_manifests(tmp_path / "draft.json", payload, [])

    assert not any(gap["parameter"] == "kernel_id" for gap in gaps)
    assert quick["inputs"]["kernels"][0]["stage"] == "perturbative_matching"
    assert full["inputs"]["kernels"][0]["stage"] == "perturbative_matching"
    assert any(edit["path"] == "inputs.kernels[0].stage" for edit in edits)


def test_plan_strict_validation_rejects_handwritten_matching_momentum_gev(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["perturbative_matching"]
    payload["inputs"]["correlators"] = []
    payload["inputs"]["artifacts"] = [
        {
            "id": "ft",
            "stage": "fourier_transform",
            "path": "ft.nc",
            "momentum": "PX2PY0PZ0",
            "volume": "S16T3",
            "lattice_spacing_fm": 0.1,
        }
    ]
    payload["inputs"]["kernels"][0]["stage"] = "perturbative_matching"
    payload["stages"] = {
        "perturbative_matching": {
            "defaults": {"momentum_gev": 2.15, "mu": 2.0},
            "jobs": [{"id": "mt", "inputs": {"quasi": "ft"}}],
        }
    }

    valid, issues = validate_candidate_payload(tmp_path / "draft.json", payload)

    assert valid is False
    assert any("stages.perturbative_matching.defaults.momentum_gev" in issue.message for issue in issues)


def test_plan_requires_patch_after_yes_to_stage_parameter_completion(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["fourier_transform"],
        },
        "inputs": {"correlators": [], "artifacts": [{"id": "rn", "path": "rn.nc", "stage": "renormalization"}], "kernels": []},
        "stages": {"fourier_transform": {"defaults": {}, "jobs": [{"id": "ft", "inputs": {"input": "rn"}}]}},
    }
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    state = PlanAgentState(path, "", payload, payload)
    state.stage_completion_checked = True
    state.stage_required_checked.add("fourier_transform")
    state.stage_optional_checked.add("fourier_transform")

    answered = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {"patches": [{"op": "add", "path": "/stages/fourier_transform/defaults/order", "value": ["LA"]}]},
    )
    assert answered["ok"] is True
    assert answered["candidate_complete"] is False
    state.parameter_completion_checked = True
    state.parameter_completion_requested = True
    blocked = _run_planning_tool(state, "build_quick_full_candidates", {})
    assert blocked["ok"] is False
    assert "still have missing parameters" in blocked["error"]


def test_plan_stage_question_accepts_free_form_subset() -> None:
    outputs: list[str] = []
    answer = _ask_plan_agent_question(
        {"question_id": "stage.add_remaining", "prompt": "Add missing stages?", "choices": ["yes", "no"]},
        input_func=lambda prompt: "I only want renormalization and fourier_transform",
        output_func=outputs.append,
    )

    assert answer == "I only want renormalization and fourier_transform"


def test_explicit_stage_subset_answer_adds_requested_stage_shells(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    answer = _run_planning_tool(
        state,
        "load_manifest",
        {},
    )
    assert "stage_completion_question_required" not in answer
    applied = _apply_user_answer_to_candidate(state, "stage.add_remaining", "I only want renormalization and fourier_transform")
    assert applied["event"] == "user_answer_applied"
    assert state.stage_completion_checked is True
    assert state.stage_completion_requested is True
    assert state.candidate_payload["metadata"]["stages"] == ["correlator_analysis", "renormalization", "fourier_transform"]
    assert state.candidate_payload["stages"]["fourier_transform"]["jobs"] == [{"id": "fourier_transform"}]


def test_plan_stage_none_answer_keeps_partial_workflow(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    applied = _apply_user_answer_to_candidate(state, "stage.add_remaining", "none")

    assert applied["event"] == "user_answer_not_applied"
    assert state.stage_completion_checked is True
    assert state.stage_completion_requested is False


def test_stage_control_question_id_is_not_rewritten_to_manifest_path() -> None:
    question_id = _manifest_question_id_from_user_input_action(
        {"question_id": "stage.add_remaining", "prompt": "This manifest is not a full canonical flow."},
        "metadata.random_seed was skipped earlier.",
    )

    assert question_id == "stage.add_remaining"


def test_unused_stage_question_is_asked_before_add_remaining(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    question = _next_questions_for_state(state)[0]

    assert question["question_id"] == "stage.unused.review"
    assert "not listed in metadata.stages" in question["prompt"]


def test_unused_stage_include_adds_metadata_stage(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    applied = _apply_user_answer_to_candidate(state, "stage.unused.review", "include")

    assert applied["event"] == "user_answer_applied"
    assert state.candidate_payload["metadata"]["stages"] == [
        "correlator_analysis",
        "renormalization",
        "review",
    ]
    assert "review" in state.candidate_payload["stages"]


def test_unused_stage_remove_drops_stage_config(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    applied = _apply_user_answer_to_candidate(state, "stage.unused.review", "remove")

    assert applied["event"] == "user_answer_applied"
    assert "review" not in state.candidate_payload["stages"]
    assert state.candidate_payload["metadata"]["stages"] == ["correlator_analysis", "renormalization"]


def test_unused_stage_question_id_is_not_rewritten_to_manifest_path() -> None:
    question_id = _manifest_question_id_from_user_input_action(
        {"question_id": "stage.unused.review", "prompt": "Include unused review?"},
        "metadata.random_seed was skipped earlier.",
    )

    assert question_id == "stage.unused.review"


def test_check_manifest_draft_reports_unused_stage(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }

    issues = check_manifest_draft(tmp_path / "draft.json", payload)

    assert any(
        issue.severity == "error" and issue.manifest_path == "stages.review" and "not listed in `metadata.stages`" in issue.message
        for issue in issues
    )


def test_plan_asks_unused_stage_before_llm(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    outputs: list[str] = []

    result = run_interactive_plan(
        manifest,
        backend="cli",
        provider="codex",
        input_func=lambda prompt: "q",
        output_func=outputs.append,
    )

    assert result is None
    joined = "\n".join(outputs)
    assert "not listed in metadata.stages" in joined
    assert "Plan cancelled" in joined


def test_stage_choice_question_id_is_not_rewritten_to_manifest_path() -> None:
    question_id = _manifest_question_id_from_user_input_action(
        {"question_id": "stage_required.renormalization", "prompt": "renormalization required choices"},
        "metadata.random_seed was skipped earlier.",
    )

    assert question_id == "stage_required.renormalization"


def test_stage_required_answer_updates_stage_defaults(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"]["defaults"] = {}
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_required.renormalization",
        "scheme=hybrid, strategy=external_denominator, zs_fm=0.2",
    )

    assert result["event"] == "user_answer_applied"
    assert state.stage_required_checked == set()
    assert any(gap["stage"] == "renormalization" for gap in _stage_parameter_gaps(state.candidate_payload))
    assert state.candidate_payload["stages"]["renormalization"]["defaults"]["scheme"] == "hybrid"
    assert state.candidate_payload["stages"]["renormalization"]["defaults"]["strategy"] == "external_denominator"
    assert state.candidate_payload["stages"]["renormalization"]["defaults"]["zs_fm"] == 0.2


def test_single_required_enum_answer_accepts_bare_value(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"]["defaults"].pop("strategy")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_required.renormalization",
        "external_denominator",
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["renormalization"]["defaults"]["strategy"] == "external_denominator"


def test_stage_required_answer_updates_job_inputs(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"] = {
        "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": 0.2},
        "jobs": [{"id": "rn"}],
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_required.renormalization",
        '{"target": "ca_pz", "denominator": "ca_p0"}',
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["renormalization"]["jobs"][0]["inputs"] == {"target": "ca_pz", "denominator": "ca_p0"}
    assert "target" not in state.candidate_payload["stages"]["renormalization"]["defaults"]


def test_extrapolation_required_answer_updates_lightcone_input(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["extrapolation"]
    payload["stages"] = {"extrapolation": {"defaults": {}, "jobs": [{"id": "ext"}]}}
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_required.extrapolation",
        '{"inputs.lightcone": ["mt_p4", "mt_p5"]}',
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["extrapolation"]["jobs"][0]["inputs"] == {"lightcone": ["mt_p4", "mt_p5"]}


def test_stage_required_answer_keeps_list_stage_fields_as_lists(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["correlator_analysis"]["defaults"].pop("fit_strategy")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_required.correlator_analysis",
        "fit_scope=3pt_ratio, fitting_form=Breit",
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["correlator_analysis"]["defaults"]["fit_scope"] == ["3pt_ratio"]
    assert state.candidate_payload["stages"]["correlator_analysis"]["defaults"].get("fit_strategy") is None
    assert state.candidate_payload["stages"]["correlator_analysis"]["defaults"]["fitting_form"] == "Breit"


def test_stage_optional_answer_keeps_fit_strategy_as_list(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_optional.correlator_analysis",
        "fit_strategy=chained, nstate=1",
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["correlator_analysis"]["defaults"]["fit_strategy"] == ["chained"]
    assert state.candidate_payload["stages"]["correlator_analysis"]["defaults"]["nstate"] == [1]


def test_unparsed_stage_answer_does_not_mark_stage_checked(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(state, "stage_required.renormalization", "hybrid ratio please")

    assert result["event"] == "user_answer_not_applied"
    assert state.stage_required_checked == set()
    assert state.candidate_payload == payload


def test_required_none_does_not_clear_existing_stage_gaps(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["fourier_transform"]
    payload["stages"] = {"fourier_transform": {"defaults": {}, "jobs": [{"id": "ft"}]}}
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(state, "stage_required.fourier_transform", "none")

    assert result["event"] == "user_answer_not_applied"
    assert state.stage_required_checked == set()


def test_none_manifest_answer_is_not_written_as_string(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["correlator_analysis"]["defaults"].pop("component")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(state, "stages.correlator_analysis.defaults.component", "none")

    assert result["event"] == "user_answer_not_applied"
    assert "component" not in state.candidate_payload["stages"]["correlator_analysis"]["defaults"]


def test_axis_description_is_not_written_to_data_path(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "inputs.correlators.0.data_path",
        "shape (64, 48), axis 0 is time, axis 1 is cfg",
    )

    assert result["event"] == "user_answer_not_applied"
    assert state.candidate_payload["inputs"]["correlators"][0]["data_path"] == "data/c2.h5"


def test_metadata_answers_can_be_applied_one_at_a_time(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"].pop("random_seed")
    payload["metadata"].pop("resample_mode")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    seed = _apply_user_answer_to_candidate(state, "metadata.random_seed", 1984)
    mode = _apply_user_answer_to_candidate(state, "metadata.resample_mode", "jk")

    assert seed["event"] == "user_answer_applied"
    assert mode["event"] == "user_answer_applied"
    assert state.candidate_payload["metadata"]["random_seed"] == 1984
    assert state.candidate_payload["metadata"]["resample_mode"] == "jk"


def test_combined_metadata_answer_updates_required_fields(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"].pop("random_seed")
    payload["metadata"].pop("resample_mode")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(state, "metadata.required", "random_seed=1984, resample_mode=jackknife")

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["metadata"]["random_seed"] == 1984
    assert state.candidate_payload["metadata"]["resample_mode"] == "jk"


def test_stage_contract_input_gap_asks_required_before_optional(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))
    state.stage_completion_checked = True

    question = _next_questions_for_state(state)[0]

    assert question["question_id"] == "stage_required.correlator_analysis"
    assert "at least one 3pt correlator" in question["prompt"]
    assert "correlator_analysis" not in state.stage_required_checked


def test_text_plan_reads_metadata_from_free_form_request(tmp_path: Path) -> None:
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF manifest from c2.h5. "
        "Use random_seed 1984 and resample_mode jk. "
        "Only correlator_analysis is required.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["metadata"]["random_seed"] == 1984
    assert payload["metadata"]["resample_mode"] == "jk"


def test_text_plan_target_observable_does_not_match_data_path(tmp_path: Path) -> None:
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a manifest with target_observable: pdf from sample_2pt.npy and current_data_path sample_current.npz. "
        "Use random_seed 1984 and resample_mode jk.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["metadata"]["target_observable"] == "pdf"
    three_point = next(item for item in payload["inputs"]["correlators"] if item["correlator_type"] == "3pt")
    assert "polarization" not in three_point


@pytest.mark.parametrize("polarization", ["unpolarized", "helicity", "transversity"])
def test_text_plan_records_explicit_polarization(tmp_path: Path, polarization: str) -> None:
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion quark PDF manifest from sample_2pt.npy and current_data_path sample_current.npz. "
        f"polarization: {polarization}. Use random_seed 1984 and resample_mode jk.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)
    three_point = next(item for item in payload["inputs"]["correlators"] if item["correlator_type"] == "3pt")

    assert three_point["polarization"] == polarization


def test_text_plan_uses_external_hadron_and_ignores_fourier_observable(tmp_path: Path) -> None:
    (tmp_path / "rn_input.nc").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a manifest with target_observable: pdf from rn_input.nc. Run fourier_transform with "
        "fourier observable: nucleon_quark_quasi_pdf, polarization: helicity, and y_grid [-0.5, 0, 0.5]. "
        "Use random_seed 1984 and resample_mode jk.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["metadata"]["target_observable"] == "pdf"
    assert "observable" not in payload["stages"]["fourier_transform"]["defaults"]
    assert payload["stages"]["fourier_transform"]["defaults"]["hadron"] == "nucleon"
    assert payload["stages"]["fourier_transform"]["defaults"]["polarization"] == "helicity"
    assert not any(gap["parameter"] == "hadron" for gap in _stage_parameter_gaps(payload, manifest))


def test_external_fourier_input_without_provenance_requires_hadron(tmp_path: Path) -> None:
    (tmp_path / "rn_input.nc").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a manifest with target_observable: pdf from rn_input.nc. Run fourier_transform with "
        "y_grid [-0.5, 0, 0.5]. Use random_seed 1984 and resample_mode jk.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert any(gap["parameter"] == "hadron" for gap in _stage_parameter_gaps(payload, manifest))


def test_gluon_text_plan_normalizes_fourier_sector(tmp_path: Path) -> None:
    (tmp_path / "rn_pz.nc").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion gluon PDF manifest from rn_pz.nc. Use random_seed 1984 and resample_mode jk. "
        "Run fourier_transform with "
        "y_grid [-0.5, 0, 0.5] and sector sea.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["metadata"]["parton"] == "gluon"
    assert payload["stages"]["fourier_transform"]["defaults"]["sector"] == "full"


def test_da_text_plan_normalizes_fourier_sector(tmp_path: Path) -> None:
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion DA manifest from rn_pz.nc. "
        "Use random_seed 1984 and resample_mode jk. "
        "Run fourier_transform with y_grid {\"start\": -1.0, \"stop\": 1.0, \"num\": 101} and sector valence.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["metadata"]["target_observable"] == "da"
    assert payload["stages"]["fourier_transform"]["defaults"]["sector"] == "full"


def test_text_plan_omits_default_fourier_coord_unit(tmp_path: Path) -> None:
    (tmp_path / "rn_pz.nc").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF manifest from rn_pz.nc. "
        "Use random_seed 1984 and resample_mode jk. "
        'Run fourier_transform with y_grid {"start": -1.0, "stop": 1.0, "num": 101}.',
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert "coord_unit" not in payload["stages"]["fourier_transform"]["defaults"]


def test_text_plan_ignores_removed_fourier_coord_unit(tmp_path: Path) -> None:
    (tmp_path / "rn_pz.nc").write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF manifest from rn_pz.nc. "
        "Use random_seed 1984 and resample_mode jk. "
        'Run fourier_transform with y_grid {"start": -1.0, "stop": 1.0, "num": 101} and coord_unit: lattice.',
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert "coord_unit" not in payload["stages"]["fourier_transform"]["defaults"]


def test_text_plan_reads_colon_json_stage_defaults(tmp_path: Path) -> None:
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF manifest from rn_pz.nc. "
        "Use random_seed 1984 and resample_mode jk. "
        "Run fourier_transform and perturbative_matching. "
        "y_grid: {\"start\": -1.0, \"stop\": 1.0, \"num\": 101}. "
        "scheme_scan: {\"zmin_fm\": [1], \"zmax_fm\": [5], \"zmax_ext_fm\": 8}. "
        "quasi_y_ls: {\"start\": -1.0, \"stop\": 1.0, \"num\": 100}. "
        "lc_x_ls: {\"start\": -1.0, \"stop\": 1.0}.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)

    assert payload["stages"]["fourier_transform"]["defaults"]["quasi_y_ls"]["num"] == 100
    assert payload["stages"]["fourier_transform"]["defaults"]["scheme_scan"]["zmax_ext_fm"] == 8
    assert payload["stages"]["perturbative_matching"]["defaults"]["lc_x_ls"] == {"start": -1.0, "stop": 1.0}
    assert "quasi_y_ls" not in payload["stages"]["perturbative_matching"]["defaults"]


def test_text_plan_reads_partial_artifact_fallback_metadata(tmp_path: Path) -> None:
    (tmp_path / "mt_p5.nc").write_text("not a real netcdf", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a partial extrapolation manifest from mt_p5.nc. "
        "mt_p5.nc: stage perturbative_matching, path mt_p5.nc. If metadata is missing, use momentum PX5PY0PZ0, volume S48T64, lattice_spacing_fm 0.0574, hadron pion, gfix CG, bz_direction X. "
        "Use random_seed 1984 and resample_mode jk. Run extrapolation.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)
    artifact = payload["inputs"]["artifacts"][0]
    ok, issues = validate_candidate_payload(manifest, payload)

    assert artifact["momentum"] == "PX5PY0PZ0"
    assert artifact["volume"] == "S48T64"
    assert artifact["lattice_spacing_fm"] == 0.0574
    assert not any("IO backends" in issue.message for issue in issues)


def test_text_plan_deduplicates_repeated_discrete_3pt_values(tmp_path: Path) -> None:
    for name in ("a060_x_p0_3pt_ts8.h5", "a060_x_p0_3pt_ts10.h5", "a060_x_p5_3pt_ts8.h5", "a060_x_p5_3pt_ts10.h5"):
        (tmp_path / name).write_text("", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF correlator_analysis manifest from "
        "a060_x_p0_3pt_ts8.h5 with bT 0, bz 0, tsep 8; "
        "a060_x_p0_3pt_ts10.h5 with bT 0, bz 0, tsep 10; "
        "a060_x_p5_3pt_ts8.h5 with bT 0, bz 0, tsep 8; "
        "a060_x_p5_3pt_ts10.h5 with bT 0, bz 0, tsep 10. "
        "Use random_seed 1984 and resample_mode jk.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)
    three_points = [item for item in payload["inputs"]["correlators"] if item["correlator_type"] == "3pt"]

    assert len(three_points) == 4
    assert {tuple(item["bT"]) for item in three_points} == {(0,)}
    assert {tuple(item["bz"]) for item in three_points} == {(0,)}
    assert {tuple(item["tsep"]) for item in three_points} == {(8,), (10,)}


def test_text_plan_reads_current_operator_for_3pt_label(tmp_path: Path) -> None:
    (tmp_path / "sample_p0_3pt_ts8.h5").write_text("", encoding="utf-8")
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion PDF correlator_analysis manifest from sample_p0_3pt_ts8.h5. "
        "Use random_seed 1984, resample_mode jk, current_operator for 3pt: current, bT 0, bz 0, tsep 8.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)
    three_point = next(item for item in payload["inputs"]["correlators"] if item["correlator_type"] == "3pt")

    assert three_point["current_operator"] == "current"


def test_text_plan_reads_correlator_required_choices(tmp_path: Path) -> None:
    np.save(tmp_path / "local_PX0PY0PZ6_2pt.npy", np.ones((64, 4)))
    np.save(tmp_path / "nonlocal_PX0PY0PZ6_2pt.npy", np.ones((1, 64, 4)))
    manifest = tmp_path / "request.txt"
    manifest.write_text(
        "Build a pion DA qda_ratio correlator_analysis manifest from local_PX0PY0PZ6_2pt.npy and nonlocal_PX0PY0PZ6_2pt.npy. "
        "Use random_seed 1984, resample_mode jk, momentum PX0PY0PZ6, lattice_spacing_fm: 0.0574, sink_operator: sink, bT 0, bz 0, bz_direction Z. "
        "fit_scope: qda_ratio. fitting_form: Breit.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(manifest)
    defaults = payload["stages"]["correlator_analysis"]["defaults"]
    nonlocal_pt2 = next(item for item in payload["inputs"]["correlators"] if item["correlator_id"] == "nonlocal_PX0PY0PZ6_2pt")

    assert defaults["fit_scope"] == ["qda_ratio"]
    assert defaults["fitting_form"] == "Breit"
    assert nonlocal_pt2["sink_operator"] == "sink"
    assert nonlocal_pt2["lattice_spacing_fm"] == 0.0574


def test_stage_parameter_gap_answer_applies_first_gap_path(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["fourier_transform"]
    payload["inputs"]["correlators"] = []
    payload["inputs"]["kernels"] = []
    payload["inputs"]["artifacts"] = [
        {
            "id": "rn",
            "stage": "renormalization",
            "path": "rn.nc",
            "momentum": "PX1PY0PZ0",
            "volume": "S16T5",
            "lattice_spacing_fm": 0.1,
            "hadron": "pion",
        }
    ]
    payload["stages"] = {"fourier_transform": {"defaults": {}, "jobs": [{"id": "ft", "inputs": {"input": "rn"}}]}}
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_params.fourier_transform.shared.quasi_y_ls",
        '{"start": -1.0, "stop": 1.0, "num": 100}',
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["fourier_transform"]["defaults"]["quasi_y_ls"]["num"] == 100


def test_stage_parameter_gap_answer_uses_matching_question_id(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"] = ["renormalization", "perturbative_matching"]
    payload["inputs"]["artifacts"] = [
        {"id": "ft", "stage": "fourier_transform", "path": "ft.nc", "momentum": "PX1PY0PZ0", "volume": "S16T5", "lattice_spacing_fm": 0.1}
    ]
    payload["inputs"]["kernels"] = [
        {"stage": "perturbative_matching", "kernel_id": "CG_gt_quark_PDF_hybrid_NLO", "kernel_path": "lamet_agent/kernels.py"}
    ]
    payload["stages"] = {
        "renormalization": {"defaults": {"scheme": "hybrid", "strategy": "external_denominator"}, "jobs": [{"id": "rn", "inputs": {"target": "ca_p1", "denominator": "ca_p0"}}]},
        "perturbative_matching": {"defaults": {"scheme": "hybrid", "kernel_id": "CG_gt_quark_PDF_hybrid_NLO"}, "jobs": [{"id": "mt_p5", "inputs": {"quasi": "ft"}}]},
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(state, "stage_params.perturbative_matching.shared.zs_fm", "0.18")

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["perturbative_matching"]["defaults"]["zs_fm"] == 0.18
    assert "zs_fm" not in state.candidate_payload["stages"]["renormalization"]["defaults"]


@pytest.mark.parametrize(
    "answer",
    ["strategy=external_denominator", '{"strategy": "external_denominator"}'],
)
def test_stage_parameter_gap_answer_extracts_named_value(tmp_path: Path, answer: str) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"]["defaults"].pop("strategy")
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_params.renormalization.shared.strategy",
        answer,
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["renormalization"]["defaults"]["strategy"] == "external_denominator"


def test_stage_optional_answer_updates_stage_defaults(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["stages"].append("fourier_transform")
    payload["stages"]["fourier_transform"] = {
        "defaults": {},
        "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_optional.fourier_transform",
        '{"quasi_y_ls": {"start": -1.0, "stop": 1.0, "num": 4}}',
    )

    assert result["event"] == "user_answer_applied"
    assert state.stage_optional_checked == {"fourier_transform"}
    assert state.candidate_payload["stages"]["fourier_transform"]["defaults"]["quasi_y_ls"]["num"] == 4


def test_da_stage_answer_normalizes_fourier_sector(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["target_observable"] = "da"
    payload["metadata"]["stages"] = ["fourier_transform"]
    payload["stages"] = {
        "fourier_transform": {
            "defaults": {},
            "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
        }
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "stage_optional.fourier_transform",
        '{"sector": "valence"}',
    )

    assert result["event"] == "user_answer_applied"
    assert state.candidate_payload["stages"]["fourier_transform"]["defaults"]["sector"] == "full"


def test_da_manifest_patch_normalizes_fourier_sector(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"]["target_observable"] = "da"
    payload["metadata"]["stages"] = ["fourier_transform"]
    payload["stages"] = {
        "fourier_transform": {
            "defaults": {},
            "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
        }
    }
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {"patches": [{"op": "add", "path": "/stages/fourier_transform/defaults/sector", "value": "valence"}], "allow_incomplete": True},
    )

    assert result["ok"] is True
    assert state.candidate_payload["stages"]["fourier_transform"]["defaults"]["sector"] == "full"


def test_manifest_patch_deduplicates_correlator_discrete_values(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["inputs"]["correlators"][0]["correlator_type"] = "3pt"
    payload["inputs"]["correlators"][0]["bT"] = [0]
    payload["inputs"]["correlators"][0]["bz"] = [0]
    payload["inputs"]["correlators"][0]["tsep"] = [8]
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {
            "patches": [
                {"op": "replace", "path": "/inputs/correlators/0/bT", "value": [0, 0, 0, 0]},
                {"op": "replace", "path": "/inputs/correlators/0/bz", "value": [0, 0]},
                {"op": "replace", "path": "/inputs/correlators/0/tsep", "value": [8, 8, 10]},
            ],
            "allow_incomplete": True,
        },
    )

    assert result["ok"] is True
    assert state.candidate_payload["inputs"]["correlators"][0]["bT"] == [0]
    assert state.candidate_payload["inputs"]["correlators"][0]["bz"] == [0]
    assert state.candidate_payload["inputs"]["correlators"][0]["tsep"] == [8, 10]


def test_manifest_confirmation_answer_is_not_applied(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))

    result = _apply_user_answer_to_candidate(
        state,
        "confirm.inputs.correlators.0.current_operator",
        "yes",
    )

    assert result["event"] == "user_answer_not_applied"
    assert state.candidate_payload == payload


def test_plan_api_answer_still_runs_conversions(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    session = _PlanAgentSession(
        backend="cli",
        manifest_path=tmp_path / "request.txt",
        manifest_text="",
        api_key=None,
        provider="codex",
        model_name=None,
        base_url=None,
    )

    session.observe({"event": "user_answer", "question_id": "stage.add_remaining", "value": "none"})

    action = session.decide()
    assert action["action"] == "call_tool"
    assert action["tool_name"] == "plan_correlator_h5_conversions"


def test_plan_stage_params_question_without_choices_accepts_free_text() -> None:
    answer = _ask_plan_agent_question(
        {"question_id": "stage_params.fourier_transform.shared.Lambda0_gev", "prompt": "Choose Fourier order."},
        input_func=lambda prompt: "LA",
        output_func=lambda text: None,
    )

    assert answer == "LA"


def test_planner_requests_missing_bz_direction_for_3pt() -> None:
    payload = {
        "metadata": {"random_seed": 1984, "resample_mode": "jk", "sample_error_mode": "covariance", "stages": []},
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "source_operator": "g5",
                    "sink_operator": "g5",
                    "current_operator": "gT_nonlocal",
                    "polarization": "unpolarized",
                    "volume": "S16T32",
                    "lattice_spacing_fm": 0.1,
                    "momentum": ["PX0PY0PZ0"],
                    "bT": [0],
                    "bz": [0],
                    "tsep": [8],
                }
            ]
        },
    }
    state = PlanAgentState(Path("draft.json"), "", payload, payload)
    questions = _next_questions_for_state(state)
    assert questions[0]["question_id"] == "inputs.correlators.0.bz_direction"


def test_correlator_h5_conversion_outputs_existing_reader_layout(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    root.mkdir()
    data_dir = root / "data"
    data_dir.mkdir()
    pt2_cfg_time = np.arange(15, dtype=float).reshape(5, 3)
    pt3_cfg_tau_z0 = np.arange(12, dtype=float).reshape(3, 4)
    pt3_cfg_tau_z1 = np.arange(12, 24, dtype=float).reshape(3, 4)
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/raw_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T3",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/raw_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0, 1],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": _required_correlator_defaults(), "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
    }
    with h5py.File(data_dir / "raw_2pt.h5", "w") as h5f:
        h5f.create_dataset("raw_pt2", data=pt2_cfg_time)
    with h5py.File(data_dir / "raw_3pt.h5", "w") as h5f:
        h5f.attrs["bz_direction"] = "Z"
        h5f.create_dataset("raw_z0", data=pt3_cfg_tau_z0)
        h5f.create_dataset("raw_z1", data=pt3_cfg_tau_z1)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    inspections = inspect_correlator_h5_files(path, payload)
    c3_inspection = next(item for item in inspections if item.correlator_id == "c3")
    assert c3_inspection.attrs["bz_direction"] == "Z"

    conversions = plan_correlator_h5_conversions(path, payload)
    assert len(conversions) == 2
    assert all(item.ambiguous for item in conversions)
    state = PlanAgentState(path, "", payload, payload, conversions=conversions)
    result = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c2",
            "datasets": [{"source": "raw_pt2", "target": "g5/g5/PX0PY0PZ0", "transpose": True}],
        },
    )
    assert result["ok"] is True
    result = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c3",
            "datasets": [
                {"source": "raw_z0", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", "transpose": True},
                {"source": "raw_z1", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz1", "transpose": True},
            ],
        },
    )
    assert result["ok"] is True
    for conversion in conversions:
        convert_correlator_h5(conversion)

    c2_output = next(item for item in conversions if item.correlator_id == "c2").output_file
    c3_output = next(item for item in conversions if item.correlator_id == "c3").output_file
    with h5py.File(c3_output) as h5f:
        assert h5f.attrs["bz_direction"] == "Z"
        assert h5f.attrs["standard_correlator_hdf5_version"] == 2
    assert np.array_equal(
        _read_2pt(c2_output, source_operator="g5", sink_operator="g5", momentum="PX0PY0PZ0"),
        pt2_cfg_time,
    )
    assert np.array_equal(
        _read_3pt(
            c3_output,
            source_operator="g5",
            sink_operator="g5",
            current_operator="gT_nonlocal",
            momentum="PX0PY0PZ0",
            bT=0,
            bz=1,
            tsep=3,
        ),
        pt3_cfg_tau_z1,
    )


def test_correlator_numpy_conversion_outputs_standard_h5_and_script(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    pt2_cfg_time = np.arange(15, dtype=float).reshape(5, 3)
    np.save(data_dir / "raw_2pt.npy", pt2_cfg_time)
    payload = _minimal_payload(root, data_path="data/raw_2pt.npy")
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    conversions = plan_correlator_h5_conversions(path, payload)
    assert len(conversions) == 1
    assert conversions[0].ambiguous
    state = PlanAgentState(path, "", payload, payload, conversions=conversions)
    result = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c2",
            "datasets": [{"source": "array", "target": "g5/g5/PX0PY0PZ0", "transpose": True}],
        },
    )
    assert result["ok"] is True
    convert_correlator_h5(state.conversions[0])

    assert Path(state.conversions[0].script_file).is_file()
    assert np.array_equal(
        _read_2pt(
            state.conversions[0].output_file,
            source_operator="g5",
            sink_operator="g5",
            momentum="PX0PY0PZ0",
        ),
        pt2_cfg_time,
    )


def test_correlator_npz_conversion_with_axis_order_and_index(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    np.savez(data_dir / "raw_3pt.npz", all_z=data)
    payload = _minimal_payload(root)
    payload["inputs"]["correlators"] = [
        {
            "correlator_id": "c3",
            "correlator_type": "3pt",
            "data_path": "data/raw_3pt.npz",
            "ensemble": "E",
            "hadron": "pion",
            "gfix": "CG",
            "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
            "momentum": ["PX0PY0PZ0"],
            "lattice_spacing_fm": 0.1,


            "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


            "bT": [0],
            "bz": [0, 1],
            "tsep": [3],
        }
    ]
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    conversions = plan_correlator_h5_conversions(path, payload)
    assert conversions[0].ambiguous
    state = PlanAgentState(path, "", payload, payload, conversions=conversions)
    targets = [
        "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0",
        "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz1",
    ]
    result = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c3",
            "datasets": [
                {"source": "all_z", "target": targets[0], "index": {"0": 0}, "transpose": True},
                {"source": "all_z", "target": targets[1], "index": {"0": 1}, "transpose": True},
            ],
        },
    )
    assert result["ok"] is True
    convert_correlator_h5(state.conversions[0])

    assert np.array_equal(
        _read_3pt(
            state.conversions[0].output_file,
            source_operator="g5",
            sink_operator="g5",
            current_operator="gT_nonlocal",
            momentum="PX0PY0PZ0",
            bT=0,
            bz=1,
            tsep=3,
        ),
        data[1],
    )


def test_text_plan_composes_2pt_current_into_standard_3pt_h5(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    c2 = np.arange(8, dtype=float).reshape(2, 4)
    current = np.full((1, 4), 2.0)
    np.save(data_dir / "sample_2pt.npy", c2)
    np.savez(data_dir / "sample_current.npz", current=current)
    request = tmp_path / "request.txt"
    request.write_text(
        f"Build a DA plan from {data_dir / 'sample_2pt.npy'} and current file {data_dir / 'sample_current.npz'}. "
        "Use ensemble planned, hadron pion, gfix GI, source_operator source, sink_operator sink, "
        "current_operator current, polarization unpolarized, volume S4T4, lattice_spacing_fm: 0.1, "
        "momentum PX0PY0PZ0, bT 0, bz 0, tsep 1, bz_direction Z.",
        encoding="utf-8",
    )

    payload, raw = load_relaxed_manifest(request)
    correlators = payload["inputs"]["correlators"]

    assert "sample_current.npz" in raw
    assert [item["correlator_type"] for item in correlators] == ["2pt", "3pt"]
    planned_correlator = correlators[1]
    assert planned_correlator["plan_sources"]["two_point"].endswith("sample_2pt.npy")
    conversions = plan_correlator_h5_conversions(request, payload)
    planned = next(item for item in conversions if item.operation == "compose_2pt_current")
    assert planned.ambiguous is False

    convert_correlator_h5(planned)
    quick, full, _edits = build_repaired_manifests(request, payload, conversions)
    for repaired in (quick, full):
        assert "plan_sources" not in json.dumps(repaired)
        repaired_3pt = next(item for item in repaired["inputs"]["correlators"] if item["correlator_id"] == "planned_3pt_from_current")
        assert repaired_3pt["data_path"].endswith("request_planned_3pt.h5")

    assert np.array_equal(
        _read_3pt(
            planned.output_file,
            source_operator="source",
            sink_operator="sink",
            current_operator="current",
            momentum="PX0PY0PZ0",
            bT=0,
            bz=0,
            tsep=1,
        ),
        np.repeat(c2[1:2] * current, 2, axis=0).T,
    )


def test_text_plan_expands_momentum_tsep_npy_template_into_standard_h5(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    inputs = tmp_path / "npy_inputs"
    inputs.mkdir()
    np.save(inputs / "a060_x_p0_2pt.npy", np.ones((64, 5), dtype=np.complex128))
    np.save(inputs / "a060_x_p5_2pt.npy", np.ones((64, 5), dtype=np.complex128) * 2)
    for mom in (0, 5):
        for tsep in (8, 10):
            np.save(inputs / f"a060_x_p{mom}_3pt_ts{tsep}.npy", np.ones((3, tsep + 1, 5), dtype=np.complex128) * (mom + tsep))
    request = tmp_path / "CGPDF.txt"
    request.write_text(
        "Analyze a coulomb-gauge fixing pion quark PDF workflow from two-point npy file and three-point npy file.\n"
        f"The two-point correlator file is {inputs / 'a060_x_p0_2pt.npy'}.\n"
        f"The three-point correlator file is {inputs}/a060_x_p{{mom}}_3pt_ts{{tsep}}.npy, where mom means the momentum and tsep means t-separation.\n"
        "Correlator_analysis, hybrid-ratio renormalization, fourier_transform, perturbative_matching and review are required for the manifest draft.\n"
        "Use ensemble planned, volume S48T64, lattice_spacing_fm: 0.0574, source_operator source, "
        "sink_operator sink, current_operator current, polarization unpolarized, bT 0.\n"
        "Review literature: true.\n",
        encoding="utf-8",
    )

    payload, _raw = load_relaxed_manifest(request)
    correlators = payload["inputs"]["correlators"]
    assert payload["metadata"]["stages"] == ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "review"]
    assert payload["stages"]["review"]["defaults"] == {"literature": True}
    assert {item["correlator_id"] for item in correlators} == {
        "a060_x_p0_2pt",
        "a060_x_p5_2pt",
        "a060_x_p0_3pt_ts8",
        "a060_x_p0_3pt_ts10",
        "a060_x_p5_3pt_ts8",
        "a060_x_p5_3pt_ts10",
    }
    conversions = plan_correlator_h5_conversions(request, payload)
    assert len(conversions) == 6
    assert all(not item.ambiguous for item in conversions)
    three_point = next(item for item in conversions if item.correlator_id == "a060_x_p5_3pt_ts10")
    assert len(three_point.datasets) == 3

    convert_correlator_h5(three_point)

    assert np.array_equal(
        _read_3pt(
            three_point.output_file,
            source_operator="source",
            sink_operator="sink",
            current_operator="current",
            momentum="PX5PY0PZ0",
            bT=0,
            bz=2,
            tsep=10,
        ),
        (np.ones((11, 5), dtype=np.complex128) * 15).T,
    )


def test_build_quick_full_candidates_plans_missing_conversions(tmp_path: Path) -> None:
    np.save(tmp_path / "sample_2pt.npy", np.ones((64, 4)))
    np.savez(tmp_path / "sample_current.npz", current=np.ones((1, 4)))
    request = tmp_path / "request.txt"
    request.write_text(
        "Build a pion PDF correlator_analysis manifest from sample_2pt.npy and current_data_path sample_current.npz. "
        "Use ensemble planned, CG, volume S48T64, lattice spacing 0.0574 fm, momentum PX0PY0PZ0, "
        "source operator source, sink operator sink, current operator current, bz_direction Z, bT 0, bz 0, tsep 1. "
        "polarization unpolarized. "
        "Use plan-only 2pt_current composition. Only correlator_analysis is required.",
        encoding="utf-8",
    )
    payload, _raw = load_relaxed_manifest(request)
    payload["metadata"]["random_seed"] = 1984
    payload["metadata"]["resample_mode"] = "jk"
    payload["metadata"]["sample_error_mode"] = "covariance"
    payload["stages"]["correlator_analysis"]["defaults"].update(_required_correlator_defaults())
    state = PlanAgentState(request, request.read_text(encoding="utf-8"), payload, copy.deepcopy(payload))
    state.stage_completion_checked = True
    state.stage_required_checked.add("correlator_analysis")
    state.stage_optional_checked.add("correlator_analysis")

    result = _run_planning_tool(state, "build_quick_full_candidates", {})

    assert result["ok"] is True
    assert state.conversions
    dumped = json.dumps(state.full)
    assert "sample_2pt.npy" not in dumped
    assert "2pt_current" not in dumped
    assert "plan_sources" not in dumped


def test_text_plan_drafts_multiple_2pt_current_components(tmp_path: Path) -> None:
    np.save(tmp_path / "sample_p0_2pt.npy", np.ones((64, 4)))
    np.save(tmp_path / "sample_p1_2pt.npy", np.ones((64, 4)))
    np.savez(tmp_path / "sample_current_V4.npz", current=np.ones((1, 4)))
    np.savez(tmp_path / "sample_current_A4.npz", current=np.ones((1, 4)))
    request = tmp_path / "request.txt"
    request.write_text(
        "Build a pion PDF correlator_analysis manifest from sample_p0_2pt.npy sample_p1_2pt.npy "
        "and current_data_path sample_current_V4.npz sample_current_A4.npz. "
        "Use ensemble planned, CG, volume S48T64, lattice spacing 0.0574 fm, bz_direction Z, bT 0, bz 0, tsep 1. "
        "Use plan-only 2pt_current composition. Only correlator_analysis is required.",
        encoding="utf-8",
    )

    payload, _raw = load_relaxed_manifest(request)
    planned = [item for item in payload["inputs"]["correlators"] if item["correlator_type"] == "3pt"]

    assert len(planned) == 4
    assert {item["current_operator"] for item in planned} == {"V4", "A4"}
    assert {item["momentum"][0] for item in planned} == {"PX0PY0PZ0", "PX1PY0PZ0"}


def test_text_plan_maps_nonlocal_qda_2pt_template_into_standard_h5(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    local = np.ones((64, 5), dtype=np.complex128)
    nonlocal_data = np.arange(3 * 64 * 5, dtype=float).reshape(3, 64, 5).astype(np.complex128)
    np.save(tmp_path / "local_PX0PY0PZ6_2pt.npy", local)
    np.save(tmp_path / "nonlocal_PX0PY0PZ6_2pt.npy", nonlocal_data)
    request = tmp_path / "qda_da.txt"
    request.write_text(
        "Build a GI pion DA qda_ratio correlator_analysis manifest.\n"
        "Use ensemble planned, volume S48T64, lattice spacing 0.0574 fm, momentum PX0PY0PZ6, "
        "source_operator source, sink_operator sink_nonlocal, bT 0, bz_direction Z.\n"
        "The local 2pt file is local_PX0PY0PZ6_2pt.npy.\n"
        "The nonlocal DA 2pt file is nonlocal_PX0PY0PZ6_2pt.npy with axes bz,time,cfg.\n"
        "Use fit_scope qda_ratio. Only correlator_analysis is required.\n",
        encoding="utf-8",
    )

    payload, _raw = load_relaxed_manifest(request)

    assert payload["stages"]["correlator_analysis"]["defaults"] == {"fit_scope": ["qda_ratio"]}
    nonlocal_correlator = next(item for item in payload["inputs"]["correlators"] if item["correlator_id"].startswith("nonlocal"))
    assert nonlocal_correlator["sink_operator"] == "sink_nonlocal"
    assert nonlocal_correlator["bz"] == [0, 1, 2]
    conversions = plan_correlator_h5_conversions(request, payload)
    mapping = next(item for item in conversions if item.correlator_id == "nonlocal_PX0PY0PZ6_2pt")
    assert len(mapping.datasets) == 3

    convert_correlator_h5(mapping)

    assert np.array_equal(
        _read_2pt(
            mapping.output_file,
            source_operator="source",
            sink_operator="sink_nonlocal",
            momentum="PX0PY0PZ6",
            bT=0,
            bz=2,
        ),
        nonlocal_data[2].T,
    )


def test_correlator_conversion_mapping_rejects_bad_shapes_and_targets(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    data = np.zeros((2, 3, 4), dtype=float)
    np.savez(data_dir / "raw_3pt.npz", all_z=data)
    payload = _minimal_payload(root)
    payload["inputs"]["correlators"] = [
        {
            "correlator_id": "c3",
            "correlator_type": "3pt",
            "data_path": "data/raw_3pt.npz",
            "ensemble": "E",
            "hadron": "pion",
            "gfix": "CG",
            "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
            "momentum": ["PX0PY0PZ0"],
            "lattice_spacing_fm": 0.1,


            "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


            "bT": [0],
            "bz": [0, 1],
            "tsep": [3],
        }
    ]
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    conversions = plan_correlator_h5_conversions(path, payload)
    state = PlanAgentState(path, "", payload, payload, conversions=conversions)

    duplicate = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c3",
            "datasets": [
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", "index": {"0": 0}, "transpose": True},
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", "index": {"0": 1}, "transpose": True},
            ],
        },
    )
    assert duplicate["ok"] is False

    bad_axis = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c3",
            "datasets": [
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", "index": {"0": 0}, "axis_order": [0, 0]},
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz1", "index": {"0": 1}, "transpose": True},
            ],
        },
    )
    assert bad_axis["ok"] is False

    bad_tau = _run_planning_tool(
        state,
        "apply_correlator_conversion_mapping",
        {
            "correlator_id": "c3",
            "datasets": [
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", "index": {"0": 0}},
                {"source": "all_z", "target": "g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz1", "index": {"0": 1}},
            ],
        },
    )
    assert bad_tau["ok"] is False


def test_cli_plan_accept_writes_quick_and_full_manifests(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {**_required_correlator_defaults(), "nstate": [2, 3]}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--provider", "codex"], input="2\nnone\nnone\na\n")

    assert result.exit_code == 0, result.output
    quick_path = root / "artifacts" / "plan_manifests" / "draft.quick.json"
    full_path = root / "artifacts" / "plan_manifests" / "draft.full.json"
    assert quick_path.is_file()
    assert full_path.is_file()
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    assert quick["stages"]["correlator_analysis"]["defaults"]["nstate"] == [2]
    assert full["metadata"]["sample_error_mode"] == "covariance"
    assert full["stages"]["correlator_analysis"]["defaults"]["model_average"] is False
    assert "pt2_windows" not in full["stages"]["correlator_analysis"]["defaults"]
    assert "pt3_windows" not in full["stages"]["correlator_analysis"]["defaults"]
    assert "pt3_tau_cuts" not in full["stages"]["correlator_analysis"]["defaults"]


def test_full_plan_variant_preserves_explicit_correlator_windows(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    explicit_pt2 = [{"tmin": 3, "tmax": 12}, {"tmin": 5, "tmax": 13}]
    explicit_pt3 = [{"tsep_ls": [8, 10], "tau_cut": 2}, {"tsep_ls": [8, 10], "tau_cut": 4}]
    payload["stages"] = {
        "correlator_analysis": {
            "defaults": {
                "pt2_windows": explicit_pt2,
                "pt3_windows": explicit_pt3,
                "nstate": [2],
            },
            "jobs": [],
        }
    }

    _quick, full, _edits = build_repaired_manifests(
        tmp_path / "draft.json", payload, []
    )

    defaults = full["stages"]["correlator_analysis"]["defaults"]
    assert defaults["pt2_windows"] == explicit_pt2
    assert defaults["pt3_windows"] == explicit_pt3


def test_cli_plan_asks_missing_random_seed_once_and_applies_answer(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "bs",
            "sample_error_mode": "covariance",
            "bs_samples": 20,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {**_required_correlator_defaults(), "nstate": [2, 3], "model_average": True},
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--provider", "codex"], input="random_seed=1984, resample_mode=bs\nnone\nnone\na\n")

    assert result.exit_code == 0, result.output
    assert "metadata required choices" in result.output
    quick = json.loads((root / "artifacts" / "plan_manifests" / "draft.quick.json").read_text(encoding="utf-8"))
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["random_seed"] == 1984
    assert quick["metadata"]["random_seed"] == 1984
    assert quick["metadata"]["resample_mode"] == "bs"
    assert quick["metadata"]["sample_error_mode"] == "covariance"
    assert quick["stages"]["correlator_analysis"]["defaults"]["model_average"] is True
    assert "Unresolved questions" not in result.output


def test_plan_rejects_malformed_llm_user_input_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": _required_correlator_defaults(), "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    actions = iter(
        [
            {"action": "request_user_input", "reason": "Need input but omitted prompt.", "args": {}},
            {"action": "call_tool", "tool_name": "load_manifest", "args": {}, "reason": "Inspect manifest."},
            {"action": "call_tool", "tool_name": "check_manifest_draft", "args": {}, "reason": "Check manifest."},
            {"action": "call_tool", "tool_name": "plan_correlator_h5_conversions", "args": {}, "reason": "Plan conversions."},
            {"action": "call_tool", "tool_name": "inspect_correlator_h5_files", "args": {}, "reason": "Inspect HDF5."},
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates."},
            {
                "action": "request_user_input",
                "reason": "Confirm whether to add downstream stages.",
                "args": {"question_id": "stage.add_remaining", "prompt": "Add extra downstream stages?"},
            },
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates after stage preference."},
            {
                "action": "request_user_input",
                "reason": "Confirm optional correlator_analysis choices.",
                "args": {"question_id": "stage_optional.correlator_analysis", "prompt": "correlator_analysis optional choices. Reply none."},
            },
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates after stage choices."},
            {"action": "propose_plan", "reason": "Ready.", "args": {"summary": "Ready after rejecting malformed question."}},
        ]
    )

    def fake_request_llm_text(**kwargs):
        del kwargs
        return json.dumps(next(actions))

    monkeypatch.setattr("lamet_agent.planning.request_llm_text", fake_request_llm_text)
    outputs: list[str] = []
    answers = iter(["a"])

    result = run_interactive_plan(
        manifest,
        backend="api",
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="test",
        input_func=lambda prompt: next(answers),
        output_func=outputs.append,
    )

    assert result is not None
    assert "Add extra downstream stages?" not in "\n".join(outputs)
    assert "optional choices" not in "\n".join(outputs)
    assert "Planner needs user input." not in "\n".join(outputs)
    assert (root / "artifacts" / "plan_manifests" / "draft.full.json").is_file()


def test_plan_applies_manifest_path_user_answer_without_llm_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": _required_correlator_defaults(), "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    actions = iter(
        [
            {
                "action": "request_user_input",
                "reason": "Need the required random seed.",
                "args": {"question_id": "random_seed", "prompt": "metadata.random_seed is required. Enter an integer seed."},
            },
            {"action": "call_tool", "tool_name": "plan_correlator_h5_conversions", "args": {}, "reason": "Plan conversions."},
            {"action": "call_tool", "tool_name": "inspect_correlator_h5_files", "args": {}, "reason": "Inspect HDF5."},
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates."},
            {
                "action": "request_user_input",
                "reason": "Confirm whether to add downstream stages.",
                "args": {"question_id": "stage.add_remaining", "prompt": "Add extra downstream stages?"},
            },
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates after stage preference."},
            {
                "action": "request_user_input",
                "reason": "Confirm optional correlator_analysis choices.",
                "args": {"question_id": "stage_optional.correlator_analysis", "prompt": "correlator_analysis optional choices. Reply none."},
            },
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates after stage choices."},
            {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build candidates after optional choices."},
            {"action": "propose_plan", "reason": "Ready.", "args": {"summary": "Ready after user seed answer."}},
        ]
    )

    def fake_request_llm_text(**kwargs):
        del kwargs
        return json.dumps(next(actions))

    answers = iter(["1999", "a"])
    monkeypatch.setattr("lamet_agent.planning.request_llm_text", fake_request_llm_text)

    result = run_interactive_plan(
        manifest,
        backend="api",
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="test",
        input_func=lambda prompt: next(answers),
        output_func=lambda text: None,
    )

    assert result is not None
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["random_seed"] == 1999


def test_cli_plan_revision_expands_fit_window_search(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {
                    **_required_correlator_defaults(),
                    "pt2_windows": [{"tmin": 3, "tmax": 12}, {"tmin": 4, "tmax": 12}],
                    "pt3_windows": [{"tsep_ls": [3], "tau_cut": 2}, {"tsep_ls": [3], "tau_cut": 3}],
                },
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--provider", "codex"], input="2\nnone\nnone\nr\nPlease broaden the fit window search.\n2\na\n")

    assert result.exit_code == 0, result.output
    assert "LLM expanded the fit-window search" in result.output
    assert "stages.correlator_analysis.defaults.pt2_windows" in result.output
    full_path = root / "artifacts" / "plan_manifests" / "draft.full.json"
    full = json.loads(full_path.read_text(encoding="utf-8"))
    defaults = full["stages"]["correlator_analysis"]["defaults"]
    assert {"tmin": 2, "tmax": 12} in defaults["pt2_windows"]
    assert {"tmin": 6, "tmax": 12} in defaults["pt2_windows"]
    assert defaults["pt3_windows"] == [
        {"tsep_ls": [3], "tau_cut": 2},
        {"tsep_ls": [3], "tau_cut": 3},
        {"tsep_ls": [3], "tau_cut": 4},
        {"tsep_ls": [3], "tau_cut": 5},
    ]
    assert defaults["model_average"] is False
    assert full["metadata"]["sample_error_mode"] == "covariance"
    assert "Quick manifest changes:" in result.output
    assert "Full manifest changes:" in result.output


def test_cli_plan_revision_can_revert_tau_cuts_after_broadening(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c2",
                    "correlator_type": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {
                    **_required_correlator_defaults(),
                    "pt2_windows": [{"tmin": 3, "tmax": 12}, {"tmin": 4, "tmax": 12}],
                    "pt3_windows": [{"tsep_ls": [3], "tau_cut": 2}, {"tsep_ls": [3], "tau_cut": 3}],
                },
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plan", str(manifest), "--provider", "codex"],
        input="2\nnone\nnone\nr\nPlease broaden the fit window search.\nr\nPlease revert the tau cuts.\na\n",
    )

    assert result.exit_code == 0, result.output
    assert "LLM reverted the tau-cut search" in result.output
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    defaults = full["stages"]["correlator_analysis"]["defaults"]
    assert defaults["pt3_windows"] == [
        {"tsep_ls": [3], "tau_cut": 2},
        {"tsep_ls": [3], "tau_cut": 3},
    ]
    assert {"tmin": 2, "tmax": 12} in defaults["pt2_windows"]


def test_manifest_json_patch_add_replace_remove_object_and_list_values() -> None:
    payload = {
        "metadata": {"stages": ["correlator_analysis"]},
        "inputs": {"artifacts": [{"id": "old"}]},
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": []}},
    }

    patched, edits = apply_manifest_json_patches(
        payload,
        [
            {"op": "add", "path": "/metadata/random_seed", "value": 1984},
            {"op": "add", "path": "/metadata/stages/-", "value": "renormalization"},
            {"op": "replace", "path": "/inputs/artifacts/0/id", "value": "new"},
            {"op": "remove", "path": "/stages/correlator_analysis/defaults"},
        ],
    )

    assert patched["metadata"]["random_seed"] == 1984
    assert patched["metadata"]["stages"] == ["correlator_analysis", "renormalization"]
    assert patched["inputs"]["artifacts"][0]["id"] == "new"
    assert "defaults" not in patched["stages"]["correlator_analysis"]
    assert len(edits) == 4


def test_manifest_json_patch_accepts_dotted_manifest_paths() -> None:
    payload = {"metadata": {"stages": ["correlator_analysis"]}, "inputs": {}, "stages": {}}

    patched, edits = apply_manifest_json_patches(
        payload,
        [{"op": "add", "path": "metadata.random_seed", "value": 1990}],
    )

    assert patched["metadata"]["random_seed"] == 1990
    assert edits[0]["path"] == "metadata.random_seed"


def test_manifest_json_patch_rejects_unsupported_or_unsafe_paths() -> None:
    payload = {"metadata": {}, "inputs": {}, "stages": {}}

    with pytest.raises(ValueError, match="Unsupported JSON Patch op"):
        apply_manifest_json_patches(payload, [{"op": "copy", "path": "/metadata/run_id", "value": "demo"}])
    with pytest.raises(ValueError, match="may only modify"):
        apply_manifest_json_patches(payload, [{"op": "add", "path": "/outside/value", "value": "demo"}])
    with pytest.raises(ValueError, match="Cannot replace missing"):
        apply_manifest_json_patches(payload, [{"op": "replace", "path": "/metadata/run_id", "value": "demo"}])


def test_validate_candidate_manifest_rejects_duplicate_job_ids(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_kernel(root)
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "c2.h5").write_text("placeholder", encoding="utf-8")
    payload = _minimal_payload(root)
    patched, _ = apply_manifest_json_patches(
        payload,
        [{"op": "replace", "path": "/stages/renormalization/jobs/0/id", "value": "ca"}],
    )

    ok, issues = validate_candidate_payload(tmp_path / "draft.json", patched)

    assert not ok
    assert any("globally unique" in issue.message for issue in issues)


def test_planning_patch_tool_rejects_invalid_candidate_without_mutating_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_kernel(root)
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "c2.h5").write_text("placeholder", encoding="utf-8")
    payload = _minimal_payload(root)
    state = PlanAgentState(
        manifest_path=tmp_path / "draft.json",
        manifest_text=json.dumps(payload),
        original_payload=payload,
        candidate_payload=json.loads(json.dumps(payload)),
    )

    observation = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {"patches": [{"op": "replace", "path": "/stages/renormalization/jobs/0/id", "value": "ca"}]},
    )

    assert observation["ok"] is False
    assert state.candidate_payload["stages"]["renormalization"]["jobs"][0]["id"] == "rn"


def test_planning_patch_tool_rejects_plan_only_conversion_fields(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, json.loads(json.dumps(payload)))

    observation = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {"patches": [{"op": "add", "path": "/inputs/correlators/0/plan_sources", "value": {"two_point": "c2.npy"}}]},
    )

    assert observation["ok"] is False
    assert "plan-only" in observation["error"]
    assert "plan_sources" not in state.candidate_payload["inputs"]["correlators"][0]


def test_correlator_manifest_answer_invalidates_planned_conversions(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, copy.deepcopy(payload))
    state.conversions = [object()]  # type: ignore[list-item]

    result = _apply_user_answer_to_candidate(state, "inputs.correlators.0.source_operator", "g5")

    assert result["event"] == "user_answer_applied"
    assert state.conversions == []


def test_cli_plan_revision_adds_renormalization_stage_from_english_instruction(
    tmp_path: Path, planning_api_stub: _PlanningApiStub
) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    for name in ("p0_2pt.h5", "p0_3pt.h5", "p5_2pt.h5", "p5_3pt.h5"):
        with h5py.File(data_dir / name, "w") as h5f:
            if "2pt" in name:
                momentum = "PX0PY0PZ0" if name.startswith("p0") else "PX5PY0PZ0"
                h5f.create_dataset(f"g5/g5/{momentum}", data=np.ones((5, 3)))
            else:
                momentum = "PX0PY0PZ0" if name.startswith("p0") else "PX5PY0PZ0"
                h5f.create_dataset(f"g5/g5/gT_nonlocal/{momentum}/tsep3/bT0/bz0", data=np.ones((4, 3)))
    payload = {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "p0_2pt",
                    "correlator_type": "2pt",
                    "data_path": "data/p0_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "p0_3pt",
                    "correlator_type": "3pt",
                    "data_path": "data/p0_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX0PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
                {
                    "correlator_id": "p5_2pt",
                    "correlator_type": "2pt",
                    "data_path": "data/p5_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX5PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                },
                {
                    "correlator_id": "p5_3pt",
                    "correlator_type": "3pt",
                    "data_path": "data/p5_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_operator": "g5", "sink_operator": "g5", "volume": "S16T5",
                    "momentum": ["PX5PY0PZ0"],
                    "lattice_spacing_fm": 0.1,


                    "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": _required_correlator_defaults(),
                "jobs": [
                    {"id": "ca_p0_fh", "correlator_ids": ["p0_2pt", "p0_3pt"], "params": {"momentum": "PX0PY0PZ0"}},
                    {"id": "ca_p5_fh", "correlator_ids": ["p5_2pt", "p5_3pt"], "params": {"momentum": "PX5PY0PZ0"}},
                ],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--provider", "codex"], input="2\nnone\nnone\nr\nPlease add the renormalization stage.\nnone\nnone\na\n")

    assert result.exit_code == 0, result.output
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["stages"] == ["correlator_analysis", "renormalization"]
    assert full["stages"]["renormalization"]["defaults"]["scheme"] == "hybrid"
    assert full["stages"]["renormalization"]["defaults"]["strategy"] == "external_denominator"
    assert full["stages"]["renormalization"]["jobs"] == [
        {"id": "rn_p5_fh", "inputs": {"target": "ca_p5_fh", "denominator": "ca_p0_fh"}}
    ]


def test_text_plan_drafts_2pt_current_composition_without_chinese_json(tmp_path: Path) -> None:
    from lamet_agent.planning.core import load_relaxed_manifest

    (tmp_path / "c2.npy").write_bytes(b"")
    (tmp_path / "current.npz").write_bytes(b"")
    request = tmp_path / "request.txt"
    chinese_prefix = "\u8bf7\u7528"
    request.write_text(
        f"{chinese_prefix} c2.npy and current.npz to compose a nonlocal disconnected 3pt for pion DA.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(request)
    correlators = payload["inputs"]["correlators"]
    assert [item["correlator_type"] for item in correlators] == ["2pt", "3pt"]
    assert correlators[1]["correlator_id"] == "planned_3pt_from_current"
    assert correlators[1]["plan_sources"]["two_point"] == "c2.npy"
    assert correlators[1]["plan_sources"]["current"] == "current.npz"
    dumped = json.dumps(payload, ensure_ascii=False)
    assert not any("\u4e00" <= char <= "\u9fff" for char in dumped)


def test_text_plan_preserves_explicit_gpd_operators(tmp_path: Path) -> None:
    from lamet_agent.planning.core import load_relaxed_manifest

    np.save(tmp_path / "gpd_PX0PY0PZ0_2pt.npy", np.ones((64, 4)))
    np.save(tmp_path / "gpd_PX1PY0PZ0_2pt.npy", np.ones((64, 4)))
    np.save(tmp_path / "gpd_PX1PY0PZ0_3pt_ts8.npy", np.ones((2, 9, 4)))
    request = tmp_path / "gpd_nonforward.txt"
    request.write_text(
        "Build a pion GPD non-forward correlator_analysis manifest. "
        "Use source operator g5, sink operator g5, current operator gt, polarization unpolarized, "
        "bT 0, bz 0, bz 1, bz_direction Z. "
        "Files: gpd_PX0PY0PZ0_2pt.npy, gpd_PX1PY0PZ0_2pt.npy, gpd_PX1PY0PZ0_3pt_ts8.npy.",
        encoding="utf-8",
    )

    payload, _text = load_relaxed_manifest(request)
    correlators = payload["inputs"]["correlators"]
    assert {item["source_operator"] for item in correlators} == {"g5"}
    assert {item["sink_operator"] for item in correlators} == {"g5"}
    assert [item["current_operator"] for item in correlators if item["correlator_type"] == "3pt"] == ["gt"]
    conversions = plan_correlator_h5_conversions(request, payload)
    targets = [dataset["target"] for mapping in conversions for dataset in mapping.datasets]
    assert "g5/g5/gt/PX1PY0PZ0/tsep8/bT0/bz0" in targets
