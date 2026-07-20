"""Agent helpers for interactive planning."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Callable

from lamet_agent.core.banner import BANNER
from lamet_agent.manifest import parse_volume

from .conversion import _dataset_names, _standard_dataset_paths, inspect_correlator_h5_files, plan_correlator_h5_conversions
from .core import (
    PlanAgentState,
    PlanIssue,
    PlanProposal,
    apply_manifest_json_patches,
    build_repaired_manifests,
    check_manifest_draft,
    load_relaxed_manifest,
    _as_list,
    _dataclass_json,
    _expand_pt2_windows,
    _expand_tau_cuts,
    _get_path_value,
    _merge_revision_edits,
    _planned_manifest_paths,
    _set_path_value,
    _stage_parameter_gaps,
    _strict_manifest_issues,
    validate_candidate_payload,
)
from .questions import (
    _ask_plan_agent_question,
    _coerce_user_answer_for_manifest_path,
    _get_dotted_path,
    _json_pointer_from_question_id,
    _manifest_question_id_from_user_input_action,
    _next_questions_for_state,
    _valid_plan_agent_question,
)
from .render import _render_proposal, _render_written_summary, write_planned_outputs


PLAN_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "request_user_input", "propose_plan", "finish"]},
        "reason": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


PLAN_TOOL_CATALOG = {
    "load_manifest": "Return the current in-memory manifest candidate and planned output paths.",
    "check_manifest_draft": "Run deterministic manifest checks that tolerate incomplete drafts.",
    "list_stage_parameter_gaps": "Return structured missing parameters and missing input roles for configured stages, including allowed options or example values.",
    "inspect_correlator_h5_files": "Summarize HDF5, NPY, and NPZ datasets/arrays referenced by inputs.correlators.",
    "plan_correlator_h5_conversions": "Detect source HDF5/NPY/NPZ files that need conversion to the standard correlator HDF5 layout.",
    "apply_correlator_conversion_mapping": "Apply one user-confirmed correlator conversion. Args must be {correlator_id, datasets:[{source,target,axis_order?,index?,transpose?}, ...]}; never pass source/target at top level.",
    "validate_candidate_manifest": "Run strict schema, DAG, and stage-local validation on the current candidate.",
    "apply_manifest_patch_to_candidate": "Apply guarded JSON Patch edits to the in-memory candidate after validation.",
    "build_quick_full_candidates": "Build quick/full manifest candidates and validate their strict schema.",
}


def _planning_system_prompt() -> str:
    return (
        "You are the planning controller for a Python LaMET workflow agent. "
        "You control the plan loop by choosing one action at a time. "
        "Use planning tools to inspect state, validate assumptions, apply candidate manifest patches, "
        "ask the user for missing intent, and then propose a plan. "
        "Your first action must be call_tool load_manifest, and you must call check_manifest_draft before asking user input. "
        "When a tool observation includes next_questions, let the controller ask those deterministic questions; do not rewrite them or invent alternate question_id values. "
        "Never claim a manifest edit was applied until a tool observation confirms it. "
        "Do not write files; final writes happen only after the user accepts. "
        "Return exactly one JSON object matching this schema: "
        + json.dumps(PLAN_ACTION_SCHEMA)
        + "\nAvailable planning tools:\n"
        + "\n".join(f"- {name}: {description}" for name, description in PLAN_TOOL_CATALOG.items())
        + "\nJSON Patch rules: edits may only target /metadata, /inputs, or /stages; use op add, replace, or remove. "
        "For request_user_input, args.prompt must be a concrete user-facing question and args.question_id must identify the decision. "
        "Ask exactly one question per request_user_input action; never combine stage choices, metadata values, parameter values, and data-axis mappings in one prompt. "
        "For ordinary manifest fields, ask for exactly one manifest field at a time and set question_id to the exact dotted manifest path, for example metadata.random_seed, inputs.correlators.0.momentum, inputs.correlators.0.source_operator, or inputs.correlators.1.current_operator. "
        "Do not ask for several ordinary manifest fields in one answer; after the user answers one field, let the automatic patch observation update the candidate before asking the next field. "
        "When multiple items are missing, ask and resolve them one at a time, starting with deterministic manifest fields such as metadata.random_seed before broader workflow choices. "
        "Prefer Yes/No or multiple-choice questions only when the answer is genuinely binary or enumerable; use free-form questions when the user may need to name a subset, axis meaning, or concrete parameter values. "
        "Keep request_user_input prompts concise: state the file shape, uncertain axes or indices, and exact answer format only. "
        "If metadata.stages is not the full canonical flow, ask whether the user wants to add extra downstream stages. "
        "Use question_id 'stage.add_remaining'. This question may be free-form when the user may want only a subset, for example: 'only add renormalization and fourier_transform'. "
        "Add only stages whose inputs can be wired unambiguously; otherwise ask another concise question. "
        "If a configured stage has missing parameters or missing required input roles, explain which stage/job is incomplete and ask a Yes/No question before patching. "
        "Use list_stage_parameter_gaps for structured missing-parameter details when available. "
        "For that question, use question_id 'stage_params.<stage>.<job_id>' when possible. "
        "If the user says yes, ask for a concrete value when it is not inferable; list allowed options for enum-like fields and give one valid example for list/dict fields. "
        "If the value is inferable from existing manifest examples or upstream metadata, patch it only after the Yes answer and state the exact manifest path/value. "
        "For missing required fields, prefer request_user_input unless the user's instruction or examples clearly establish the value. "
        "For stage additions, preserve existing ids and wire jobs through existing upstream job ids. "
        "For inputs.kernels[].stage, treat legacy value 'matching' as an alias of 'perturbative_matching'; do not ask the user to rename it. "
        "The written quick/full manifests normalize that alias to 'perturbative_matching'. "
        "For correlator data conversion, inspect HDF5/NPY/NPZ inputs and never guess ambiguous axes or source keys. "
        "Use multiple-choice questions for simple axis/index choices, but use free-form questions for high-dimensional mappings where the user must describe source, target, cfg/time or cfg/tau axes, z/bz ordering, momentum selection, optional axis_order, optional index selections, and transpose. "
        "When the user gives an unambiguous mapping, call apply_correlator_conversion_mapping with args.correlator_id and args.datasets as a non-empty list of dataset mappings. "
        "Each dataset mapping must include source and target, with optional axis_order, index, and transpose. "
        "axis_order is zero-based and may name either the remaining array axes after index selection, such as [0,1], or the original source axes, such as [3,4] after fixing axes 0,1,2. "
        "Do not use one-based axis_order such as [1,2] for a two-dimensional post-index dataset. "
        "Do not pass source, target, axis_order, index, or transpose directly in top-level args. "
        "For multi-bz 3pt data, include one datasets item per standard bz target in a single tool call when practical. "
        "Do not create a custom conversion section in the manifest and do not encode conversion mappings as JSON Patch manifest edits. "
        "Only use JSON Patch for ordinary manifest fields such as metadata, inputs, and stages. "
        "If a correlator data file is not .npy, .npz, .h5, or .hdf5, tell the user the format is unsupported and strongly recommend converting to standard .h5."
    )


def _initial_planning_user_prompt(manifest_path: Path, manifest_text: str) -> str:
    return json.dumps(
        {
            "task": "Prepare this LaMET analysis manifest for execution.",
            "manifest_path": str(manifest_path),
            "manifest_text": manifest_text,
            "stage_ids": ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"],
            "stage_completion_policy": (
                "The canonical full flow is correlator_analysis -> renormalization -> fourier_transform -> perturbative_matching -> extrapolation -> review. "
                "If the manifest contains only a prefix or subset, ask whether to add extra stages before proposing a plan; allow a free-form subset such as only renormalization and fourier_transform."
            ),
            "stage_parameter_guidance": {
                "correlator_analysis": {
                    "common_defaults": {
                        "nstate": [2],
                        "fit_scope": ["3pt_ratio"],
                        "fit_strategy": ["joint"],
                        "fitting_form": "Breit",
                    },
                    "automatic_windows": (
                        "Omit pt2_windows, pt3_windows, and pt3_tau_cuts to let the stage "
                        "generate bounded data-driven candidates. Preserve any explicitly "
                        "authored window lists exactly."
                    ),
                    "options": {
                        "fit_scope": ["3pt_ratio", "FH", "3pt_ratio+FH", "qda_ratio"],
                        "fit_strategy": ["joint", "chained"],
                        "fitting_form": ["Breit", "NonBreit"],
                        "component": ["re", "im", "both"],
                    },
                    "qda_ratio_inputs": (
                        "one ordinary 2pt with local source/sink operators and one qDA 2pt "
                        "with a nonlocal operator plus bT/bz metadata"
                    ),
                },
                "renormalization": {
                    "required": {"scheme": "ratio | hybrid_ratio | hybrid_self_renormalization"},
                    "branches": {
                        "ratio": {"inputs": ["target", "denominator"]},
                        "hybrid_ratio": {"inputs": ["target", "denominator"], "zs_fm": 0.2},
                        "hybrid_self_renormalization": {"fit_inputs": ["reference"], "apply_inputs": ["target", "zR"]},
                    },
                    "optional": {
                        "normalization": True,
                        "hybrid_scheme_parameters": {"m0_gev": 0.0, "delta_m_gev": 0.0},
                    },
                },
                "fourier_transform": {
                    "required": {"order": ["LA", "NLA"], "coord_unit": "lattice", "sector": "valence", "y_grid": {"start": -2.0, "stop": 2.0, "num": 100}, "momentum_gev": 2.15},
                    "options": {"order": ["LA", "NLA"], "sector_pdf": ["valence", "total", "full", "sea"], "part": ["re", "im", "both"]},
                },
                "perturbative_matching": {
                    "required": {"kernel_id": "declared inputs.kernels kernel_id", "momentum_gev": 2.15, "mu": 2.0, "component": "re", "zs_fm": "required for hybrid kernels"},
                    "options": {"component": ["re", "im"]},
                },
                "extrapolation": {
                    "required": {"inputs": {"lightcone": ["matching_job_1", "matching_job_2"]}},
                    "defaults": {"lowest_lattice_spacing_order": 2, "highest_momentum_order": 2, "pdep_gev": [1.5, 2.0, 2.5]},
                },
                "review": {"required": "none"},
            },
            "common_stage_contracts": {
                "renormalization": {
                    "ratio_inputs": {
                        "target": "upstream bare matrix-element job",
                        "denominator": "zero-momentum/reference bare matrix-element job",
                    },
                    "ratio_defaults": {"scheme": "ratio"},
                    "hybrid_ratio_defaults": {
                        "scheme": "hybrid_ratio",
                        "zs_fm": "required",
                        "scheme_parameters": {"m0_gev": 0.0, "delta_m_gev": 0.0},
                    },
                    "hybrid_self_renormalization_inputs": {"fit": ["reference"], "apply": ["target", "zR"]},
                },
                "fourier_transform": {"inputs": {"input": "renormalized matrix-element job or artifact"}},
                "perturbative_matching": {"inputs": {"quasi": "Fourier transform job or artifact"}},
            },
            "correlator_conversion_contract": {
                "standard_2pt_h5": "source_operator/sink_operator/momentum with dataset shape (Lt, n_cfg)",
                "qda_2pt_h5": "source_operator/sink_operator/momentum/bT<bT>/bz<bz> with dataset shape (Lt, n_cfg); bT/bz are selectors, not operator-name suffixes",
                "standard_3pt_h5": "source_operator/sink_operator/current_operator/momentum/tsep<tsep>/bT<bT>/bz<bz> with dataset shape (tsep+1, n_cfg)",
                "bz_direction": "required 3pt manifest provenance: X, Y, Z, XY, XZ, YZ, or XYZ; it is not an HDF5 path layer",
                "npy_source": "single array; source may be 'array'; user must identify cfg/time or cfg/tau axes and any selected momentum/z indices",
                "npz_source": "source must be an NPZ key; user must map each key to one standard target",
                "apply_conversion_tool_args": {
                    "correlator_id": "correlator_id from inputs.correlators",
                    "datasets": [
                        {
                            "source": "HDF5 dataset path, NPZ key, or 'array' for NPY",
                            "target": "one standard target path",
                            "axis_order": "optional zero-based list producing standard (time_or_tau, cfg) order; may use remaining axes after index selection or original source axes",
                            "index": "optional object mapping source axis number to selected index before axis_order/transpose",
                            "transpose": "optional final transpose boolean",
                        }
                    ],
                },
                "mapping_item": {
                    "source": "HDF5 dataset path, NPZ key, or 'array' for NPY",
                    "target": "one standard target path",
                    "axis_order": "optional zero-based list producing standard (time_or_tau, cfg) order; may use remaining axes after index selection or original source axes",
                    "index": "optional object mapping source axis number to selected index before transpose",
                    "transpose": "optional final transpose boolean",
                },
            },
        },
        indent=2,
    )


class _PlanAgentSession:
    def __init__(
        self,
        *,
        backend: str,
        manifest_path: Path,
        manifest_text: str,
        api_key: str | None,
        provider: str | None,
        model_name: str | None,
        base_url: str | None,
    ) -> None:
        self.backend = backend
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": _planning_system_prompt()},
            {"role": "user", "content": _initial_planning_user_prompt(manifest_path, manifest_text)},
        ]
        self.mock_phase = "load"
        self.last_revision: str | None = None

    def observe(self, observation: dict[str, Any]) -> None:
        self.messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False, indent=2)})
        if observation.get("event") == "user_revision":
            self.last_revision = str(observation.get("text", ""))
            self.mock_phase = "mock_revision"
        elif observation.get("event") == "user_answer":
            if observation.get("question_id") == "stage.add_remaining":
                self.mock_phase = "build"
            elif str(observation.get("question_id", "")).startswith("stage_params."):
                value = str(observation.get("value", "")).strip().lower()
                self.mock_phase = "blocked" if value in {"no", "n", "false", "0"} else "build"
            else:
                self.mock_phase = "mock_answer"
        elif observation.get("event") == "question_skipped":
            self.mock_phase = "conversions"
        elif "not the full canonical stage flow" in str(observation.get("error", "")):
            self.mock_phase = "stage_completion"
        elif "still have missing parameters or input roles" in str(observation.get("error", "")):
            self.mock_phase = "blocked"
        elif "missing parameters or input roles" in str(observation.get("error", "")):
            self.mock_phase = "parameter_completion"

    def decide(self) -> dict[str, Any]:
        if self.backend == "mock":
            return self._mock_decide()
        from lamet_agent import planning as planning_api

        text = planning_api.request_llm_text(
            backend=self.backend,
            messages=self.messages,
            api_key=self.api_key,
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.base_url,
        )
        action = _parse_json_object(text)
        self.messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        return action

    def _mock_decide(self) -> dict[str, Any]:
        phase = self.mock_phase
        if phase == "load":
            self.mock_phase = "check"
            return {"action": "call_tool", "tool_name": "load_manifest", "args": {}, "reason": "Inspect the draft manifest."}
        if phase == "check":
            self.mock_phase = "maybe_seed"
            return {"action": "call_tool", "tool_name": "check_manifest_draft", "args": {}, "reason": "Find deterministic manifest issues."}
        if phase == "maybe_seed":
            self.mock_phase = "conversions"
            return {
                "action": "request_user_input",
                "reason": "metadata.random_seed is required when absent.",
                "args": {
                    "question_id": "metadata.random_seed",
                    "prompt": "metadata.random_seed is required. Which seed should be used?",
                    "choices": [
                        {"label": "1", "value": 1984, "description": "Use 1984, matching the repository examples."},
                        {"label": "2", "value": 20260707, "description": "Use a date-based seed for this planning run."},
                        {"label": "3", "value": "__custom_int__", "description": "Enter a custom positive integer seed."},
                    ],
                    "custom_hint": "Enter random_seed as an integer: ",
                    "skip_if_present": "metadata.random_seed",
                },
            }
        if phase == "mock_answer":
            self.mock_phase = "conversions"
            value = self._latest_user_answer()
            return {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {"patches": [{"op": "add", "path": "/metadata/random_seed", "value": int(value)}]},
                "reason": "Apply the user-selected random seed.",
            }
        if phase == "conversions":
            self.mock_phase = "inspect"
            return {"action": "call_tool", "tool_name": "plan_correlator_h5_conversions", "args": {}, "reason": "Plan any correlator data conversions."}
        if phase == "inspect":
            self.mock_phase = "build"
            return {"action": "call_tool", "tool_name": "inspect_correlator_h5_files", "args": {}, "reason": "Inspect correlator data inputs."}
        if phase == "build":
            self.mock_phase = "propose"
            return {"action": "call_tool", "tool_name": "build_quick_full_candidates", "args": {}, "reason": "Build quick and full manifest candidates."}
        if phase == "stage_completion":
            return {
                "action": "request_user_input",
                "reason": "The manifest is not the full canonical stage flow.",
                "args": {
                    "question_id": "stage.add_remaining",
                    "prompt": "This manifest is not a full canonical flow. Add extra downstream stages?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add downstream stages when inputs can be wired unambiguously."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest as a partial workflow."},
                    ],
                },
            }
        if phase == "parameter_completion":
            return {
                "action": "request_user_input",
                "reason": "A configured stage is missing required parameters or input roles.",
                "args": {
                    "question_id": "stage_params.missing",
                    "prompt": "A configured stage is missing required parameters. Add them before building manifests?",
                    "choices": [
                        {"label": "1", "value": "yes", "description": "Yes, add the missing parameters using explicit values or examples."},
                        {"label": "2", "value": "no", "description": "No, keep the manifest unchanged."},
                    ],
                },
            }
        if phase == "mock_revision":
            self.mock_phase = "build"
            note = self.last_revision or ""
            text = note.lower()
            suppressions = []
            if ("tau" in text or "pt3_tau_cuts" in text) and (
                "改回" in note or "恢复" in note or "撤回" in note or "undo" in text or "revert" in text
            ):
                suppressions.append("stages.correlator_analysis.defaults.pt3_tau_cuts")
            return {
                "action": "call_tool",
                "tool_name": "apply_manifest_patch_to_candidate",
                "args": {
                    "patches": "__mock_revision__",
                    "revision": note,
                    "suppress_full_expansions": suppressions,
                },
                "reason": "Apply the user's revision as candidate manifest patches.",
            }
        if phase == "blocked":
            self.mock_phase = "done"
            return {
                "action": "finish",
                "reason": "Configured stages still have missing parameters or input roles. No manifest files were written.",
                "args": {"error": True},
            }
        self.mock_phase = "done"
        return {"action": "propose_plan", "reason": "Present the latest validated candidate.", "args": {"summary": "Mock planning summary."}}

    def _latest_user_answer(self) -> Any:
        for message in reversed(self.messages):
            try:
                observation = json.loads(message["content"]).get("observation", {})
            except Exception:
                continue
            if observation.get("event") == "user_answer":
                return observation.get("value")
        return 1984


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _apply_user_answer_to_candidate(state: PlanAgentState, question_id: str, value: Any) -> dict[str, Any]:
    """Apply direct answers to manifest-path questions through the same patch guardrails."""
    if question_id == "stage.add_remaining":
        state.stage_completion_checked = True
        text = str(value).strip().lower()
        negative = text in {"no", "n", "false", "0"} or "keep" in text and "partial" in text or "不" in text and ("加" in text or "添加" in text)
        state.stage_completion_requested = not negative and text in {"yes", "y", "true", "1"}
        return {"event": "user_answer_not_applied", "question_id": question_id, "value": value, "reason": "stage completion preference recorded for the planning agent."}
    if question_id.startswith("stage_params."):
        state.parameter_completion_checked = True
        state.parameter_completion_requested = str(value).strip().lower() in {"yes", "y", "true", "1"}
        return {"event": "user_answer_not_applied", "question_id": question_id, "value": value, "reason": "stage parameter completion preference recorded for the planning agent."}
    match = re.fullmatch(r"inputs\.correlators\.\d+\.([A-Za-z_][A-Za-z0-9_]*)", question_id)
    if match and match.group(1) not in {
        "correlator_id",
        "correlator_type",
        "data_path",
        "ensemble",
        "hadron",
        "gfix",
        "source_operator",
        "sink_operator",
        "momentum",
        "lattice_spacing_fm",
        "volume",
        "current_operator",
        "bz_direction",
        "bT",
        "bz",
        "tsep",
    }:
        return {"event": "user_answer_not_applied", "question_id": question_id, "value": value, "reason": "question_id is not a manifest correlator field."}
    pointer = _json_pointer_from_question_id(question_id)
    if pointer is None:
        return {"event": "user_answer_not_applied", "question_id": question_id, "reason": "question_id is not a manifest path."}
    try:
        coerced = _coerce_user_answer_for_manifest_path(question_id, value)
    except (TypeError, ValueError):
        return {
            "event": "user_answer_not_applied",
            "question_id": question_id,
            "reason": f"Answer {value!r} could not be converted to the required manifest value type.",
        }
    op = "replace" if _get_dotted_path(state.candidate_payload, question_id) is not None else "add"
    observation = _run_planning_tool(
        state,
        "apply_manifest_patch_to_candidate",
        {
            "patches": [
                {
                    "op": op,
                    "path": pointer,
                    "value": coerced,
                    "note": "Applied user answer from planner question.",
                }
            ]
        },
    )
    observation["event"] = "user_answer_applied"
    observation["question_id"] = question_id
    observation["value"] = coerced
    return observation


def _mock_revision_patches(state: PlanAgentState, note: str) -> list[dict[str, Any]]:
    """Return deterministic mock patches so tests can exercise the agent patch path."""
    text = note.lower()
    payload = state.candidate_payload
    if "renormalization" in text or "重整" in note:
        stages = payload.get("stages", {})
        metadata = payload.get("metadata", {})
        order = list(metadata.get("stages", [])) if isinstance(metadata, dict) and isinstance(metadata.get("stages"), list) else []
        jobs = stages.get("correlator_analysis", {}).get("jobs", []) if isinstance(stages, dict) else []
        denominator = None
        targets: list[str] = []
        for job in jobs if isinstance(jobs, list) else []:
            if not isinstance(job, dict) or not isinstance(job.get("id"), str):
                continue
            job_id = job["id"]
            if "p0" in job_id:
                denominator = job_id
            elif re.search(r"p[1-9]", job_id):
                targets.append(job_id)
        denominator = denominator or (jobs[0]["id"] if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict) else "ca")
        targets = targets or [job["id"] for job in jobs[1:] if isinstance(job, dict) and isinstance(job.get("id"), str)]
        renorm_jobs = [
            {"id": target.replace("ca_", "rn_", 1) if target.startswith("ca_") else f"rn_{target}", "inputs": {"target": target, "denominator": denominator}}
            for target in targets
        ]
        if "renormalization" not in order:
            index = order.index("correlator_analysis") + 1 if "correlator_analysis" in order else len(order)
            order.insert(index, "renormalization")
        return [
            {"op": "replace", "path": "/metadata/stages", "value": order},
            {
                "op": "add",
                "path": "/stages/renormalization",
                "value": {
                    "defaults": {
                        "normalization": False,
                        "scheme": "hybrid_ratio",
                        "zs_fm": 0.1722,
                        "scheme_parameters": {"m0_gev": 0.0, "delta_m_gev": 0.0},
                    },
                    "jobs": renorm_jobs,
                },
            },
        ]
    if ("fit window" in text or "window" in text or "窗口" in note) and ("search" in text or "scan" in text or "多" in note or "加" in note):
        defaults = payload.get("stages", {}).get("correlator_analysis", {}).get("defaults", {})
        return [
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt2_windows",
                "value": _expand_pt2_windows(defaults.get("pt2_windows")),
                "note": "LLM expanded the fit-window search.",
            },
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt3_tau_cuts",
                "value": _expand_tau_cuts(defaults.get("pt3_tau_cuts")),
                "note": "LLM expanded the fit-window search.",
            },
        ]
    if ("tau" in text or "pt3_tau_cuts" in text) and ("改回" in note or "恢复" in note or "撤回" in note or "undo" in text or "revert" in text):
        original = _get_path_value(state.original_payload, "stages.correlator_analysis.defaults.pt3_tau_cuts")
        return [
            {
                "op": "replace",
                "path": "/stages/correlator_analysis/defaults/pt3_tau_cuts",
                "value": original,
                "note": "LLM reverted the tau-cut search.",
            }
        ]
    return []


def _run_planning_tool(state: PlanAgentState, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "load_manifest":
        quick_path, full_path = _planned_manifest_paths(state.manifest_path, state.candidate_payload)
        state.quick_path = quick_path
        state.full_path = full_path
        canonical_stages = ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"]
        metadata = state.candidate_payload.get("metadata", {})
        configured_stages = metadata.get("stages", []) if isinstance(metadata, dict) else []
        configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
        stage_parameter_gaps = _stage_parameter_gaps(state.candidate_payload)
        return {
            "tool_name": tool_name,
            "manifest": state.candidate_payload,
            "quick_manifest_path": str(quick_path),
            "full_manifest_path": str(full_path),
            "canonical_stage_flow": canonical_stages,
            "configured_stages": configured_stage_list,
            "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
            "stage_completion_question_required": configured_stage_list != canonical_stages,
            "stage_parameter_gaps": stage_parameter_gaps,
            "stage_parameter_question_required": bool(stage_parameter_gaps),
            "next_questions": _next_questions_for_state(state),
        }
    if tool_name == "check_manifest_draft":
        state.issues = check_manifest_draft(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "issues": _dataclass_json(state.issues), "next_questions": _next_questions_for_state(state)}
    if tool_name == "list_stage_parameter_gaps":
        return {"tool_name": tool_name, "stage_parameter_gaps": _stage_parameter_gaps(state.candidate_payload), "next_questions": _next_questions_for_state(state)}
    if tool_name == "inspect_correlator_h5_files":
        state.inspections = inspect_correlator_h5_files(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "h5_inspections": _dataclass_json(state.inspections)}
    if tool_name == "plan_correlator_h5_conversions":
        state.conversions = plan_correlator_h5_conversions(state.manifest_path, state.candidate_payload)
        return {"tool_name": tool_name, "planned_data_conversions": _dataclass_json(state.conversions)}
    if tool_name == "apply_correlator_conversion_mapping":
        correlator_id = str(args.get("correlator_id") or "")
        datasets = args.get("datasets")
        if not correlator_id or not isinstance(datasets, list) or not datasets:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Expected args format: {'correlator_id': '...', 'datasets': [{'source': 'array', 'target': '...', 'axis_order': [..], 'index': {'0': 0}, 'transpose': false}, ...]}. Do not pass source/target at top level.",
            }
        correlators = state.candidate_payload.get("inputs", {}).get("correlators", [])
        correlator = next((item for item in correlators if isinstance(item, dict) and str(item.get("correlator_id")) == correlator_id), None)
        mapping = next((item for item in state.conversions if item.correlator_id == correlator_id), None)
        if correlator is None or mapping is None:
            return {"tool_name": tool_name, "ok": False, "error": f"Unknown correlator conversion {correlator_id!r}."}
        source = Path(mapping.source_file)
        names = _dataset_names(source)
        targets = set(_standard_dataset_paths(correlator))
        cleaned = []
        seen_targets: set[str] = set()
        for item in datasets:
            if not isinstance(item, dict):
                return {"tool_name": tool_name, "ok": False, "error": "Each dataset mapping must be an object."}
            source_name = str(item.get("source") or "array")
            target = str(item.get("target") or "")
            if target not in targets:
                return {"tool_name": tool_name, "ok": False, "error": f"Target {target!r} is not one of the standard targets {sorted(targets)}."}
            if source_name not in names:
                return {"tool_name": tool_name, "ok": False, "error": f"Source {source_name!r} is not in {sorted(names)}."}
            if target in seen_targets:
                return {"tool_name": tool_name, "ok": False, "error": f"Target {target!r} is mapped more than once."}
            seen_targets.add(target)
            shape = list(names[source_name])
            original_shape = list(shape)
            fixed_axes = {int(axis) for axis in (item.get("index") or {})}
            for axis, index in sorted((item.get("index") or {}).items(), key=lambda pair: int(pair[0]), reverse=True):
                axis_i = int(axis)
                index_i = int(index)
                if axis_i < 0 or axis_i >= len(shape):
                    return {"tool_name": tool_name, "ok": False, "error": f"Index axis {axis_i} is out of bounds for source {source_name!r} shape {original_shape}."}
                if index_i < 0 or index_i >= shape[axis_i]:
                    return {"tool_name": tool_name, "ok": False, "error": f"Index {index_i} is out of bounds for axis {axis_i} of source {source_name!r} shape {original_shape}."}
                shape.pop(axis_i)
            if item.get("axis_order") is not None:
                axes = [int(axis) for axis in item["axis_order"]]
                if len(set(axes)) != len(axes):
                    return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} has duplicate axes."}
                if sorted(axes) == list(range(1, len(shape) + 1)):
                    axes = [axis - 1 for axis in axes]
                elif axes and max(axes) >= len(shape):
                    remaining_axes = [axis for axis in range(len(original_shape)) if axis not in fixed_axes]
                    if any(axis not in remaining_axes for axis in axes):
                        return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} is not compatible with index-fixed axes {sorted(fixed_axes)} and source shape {original_shape}."}
                    axes = [remaining_axes.index(axis) for axis in axes]
                if sorted(axes) != list(range(len(shape))):
                    return {"tool_name": tool_name, "ok": False, "error": f"axis_order {axes} is not a permutation of remaining axes for shape {shape}."}
                shape = [shape[axis] for axis in axes]
            if item.get("transpose"):
                shape = list(reversed(shape))
            if len(shape) != 2:
                return {"tool_name": tool_name, "ok": False, "error": f"Mapped dataset {target!r} has final shape {shape}; expected a 2D standard correlator dataset."}
            if correlator.get("correlator_type") == "3pt":
                match = re.search(r"/tsep(\d+)/", f"/{target}/")
                if match and shape[0] != int(match.group(1)) + 1:
                    return {"tool_name": tool_name, "ok": False, "error": f"Mapped 3pt dataset {target!r} has tau length {shape[0]}; expected tsep+1={int(match.group(1)) + 1}."}
            else:
                try:
                    temporal_extent = parse_volume(str(correlator.get("volume", "")))[1]
                except ValueError as exc:
                    return {"tool_name": tool_name, "ok": False, "error": str(exc)}
                if shape[0] != temporal_extent:
                    return {
                        "tool_name": tool_name,
                        "ok": False,
                        "error": (
                            f"Mapped 2pt dataset {target!r} has Lt={shape[0]}; "
                            f"expected {temporal_extent} from manifest volume."
                        ),
                    }
            out = {"source": source_name, "target": target, "transpose": bool(item.get("transpose", False))}
            if item.get("axis_order") is not None:
                out["axis_order"] = [int(axis) for axis in item["axis_order"]]
            if item.get("index") is not None:
                out["index"] = {str(axis): int(index) for axis, index in item["index"].items()}
            cleaned.append(out)
        if seen_targets != targets:
            missing = sorted(targets - seen_targets)
            extra = sorted(seen_targets - targets)
            return {"tool_name": tool_name, "ok": False, "error": f"Dataset mappings must cover every standard target exactly once. Missing={missing}; extra={extra}."}
        mapping.datasets = cleaned
        mapping.ambiguous = False
        mapping.reason = None
        state.quick = None
        state.full = None
        return {"tool_name": tool_name, "ok": True, "conversion": _dataclass_json(mapping)}
    if tool_name == "validate_candidate_manifest":
        ok, issues = validate_candidate_payload(state.manifest_path, state.candidate_payload)
        state.issues = issues
        return {"tool_name": tool_name, "ok": ok, "issues": _dataclass_json(issues)}
    if tool_name == "apply_manifest_patch_to_candidate":
        patches = args.get("patches", [])
        if patches == "__mock_revision__":
            patches = _mock_revision_patches(state, str(args.get("revision") or ""))
        if not isinstance(patches, list):
            return {"tool_name": tool_name, "ok": False, "error": "patches must be a list of JSON Patch objects."}
        try:
            candidate, edits = apply_manifest_json_patches(state.candidate_payload, patches)
        except ValueError as exc:
            return {"tool_name": tool_name, "ok": False, "error": str(exc)}
        ok, issues = validate_candidate_payload(state.manifest_path, candidate)
        correlators = candidate.get("inputs", {}).get("correlators", [])
        incomplete_correlator = isinstance(correlators, list) and any(isinstance(item, dict) and not _standard_dataset_paths(item) for item in correlators)
        incomplete_stage_params = bool(_stage_parameter_gaps(candidate))
        incomplete_kernel = any(
            issue.severity == "error" and ("kernel_id" in issue.message or "kernel_parameters" in issue.message or "zs_fm" in issue.message)
            for issue in issues
        )
        if not ok and not incomplete_correlator and not incomplete_stage_params and not incomplete_kernel and any(issue.severity == "error" and "Field required" not in issue.message for issue in issues):
            return {"tool_name": tool_name, "ok": False, "issues": _dataclass_json(issues), "edits": edits}
        state.candidate_payload = candidate
        state.manifest_edits.extend(edits)
        suppressions = args.get("suppress_full_expansions")
        if isinstance(suppressions, list):
            state.suppressed_full_expansions.update(str(item) for item in suppressions if isinstance(item, str))
        state.issues = issues
        state.quick = None
        state.full = None
        return {"tool_name": tool_name, "ok": True, "candidate_complete": ok, "edits": edits, "issues": _dataclass_json(issues)}
    if tool_name == "build_quick_full_candidates":
        canonical_stages = ["correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation", "review"]
        metadata = state.candidate_payload.get("metadata", {})
        configured_stages = metadata.get("stages", []) if isinstance(metadata, dict) else []
        configured_stage_list = [stage for stage in configured_stages if isinstance(stage, str)] if isinstance(configured_stages, list) else []
        original_metadata = state.original_payload.get("metadata", {})
        original_stages = original_metadata.get("stages", []) if isinstance(original_metadata, dict) else []
        original_stage_list = [stage for stage in original_stages if isinstance(stage, str)] if isinstance(original_stages, list) else []
        if configured_stage_list != canonical_stages and not state.stage_completion_checked:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "This manifest is not the full canonical stage flow. Ask the user first with question_id='stage.add_remaining' whether to add extra downstream stages; allow a free-form subset such as renormalization and fourier_transform.",
                "canonical_stage_flow": canonical_stages,
                "configured_stages": configured_stage_list,
                "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
                "next_questions": _next_questions_for_state(state),
            }
        if state.stage_completion_requested and configured_stage_list == original_stage_list:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "The user answered yes to adding stages, but metadata.stages has not changed. Patch the requested stages first or ask a follow-up question.",
                "configured_stages": configured_stage_list,
                "missing_canonical_stages": [stage for stage in canonical_stages if stage not in configured_stage_list],
            }
        parameter_gaps = _stage_parameter_gaps(state.candidate_payload)
        if parameter_gaps and not state.parameter_completion_checked:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Configured stages have missing parameters or input roles. Explain the gaps and ask a Yes/No question first with question_id like 'stage_params.<stage>.<job_id>'.",
                "stage_parameter_gaps": parameter_gaps,
                "next_questions": _next_questions_for_state(state),
            }
        if parameter_gaps:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Configured stages still have missing parameters or input roles. Patch the missing manifest paths first, ask for concrete values, or quit without writing files.",
                "stage_parameter_gaps": parameter_gaps,
                "next_questions": _next_questions_for_state(state),
            }
        ambiguous = [item for item in state.conversions if item.ambiguous]
        if ambiguous:
            return {
                "tool_name": tool_name,
                "ok": False,
                "error": "Ambiguous correlator conversions must be resolved before building quick/full manifests.",
                "ambiguous_conversions": _dataclass_json(ambiguous),
            }
        quick, full, edits = build_repaired_manifests(
            state.manifest_path,
            state.candidate_payload,
            state.conversions,
            suppressed_full_expansions=state.suppressed_full_expansions,
        )
        quick_issues = _strict_manifest_issues(quick)
        full_issues = _strict_manifest_issues(full)
        if quick_issues or full_issues:
            return {
                "tool_name": tool_name,
                "ok": False,
                "quick_issues": _dataclass_json(quick_issues),
                "full_issues": _dataclass_json(full_issues),
            }
        state.quick = quick
        state.full = full
        state.quick_path, state.full_path = _planned_manifest_paths(state.manifest_path, state.candidate_payload)
        for edit in edits:
            _merge_revision_edits(state.manifest_edits, [edit])
        return {
            "tool_name": tool_name,
            "ok": True,
            "deterministic_manifest_edits": edits,
            "quick_manifest_path": str(state.quick_path),
            "full_manifest_path": str(state.full_path),
        }
    return {"tool_name": tool_name, "ok": False, "error": f"Unknown planning tool: {tool_name}"}


def run_interactive_plan(
    manifest_path: Path,
    *,
    backend: str,
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> PlanRunResult | None:
    """Run the terminal planning loop under LLM/tool control."""
    payload, manifest_text = load_relaxed_manifest(manifest_path)
    state = PlanAgentState(
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        original_payload=copy.deepcopy(payload),
        candidate_payload=copy.deepcopy(payload),
    )
    session = _PlanAgentSession(
        backend=backend,
        manifest_path=manifest_path,
        manifest_text=manifest_text,
        api_key=api_key,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    output_func(BANNER)

    for _ in range(60):
        action = session.decide()
        action_type = action.get("action")
        reason = str(action.get("reason", ""))
        args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}

        if action_type == "call_tool":
            tool_name = str(action.get("tool_name") or "")
            if not tool_name:
                session.observe({"event": "invalid_action", "action": action, "error": "call_tool requires tool_name."})
                continue
            observation = _run_planning_tool(state, tool_name, args)
            session.observe(observation)
            next_questions = observation.get("next_questions")
            if tool_name != "load_manifest" and isinstance(next_questions, list) and next_questions:
                question = next_questions[0]
                if _valid_plan_agent_question(question):
                    skip_path = question.get("skip_if_present")
                    if isinstance(skip_path, str) and _get_dotted_path(state.candidate_payload, skip_path) is not None:
                        session.observe({"event": "question_skipped", "reason": f"{skip_path} is already present."})
                        continue
                    answer = _ask_plan_agent_question(question, input_func, output_func)
                    if answer == "quit":
                        output_func("Plan cancelled; no files were written.")
                        return None
                    question_id = str(question.get("question_id"))
                    session.observe({"event": "user_answer", "question_id": question_id, "value": answer})
                    applied = _apply_user_answer_to_candidate(state, question_id, answer)
                    session.observe(applied)
            continue

        if action_type == "request_user_input":
            if not _valid_plan_agent_question(args):
                session.observe(
                    {
                        "event": "user_input_rejected",
                        "error": "request_user_input requires args.question_id and a concrete args.prompt. Do not ask the terminal until you can state the exact question.",
                        "action": action,
                    }
                )
                continue
            skip_path = args.get("skip_if_present")
            if isinstance(skip_path, str) and _get_dotted_path(state.candidate_payload, skip_path) is not None:
                session.observe({"event": "question_skipped", "reason": f"{skip_path} is already present."})
                continue
            raw_question_id = str(args.get("question_id"))
            if raw_question_id == "stage.add_remaining" and state.stage_completion_checked:
                session.observe({"event": "question_skipped", "reason": "stage.add_remaining was already answered."})
                continue
            if raw_question_id.startswith("stage_params.") and state.parameter_completion_checked:
                session.observe({"event": "question_skipped", "reason": f"{raw_question_id} was already answered."})
                continue
            answer = _ask_plan_agent_question(args, input_func, output_func)
            if answer == "quit":
                output_func("Plan cancelled; no files were written.")
                return None
            question_id = _manifest_question_id_from_user_input_action(args, reason) or str(args.get("question_id"))
            normalized_question_id = re.sub(r"\[(\d+)\]", r".\1", question_id)
            if _json_pointer_from_question_id(question_id) is None:
                text = f"{args.get('prompt', '')}\n{reason}"
                match = re.search(
                    r"correlator\s+['\"]([^'\"]+)['\"].*?['\"](momentum|source_operator|sink_operator|current_operator|bz_direction|volume|bT|bz|tsep|lattice_spacing_fm)['\"]",
                    text,
                    flags=re.I | re.S,
                )
                correlators = state.candidate_payload.get("inputs", {}).get("correlators", [])
                if match and isinstance(correlators, list):
                    for index, item in enumerate(correlators):
                        if isinstance(item, dict) and str(item.get("correlator_id")) == match.group(1):
                            question_id = f"inputs.correlators.{index}.{match.group(2)}"
                            break
                if _json_pointer_from_question_id(question_id) is None:
                    text_lower = text.lower()
                    for gap in _stage_parameter_gaps(state.candidate_payload):
                        stage = str(gap.get("stage", ""))
                        job_id = str(gap.get("job_id", ""))
                        parameter = str(gap.get("parameter", ""))
                        if stage.lower() in text_lower and job_id.lower() in text_lower and parameter.lower() in text_lower:
                            question_id = str(gap.get("path"))
                            break
                if _json_pointer_from_question_id(question_id) is None:
                    kernels = state.candidate_payload.get("inputs", {}).get("kernels", [])
                    text_lower = text.lower()
                    if isinstance(kernels, list) and len(kernels) == 1:
                        if "zs_fm" in text_lower:
                            if "renormalization" in text_lower:
                                question_id = "stages.renormalization.defaults.zs_fm"
                            else:
                                question_id = "stages.perturbative_matching.defaults.zs_fm"
                        elif "kernel_id" in text_lower:
                            question_id = "inputs.kernels.0.kernel_id"
            session.observe({"event": "user_answer", "question_id": question_id, "value": answer})
            applied = _apply_user_answer_to_candidate(state, question_id, answer)
            session.observe(applied)
            continue

        if action_type == "propose_plan":
            if state.quick is None or state.full is None or state.quick_path is None or state.full_path is None:
                session.observe(
                    {
                        "event": "proposal_rejected",
                        "error": "No validated quick/full manifest candidates are available. Call build_quick_full_candidates first.",
                    }
                )
                continue
            proposal = PlanProposal(
                report=str(args.get("summary") or reason or "Planning proposal is ready."),
                manifest_edits=state.manifest_edits,
                quick_manifest_path=str(state.quick_path),
                full_manifest_path=str(state.full_path),
                data_conversions=state.conversions,
            )
            output_func(_render_proposal(proposal, state.issues))
            answer = input_func("Accept these modifications and write files? [a]ccept/[r]evise/[q]uit: ").strip().lower()
            if answer in {"a", "accept", "y", "yes"}:
                result = write_planned_outputs(
                    state.original_payload,
                    state.quick,
                    state.full,
                    state.conversions,
                    state.quick_path,
                    state.full_path,
                )
                output_func(_render_written_summary(result))
                return result
            if answer in {"r", "revise"}:
                note = input_func("Revision instruction: ").strip()
                if note:
                    if "stage" in note.lower() or "阶段" in note:
                        state.stage_completion_checked = True
                    session.observe({"event": "user_revision", "text": note})
                output_func("")
                output_func("")
                continue
            if answer in {"q", "quit", "n", "no"}:
                output_func("Plan cancelled; no files were written.")
                return None
            output_func("Please enter accept, revise, or quit.")
            session.observe({"event": "proposal_response_invalid", "value": answer})
            continue

        if action_type == "finish":
            if args.get("error"):
                raise ValueError(str(reason or "Plan finished with a blocking error."))
            output_func(str(reason or "Plan finished without writing files."))
            return None

        session.observe({"event": "invalid_action", "action": action, "error": "Unknown planning action."})

    raise ValueError("Planning agent exceeded the maximum number of action steps.")
