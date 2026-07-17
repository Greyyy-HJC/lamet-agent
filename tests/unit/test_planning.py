from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from lamet_agent.cli import app
from lamet_agent.planning import (
    PlanAgentState,
    _ask_plan_agent_question,
    _apply_user_answer_to_candidate,
    _run_planning_tool,
    _stage_parameter_gaps,
    _next_questions_for_state,
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
from lamet_agent.stages.correlator.functions import _read_2pt, _read_3pt


def _write_kernel(root: Path) -> None:
    (root / "src" / "lamet_agent").mkdir(parents=True)
    (root / "src" / "lamet_agent" / "kernels.py").write_text("# test kernel\n", encoding="utf-8")


def _minimal_payload(root: Path, data_path: str = "data/c2.h5") -> dict:
    return {
        "metadata": {
            "run_id": "demo",
            "root_directory": str(root),
            "artifacts_directory": "artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
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
                    "kernel_path": "src/lamet_agent/kernels.py",
                    "scheme": "ratio",
                    "kernel_parameters": {},
                }
            ],
        },
        "stages": {
            "correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2"], "params": {"momentum": "PX0PY0PZ0"}}]},
            "renormalization": {
                "defaults": {"scheme": "hybrid_ratio", "zs_fm": 0.2},
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

    listed = _run_planning_tool(state, "list_stage_parameter_gaps", {})
    gaps = listed["stage_parameter_gaps"]
    assert any(gap["parameter"] == "order" and '"LA"' in gap["suggested_fix"] for gap in gaps)
    assert any(gap["parameter"] == "y_grid" and "start" in gap["suggested_fix"] for gap in gaps)

    blocked = _run_planning_tool(state, "build_quick_full_candidates", {})
    assert blocked["ok"] is False
    assert "missing parameters" in blocked["error"]
    assert blocked["next_questions"][0]["question_id"] == "stage_params.fourier_transform.ft"


def test_planning_reports_legacy_zs_locations_and_flat_parameter_gaps(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write_kernel(root)
    payload = _minimal_payload(root)
    payload["inputs"]["kernels"][0]["kernel_parameters"] = {"zs_fm": 0.2}
    payload["stages"]["renormalization"]["defaults"].pop("zs_fm")
    payload["stages"]["renormalization"]["defaults"]["scheme_parameters"] = {"zs_fm": 0.2}

    issues = check_manifest_draft(tmp_path / "draft.json", payload)
    gaps = _stage_parameter_gaps(payload)

    issue_paths = {issue.manifest_path for issue in issues}
    assert "inputs.kernels[0].kernel_parameters.zs_fm" in issue_paths
    assert "stages.renormalization.defaults.scheme_parameters.zs_fm" in issue_paths
    assert any(gap["path"] == "stages.renormalization.defaults.zs_fm" for gap in gaps)


def test_planning_accepts_ratio_without_hybrid_parameters(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["stages"]["renormalization"]["defaults"] = {"scheme": "ratio"}

    gaps = _stage_parameter_gaps(payload)

    assert not any(gap["stage"] == "renormalization" for gap in gaps)


def test_planning_distinguishes_hybrid_self_renormalization_fit_jobs(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["inputs"]["kernels"] = [{
        "stage": "renormalization",
        "kernel_id": "ZMSbar_pdf",
        "kernel_path": "src/lamet_agent/kernels.py",
        "scheme": "hybrid_self_renormalization",
        "kernel_parameters": {},
    }]
    payload["stages"]["renormalization"] = {
        "defaults": {"scheme": "hybrid_self_renormalization"},
        "jobs": [
            {
                "id": "rn_fit",
                "inputs": {"reference": "ca"},
                "params": {"scheme_parameters": {"LambdaQCD_gev": 0.1, "d": -0.08183}},
            }
        ],
    }

    gaps = _stage_parameter_gaps(payload)

    assert not any(gap["stage"] == "renormalization" for gap in gaps)


def test_plan_load_manifest_reports_deterministic_random_seed_question(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["metadata"].pop("random_seed", None)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    loaded = _run_planning_tool(state, "load_manifest", {})

    assert loaded["next_questions"][0]["question_id"] == "metadata.random_seed"


def test_plan_reports_correlator_metadata_question_before_ambiguous_paths(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    payload["inputs"]["correlators"][0].pop("momentum", None)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    loaded = _run_planning_tool(state, "load_manifest", {})

    assert loaded["next_questions"][0]["question_id"] == "inputs.correlators.0.momentum"


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
            "defaults": {"mu": 2.0, "component": "re"},
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
            "defaults": {"momentum_gev": 2.15, "mu": 2.0, "component": "re"},
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


def test_plan_stage_subset_answer_does_not_trigger_full_stage_gate(tmp_path: Path) -> None:
    payload = _minimal_payload(tmp_path)
    state = PlanAgentState(tmp_path / "draft.json", "", payload, payload)

    result = state.stage_completion_checked
    answer = _run_planning_tool(
        state,
        "load_manifest",
        {},
    )
    assert answer["stage_completion_question_required"] is True
    applied = _apply_user_answer_to_candidate(state, "stage.add_remaining", "I only want renormalization and fourier_transform")
    assert applied["event"] == "user_answer_not_applied"
    assert state.stage_completion_checked is True
    assert state.stage_completion_requested is False
    assert result is False


def test_plan_stage_params_question_without_choices_accepts_free_text() -> None:
    answer = _ask_plan_agent_question(
        {"question_id": "stage_params.fourier_transform.ft", "prompt": "Choose Fourier order."},
        input_func=lambda prompt: "LA",
        output_func=lambda text: None,
    )

    assert answer == "LA"


def test_planner_requests_missing_bz_direction_for_3pt() -> None:
    payload = {
        "metadata": {"random_seed": 1984, "stages": []},
        "inputs": {
            "correlators": [
                {
                    "correlator_id": "c3",
                    "correlator_type": "3pt",
                    "source_operator": "g5",
                    "sink_operator": "g5",
                    "current_operator": "gT_nonlocal",
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0, 1],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
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


            "current_operator": "gT_nonlocal", "bz_direction": "Z",


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


            "current_operator": "gT_nonlocal", "bz_direction": "Z",


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


def test_cli_plan_mock_accept_writes_quick_and_full_manifests(tmp_path: Path) -> None:
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {"nstate": [2, 3]}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="2\na\n")

    assert result.exit_code == 0, result.output
    quick_path = root / "artifacts" / "plan_manifests" / "draft.quick.json"
    full_path = root / "artifacts" / "plan_manifests" / "draft.full.json"
    assert quick_path.is_file()
    assert full_path.is_file()
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    assert quick["stages"]["correlator_analysis"]["defaults"]["nstate"] == [2]
    assert full["metadata"]["sample_error_mode"] == "covariance"
    assert full["stages"]["correlator_analysis"]["defaults"]["model_average"] is True


def test_cli_plan_asks_missing_random_seed_once_and_applies_answer(tmp_path: Path) -> None:
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


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
                "defaults": {"nstate": [2, 3], "model_average": True},
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="1\n2\na\n")

    assert result.exit_code == 0, result.output
    assert "metadata.random_seed is required" in result.output
    quick = json.loads((root / "artifacts" / "plan_manifests" / "draft.quick.json").read_text(encoding="utf-8"))
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["random_seed"] == 1984
    assert quick["metadata"]["random_seed"] == 1984
    assert quick["metadata"]["resample_mode"] == "jk"
    assert quick["metadata"]["sample_error_mode"] == "mean"
    assert quick["stages"]["correlator_analysis"]["defaults"]["model_average"] is False
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
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
            {"action": "propose_plan", "reason": "Ready.", "args": {"summary": "Ready after rejecting malformed question."}},
        ]
    )

    def fake_request_llm_text(**kwargs):
        del kwargs
        return json.dumps(next(actions))

    monkeypatch.setattr("lamet_agent.planning.request_llm_text", fake_request_llm_text)
    outputs: list[str] = []

    result = run_interactive_plan(
        manifest,
        backend="api",
        provider="deepseek",
        model_name="deepseek-chat",
        api_key="test",
        input_func=lambda prompt: "a",
        output_func=outputs.append,
    )

    assert result is not None
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


                    "bT": [0],
                    "bz": [0],
                    "tsep": [3],
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}]}},
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
            {"action": "propose_plan", "reason": "Ready.", "args": {"summary": "Ready after user seed answer."}},
        ]
    )

    def fake_request_llm_text(**kwargs):
        del kwargs
        return json.dumps(next(actions))

    answers = iter(["1999", "no", "a"])
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


def test_cli_plan_revision_expands_fit_window_search(tmp_path: Path) -> None:
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


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
                    "pt2_windows": [{"tmin": 3, "tmax": 12}, {"tmin": 4, "tmax": 12}],
                    "pt3_tau_cuts": [2, 3],
                },
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"], "params": {"momentum": "PX0PY0PZ0"}}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="2\nr\n帮我多加几个 fit window 的搜索吧\n2\na\n")

    assert result.exit_code == 0, result.output
    assert "LLM expanded the fit-window search" in result.output
    assert "stages.correlator_analysis.defaults.pt2_windows" in result.output
    full_path = root / "artifacts" / "plan_manifests" / "draft.full.json"
    full = json.loads(full_path.read_text(encoding="utf-8"))
    defaults = full["stages"]["correlator_analysis"]["defaults"]
    assert {"tmin": 2, "tmax": 12} in defaults["pt2_windows"]
    assert {"tmin": 6, "tmax": 12} in defaults["pt2_windows"]
    assert defaults["pt3_tau_cuts"] == [2, 3, 4, 5, 6, 7]
    assert defaults["model_average"] is True
    assert full["metadata"]["sample_error_mode"] == "covariance"
    assert "Quick manifest changes:" in result.output
    assert "Full manifest changes:" in result.output


def test_cli_plan_revision_can_revert_tau_cuts_after_broadening(tmp_path: Path) -> None:
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


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
                    "pt2_windows": [{"tmin": 3, "tmax": 12}, {"tmin": 4, "tmax": 12}],
                    "pt3_tau_cuts": [2, 3],
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
        ["plan", str(manifest), "--backend", "mock"],
        input="2\nr\n帮我多加几个 fit window 的搜索吧\nr\ntau cuts 改回去吧\na\n",
    )

    assert result.exit_code == 0, result.output
    assert "LLM reverted the tau-cut search" in result.output
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    defaults = full["stages"]["correlator_analysis"]["defaults"]
    assert defaults["pt3_tau_cuts"] == [2, 3]
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


def test_cli_plan_mock_revision_adds_renormalization_stage_from_chinese_instruction(tmp_path: Path) -> None:
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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


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


                    "current_operator": "gT_nonlocal", "bz_direction": "Z",


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
                "defaults": {"fit_scope": ["ratio"]},
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
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="2\nr\n加上 renormalization 的 stage 吧\na\n")

    assert result.exit_code == 0, result.output
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["stages"] == ["correlator_analysis", "renormalization"]
    assert full["stages"]["renormalization"]["defaults"]["scheme"] == "hybrid_ratio"
    assert full["stages"]["renormalization"]["jobs"] == [
        {"id": "rn_p5_fh", "inputs": {"target": "ca_p5_fh", "denominator": "ca_p0_fh"}}
    ]
