from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from lamet_agent.cli import app
from lamet_agent.planning import (
    PlanAgentState,
    _run_planning_tool,
    apply_manifest_json_patches,
    check_manifest_draft,
    convert_correlator_h5,
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
                    "kind": "2pt",
                    "data_path": data_path,
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                }
            ],
            "artifacts": [],
            "kernels": [
                {
                    "stage": "perturbative_matching",
                    "kernel_id": "CG_gt_PDF_hybrid",
                    "kernel_path": "src/lamet_agent/kernels.py",
                    "scheme": "ratio",
                    "kernel_parameters": {"zs_fm": 0.2},
                }
            ],
        },
        "stages": {
            "correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2"]}]},
            "renormalization": {
                "defaults": {"scheme": "hybrid_ratio", "scheme_parameters": {"zs_fm": 0.2}},
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
                    "kind": "2pt",
                    "data_path": "data/raw_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/raw_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0, 1],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
    }
    with h5py.File(data_dir / "raw_2pt.h5", "w") as h5f:
        h5f.create_dataset("raw_pt2", data=pt2_cfg_time)
    with h5py.File(data_dir / "raw_3pt.h5", "w") as h5f:
        h5f.create_dataset("raw_z0", data=pt3_cfg_tau_z0)
        h5f.create_dataset("raw_z1", data=pt3_cfg_tau_z1)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    conversions = plan_correlator_h5_conversions(path, payload)
    assert len(conversions) == 2
    assert all(not item.ambiguous for item in conversions)
    for conversion in conversions:
        convert_correlator_h5(conversion)

    c2_output = next(item for item in conversions if item.correlator_id == "c2").output_file
    c3_output = next(item for item in conversions if item.correlator_id == "c3").output_file
    assert np.array_equal(_read_2pt(c2_output, source_sink="SS", gamma="5", momentum="PX0PY0PZ0"), pt2_cfg_time)
    assert np.array_equal(
        _read_3pt(
            c3_output,
            source_sink="SS",
            gamma="T",
            momentum="PX0PY0PZ0",
            b_dir="b_X",
            eta="eta0",
            bt="bT0",
            bz="bz1",
            tsep=3,
        ),
        pt3_cfg_tau_z1,
    )


def test_cli_plan_mock_accept_writes_quick_and_full_manifests(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {"nstate": [2, 3]}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="a\n")

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
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {"nstate": [2, 3], "model_average": True},
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="1\na\n")

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
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
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
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}]}},
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


def test_cli_plan_revision_expands_fit_window_search(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "repo"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "c2.h5", "w") as h5f:
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
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
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="r\n帮我多加几个 fit window 的搜索吧\na\n")

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
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=np.ones((5, 3)))
    with h5py.File(data_dir / "c3.h5", "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/c2.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "c3",
                    "kind": "3pt",
                    "data_path": "data/c3.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
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
                "jobs": [{"id": "ca", "correlator_ids": ["c2", "c3"]}],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plan", str(manifest), "--backend", "mock"],
        input="r\n帮我多加几个 fit window 的搜索吧\nr\ntau cuts 改回去吧\na\n",
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
                h5f.create_dataset(f"SS/5/{momentum}", data=np.ones((5, 3)))
            else:
                momentum = "PX0PY0PZ0" if name.startswith("p0") else "PX5PY0PZ0"
                h5f.create_dataset(f"SS/T/{momentum}/b_X/eta0/bT0/bz0", data=np.ones((4, 3)))
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
                    "kind": "2pt",
                    "data_path": "data/p0_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "p0_3pt",
                    "kind": "3pt",
                    "data_path": "data/p0_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX0PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 0.0,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
                {
                    "correlator_id": "p5_2pt",
                    "kind": "2pt",
                    "data_path": "data/p5_2pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX5PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 2.15,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                },
                {
                    "correlator_id": "p5_3pt",
                    "kind": "3pt",
                    "data_path": "data/p5_3pt.h5",
                    "ensemble": "E",
                    "hadron": "pion",
                    "gfix": "CG",
                    "source_sink": "SS",
                    "momentum": "PX5PY0PZ0",
                    "a_fm": 0.1,
                    "pz_gev": 2.15,
                    "src_gamma": "5",
                    "sink_gamma": "5",
                    "current_gamma": "T",
                    "z_direction": "X",
                    "eta": "eta0",
                    "bt": [0],
                    "bz": [0],
                    "tsep": 3,
                },
            ],
            "artifacts": [],
            "kernels": [],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {"fit_scope": ["ratio"]},
                "jobs": [
                    {"id": "ca_p0_fh", "correlator_ids": ["p0_2pt", "p0_3pt"]},
                    {"id": "ca_p5_fh", "correlator_ids": ["p5_2pt", "p5_3pt"]},
                ],
            }
        },
    }
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["plan", str(manifest), "--backend", "mock"], input="r\n加上 renormalization 的 stage 吧\na\n")

    assert result.exit_code == 0, result.output
    full = json.loads((root / "artifacts" / "plan_manifests" / "draft.full.json").read_text(encoding="utf-8"))
    assert full["metadata"]["stages"] == ["correlator_analysis", "renormalization"]
    assert full["stages"]["renormalization"]["defaults"]["scheme"] == "hybrid_ratio"
    assert full["stages"]["renormalization"]["jobs"] == [
        {"id": "rn_p5_fh", "inputs": {"target": "ca_p5_fh", "denominator": "ca_p0_fh"}}
    ]
