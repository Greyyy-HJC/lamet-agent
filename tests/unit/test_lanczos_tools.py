"""Tests for the correlator-stage Lanczos analysis path."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from lamet_agent.__main__ import app
from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import (
    prepare_tool_args,
    required_job_tool_sequence,
    resolve_job_tools,
    validate_stage_inputs,
)
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.manifest_params import resolve_stage_params
from lamet_agent.planning.agent import run_interactive_plan
from lamet_agent.planning.core import _stage_parameter_gaps
from lamet_agent.stages.correlator import functions as correlator_functions
from lamet_agent.stages.correlator.functions import (
    STAGE_TOOLS,
    inspect_lanczos_inputs,
    run_lanczos_analysis,
)
from lamet_agent.stages.correlator.lanczos import (
    analyze_threept,
    analyze_twopt,
    median_threept_matrix,
    median_twopt_energies,
    plan_tsep_tau_conversion,
)


def _exact_correlators(n_cfg: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = np.asarray([0.25, 0.7])
    overlaps_squared = np.asarray([1.0, 0.3])
    transfer_values = np.exp(-energies)
    time = np.arange(8)
    c2_mean = np.sum(
        overlaps_squared[:, None] * np.exp(-energies[:, None] * time), axis=0
    )
    current = np.asarray([[0.8, 0.1], [0.1, 0.4]])
    overlaps = np.sqrt(overlaps_squared)
    c3_mean = np.empty((3, 3), dtype=float)
    for sigma in range(3):
        for tau in range(3):
            c3_mean[sigma, tau] = np.sum(
                overlaps[:, None]
                * transfer_values[:, None] ** sigma
                * current
                * overlaps[None, :]
                * transfer_values[None, :] ** tau
            )
    return (
        np.tile(c2_mean, (n_cfg, 1)),
        np.tile(c3_mean, (n_cfg, 1, 1)),
        current,
    )


def _write_ordinary_h5(tmp_path: Path) -> tuple[str, str, np.ndarray]:
    n_cfg = 6
    energies = np.asarray([0.25, 0.7])
    overlaps_squared = np.asarray([1.0, 0.3])
    overlaps = np.sqrt(overlaps_squared)
    current = np.asarray([[0.8 + 0.25j, 0.1 - 0.05j], [0.1 + 0.02j, 0.4 - 0.1j]])
    time = np.arange(14)
    c2 = np.tile(
        np.sum(overlaps_squared[:, None] * np.exp(-energies[:, None] * time), axis=0),
        (n_cfg, 1),
    )
    pt2_path = tmp_path / "ordinary_c2.h5"
    pt3_path = tmp_path / "ordinary_c3.h5"
    with h5py.File(pt2_path, "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=c2.T)
    with h5py.File(pt3_path, "w") as h5f:
        for tsep in (4, 6, 8, 12):
            values = np.empty((tsep + 1, n_cfg), dtype=complex)
            for tau in range(tsep + 1):
                sigma = tsep - tau
                mean = np.sum(
                    overlaps[:, None]
                    * np.exp(-energies[:, None] * sigma)
                    * current
                    * overlaps[None, :]
                    * np.exp(-energies[None, :] * tau)
                )
                values[tau] = mean
            h5f.create_dataset(
                f"g5/g5/J/PX0PY0PZ0/tsep{tsep}/bT0/bz0", data=values
            )
    return str(pt2_path), str(pt3_path), current


def _manifest(
    pt2_path: str,
    pt3_path: str,
    *,
    tseps: list[int] | None = None,
    volume: str = "S8T14",
) -> AnalysisManifest:
    declared_tseps = [4, 6, 8, 12] if tseps is None else tseps
    return AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "lanczos",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 3,
                "stages": ["correlator_analysis"],
            },
            "inputs": {
                "correlators": [
                    {
                        "correlator_id": "c2",
                        "correlator_type": "2pt",
                        "data_path": pt2_path,
                        "ensemble": "toy",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "volume": volume,
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                    },
                    {
                        "correlator_id": "c3",
                        "correlator_type": "3pt",
                        "data_path": pt3_path,
                        "ensemble": "toy",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "current_operator": "J",
                        "polarization": "unpolarized",
                        "bz_direction": "Z",
                        "volume": volume,
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                        "bT": [0],
                        "bz": [0],
                        "tsep": declared_tseps,
                    },
                ],
                "artifacts": [],
                "kernels": [],
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {
                        "analysis_method": "lanczos",
                        "component": "re",
                        "fit_scope": ["3pt_matrix"],
                        "fitting_form": "Breit",
                        "nstate": [2],
                        "lanczos_inner_samples": 4,
                    },
                    "jobs": [
                        {
                            "id": "matrix",
                            "correlator_ids": ["c2", "c3"],
                            "params": {"momentum": "PX0PY0PZ0"},
                        }
                    ],
                }
            },
        }
    )


def test_exact_lanczos_recovers_two_state_spectrum_and_matrix() -> None:
    c2, c3, current = _exact_correlators()
    twopt = analyze_twopt(c2, 6, seed=0, max_iterations=3)
    energies = median_twopt_energies(twopt, max_states=2)
    assert energies[-1] == pytest.approx([0.25, 0.7])

    threept = analyze_threept(c3, c2, c2, 6, seed=0, max_iterations=2)
    matrix = median_threept_matrix(threept, iteration=2, max_states=2)
    assert matrix == pytest.approx(current)


def test_standard_manifest_contract_and_lanczos_tool_routing(tmp_path: Path) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path)
    job = manifest.stages["correlator_analysis"].jobs[0]
    params = resolve_stage_params(
        "correlator_analysis", manifest.stages["correlator_analysis"].defaults, job.params
    )
    assert params["lanczos_precision"] == 0
    assert "lanczos_iterations" not in params
    manifest.metadata.workers = 3
    assert validate_stage_inputs("correlator_analysis", manifest, job) == []
    assert set(resolve_job_tools("correlator_analysis", job, params, stage_tools=STAGE_TOOLS)) == {
        "inspect_lanczos_inputs",
        "run_lanczos_analysis",
    }
    assert required_job_tool_sequence("correlator_analysis", job, params) == (
        "inspect_lanczos_inputs",
        "run_lanczos_analysis",
    )
    args = prepare_tool_args(
        "run_lanczos_analysis",
        {},
        manifest=manifest,
        stage="correlator_analysis",
        job=job,
        effective_params=params,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert args["pt3_paths"] == {str(tsep): pt3_path for tsep in (4, 6, 8, 12)}
    assert args["tsep_ls"] == [4, 6, 8, 12]
    assert args["z_values"] == [0]
    assert "save_path" not in args
    assert args["artifacts_dir"] == str(tmp_path / "artifacts")
    assert args["job_id"] == "matrix"
    assert args["workers"] == 3


def test_lanczos_precision_zero_prints_nonblocking_plan_and_validate_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path)
    manifest.metadata.root_directory = str(Path.cwd())
    manifest_path = tmp_path / "lanczos.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2),
        encoding="utf-8",
    )

    validation = CliRunner().invoke(app, ["validate", str(manifest_path)])

    assert validation.exit_code == 0, validation.output
    assert '"status": "valid"' in validation.output
    assert "lanczos_precision=0" in validation.output
    assert "NumPy double precision" in validation.output

    monkeypatch.setattr(
        "lamet_agent.planning.agent._PlanAgentSession.decide",
        lambda _self: {
            "action": "finish",
            "reason": "Warning smoke test complete.",
            "args": {},
        },
    )
    outputs: list[str] = []

    result = run_interactive_plan(
        manifest_path,
        backend="mock",
        output_func=outputs.append,
    )

    assert result is None
    assert "lanczos_precision=0" in "\n".join(outputs)
    assert "NumPy double precision" in "\n".join(outputs)


def test_explicit_lanczos_precision_suppresses_double_precision_warning(
    tmp_path: Path,
) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path)
    manifest.metadata.root_directory = str(Path.cwd())
    manifest.stages["correlator_analysis"].defaults["lanczos_precision"] = 100
    manifest_path = tmp_path / "lanczos_high_precision.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2),
        encoding="utf-8",
    )

    validation = CliRunner().invoke(app, ["validate", str(manifest_path)])

    assert validation.exit_code == 0, validation.output
    assert "lanczos_precision=0" not in validation.output


def test_manifest_rejects_user_authored_lanczos_iterations(tmp_path: Path) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path)
    manifest.stages["correlator_analysis"].defaults["lanczos_iterations"] = 2
    job = manifest.stages["correlator_analysis"].jobs[0]

    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "lanczos_iterations is determined automatically from the available 2pt times "
        "and complete 3pt square; remove this parameter."
    ]


def test_manifest_validation_rejects_tseps_that_cannot_form_complete_square(
    tmp_path: Path,
) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path, tseps=[5, 7])
    job = manifest.stages["correlator_analysis"].jobs[0]
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "The standard tsep/tau 3pt data cannot form a complete Lanczos square."
    ]


def test_method_specific_contract_rejects_parameters_from_the_other_method(
    tmp_path: Path,
) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    lanczos_manifest = _manifest(pt2_path, pt3_path)
    lanczos_stage = lanczos_manifest.stages["correlator_analysis"]
    lanczos_stage.defaults["fit_strategy"] = ["joint"]
    lanczos_job = lanczos_stage.jobs[0]
    assert validate_stage_inputs("correlator_analysis", lanczos_manifest, lanczos_job) == [
        "analysis_method='lanczos' does not accept spectral-fit-only parameter 'fit_strategy'."
    ]

    spectral_manifest = _manifest(pt2_path, pt3_path)
    spectral_stage = spectral_manifest.stages["correlator_analysis"]
    spectral_stage.defaults = {
        "analysis_method": "spectral_fit",
        "component": "re",
        "fit_scope": ["3pt_ratio"],
        "fit_strategy": ["joint"],
        "fitting_form": "Breit",
        "model_average": False,
        "nstate": [2],
        "posterior_prior_error_scale": 3.0,
        "q_min": 0.05,
        "lanczos_t0": 1,
    }
    spectral_job = spectral_stage.jobs[0]
    assert validate_stage_inputs("correlator_analysis", spectral_manifest, spectral_job) == [
        "analysis_method='spectral_fit' does not accept Lanczos-only parameter 'lanczos_t0'."
    ]


def test_planning_contract_handles_dict_correlators_and_reports_method_mismatch(
    tmp_path: Path,
) -> None:
    pt2_path, pt3_path, _current = _write_ordinary_h5(tmp_path)
    manifest = _manifest(pt2_path, pt3_path)
    manifest.stages["correlator_analysis"].defaults["fit_strategy"] = ["joint"]
    gaps = _stage_parameter_gaps(manifest.model_dump(mode="json"))
    assert any(
        gap["parameter"] == "fit_strategy"
        and "does not accept spectral-fit-only" in gap["message"]
        for gap in gaps
    )


def test_ordinary_tsep_conversion_trims_and_warns_about_discarded_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pt2_path, pt3_path, current = _write_ordinary_h5(tmp_path)
    paths = {str(tsep): pt3_path for tsep in (4, 6, 8, 12)}
    with pytest.warns(UserWarning, match="uses 4 of 34"):
        inspection = inspect_lanczos_inputs(
            {},
            pt2_path=pt2_path,
            pt3_paths=paths,
            tsep_ls=[4, 6, 8, 12],
            source_operator="g5",
            sink_operator="g5",
            current_operator="J",
            momentum="PX0PY0PZ0",
            fitting_form="Breit",
            fit_scope="3pt_matrix",
            z_values=[0],
            bT=0,
            temporal_extent=14,
        )
    assert inspection["status"] == "valid_with_discarded_points"
    assert inspection["lanczos_t0"] == 2
    assert inspection["lanczos_time_step"] == 2
    assert inspection["source_2pt_indices"] == [4, 6, 8, 10]
    assert inspection["sampling_plan"]["selected_tseps"] == [4, 6, 8]
    assert inspection["sampling_plan"]["used_point_count"] == 4
    assert inspection["sampling_plan"]["discarded_point_count"] == 30

    worker_counts: list[int] = []

    class _ImmediateFuture:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class _ImmediateExecutor:
        def __init__(self, *, max_workers: int) -> None:
            worker_counts.append(max_workers)

        def __enter__(self) -> "_ImmediateExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
            return False

        def submit(self, function: object, *args: object) -> _ImmediateFuture:
            return _ImmediateFuture(function(*args))  # type: ignore[operator]

    monkeypatch.setattr(correlator_functions, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(correlator_functions, "as_completed", lambda futures: iter(futures))

    with pytest.warns(UserWarning, match="30 points are discarded"):
        store: dict = {}
        result = run_lanczos_analysis(
            store,
            pt2_path=pt2_path,
            pt3_paths=paths,
            tsep_ls=[4, 6, 8, 12],
            source_operator="g5",
            sink_operator="g5",
            current_operator="J",
            momentum="PX0PY0PZ0",
            fitting_form="Breit",
            fit_scope="3pt_matrix",
            nstate=2,
            z_values=[0],
            bT=0,
            bz_direction="Z",
            part="both",
            resample_mode="jk",
            sample_error_mode="covariance",
            n_boot=None,
            seed=0,
            bin_size=1,
            lanczos_inner_samples=4,
            lanczos_precision=0,
            workers=2,
            ensemble="toy",
            tag="ordinary",
            temporal_extent=14,
            artifacts_dir=tmp_path / "artifacts",
        )
    output = EnsembleData.from_netcdf(result["netcdf_path"])
    assert output.array.values[:, 0] == pytest.approx(np.full(6, current[0, 0]))
    assert list(store["lanczos_state_matrices"].coords["component"].values) == ["re", "im"]
    assert worker_counts == [2]
    assert result["sampling_plan"]["discarded_point_count"] == 30
    assert Path(result["state_matrix_netcdf"]).is_file()


def test_tsep_conversion_plan_uses_t0_and_sparse_transfer_power() -> None:
    plan = plan_tsep_tau_conversion(
        [4, 6, 8, 12],
        source_times=14,
        sink_times=14,
        requested_iterations=2,
    )
    assert (plan["t0"], plan["time_step"], plan["iterations"]) == (2, 2, 2)
    assert [(point["tsep"], point["tau"]) for point in plan["used_points"]] == [
        (4, 2),
        (6, 4),
        (6, 2),
        (8, 4),
    ]


def test_lanczos_twopoint_tool_writes_iteration_spectrum(tmp_path: Path) -> None:
    pt2_path, _pt3_path, _current = _write_ordinary_h5(tmp_path)
    store: dict = {}
    result = run_lanczos_analysis(
        store,
        pt2_path=pt2_path,
        source_operator="g5",
        sink_operator="g5",
        momentum="PX0PY0PZ0",
        fitting_form="Breit",
        fit_scope="2pt_spectrum",
        nstate=2,
        part="re",
        resample_mode="jk",
        sample_error_mode="covariance",
        n_boot=None,
        seed=0,
        bin_size=1,
        lanczos_inner_samples=4,
        lanczos_precision=0,
        ensemble="toy",
        tag="spectrum",
        temporal_extent=14,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert Path(result["netcdf_path"]).is_file()
    assert isinstance(store["output"], EnsembleData)
    assert result["iterations"] == 7
    energies = store["output"].array.sel(channel="source", iteration=3).values
    assert energies == pytest.approx(np.tile([0.25, 0.7], (6, 1)))
