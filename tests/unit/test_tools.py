"""Unit tests for job-aware core tool preparation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from lamet_agent.core.tools import (
    prepare_tool_args,
    required_job_tool_sequence,
    resolve_job_tools,
    resolve_plot_save_path,
    validate_stage_inputs,
)
from lamet_agent.manifest import AnalysisManifest, derive_job_kinematics, validate_manifest_file
from lamet_agent.manifest_params import merge_stage_params
from lamet_agent.stages.matching.skills import effective_matching_params


def _manifest():
    return validate_manifest_file(Path("examples/pion_pdf_cg_manifest.json"))


def test_resolve_renormalization_job_tools_by_scheme_and_roles() -> None:
    stage_tools = {
        name: lambda store: {}
        for name in (
            "apply_ratio_scheme_renormalization",
            "fit_self_renormalization_factor",
            "apply_self_renormalization",
            "plot_self_renormalization_diagnostics",
            "plot_renormalized_matrix_element",
            "load_bare_matrix_element",
        )
    }
    pdf_manifest = _manifest()
    ratio_job = pdf_manifest.stages["renormalization"].jobs[0]
    ratio_params = merge_stage_params(pdf_manifest.stages["renormalization"].defaults, ratio_job.params)
    assert set(resolve_job_tools(
        "renormalization", ratio_job, ratio_params, stage_tools=stage_tools,
    )) == {"apply_ratio_scheme_renormalization", "plot_renormalized_matrix_element"}
    assert required_job_tool_sequence("renormalization", ratio_job, ratio_params) == (
        "apply_ratio_scheme_renormalization",
        "plot_renormalized_matrix_element",
    )

    da_manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    fit_job, apply_job = da_manifest.stages["renormalization"].jobs[:2]
    defaults = da_manifest.stages["renormalization"].defaults
    assert set(resolve_job_tools(
        "renormalization", fit_job, merge_stage_params(defaults, fit_job.params), stage_tools=stage_tools,
    )) == {"fit_self_renormalization_factor", "plot_self_renormalization_diagnostics"}
    assert set(resolve_job_tools(
        "renormalization", apply_job, merge_stage_params(defaults, apply_job.params), stage_tools=stage_tools,
    )) == {
        "apply_self_renormalization",
        "plot_self_renormalization_diagnostics",
        "plot_renormalized_matrix_element",
    }
    assert required_job_tool_sequence(
        "renormalization", fit_job, merge_stage_params(defaults, fit_job.params)
    ) == ("fit_self_renormalization_factor", "plot_self_renormalization_diagnostics")
    assert required_job_tool_sequence(
        "renormalization", apply_job, merge_stage_params(defaults, apply_job.params)
    ) == (
        "apply_self_renormalization",
        "plot_self_renormalization_diagnostics",
        "plot_renormalized_matrix_element",
    )


def test_hybrid_scheme_self_renormalization_routes_denominator_apply_job(tmp_path: Path) -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    defaults = manifest.stages["renormalization"].defaults
    defaults["scheme"] = "hybrid"
    defaults["zs_fm"] = 0.2
    job = manifest.stages["renormalization"].jobs[1]
    job.inputs["denominator"] = "pion_zero_momentum_reference"
    params = merge_stage_params(defaults, job.params)

    assert validate_stage_inputs("renormalization", manifest, job) == []
    assert required_job_tool_sequence("renormalization", job, params)[0] == "apply_self_renormalization"
    args = prepare_tool_args(
        "apply_self_renormalization",
        {},
        manifest=manifest,
        stage="renormalization",
        job=job,
        effective_params=params,
        artifacts_dir=tmp_path,
    )
    assert args["scheme"] == "hybrid"
    assert args["strategy"] == "self_renormalization"
    assert args["denominator"] == "denominator"
    assert args["zs_fm"] == pytest.approx(0.2)


def test_qda_ratio_correlator_job_uses_unified_tool_contract() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "da",
                "root_directory": ".",
                "artifacts_directory": "artifacts",
                "target_observable": "da",
                "parton": "quark",
                "resample_mode": "bs",
                "bs_samples": 8,
                "random_seed": 1984,
                "workers": 4,
                "stages": ["correlator_analysis"],
            },
            "inputs": {
                "correlators": [
                    {
                        "correlator_id": "c2_local",
                        "correlator_type": "2pt",
                        "data_path": "c2.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "volume": "S48T64",
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                    },
                    {
                        "correlator_id": "c2_qda",
                        "correlator_type": "2pt",
                        "data_path": "qda.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "gT5_nonlocal",
                        "bz_direction": "Z",
                        "bT": [0],
                        "bz": [1, 2],
                        "volume": "S48T64",
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                    }
                ],
                "artifacts": [],
                "kernels": [],
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {"fit_scope": ["qda_ratio"], "momentum": "PX0PY0PZ0", "model_average": True},
                    "jobs": [{"id": "ca", "correlator_ids": ["c2_local", "c2_qda"], "params": {}}],
                }
            },
        }
    )
    job = manifest.stages["correlator_analysis"].jobs[0]
    params = merge_stage_params(manifest.stages["correlator_analysis"].defaults, job.params)
    assert validate_stage_inputs("correlator_analysis", manifest, job) == []
    assert set(resolve_job_tools(
        "correlator_analysis",
        job,
        params,
        stage_tools={"fit_bare_matrix_grid": lambda store: {}},
    )) == {"fit_bare_matrix_grid"}
    assert required_job_tool_sequence("correlator_analysis", job, params) == ()
    args = prepare_tool_args(
        "fit_bare_matrix_grid",
        {"z_values": [99]},
        manifest=manifest,
        stage="correlator_analysis",
        job=job,
        effective_params=params,
        artifacts_dir=Path("artifacts/correlator_analysis"),
    )
    assert args["z_values"] == [1, 2]
    assert args["pt2_path"].endswith("c2.h5")
    assert args["sink_operator"] == "g5"
    assert args["qda_path"].endswith("qda.h5")
    assert args["qda_sink_operator"] == "gT5_nonlocal"
    assert args["fit_scope"] == "qda_ratio"
    assert args["qda_denominator_mode"] == "local"
    assert "pt2_bT" not in args
    assert "pt2_bz" not in args
    assert args["model_average"] is True
    assert args["workers"] == 4
    missing_local = job.model_copy(update={"correlator_ids": ["c2_qda"]})
    assert validate_stage_inputs("correlator_analysis", manifest, missing_local) == [
        "A qda_ratio job without an ordinary local 2pt correlator requires "
        "bz=0 in the nonlocal qDA 2pt grid."
    ]

    qda_input = next(item for item in manifest.correlators if item.correlator_id == "c2_qda")
    qda_input.bz = [0, 1, 2]
    assert validate_stage_inputs("correlator_analysis", manifest, missing_local) == []
    fallback_params = merge_stage_params(
        manifest.stages["correlator_analysis"].defaults, missing_local.params
    )
    fallback_args = prepare_tool_args(
        "fit_bare_matrix_grid",
        {},
        manifest=manifest,
        stage="correlator_analysis",
        job=missing_local,
        effective_params=fallback_params,
        artifacts_dir=Path("artifacts/correlator_analysis"),
    )
    assert fallback_args["pt2_path"].endswith("qda.h5")
    assert fallback_args["source_operator"] == "g5"
    assert fallback_args["sink_operator"] == "gT5_nonlocal"
    assert fallback_args["pt2_bT"] == 0
    assert fallback_args["pt2_bz"] == 0
    assert fallback_args["qda_denominator_mode"] == "nonlocal_bz0"
    assert fallback_args["z_values"] == [0, 1, 2]


def test_3pt_ratio_correlator_job_keeps_default_tool_routing() -> None:
    manifest = validate_manifest_file(Path("examples/pion_pdf_cg_manifest.json"))
    job = manifest.stages["correlator_analysis"].jobs[0]
    params = merge_stage_params(manifest.stages["correlator_analysis"].defaults, job.params)
    tools = {
        "inspect_correlator_scale": lambda store: {},
        "tune_bare_matrix": lambda store: {},
        "fit_bare_matrix_grid": lambda store: {},
    }
    assert required_job_tool_sequence("correlator_analysis", job, params) == ()
    assert set(resolve_job_tools("correlator_analysis", job, params, stage_tools=tools)) == set(tools)


def test_legacy_self_renormalization_scheme_returns_migration_error() -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    manifest.stages["renormalization"].defaults["scheme"] = "self_renormalization"
    issue = validate_stage_inputs("renormalization", manifest, manifest.stages["renormalization"].jobs[0])

    assert issue == [
        "renormalization scheme 'self_renormalization' is no longer supported; "
        "use scheme='ratio' with strategy='self_renormalization'."
    ]


def test_legacy_ratio_strategy_returns_migration_error() -> None:
    manifest = _manifest()
    manifest.stages["renormalization"].defaults["strategy"] = "ratio"
    job = manifest.stages["renormalization"].jobs[0]

    assert validate_stage_inputs("renormalization", manifest, job) == [
        "renormalization strategy 'ratio' is no longer supported; "
        "use strategy='external_denominator'."
    ]


def test_hybrid_self_fit_rejects_fixed_m0() -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    job = manifest.stages["renormalization"].jobs[0]
    job.params["scheme_parameters"]["m0_gev"] = -0.094

    assert validate_stage_inputs("renormalization", manifest, job) == [
        "self_renormalization fit jobs determine the reference m0; "
        "remove scheme_parameters.m0_gev here (apply jobs may override target m0_gev)."
    ]


def test_hybrid_self_rejects_unknown_z_coverage_policy() -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    job = manifest.stages["renormalization"].jobs[1]
    job.params["scheme_parameters"]["z_coverage_policy"] = "freeze"

    assert validate_stage_inputs("renormalization", manifest, job) == [
        "self_renormalization z_coverage_policy must be "
        "'strict', 'intersection', or 'extrapolate'."
    ]


def test_hybrid_self_requires_explicit_lambdaqcd_gev() -> None:
    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    job = manifest.stages["renormalization"].jobs[1]
    manifest.stages["renormalization"].defaults["scheme_parameters"].pop("LambdaQCD_gev")

    assert validate_stage_inputs("renormalization", manifest, job) == [
        "self_renormalization requires scheme_parameters.LambdaQCD_gev "
        "on every fit and apply job."
    ]


def test_resolve_plot_save_path_uses_artifact_directory(tmp_path: Path) -> None:
    assert resolve_plot_save_path("elsewhere/fit.png", artifacts_dir=tmp_path) == str(tmp_path / "fit")
    assert resolve_plot_save_path(None, artifacts_dir=tmp_path) == str(tmp_path / "fit_on_data")


def test_prepare_correlator_tuning_args_from_job_sources(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[1]
    effective = manifest.stages["correlator_analysis"].defaults
    args = prepare_tool_args(
        "tune_bare_matrix", {}, manifest=manifest, stage="correlator_analysis", job=job,
        effective_params=effective, artifacts_dir=tmp_path,
    )
    pt3 = next(
        item
        for item in manifest.correlators
        if item.correlator_id in job.correlator_ids and item.correlator_type == "3pt"
    )
    assert args["momentum"] == derive_job_kinematics(manifest, job)["momentum"]
    assert args["tsep_ls"] == pt3.tsep
    assert args["z_values"] == pt3.bz
    assert args["bz_direction"] == pt3.bz_direction
    assert "tune_z_values" not in args
    assert args["nstate_values"] == effective["nstate"]
    assert args["fit_strategies"] == effective["fit_strategy"]
    assert args["fit_scope_values"] == effective["fit_scope"]
    assert set(args["pt3_paths"]) == {str(tsep) for tsep in pt3.tsep}
    assert args["resample_mode"] == manifest.metadata.resample_mode
    assert args["sample_error_mode"] == manifest.metadata.sample_error_mode
    assert args["seed"] == manifest.metadata.random_seed
    assert "workers" not in args
    if manifest.metadata.resample_mode == "bs":
        assert args["n_boot"] == manifest.metadata.bs_samples
    else:
        assert "n_boot" not in args


def test_prepare_correlator_args_injects_bs_samples_for_bootstrap_mode(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "bs",
                "root_directory": ".",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "bs",
                "random_seed": 1984,
                "bs_samples": 500,
                "stages": ["correlator_analysis"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [],
                "kernels": [],
            },
            "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]}},
        }
    )
    args = prepare_tool_args(
        "tune_bare_matrix",
        {},
        manifest=manifest,
        stage="correlator_analysis",
        job=manifest.stages["correlator_analysis"].jobs[0],
        effective_params={},
        artifacts_dir=tmp_path,
    )
    assert args["n_boot"] == 500
    assert args["seed"] == 1984
    assert "tune_z_values" not in args


def test_prepare_correlator_terminal_args_use_job_artifact_path(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    args = prepare_tool_args(
        "fit_bare_matrix_grid", {"nstate": 2, "fit_strategy": "joint", "model_average": True},
        manifest=manifest, stage="correlator_analysis", job=job,
        effective_params=manifest.stages["correlator_analysis"].defaults,
        artifacts_dir=tmp_path,
    )
    assert args["save_path"] == str(tmp_path / job.id)
    assert args["job_id"] == job.id
    kinematics = derive_job_kinematics(manifest, job)
    assert args["lattice_spacing_fm"] == kinematics["lattice_spacing_fm"]
    if manifest.stages["correlator_analysis"].defaults.get("model_average", False):
        assert args["nstate_values"] == manifest.stages["correlator_analysis"].defaults["nstate"]
        assert "nstate" not in args
    else:
        assert args["nstate"] == 2
    assert args["fit_scope"] == "3pt_ratio"
    assert args["model_average"] == manifest.stages["correlator_analysis"].defaults.get("model_average", False)
    assert args["workers"] == manifest.metadata.workers
    pt3 = next(
        item
        for item in manifest.correlators
        if item.correlator_id in job.correlator_ids and item.correlator_type == "3pt"
    )
    assert args["bz_direction"] == pt3.bz_direction


def test_correlator_stage_rejects_removed_variant_parameter() -> None:
    manifest = _manifest()
    manifest.stages["correlator_analysis"].defaults["variant"] = "free"
    job = manifest.stages["correlator_analysis"].jobs[0]
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "variant is not a supported correlator_analysis parameter."
    ]


def test_correlator_stage_allows_tune_z_parameter() -> None:
    manifest = _manifest()
    manifest.stages["correlator_analysis"].jobs[0].params["tune_z"] = 2
    job = manifest.stages["correlator_analysis"].jobs[0]

    assert validate_stage_inputs("correlator_analysis", manifest, job) == []


def test_metadata_workers_override_stage_params_for_sample_fit_tools(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.metadata.workers = 3
    correlator_job = manifest.stages["correlator_analysis"].jobs[0]
    correlator_args = prepare_tool_args(
        "fit_bare_matrix_grid",
        {},
        manifest=manifest,
        stage="correlator_analysis",
        job=correlator_job,
        effective_params={**manifest.stages["correlator_analysis"].defaults, "workers": 99},
        artifacts_dir=tmp_path,
    )
    fourier_job = manifest.stages["fourier_transform"].jobs[0]
    fourier_args = prepare_tool_args(
        "run_fourier_transform",
        {},
        manifest=manifest,
        stage="fourier_transform",
        job=fourier_job,
        effective_params={**manifest.stages["fourier_transform"].defaults, "workers": 99},
        artifacts_dir=tmp_path,
        store={"input": SimpleNamespace(attrs={})},
    )
    da_manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "da",
                "root_directory": ".",
                "artifacts_directory": "artifacts",
                "target_observable": "da",
                "parton": "quark",
                "resample_mode": "bs",
                "bs_samples": 8,
                "random_seed": 1984,
                "workers": 3,
                "stages": ["correlator_analysis"],
            },
            "inputs": {
                "correlators": [
                    {
                        "correlator_id": "c2_local",
                        "correlator_type": "2pt",
                        "data_path": "c2.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "volume": "S48T64",
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                    },
                    {
                        "correlator_id": "c2_qda",
                        "correlator_type": "2pt",
                        "data_path": "qda.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "gT5_nonlocal",
                        "bz_direction": "Z",
                        "bT": [0],
                        "bz": [0, 1],
                        "volume": "S48T64",
                        "lattice_spacing_fm": 0.1,
                        "momentum": ["PX0PY0PZ0"],
                    }
                ],
                "artifacts": [],
                "kernels": [],
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {"fit_scope": ["qda_ratio"], "momentum": "PX0PY0PZ0"},
                    "jobs": [{"id": "ca", "correlator_ids": ["c2_local", "c2_qda"], "params": {}}],
                }
            },
        }
    )
    da_job = da_manifest.stages["correlator_analysis"].jobs[0]
    da_args = prepare_tool_args(
        "fit_bare_matrix_grid",
        {},
        manifest=da_manifest,
        stage="correlator_analysis",
        job=da_job,
        effective_params={**da_manifest.stages["correlator_analysis"].defaults, "workers": 99},
        artifacts_dir=tmp_path,
    )

    assert correlator_args["workers"] == 3
    assert fourier_args["workers"] == 3
    assert da_args["workers"] == 3


def test_prepare_correlator_terminal_args_pass_nstate_values_when_not_selected(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    args = prepare_tool_args(
        "fit_bare_matrix_grid", {"fit_strategy": "joint"},
        manifest=manifest, stage="correlator_analysis", job=job,
        effective_params=manifest.stages["correlator_analysis"].defaults,
        artifacts_dir=tmp_path,
    )
    assert args["nstate_values"] == manifest.stages["correlator_analysis"].defaults["nstate"]
    assert "nstate" not in args


def test_prepare_correlator_model_average_keeps_fit_function_scan(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    effective = {
        **manifest.stages["correlator_analysis"].defaults,
        "model_average": True,
        "nstate": [2, 3],
        "prior_width": [0.5, 1.0, 2.0],
    }
    args = prepare_tool_args(
        "fit_bare_matrix_grid",
        {"nstate": 2, "prior_width": 2.0, "tmin": 4, "tmax": 12, "tau_cut": 2},
        manifest=manifest,
        stage="correlator_analysis",
        job=job,
        effective_params=effective,
        artifacts_dir=tmp_path,
    )
    assert args["model_average"] is True
    assert args["nstate_values"] == [2, 3]
    assert "nstate" not in args
    assert args["prior_width"] == [0.5, 1.0, 2.0]
    assert args["pt2_window"] == {"tmin": 4, "tmax": 12}
    assert args["pt3_window"] == {"tsep_ls": [8, 10, 12], "tau_cut": 2}


def test_prepare_correlator_terminal_args_keep_scalar_fit_scope(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    args = prepare_tool_args(
        "fit_bare_matrix_grid", {"nstate": 2, "fit_strategy": "joint", "fit_scope": "FH"},
        manifest=manifest, stage="correlator_analysis", job=job,
        effective_params=manifest.stages["correlator_analysis"].defaults,
        artifacts_dir=tmp_path,
    )
    assert args["fit_scope"] == "FH"


def test_prepare_nonbreit_correlator_args_match_initial_final_momenta(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "nonbreit",
                "root_directory": str(tmp_path),
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
                        "correlator_id": "pt2_i",
                        "correlator_type": "2pt",
                        "data_path": "pt2.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "volume": "S16T32",
                        "momentum": ["PX0PY0PZ0", "PX0PY0PZ1"],
                        "lattice_spacing_fm": 0.1,
                    },
                    {
                        "correlator_id": "pt3_fi",
                        "correlator_type": "3pt",
                        "data_path": "pt3.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_operator": "g5",
                        "sink_operator": "g5",
                        "current_operator": "gT_nonlocal", "bz_direction": "Z",
                        "volume": "S16T32",
                        "momentum": ["PX0PY0PZ1"],
                        "lattice_spacing_fm": 0.1,
                        "bT": [0],
                        "bz": [0],
                        "tsep": [8],
                    },
                ]
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {
                        "fitting_form": "NonBreit",
                        "initial_momentum": "PX0PY0PZ0",
                        "final_momentum": "PX0PY0PZ1",
                    },
                    "jobs": [{"id": "ca", "correlator_ids": ["pt2_i", "pt3_fi"]}],
                }
            },
        }
    )
    manifest._root_directory = tmp_path
    manifest._artifacts_directory = tmp_path / "artifacts"
    job = manifest.stages["correlator_analysis"].jobs[0]
    assert validate_stage_inputs("correlator_analysis", manifest, job) == []
    args = prepare_tool_args(
        "fit_bare_matrix_grid", {},
        manifest=manifest,
        stage="correlator_analysis",
        job=job,
        effective_params=manifest.stages["correlator_analysis"].defaults,
        artifacts_dir=tmp_path,
    )
    assert args["pt2_path"].endswith("pt2.h5")
    assert args["pt2_out_path"].endswith("pt2.h5")
    assert args["initial_momentum"] == "PX0PY0PZ0"
    assert args["final_momentum"] == "PX0PY0PZ1"
    assert args["initial_momentum_gev"] == 0.0
    assert args["final_momentum_gev"] == pytest.approx(manifest.correlators[0].momentum_gev("PX0PY0PZ1"))


def test_nonbreit_requires_initial_and_final_momentum(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0].model_copy(update={"params": {"fitting_form": "NonBreit"}})
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "A NonBreit correlator_analysis job requires params.initial_momentum and params.final_momentum."
    ]


def test_correlator_fit_scope_accepts_public_names_and_rejects_old_names() -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    for scope in ("3pt_ratio", "FH", "3pt_ratio+FH"):
        manifest.stages["correlator_analysis"].defaults["fit_scope"] = [scope]
        assert validate_stage_inputs("correlator_analysis", manifest, job) == []
    manifest.stages["correlator_analysis"].defaults["fit_scope"] = ["ratio"]
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "fit_scope must contain only '3pt_ratio', 'FH', '3pt_ratio+FH', or 'qda_ratio'."
    ]
    manifest.stages["correlator_analysis"].defaults["fit_scope"] = ["qda_ratio", "3pt_ratio"]
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "fit_scope='qda_ratio' cannot be mixed with 3pt/FH scopes in one job."
    ]


def test_prepare_renormalization_args_bind_roles_and_scheme(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    effective = merge_stage_params(manifest.stages["renormalization"].defaults, job.params)
    args = prepare_tool_args(
        "apply_ratio_scheme_renormalization", {}, manifest=manifest, stage="renormalization", job=job,
        effective_params=effective,
        artifacts_dir=tmp_path,
    )
    assert args["target"] == "target"
    assert args["denominator"] == "denominator"
    assert args["scheme"] == effective["scheme"]
    assert args["scheme_parameters"]["zs_fm"] == effective["zs_fm"]
    assert args["scheme_parameters"]["m0_gev"] == effective["scheme_parameters"]["m0_gev"]
    assert args["scheme_parameters"]["delta_m_gev"] == effective["scheme_parameters"]["delta_m_gev"]
    assert args["save_path"] == str(tmp_path / job.id)
    assert "normalization" not in args


def test_prepare_renormalization_args_filters_normalization_manifest_flag(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    effective = merge_stage_params(
        manifest.stages["renormalization"].defaults,
        {**job.params, "normalization": True},
    )
    args = prepare_tool_args(
        "apply_ratio_scheme_renormalization",
        {},
        manifest=manifest,
        stage="renormalization",
        job=job,
        effective_params=effective,
        artifacts_dir=tmp_path,
    )
    assert "normalization" not in args


def test_prepare_ratio_renormalization_args_do_not_require_hybrid_parameters(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    effective = merge_stage_params(
        manifest.stages["renormalization"].defaults,
        {
            **job.params,
            "scheme": "ratio",
            "scheme_parameters": {"m0_gev": 9.0, "delta_m_gev": 8.0},
        },
    )
    effective.pop("zs_fm", None)

    args = prepare_tool_args(
        "apply_ratio_scheme_renormalization",
        {},
        manifest=manifest,
        stage="renormalization",
        job=job,
        effective_params=effective,
        artifacts_dir=tmp_path,
    )

    assert args["scheme"] == "ratio"
    assert args["scheme_parameters"] == {}
    assert args["target"] == "target"
    assert args["denominator"] == "denominator"


@pytest.mark.parametrize("key", ["LambdaQCD_gev", "d", "svdcut", "z_coverage_policy"])
def test_ratio_schemes_reject_hybrid_self_only_scheme_parameters(key: str) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    manifest.stages["renormalization"].defaults["scheme_parameters"][key] = 0.1

    assert validate_stage_inputs("renormalization", manifest, job) == [
        f"strategy 'external_denominator' does not accept self-renormalization scheme_parameters: {key}."
    ]


def test_ratio_renormalization_stage_accepts_target_and_denominator_without_zs() -> None:
    manifest = _manifest()
    manifest.stages["renormalization"].defaults["scheme"] = "ratio"
    manifest.stages["renormalization"].defaults.pop("zs_fm", None)
    job = manifest.stages["renormalization"].jobs[0]

    assert validate_stage_inputs("renormalization", manifest, job) == []


def test_ratio_strategy_rejects_msbar_scheme() -> None:
    manifest = _manifest()
    manifest.stages["renormalization"].defaults["scheme"] = "msbar"
    job = manifest.stages["renormalization"].jobs[0]

    assert validate_stage_inputs("renormalization", manifest, job) == [
        "renormalization strategy 'external_denominator' does not implement scheme 'msbar'."
    ]


def test_matching_scheme_must_match_kernel_id() -> None:
    manifest = _manifest()
    manifest.stages["perturbative_matching"].defaults["scheme"] = "ratio"
    job = manifest.stages["perturbative_matching"].jobs[0]

    assert validate_stage_inputs("perturbative_matching", manifest, job) == [
        "Matching scheme 'ratio' does not match kernel_id 'CG_gt_quark_PDF_hybrid_NLO', "
        "which encodes scheme 'hybrid'."
    ]


def test_prepare_self_renormalization_args_bind_kernel_and_roles(tmp_path: Path) -> None:
    artifacts = [
        {
            "id": artifact_id,
            "stage": "correlator_analysis",
            "path": f"{artifact_id}.nc",
            "ensemble": ensemble,
            "hadron": "pion",
            "gfix": "GI",
            "momentum": momentum,
            "volume": "S96T192",
            "lattice_spacing_fm": spacing,
        }
        for artifact_id, ensemble, momentum, spacing in (
            ("bare_pdf_reference", "a06m130", "PX0PY0PZ0", 0.0574),
            ("bare_da_mom6_a06", "a06m130", "PX0PY0PZ6", 0.0574),
            ("bare_da_mom6_a09", "a09m130", "PX0PY0PZ6", 0.0882),
            ("bare_da_mom6_a12", "a12m130", "PX0PY0PZ6", 0.1213),
        )
    ]
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "self-renorm",
                "root_directory": str(tmp_path),
                "target_observable": "da",
                "parton": "quark",
                "resample_mode": "jk",
                "random_seed": 1984,
                "stages": ["renormalization"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": artifacts,
                "kernels": [
                    {
                        "stage": "renormalization",
                        "kernel_id": "ZMSbar_da",
                        "kernel_path": "src/lamet_agent/kernels.py",
                        "kernel_parameters": {"mu": 2.0},
                    }
                ],
            },
            "stages": {
                "renormalization": {
                    "defaults": {
                        "scheme": "ratio",
                        "strategy": "self_renormalization",
                        "mu": 2.0,
                        "scheme_parameters": {"LambdaQCD_gev": 0.12},
                    },
                    "jobs": [
                        {
                            "id": "rn_zR_fit",
                            "inputs": {"reference": "bare_pdf_reference"},
                            "params": {
                                "scheme_parameters": {
                                    "d": -0.08183,
                                    "svdcut": 1e-12,
                                }
                            },
                        },
                        {
                            "id": "rn_mom6_a06",
                            "inputs": {"target": "bare_da_mom6_a06", "zR": "rn_zR_fit"},
                            "params": {
                                "scheme_parameters": {
                                    "d": 0.19,
                                    "m0_gev": -0.094,
                                    "z_coverage_policy": "intersection",
                                }
                            },
                        },
                        {
                            "id": "rn_mom6_a09",
                            "inputs": {"target": "bare_da_mom6_a09", "zR": "rn_zR_fit"},
                            "params": {
                                "scheme_parameters": {
                                    "d": 0.19,
                                    "m0_gev": -0.094,
                                    "z_coverage_policy": "intersection",
                                }
                            },
                        },
                        {
                            "id": "rn_mom6_a12",
                            "inputs": {"target": "bare_da_mom6_a12", "zR": "rn_zR_fit"},
                            "params": {
                                "scheme_parameters": {
                                    "d": 0.19,
                                    "m0_gev": -0.094,
                                    "z_coverage_policy": "intersection",
                                }
                            },
                        },
                    ],
                }
            },
        }
    )
    fit_job = manifest.stages["renormalization"].jobs[0]
    apply_job = manifest.stages["renormalization"].jobs[1]
    fit_effective = merge_stage_params(manifest.stages["renormalization"].defaults, fit_job.params)
    apply_effective = merge_stage_params(manifest.stages["renormalization"].defaults, apply_job.params)

    assert set(fit_job.inputs) == {"reference"}
    assert set(apply_job.inputs) == {"target", "zR"}
    assert validate_stage_inputs("renormalization", manifest, fit_job) == []
    assert validate_stage_inputs("renormalization", manifest, apply_job) == []

    fit_args = prepare_tool_args(
        "fit_self_renormalization_factor",
        {"reference": "wrong", "order": None, "Nf": None, "save_path": None},
        manifest=manifest,
        stage="renormalization",
        job=fit_job,
        effective_params=fit_effective,
        artifacts_dir=tmp_path,
    )
    assert fit_args["reference"] == "reference"
    assert fit_args["kernel_id"] == "ZMSbar_da"
    assert fit_args["d"] == -0.08183
    assert "m0_gev" not in fit_args
    assert "d_fit" not in fit_args
    assert "n_m0" not in fit_args
    assert fit_args["mu"] == 2.0
    assert fit_args["LambdaQCD_gev"] == 0.12
    assert "order" not in fit_args
    assert "Nf" not in fit_args
    assert fit_args["svdcut"] == 1e-12
    assert "z_coverage_policy" not in fit_args
    assert fit_args["save_path"] == str(tmp_path / "rn_zR_fit")
    # Fit-job params carry required d (PDF); m0_gev omitted → fit.
    assert fit_effective["scheme_parameters"]["d"] == -0.08183
    assert "m0_gev" not in fit_effective["scheme_parameters"]

    apply_args = prepare_tool_args(
        "apply_self_renormalization",
        {
            "target": "bare_da_mom6_a06",
            "zR": "rn_zR_fit",
            "kernel_id": None,
            "order": None,
            "Nf": None,
            "save_path": None,
        },
        manifest=manifest,
        stage="renormalization",
        job=apply_job,
        effective_params=apply_effective,
        artifacts_dir=tmp_path,
    )
    assert apply_args["target"] == "target"
    assert apply_args["zR"] == "zR"
    assert apply_args["kernel_id"] == "ZMSbar_da"
    assert apply_args["mu"] == 2.0
    assert apply_args["LambdaQCD_gev"] == 0.12
    assert "order" not in apply_args
    assert "Nf" not in apply_args
    assert apply_args["z_coverage_policy"] == "intersection"
    assert apply_args["d"] == 0.19
    assert apply_args["m0_gev"] == -0.094
    assert apply_args["metadata"]["ensemble"] == "a06m130"
    assert apply_args["metadata"]["momentum"] == "PX0PY0PZ6"
    assert apply_args["metadata"]["hadron"] == "pion"
    assert apply_args["metadata"]["gfix"] == "GI"
    assert apply_args["metadata"]["momentum_gev"] == pytest.approx(
        derive_job_kinematics(manifest, apply_job)["momentum_gev"]
    )
    assert apply_args["save_path"] == str(tmp_path / "rn_mom6_a06")

    fit_diag = prepare_tool_args(
        "plot_self_renormalization_diagnostics",
        {},
        manifest=manifest,
        stage="renormalization",
        job=fit_job,
        effective_params=fit_effective,
        artifacts_dir=tmp_path,
    )
    assert fit_diag["mode"] == "fit"
    assert fit_diag["zR"] == "zR"
    assert fit_diag["fit"] == "self_renorm_fit"
    assert "order" not in fit_diag
    assert "Nf" not in fit_diag
    assert "target" not in fit_diag
    assert "include_discrete_effect" not in fit_diag

    apply_diag = prepare_tool_args(
        "plot_self_renormalization_diagnostics",
        {},
        manifest=manifest,
        stage="renormalization",
        job=apply_job,
        effective_params=apply_effective,
        artifacts_dir=tmp_path,
    )
    assert apply_diag["mode"] == "apply"
    assert apply_diag["target"] == "target"
    assert "order" not in apply_diag
    assert "Nf" not in apply_diag
    assert apply_diag["z_coverage_policy"] == "intersection"
    assert apply_diag["include_discrete_effect"] is False
    assert apply_diag["sibling_artifacts"] == []

    last_apply = manifest.stages["renormalization"].jobs[-1]
    last_effective = merge_stage_params(manifest.stages["renormalization"].defaults, last_apply.params)
    for job_id in ("rn_mom6_a06", "rn_mom6_a09", "rn_mom6_a12"):
        (tmp_path / f"{job_id}.nc").write_text("placeholder", encoding="utf-8")
    last_diag = prepare_tool_args(
        "plot_self_renormalization_diagnostics",
        {},
        manifest=manifest,
        stage="renormalization",
        job=last_apply,
        effective_params=last_effective,
        artifacts_dir=tmp_path,
    )
    assert last_diag["mode"] == "apply"
    assert last_diag["include_discrete_effect"] is True
    assert len(last_diag["sibling_artifacts"]) == 3


def test_prepare_fourier_args_from_job_and_upstream_metadata(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["fourier_transform"].jobs[0]
    source = SimpleNamespace(
        attrs={
            "lattice_spacing_fm": "0.0574",
            "momentum_gev": "2.15",
            "bz_direction": "X",
            "hadron": "pion",
            "gfix": "CG",
        }
    )
    effective = merge_stage_params(
        manifest.stages["fourier_transform"].defaults,
        {
            **job.params,
            "psi1_flavor_class": "light",
            "psi2_flavor_class": "heavy",
            "symmetry_guarantee": False,
        },
    )
    args = prepare_tool_args(
        "run_fourier_transform", {}, manifest=manifest, stage="fourier_transform", job=job,
        effective_params=effective, artifacts_dir=tmp_path, store={"input": source},
    )
    assert args["method"] == "CG"
    kinematics = derive_job_kinematics(manifest, job)
    assert args["lattice_spacing_fm"] == kinematics["lattice_spacing_fm"]
    assert args["momentum_gev"] == pytest.approx(kinematics["momentum_gev"])
    assert args["bz_direction"] == source.attrs["bz_direction"]
    assert args["psi1_flavor_class"] == "light"
    assert args["psi2_flavor_class"] == "heavy"
    assert args["symmetry_guarantee"] is False
    assert "phase_shift" not in args
    assert args["workers"] == manifest.metadata.workers
    assert args["save_path"] == str(tmp_path / job.id)


def test_prepare_fourier_args_passes_lambda0_gev(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["fourier_transform"].jobs[0]
    effective = merge_stage_params(manifest.stages["fourier_transform"].defaults, job.params)
    effective["Lambda0_gev"] = 0.37
    args = prepare_tool_args(
        "run_fourier_transform",
        {},
        manifest=manifest,
        stage="fourier_transform",
        job=job,
        effective_params=effective,
        artifacts_dir=tmp_path,
        store={"input": SimpleNamespace(attrs={})},
    )

    assert args["Lambda0_gev"] == 0.37
    assert "Lambda0" not in args


def test_fourier_symmetry_guarantee_requires_boolean() -> None:
    manifest = _manifest()
    job = manifest.stages["fourier_transform"].jobs[0]
    manifest.stages["fourier_transform"].defaults["symmetry_guarantee"] = "true"

    assert validate_stage_inputs("fourier_transform", manifest, job) == [
        "Fourier symmetry_guarantee must be a boolean."
    ]


def test_fourier_sector_options_depend_on_observable() -> None:
    manifest = _manifest()
    job = manifest.stages["fourier_transform"].jobs[0]
    manifest.stages["fourier_transform"].defaults["sector"] = "singlet"
    assert validate_stage_inputs("fourier_transform", manifest, job) == []

    manifest.metadata.target_observable = "gpd"
    assert validate_stage_inputs("fourier_transform", manifest, job) == []

    manifest.stages["fourier_transform"].defaults["sector"] = "total"
    assert validate_stage_inputs("fourier_transform", manifest, job) == [
        "Fourier sector must be one of ['full', 'sea', 'singlet', 'valence']."
    ]

    manifest = validate_manifest_file(Path("examples/pion_da_gi_manifest.json"))
    job = manifest.stages["fourier_transform"].jobs[0]
    assert validate_stage_inputs("fourier_transform", manifest, job) == []

    manifest.stages["fourier_transform"].defaults["sector"] = "singlet"
    assert validate_stage_inputs("fourier_transform", manifest, job) == [
        "Fourier sector must be one of ['full']."
    ]


def test_prepare_matching_resolves_logical_kernel(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["perturbative_matching"].jobs[0]
    effective = effective_matching_params(manifest, job)
    args = prepare_tool_args(
        "build_matching_kernel", {}, manifest=manifest, stage="perturbative_matching", job=job,
        effective_params=effective,
        artifacts_dir=tmp_path, store={"quasi": object()},
    )
    assert args["kernel_id"] == effective["kernel_id"]
    assert args["momentum_gev"] == pytest.approx(derive_job_kinematics(manifest, job)["momentum_gev"])
    assert args["zs_fm"] == effective["zs_fm"]


def test_job_zs_fm_overrides_stage_defaults_for_both_hybrid_stages(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.stages["renormalization"].defaults["zs_fm"] = 0.1
    renorm_job = manifest.stages["renormalization"].jobs[0]
    renorm_job.params["zs_fm"] = 0.2
    renorm_args = prepare_tool_args(
        "apply_ratio_scheme_renormalization",
        {},
        manifest=manifest,
        stage="renormalization",
        job=renorm_job,
        effective_params=merge_stage_params(
            manifest.stages["renormalization"].defaults, renorm_job.params
        ),
        artifacts_dir=tmp_path,
    )

    manifest.stages["perturbative_matching"].defaults["zs_fm"] = 0.3
    matching_job = manifest.stages["perturbative_matching"].jobs[0]
    matching_job.params["zs_fm"] = 0.4
    matching_args = prepare_tool_args(
        "build_matching_kernel",
        {},
        manifest=manifest,
        stage="perturbative_matching",
        job=matching_job,
        effective_params=effective_matching_params(manifest, matching_job),
        artifacts_dir=tmp_path,
        store={"quasi": object()},
    )

    assert renorm_args["scheme_parameters"]["zs_fm"] == 0.2
    assert matching_args["zs_fm"] == 0.4


def test_prepare_matching_plot_limits(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["perturbative_matching"].jobs[0]
    effective = {**effective_matching_params(manifest, job), "plot": {"xlim": [-1.0, 2.0], "ylim": [-0.2, 2.5]}}
    args = prepare_tool_args(
        "plot_matched_pdf", {}, manifest=manifest, stage="perturbative_matching", job=job,
        effective_params=effective, artifacts_dir=tmp_path, store={"quasi": object()},
    )
    assert args["xlim"] == [-1.0, 2.0]
    assert args["ylim"] == [-0.2, 2.5]


def test_new_downstream_job_validators_accept_full_manifest() -> None:
    manifest = _manifest()
    for stage in ("fourier_transform", "perturbative_matching"):
        job = manifest.stages[stage].jobs[0]
        assert validate_stage_inputs(stage, manifest, job) == []


def test_extrapolation_tool_args_use_allow_order_lists(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "metadata": {
                "run_id": "ex",
                "root_directory": ".",
                "artifacts_directory": "artifacts",
                "target_observable": "pdf",
                "parton": "quark",
                "resample_mode": "jk",
                "sample_error_mode": "covariance",
                "random_seed": 1984,
                "workers": 4,
                "stages": ["extrapolation"],
            },
            "inputs": {
                "correlators": [],
                "artifacts": [
                    {"id": "mt1", "stage": "perturbative_matching", "path": "mt1.nc"},
                    {"id": "mt2", "stage": "perturbative_matching", "path": "mt2.nc"},
                ],
                "kernels": [],
            },
            "stages": {
                "extrapolation": {
                    "defaults": {
                        "allow_order_a": [1, 2],
                        "allow_order_1overp": [2, 4],
                        "allow_order_ap": [2],
                        "fitting_param_xdep": [False, True, True],
                        "pdep_gev": [1.5, 2.0],
                    },
                    "jobs": [{"id": "extrapolate_all", "inputs": {"lightcone": ["mt1", "mt2"]}}],
                }
            },
        }
    )
    job = manifest.stages["extrapolation"].jobs[0]
    args = prepare_tool_args(
        "run_extrapolation",
        {},
        manifest=manifest,
        stage="extrapolation",
        job=job,
        effective_params=merge_stage_params(manifest.stages["extrapolation"].defaults, job.params),
        artifacts_dir=tmp_path,
        store={},
    )
    assert args["allow_order_a"] == [1, 2]
    assert args["allow_order_1overp"] == [2, 4]
    assert args["allow_order_ap"] == [2]
    assert args["fitting_param_xdep"] == [False, True, True]
    assert args["workers"] == 4


def test_hybrid_stage_validators_use_flat_effective_zs_fm() -> None:
    manifest = _manifest()
    renorm_job = manifest.stages["renormalization"].jobs[0]
    matching_job = manifest.stages["perturbative_matching"].jobs[0]
    renorm_job.params.pop("zs_fm")
    matching_job.params.pop("zs_fm")
    manifest.stages["renormalization"].defaults["zs_fm"] = 0.2
    manifest.stages["perturbative_matching"].defaults["zs_fm"] = 0.2

    assert validate_stage_inputs("renormalization", manifest, renorm_job) == []
    assert validate_stage_inputs("perturbative_matching", manifest, matching_job) == []

    manifest.stages["renormalization"].defaults.pop("zs_fm")
    manifest.stages["perturbative_matching"].defaults.pop("zs_fm")
    assert "flat parameter zs_fm" in validate_stage_inputs("renormalization", manifest, renorm_job)[0]
    assert "flat parameter zs_fm" in validate_stage_inputs("perturbative_matching", manifest, matching_job)[0]


@pytest.mark.parametrize(
    "path",
    [
        Path("examples/sample_manifest.jsonc"),
        Path("examples/partial_sample_manifest.jsonc"),
        Path("examples/pion_pdf_cg_manifest.json"),
        Path("examples/pion_pdf_gi_manifest.json"),
    ],
)
def test_example_manifests_validate(path: Path) -> None:
    manifest = validate_manifest_file(path)
    for stage_id, stage_cfg in manifest.stages.items():
        if stage_id == "extrapolation":
            continue
        for job in stage_cfg.jobs:
            assert validate_stage_inputs(stage_id, manifest, job) == []
