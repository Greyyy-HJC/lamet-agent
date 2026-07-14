"""Unit tests for job-aware core tool preparation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from lamet_agent.core.tools import prepare_tool_args, resolve_plot_save_path, validate_stage_inputs
from lamet_agent.manifest import validate_manifest_file
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.matching.skills import effective_matching_params


def _manifest():
    return validate_manifest_file(Path("examples/cg_pion_pdf_manifest.json"))


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
    assert args["momentum"] == "PX5PY0PZ0"
    assert args["tsep_ls"] == [8, 10, 12]
    assert args["z_values"] == list(range(25))
    assert "tune_z_values" not in args
    assert args["nstate_values"] == effective["nstate"]
    assert args["fit_strategies"] == effective["fit_strategy"]
    assert args["fit_scope_values"] == ["ratio"]
    assert args["pt3_paths"]["8"].endswith("_3pt_ts8.h5")
    assert args["resample_mode"] == "jk"
    assert args["sample_error_mode"] == manifest.metadata.sample_error_mode
    assert args["seed"] == manifest.metadata.random_seed
    assert "workers" not in args
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
            "inputs": {"correlators": [], "artifacts": [], "kernels": []},
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
    assert args["save_path"] == str(tmp_path / "ca_p0")
    assert args["job_id"] == "ca_p0"
    assert args["a_fm"] == 0.0574
    if manifest.stages["correlator_analysis"].defaults["model_average"]:
        assert args["nstate_values"] == manifest.stages["correlator_analysis"].defaults["nstate"]
        assert "nstate" not in args
    else:
        assert args["nstate"] == 2
    assert "fit_scope" not in args
    assert args["model_average"] == manifest.stages["correlator_analysis"].defaults["model_average"]
    assert args["workers"] == 1


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

    assert correlator_args["workers"] == 3
    assert fourier_args["workers"] == 3


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
                        "kind": "2pt",
                        "data_path": "pt2_i.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_sink": "SS",
                        "momentum": "PX0PY0PZ0",
                        "a_fm": 0.1,
                        "pz_gev": 0.0,
                        "src_gamma": "5",
                        "sink_gamma": "5",
                    },
                    {
                        "correlator_id": "pt2_f",
                        "kind": "2pt",
                        "data_path": "pt2_f.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_sink": "SS",
                        "momentum": "PX0PY0PZ1",
                        "a_fm": 0.1,
                        "pz_gev": 0.5,
                        "src_gamma": "5",
                        "sink_gamma": "5",
                    },
                    {
                        "correlator_id": "pt3_fi",
                        "kind": "3pt",
                        "data_path": "pt3.h5",
                        "ensemble": "E",
                        "hadron": "pion",
                        "gfix": "GI",
                        "source_sink": "SS",
                        "momentum": "PX0PY0PZ1",
                        "a_fm": 0.1,
                        "pz_gev": 0.0,
                        "pz_out_gev": 0.5,
                        "src_gamma": "5",
                        "sink_gamma": "5",
                        "current_gamma": "T",
                        "z_direction": "Z",
                        "eta": "eta0",
                        "bt": [0],
                        "bz": [0],
                        "tsep": 8,
                    },
                ]
            },
            "stages": {
                "correlator_analysis": {
                    "defaults": {"fitting_form": "NonBreit"},
                    "jobs": [{"id": "ca", "correlator_ids": ["pt2_i", "pt2_f", "pt3_fi"]}],
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
    assert args["pt2_path"].endswith("pt2_i.h5")
    assert args["pt2_out_path"].endswith("pt2_f.h5")
    assert args["momentum"] == "PX0PY0PZ0"
    assert args["momentum_out"] == "PX0PY0PZ1"
    assert args["pt3_momentum"] == "PX0PY0PZ1"
    assert args["pz_gev"] == 0.0
    assert args["pz_out_gev"] == 0.5


def test_nonbreit_requires_two_two_point_correlators(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0].model_copy(update={"params": {"fitting_form": "NonBreit"}})
    assert validate_stage_inputs("correlator_analysis", manifest, job) == [
        "A NonBreit correlator_analysis job requires exactly two 2pt correlators."
    ]


def test_prepare_renormalization_args_bind_roles_and_scheme(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    args = prepare_tool_args(
        "apply_ratio_scheme_renormalization", {}, manifest=manifest, stage="renormalization", job=job,
        effective_params={**manifest.stages["renormalization"].defaults, **job.params},
        artifacts_dir=tmp_path,
    )
    assert args["target"] == "target"
    assert args["denominator"] == "denominator"
    assert args["scheme"] == "hybrid_ratio"
    assert args["scheme_parameters"]["zs_fm"] == job.params["zs_fm"]
    assert args["scheme_parameters"]["m0_gev"] == manifest.stages["renormalization"].defaults["scheme_parameters"]["m0_gev"]
    assert args["scheme_parameters"]["delta_m_gev"] == manifest.stages["renormalization"].defaults["scheme_parameters"]["delta_m_gev"]
    assert args["save_path"] == str(tmp_path / "rn_p5")
    assert "normalization" not in args


def test_prepare_renormalization_args_filters_normalization_manifest_flag(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    effective = {**manifest.stages["renormalization"].defaults, **job.params, "normalization": True}
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


def test_prepare_self_renormalization_args_bind_kernel_and_roles(tmp_path: Path) -> None:
    manifest = validate_manifest_file(Path("examples/temp_self_renorm_manifest.json"))
    fit_job = manifest.stages["renormalization"].jobs[0]
    apply_job = manifest.stages["renormalization"].jobs[1]
    fit_effective = {**manifest.stages["renormalization"].defaults, **fit_job.params}
    apply_effective = {**manifest.stages["renormalization"].defaults, **apply_job.params}

    assert set(fit_job.inputs) == {"reference"}
    assert set(apply_job.inputs) == {"target", "zR"}
    assert validate_stage_inputs("renormalization", manifest, fit_job) == []
    assert validate_stage_inputs("renormalization", manifest, apply_job) == []

    fit_args = prepare_tool_args(
        "fit_self_renormalization_factor",
        {},
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
    assert fit_args["svdcut"] == 1e-12
    assert fit_args["save_path"] == str(tmp_path / "rn_zR_fit")
    # Fit-job params carry required d (PDF); m0_gev omitted → fit.
    assert fit_effective["d"] == -0.08183
    assert "m0_gev" not in fit_effective

    apply_args = prepare_tool_args(
        "apply_self_renormalization",
        {},
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
    assert apply_args["d"] == 0.19
    assert apply_args["m0_gev"] == -0.094
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
    assert apply_diag["include_discrete_effect"] is False
    assert apply_diag["sibling_artifacts"] == []

    last_apply = manifest.stages["renormalization"].jobs[-1]
    last_effective = {**manifest.stages["renormalization"].defaults, **last_apply.params}
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
    source = SimpleNamespace(attrs={"a_fm": "0.0574", "pz_gev": "2.15", "hadron": "pion", "gfix": "CG"})
    effective = {**manifest.stages["fourier_transform"].defaults, **job.params}
    args = prepare_tool_args(
        "run_fourier_transform", {}, manifest=manifest, stage="fourier_transform", job=job,
        effective_params=effective, artifacts_dir=tmp_path, store={"input": source},
    )
    assert args["method"] == "CG"
    assert args["observable"] == "pion_quark_quasi_pdf"
    assert args["a_fm"] == "0.0574"
    assert args["pz_gev"] == 2.15
    assert args["workers"] == 1
    assert args["save_path"] == str(tmp_path / "ft_p5")


def test_prepare_partial_fourier_loader_uses_external_artifact(tmp_path: Path) -> None:
    manifest = validate_manifest_file(Path("examples/partial_cg_pion_pdf_manifest.json"))
    job = manifest.stages["fourier_transform"].jobs[0]
    source = manifest.inputs.artifacts[0]
    args = prepare_tool_args(
        "load_renormalized_matrix_element_samples", {}, manifest=manifest,
        stage="fourier_transform", job=job,
        effective_params={**manifest.stages["fourier_transform"].defaults, **job.params},
        artifacts_dir=tmp_path, store={"input": source},
    )
    assert args["path"] == source.path


def test_prepare_partial_fourier_loader_uses_manifest_artifact_after_hydration(tmp_path: Path) -> None:
    manifest = validate_manifest_file(Path("examples/partial_cg_pion_pdf_manifest.json"))
    job = manifest.stages["fourier_transform"].jobs[0]
    source = manifest.inputs.artifacts[0]
    from lamet_agent.core.data import EnsembleData

    quasi = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[[1.0 + 0.1j]],
        dims=("z",),
        coords={"z": [0.0]},
        name="renormalized_matrix_element",
    )
    args = prepare_tool_args(
        "load_renormalized_matrix_element_samples",
        {},
        manifest=manifest,
        stage="fourier_transform",
        job=job,
        effective_params={**manifest.stages["fourier_transform"].defaults, **job.params},
        artifacts_dir=tmp_path,
        store={"input": quasi, "matrix_element_data": quasi},
    )
    assert args["path"] == source.path
    assert args["resample_mode"] == "bs"


def test_prepare_matching_resolves_logical_kernel(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["perturbative_matching"].jobs[0]
    args = prepare_tool_args(
        "build_matching_kernel", {}, manifest=manifest, stage="perturbative_matching", job=job,
        effective_params=effective_matching_params(manifest, job),
        artifacts_dir=tmp_path, store={"quasi": object()},
    )
    assert args["kernel_id"] == "CG_gt_qPDF_hybrid_NLO"
    assert args["pz_gev"] == 2.15
    assert args["zs_fm"] == 0.1722


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
        effective_params={**manifest.stages["renormalization"].defaults, **renorm_job.params},
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
        Path("examples/cg_pion_pdf_manifest.json"),
        Path("examples/gi_pion_pdf_manifest.json"),
        Path("examples/partial_cg_pion_pdf_manifest.json"),
    ],
)
def test_example_manifests_validate(path: Path) -> None:
    manifest = validate_manifest_file(path)
    for stage_id, stage_cfg in manifest.stages.items():
        if stage_id == "extrapolation":
            continue
        for job in stage_cfg.jobs:
            assert validate_stage_inputs(stage_id, manifest, job) == []
