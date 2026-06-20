"""Unit tests for job-aware core tool preparation."""

from pathlib import Path
from types import SimpleNamespace

from lamet_agent.core.tools import prepare_tool_args, resolve_plot_save_path, validate_stage_inputs
from lamet_agent.manifest import validate_manifest_file
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
    assert args["nstate_values"] == [2]
    assert args["fit_strategies"] == ["joint"]
    assert args["pt3_paths"]["8"].endswith("_3pt_ts8.h5")


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
    assert args["nstate"] == 2
    assert args["model_average"] is False


def test_prepare_renormalization_args_bind_roles_and_scheme(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["renormalization"].jobs[0]
    args = prepare_tool_args(
        "apply_ratio_scheme_renormalization", {}, manifest=manifest, stage="renormalization", job=job,
        effective_params=manifest.stages["renormalization"].defaults,
        artifacts_dir=tmp_path,
    )
    assert args["target"] == "target"
    assert args["denominator"] == "denominator"
    assert args["scheme"] == "hybrid_ratio"
    assert args["scheme_parameters"] == manifest.stages["renormalization"].defaults["scheme_parameters"]
    assert args["save_path"] == str(tmp_path / "rn_p5")


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
    assert args["resample_mode"] == "jk"


def test_prepare_matching_resolves_logical_kernel(tmp_path: Path) -> None:
    manifest = _manifest()
    job = manifest.stages["perturbative_matching"].jobs[0]
    args = prepare_tool_args(
        "build_matching_kernel", {}, manifest=manifest, stage="perturbative_matching", job=job,
        effective_params=effective_matching_params(manifest, job),
        artifacts_dir=tmp_path, store={"quasi": object()},
    )
    assert args["kernel_id"] == "CG_gt_PDF_hybrid"
    assert args["pz_gev"] == 2.15
    assert args["zs_fm"] == 0.1722


def test_new_downstream_job_validators_accept_full_manifest() -> None:
    manifest = _manifest()
    for stage in ("fourier_transform", "perturbative_matching"):
        job = manifest.stages[stage].jobs[0]
        assert validate_stage_inputs(stage, manifest, job) == []
