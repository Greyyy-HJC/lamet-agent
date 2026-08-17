import pytest
from pydantic import ValidationError

from lamet_agent.manifest import AnalysisManifest
from lamet_agent.manifest_params import (
    ListItems,
    ParameterSpec,
    STAGE_PARAM_CONTRACTS,
    get_stage_parameter_contract,
    merge_stage_params,
    validate_stage_parameter_mapping,
)


def test_stage_params_merge_recursively_without_mutating_inputs() -> None:
    defaults = {
        "scheme_parameters": {"LambdaQCD_gev": 0.1, "d": -0.08},
        "order": ["LA", "NLA"],
    }
    overrides = {
        "scheme_parameters": {"d": 0.19, "m0_gev": -0.094},
        "order": ["NLA"],
    }

    merged = merge_stage_params(defaults, overrides)

    assert merged == {
        "scheme_parameters": {
            "LambdaQCD_gev": 0.1,
            "d": 0.19,
            "m0_gev": -0.094,
        },
        "order": ["NLA"],
    }
    assert defaults["scheme_parameters"] == {"LambdaQCD_gev": 0.1, "d": -0.08}


def _payload() -> dict:
    return {
        "metadata": {
            "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
            "parton": "quark", "resample_mode": "jk", "random_seed": 1984, "stages": ["correlator_analysis"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca"}]}},
    }


def test_manifest_schema_uses_metadata_inputs_and_stage_jobs() -> None:
    manifest = AnalysisManifest.model_validate(_payload())
    assert manifest.run_id == "demo"
    assert manifest.metadata.workers == 1
    assert manifest.stages["correlator_analysis"].jobs[0].id == "ca"


def test_manifest_rejects_removed_correlator_analysis_mode() -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["defaults"]["analysis_mode"] = "2pt_ratio"
    with pytest.raises(ValidationError, match="analysis_mode"):
        AnalysisManifest.model_validate(payload)


@pytest.mark.parametrize("parameter", ["reference_z", "z_values"])
def test_manifest_rejects_runner_owned_qda_grid_parameters(parameter: str) -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["defaults"][parameter] = 0
    with pytest.raises(ValidationError, match=parameter):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_positive_workers() -> None:
    payload = _payload()
    payload["metadata"]["workers"] = 4
    assert AnalysisManifest.model_validate(payload).metadata.workers == 4


@pytest.mark.parametrize("workers", [0, -1, 1.5, "2"])
def test_manifest_rejects_invalid_workers(workers) -> None:
    payload = _payload()
    payload["metadata"]["workers"] = workers
    with pytest.raises(ValidationError, match="workers"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_forward_job_reference() -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["jobs"][0]["inputs"] = {"input": "later"}
    with pytest.raises(ValidationError, match="unavailable upstream"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_bs_mode_without_bs_samples() -> None:
    payload = _payload()
    payload["metadata"]["resample_mode"] = "bs"
    with pytest.raises(ValidationError, match="bs_samples"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_bs_mode_with_bs_samples() -> None:
    payload = _payload()
    payload["metadata"]["resample_mode"] = "bs"
    payload["metadata"]["bs_samples"] = 500
    manifest = AnalysisManifest.model_validate(payload)
    assert manifest.metadata.bs_samples == 500


def _fourier_payload() -> dict:
    payload = _payload()
    payload["metadata"]["stages"] = ["fourier_transform"]
    payload["inputs"]["artifacts"] = [
        {"id": "rn", "stage": "renormalization", "path": "rn.nc"}
    ]
    payload["stages"] = {
        "fourier_transform": {
            "defaults": {"Lambda0_gev": 0.0},
            "jobs": [{"id": "ft", "inputs": {"input": "rn"}}],
        }
    }
    return payload


def test_manifest_rejects_legacy_fourier_lambda0_in_defaults() -> None:
    payload = _fourier_payload()
    payload["stages"]["fourier_transform"]["defaults"] = {"Lambda0": 0.1}

    with pytest.raises(ValidationError, match=r"fourier_transform\.defaults\.Lambda0"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_legacy_fourier_lambda0_in_job_params() -> None:
    payload = _fourier_payload()
    payload["stages"]["fourier_transform"]["jobs"][0]["params"] = {"Lambda0": 0.1}

    with pytest.raises(ValidationError, match=r"fourier_transform\.jobs\[0\]\.params\.Lambda0"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_fourier_lambda0_gev() -> None:
    AnalysisManifest.model_validate(_fourier_payload())


def test_every_stage_contract_is_stage_owned_and_documents_physics() -> None:
    assert STAGE_PARAM_CONTRACTS == {
        "correlator_analysis": "lamet_agent.stages.correlator.validation:STAGE_PARAM_CONTRACT",
        "renormalization": "lamet_agent.stages.renorm.validation:STAGE_PARAM_CONTRACT",
        "fourier_transform": "lamet_agent.stages.fourier.validation:STAGE_PARAM_CONTRACT",
        "perturbative_matching": "lamet_agent.stages.matching.validation:STAGE_PARAM_CONTRACT",
        "extrapolation": "lamet_agent.stages.extrapolation.validation:STAGE_PARAM_CONTRACT",
        "review": "lamet_agent.stages.review.validation:STAGE_PARAM_CONTRACT",
    }
    for stage in STAGE_PARAM_CONTRACTS:
        stage_contract = get_stage_parameter_contract(stage)
        assert stage_contract.summary
        assert stage_contract.physics
        assert all(
            isinstance(spec, (ParameterSpec, ListItems))
            for spec in stage_contract.schema.values()
        )

    contract = get_stage_parameter_contract("fourier_transform")
    sector = contract.schema["sector"]
    y_grid = contract.schema["y_grid"]

    assert isinstance(sector, ParameterSpec)
    assert isinstance(y_grid, ParameterSpec)
    assert y_grid.required is True
    assert callable(y_grid.validator)
    assert sector.choices == ("sea", "valence", "singlet", "full")
    assert "negative-x extension" in sector.physics
    assert not any(item.code == "fourier.y_grid.required" for item in contract.constraints)
    assert any(
        item.code == "fourier.sector.manual_projection_conflict"
        and "normalization" in item.physics
        and callable(item.check)
        for item in contract.constraints
    )


def test_fourier_parameter_type_error_includes_parameter_physics() -> None:
    payload = _fourier_payload()
    payload["stages"]["fourier_transform"]["defaults"]["symmetry_guarantee"] = "true"

    with pytest.raises(ValidationError) as exc_info:
        AnalysisManifest.model_validate(payload)

    message = str(exc_info.value)
    assert "symmetry_guarantee" in message
    assert "must be bool" in message
    assert "DA phase rotation and symmetry projection" in message


def test_manifest_rejects_unknown_defaults_and_job_params_together() -> None:
    payload = _fourier_payload()
    payload["stages"]["fourier_transform"]["defaults"]["posterior_prior_eror_scale"] = 3.0
    payload["stages"]["fourier_transform"]["jobs"][0]["params"] = {"unused_knob": True}

    with pytest.raises(ValidationError) as exc_info:
        AnalysisManifest.model_validate(payload)

    message = str(exc_info.value)
    assert "stages.fourier_transform.defaults.posterior_prior_eror_scale" in message
    assert "did you mean 'posterior_prior_error_scale'?" in message
    assert "stages.fourier_transform.jobs[0].params.unused_knob" in message


def test_manifest_rejects_runner_derived_stage_kinematics() -> None:
    payload = _fourier_payload()
    payload["stages"]["fourier_transform"]["defaults"].update(
        {"pz_gev": 2.1, "momentum_gev": 2.2}
    )

    with pytest.raises(ValidationError) as exc_info:
        AnalysisManifest.model_validate(payload)

    message = str(exc_info.value)
    assert "stages.fourier_transform.defaults.pz_gev" in message
    assert "stages.fourier_transform.defaults.momentum_gev" in message
    assert "runner-derived from upstream discrete momentum, volume, and lattice_spacing_fm" in message
    assert "inputs.artifacts[]" in message


def test_manifest_rejects_run_wide_stage_parameter() -> None:
    payload = _payload()
    payload["stages"]["correlator_analysis"]["defaults"]["workers"] = 4

    with pytest.raises(ValidationError, match=r"metadata\.workers"):
        AnalysisManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("stage", "defaults", "expected_path"),
    [
        (
            "correlator_analysis",
            {"pt2_windows": [{"tmin": 2, "tmax": 8, "tmiin": 3}]},
            r"stages\.correlator_analysis\.defaults\.pt2_windows\[0\]\.tmiin",
        ),
        (
            "fourier_transform",
            {"y_grid": {"start": -1.0, "stop": 1.0, "numm": 10}},
            r"stages\.fourier_transform\.defaults\.y_grid\.numm",
        ),
        (
            "fourier_transform",
            {"scheme_scan": {"zmin_values": [1.0], "smoth": "linear"}},
            r"stages\.fourier_transform\.defaults\.scheme_scan\.smoth",
        ),
        (
            "perturbative_matching",
            {"plot": {"xlim": [-1.0, 1.0], "ylimm": [-0.2, 1.0]}},
            r"stages\.perturbative_matching\.defaults\.plot\.ylimm",
        ),
    ],
)
def test_manifest_recursively_rejects_unknown_nested_stage_parameters(
    stage: str,
    defaults: dict,
    expected_path: str,
) -> None:
    payload = _payload()
    payload["metadata"]["stages"] = [stage]
    payload["stages"] = {stage: {"defaults": defaults, "jobs": [{"id": "job"}]}}

    with pytest.raises(ValidationError, match=expected_path):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_parameters_for_parameterless_stage() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["review"]
    payload["stages"] = {
        "review": {"defaults": {"freeform": True}, "jobs": [{"id": "review"}]}
    }

    with pytest.raises(ValidationError, match=r"stages\.review\.defaults\.freeform"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_review_literature_settings() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["review"]
    payload["stages"] = {
        "review": {
            "defaults": {"literature": True, "literature_max_papers": 4},
            "jobs": [{"id": "review"}],
        }
    }

    manifest = AnalysisManifest.model_validate(payload)

    assert manifest.stages["review"].defaults["literature"] is True
    assert manifest.stages["review"].defaults["literature_max_papers"] == 4


def test_manifest_rejects_unused_stage_configuration() -> None:
    payload = _payload()
    payload["stages"]["review"] = {
        "defaults": {"literature": False, "literature_max_papers": 4},
        "jobs": [{"id": "review"}],
    }

    with pytest.raises(ValidationError, match="unused stages"):
        AnalysisManifest.model_validate(payload)


def test_stage_parameter_contract_fails_closed_when_registry_entry_is_missing(monkeypatch) -> None:
    monkeypatch.delitem(STAGE_PARAM_CONTRACTS, "review")

    with pytest.raises(ValueError, match="must be registered in STAGE_PARAM_CONTRACTS"):
        validate_stage_parameter_mapping("review", {}, path="stages.review.defaults")


def test_manifest_rejects_zs_fm_in_kernel_parameters() -> None:
    payload = _payload()
    payload["inputs"]["kernels"] = [
        {
            "stage": "perturbative_matching",
            "kernel_id": "CG_gt_quark_PDF_hybrid_NLO",
            "kernel_path": "kernels.py",
            "kernel_parameters": {"zs_fm": 0.2},
        }
    ]
    with pytest.raises(ValidationError, match=r"inputs\.kernels\[0\]\.kernel_parameters\.zs_fm"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_stage_scheme_on_kernel_declaration() -> None:
    payload = _payload()
    payload["inputs"]["kernels"] = [
        {
            "stage": "perturbative_matching",
            "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
            "kernel_path": "kernels.py",
            "scheme": "ratio",
        }
    ]

    with pytest.raises(ValidationError, match=r"inputs\.kernels\[0\]\.scheme is no longer supported"):
        AnalysisManifest.model_validate(payload)


def _hybrid_self_payload() -> dict:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization"]
    payload["inputs"]["artifacts"] = [
        {"id": "reference", "stage": "correlator_analysis", "path": "reference.nc"}
    ]
    payload["inputs"]["kernels"] = [
        {
            "stage": "renormalization",
            "kernel_id": "ZMSbar_da",
            "kernel_path": "kernels.py",
            "kernel_parameters": {"mu": 2.0},
        }
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "ratio", "strategy": "self_renormalization"},
            "jobs": [
                {
                    "id": "fit",
                    "inputs": {"reference": "reference"},
                    "params": {"scheme_parameters": {"LambdaQCD_gev": 0.12, "d": -0.08183}},
                }
            ],
        }
    }
    return payload


@pytest.mark.parametrize(
    "key",
    ["alpha_s", "b0", "cf", "f1_extension_zmin_fm", "k", "lqcd", "Nf", "order", "zms_kind", "zr_zmax_fm"],
)
def test_manifest_rejects_removed_hybrid_self_parameters(key: str) -> None:
    payload = _hybrid_self_payload()
    payload["stages"]["renormalization"]["defaults"][key] = 0.1

    with pytest.raises(ValidationError, match=rf"renormalization\.defaults\.{key}"):
        AnalysisManifest.model_validate(payload)


@pytest.mark.parametrize("key", ["LambdaQCD_gev", "d", "m0_gev", "svdcut", "z_coverage_policy"])
def test_manifest_rejects_flat_hybrid_self_parameters(key: str) -> None:
    payload = _hybrid_self_payload()
    payload["stages"]["renormalization"]["defaults"][key] = 0.1

    with pytest.raises(ValidationError, match=rf"renormalization\.defaults\.{key}"):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_hybrid_self_scheme_parameters() -> None:
    manifest = AnalysisManifest.model_validate(_hybrid_self_payload())

    params = manifest.stages["renormalization"].jobs[0].params["scheme_parameters"]
    assert params["LambdaQCD_gev"] == pytest.approx(0.12)
    assert params["d"] == pytest.approx(-0.08183)


def test_manifest_rejects_legacy_lambdaqcd_name() -> None:
    payload = _hybrid_self_payload()
    params = payload["stages"]["renormalization"]["jobs"][0]["params"]["scheme_parameters"]
    params["LambdaQCD"] = params.pop("LambdaQCD_gev")

    with pytest.raises(ValidationError, match="LambdaQCD_gev"):
        AnalysisManifest.model_validate(payload)


@pytest.mark.parametrize("key", ["alpha_s", "LambdaQCD", "LambdaQCD_gev", "Nf", "order"])
def test_manifest_rejects_running_parameters_in_renormalization_kernel_parameters(key: str) -> None:
    payload = _hybrid_self_payload()
    payload["inputs"]["kernels"][0]["kernel_parameters"][key] = 0.332

    with pytest.raises(
        ValidationError,
        match=rf"inputs\.kernels\[0\]\.kernel_parameters\.{key}",
    ):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_zs_fm_in_renormalization_scheme_parameters() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization"]
    payload["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "scheme_parameters": {"zs_fm": 0.2}},
            "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
        }
    }
    with pytest.raises(ValidationError, match=r"renormalization\.defaults\.scheme_parameters\.zs_fm"):
        AnalysisManifest.model_validate(payload)


def test_manifest_rejects_zs_fm_in_renormalization_job_scheme_parameters() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization"]
    payload["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": 0.2},
            "jobs": [
                {
                    "id": "rn",
                    "inputs": {"target": "target", "denominator": "denominator"},
                    "params": {"scheme_parameters": {"zs_fm": 0.3}},
                }
            ],
        }
    }

    with pytest.raises(
        ValidationError,
        match=r"renormalization\.jobs\[0\]\.params\.scheme_parameters\.zs_fm",
    ):
        AnalysisManifest.model_validate(payload)


def test_manifest_accepts_flat_zs_fm_defaults_and_job_overrides() -> None:
    payload = _payload()
    payload["metadata"]["stages"] = ["renormalization", "perturbative_matching"]
    payload["inputs"]["artifacts"] = [
        {"id": "target", "stage": "correlator_analysis", "path": "target.nc"},
        {"id": "denominator", "stage": "correlator_analysis", "path": "denominator.nc"},
    ]
    payload["stages"] = {
        "renormalization": {
            "defaults": {"scheme": "hybrid", "strategy": "external_denominator", "zs_fm": 0.2},
            "jobs": [{"id": "rn", "inputs": {"target": "target", "denominator": "denominator"}}],
        },
        "perturbative_matching": {
            "defaults": {"scheme": "hybrid", "zs_fm": 0.2},
            "jobs": [{"id": "mt", "inputs": {"quasi": "rn"}, "params": {"zs_fm": 0.3}}],
        },
    }

    manifest = AnalysisManifest.model_validate(payload)

    assert manifest.stages["renormalization"].defaults["zs_fm"] == 0.2
    assert manifest.stages["perturbative_matching"].jobs[0].params["zs_fm"] == 0.3
