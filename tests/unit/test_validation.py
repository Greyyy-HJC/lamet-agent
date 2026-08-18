import json
import math
from pathlib import Path

import pytest

from lamet_agent.manifest import (
    HBAR_C_GEV_FM,
    AnalysisManifest,
    ArtifactInput,
    ManifestPathError,
    CorrelatorInput,
    derive_job_kinematics,
    parse_momentum,
    physical_momentum_gev,
    resolve_artifact_metadata,
    resolve_manifest_artifact_metadata,
    validate_manifest_file,
    validate_manifest_paths,
)
from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import validate_stage_diagnostics, validate_stage_inputs


def _correlator_payload(correlator_type: str = "2pt") -> dict:
    payload = {
        "correlator_id": "c",
        "correlator_type": correlator_type,
        "data_path": "data.h5",
        "ensemble": "E",
        "hadron": "pion",
        "gfix": "CG",
        "source_operator": "g5",
        "sink_operator": "g5",
        "volume": "S48T64",
        "lattice_spacing_fm": 0.0574,
        "momentum": ["PX0PY0PZ0", "PX5PY0PZ0"],
    }
    if correlator_type == "3pt":
        payload.update(
            {
                "current_operator": "gT_nonlocal", "polarization": "unpolarized", "bz_direction": "Z",
                "tsep": [8, 10, 12],
                "bT": [0],
                "bz": [0, 1],
            }
        )
    return payload


def _path_validation_payload(root_directory: str) -> dict:
    correlator = _correlator_payload()
    correlator["data_path"] = "data.h5"
    return {
        "metadata": {
            "run_id": "paths",
            "root_directory": root_directory,
            "artifacts_directory": "new-artifacts",
            "target_observable": "pdf",
            "parton": "quark",
            "resample_mode": "jk",
            "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [correlator],
            "artifacts": [
                {"id": "external", "stage": "renormalization", "path": "external.bin"}
            ],
            "kernels": [
                {
                    "stage": "perturbative_matching",
                    "kernel_id": "CG_gt_quark_PDF_ratio_NLO",
                    "kernel_path": "kernel.py",
                }
            ],
        },
        "stages": {
            "correlator_analysis": {
                "defaults": {},
                "jobs": [{"id": "ca", "correlator_ids": ["c"]}],
            }
        },
    }


def test_validate_manifest_paths_accepts_relative_project_root_and_missing_output_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    examples = root / "examples"
    examples.mkdir(parents=True)
    (root / "data.h5").write_bytes(b"data")
    (root / "external.bin").write_bytes(b"artifact")
    (root / "kernel.py").write_text("# kernel\n", encoding="utf-8")
    manifest_path = examples / "manifest.json"
    manifest_path.write_text(json.dumps(_path_validation_payload("..")), encoding="utf-8")

    manifest = validate_manifest_file(manifest_path)
    validate_manifest_paths(manifest, project_root=root)

    assert manifest.root_directory == root.resolve()
    assert not manifest.artifacts_directory.exists()


def test_validate_manifest_paths_reports_all_missing_or_non_file_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "data.h5").mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(_path_validation_payload(".")), encoding="utf-8")
    manifest = validate_manifest_file(manifest_path)

    with pytest.raises(ManifestPathError) as exc_info:
        validate_manifest_paths(manifest, project_root=root)

    message = str(exc_info.value)
    assert "inputs.correlators[0].data_path is not a file" in message
    assert "inputs.artifacts[0].path does not exist" in message
    assert "inputs.kernels[0].kernel_path does not exist" in message


def test_validate_manifest_paths_rejects_wrong_or_missing_project_root(
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "project"
    expected_root.mkdir()
    missing_root = tmp_path / "missing-project"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_path_validation_payload(str(missing_root))),
        encoding="utf-8",
    )
    manifest = validate_manifest_file(manifest_path)

    with pytest.raises(ManifestPathError) as exc_info:
        validate_manifest_paths(manifest, project_root=expected_root)

    assert exc_info.value.issues == (
        "metadata.root_directory must resolve to the lamet-agent project root: "
        f"expected {expected_root.resolve()}, got {missing_root.resolve()}",
    )


def test_validate_manifest_resolves_root_relative_source_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    examples = root / "examples"
    examples.mkdir(parents=True)
    payload = {
        "metadata": {
            "run_id": "demo", "root_directory": "..", "artifacts_directory": "runs/artifacts",
            "target_observable": "pdf", "parton": "quark", "resample_mode": "jk", "sample_error_mode": "covariance",
            "random_seed": 1984,
            "stages": ["correlator_analysis"],
        },
        "inputs": {
            "correlators": [{
                "correlator_id": "c2", "correlator_type": "2pt", "data_path": "data/c2.h5",
                "ensemble": "E", "hadron": "pion", "gfix": "CG",
                "source_operator": "g5", "sink_operator": "g5", "volume": "S16T32",
                "momentum": ["PX0PY0PZ0"], "lattice_spacing_fm": 0.1,
            }],
            "artifacts": [], "kernels": [],
        },
        "stages": {"correlator_analysis": {"defaults": {}, "jobs": [{"id": "ca", "correlator_ids": ["c2"]}]}},
    }
    path = examples / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = validate_manifest_file(path)
    assert manifest.root_directory == root.resolve()
    assert manifest.artifacts_directory == (root / "runs" / "artifacts").resolve()
    assert manifest.correlators[0].data_path == str((root / "data" / "c2.h5").resolve())


@pytest.mark.parametrize("field", ["momentum", "tsep"])
def test_correlator_setting_fields_require_lists(field: str) -> None:
    payload = _correlator_payload("3pt")
    payload[field] = payload[field][0]
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("volume", "48x64"),
        ("momentum", ["P5"]),
        ("momentum", ["PX0PY0PZ0", "PX0PY0PZ0"]),
        ("tsep", [8, 8]),
        ("bT", [0, 0]),
        ("bz", [0, 0]),
    ],
)
def test_correlator_rejects_invalid_or_duplicate_settings(field: str, value: object) -> None:
    payload = _correlator_payload("3pt")
    payload[field] = value
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(payload)


@pytest.mark.parametrize("direction", ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"])
def test_correlator_accepts_canonical_bz_directions(direction: str) -> None:
    payload = _correlator_payload("3pt")
    payload["bz_direction"] = direction
    assert CorrelatorInput.model_validate(payload).bz_direction == direction


@pytest.mark.parametrize("polarization", ["unpolarized", "helicity", "transversity"])
def test_3pt_correlator_polarization(polarization: str) -> None:
    payload = _correlator_payload("3pt")
    payload["polarization"] = polarization

    assert CorrelatorInput.model_validate(payload).polarization == polarization

def test_3pt_correlator_requires_polarization() -> None:
    payload = _correlator_payload("3pt")
    payload.pop("polarization")
    with pytest.raises(ValueError, match="polarization"):
        CorrelatorInput.model_validate(payload)



def test_correlator_rejects_unknown_polarization() -> None:
    payload = _correlator_payload("3pt")
    payload["polarization"] = "polarized"

    with pytest.raises(ValueError, match="polarization"):
        CorrelatorInput.model_validate(payload)


@pytest.mark.parametrize("direction", ["x", "YX", "XX", "XYZW", "longitudinal", ""])
def test_correlator_rejects_noncanonical_bz_directions(direction: str) -> None:
    payload = _correlator_payload("3pt")
    payload["bz_direction"] = direction
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(payload)


@pytest.mark.parametrize(
    "removed_field",
    ["kind", "source_sink", "src_gamma", "sink_gamma", "current_gamma", "distribution_type", "a_fm", "pz_gev", "z_direction", "eta", "bt"],
)
def test_correlator_rejects_removed_fields(removed_field: str) -> None:
    payload = _correlator_payload("3pt")
    payload[removed_field] = "removed"
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(payload)


def test_correlator_rejects_4pt_and_enforces_conditional_fields() -> None:
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(_correlator_payload("4pt"))
    missing_current = _correlator_payload("3pt")
    missing_current.pop("current_operator")
    with pytest.raises(ValueError, match="current_operator"):
        CorrelatorInput.model_validate(missing_current)
    missing_direction = _correlator_payload("3pt")
    missing_direction.pop("bz_direction")
    with pytest.raises(ValueError, match="bz_direction"):
        CorrelatorInput.model_validate(missing_direction)
    extra_current = _correlator_payload("2pt")
    extra_current["current_operator"] = "gT_nonlocal"
    with pytest.raises(ValueError, match="only valid for 3pt"):
        CorrelatorInput.model_validate(extra_current)
    extra_polarization = _correlator_payload("2pt")
    extra_polarization["polarization"] = "helicity"
    with pytest.raises(ValueError, match="only valid for 3pt"):
        CorrelatorInput.model_validate(extra_polarization)
    da_like = _correlator_payload("2pt")
    da_like["bz_direction"] = "Z"
    da_like["bT"] = [0]
    da_like["bz"] = [0, 1]
    assert CorrelatorInput.model_validate(da_like).bz == [0, 1]
    extra_tsep = _correlator_payload("2pt")
    extra_tsep["tsep"] = [8]
    with pytest.raises(ValueError, match="only valid for 3pt"):
        CorrelatorInput.model_validate(extra_tsep)


def test_momentum_helpers_cover_zero_negative_axes_and_xyz_norm() -> None:
    assert parse_momentum("PX0PY0PZ0") == (0, 0, 0)
    assert parse_momentum("PX-2PY3PZ-4") == (-2, 3, -4)
    assert physical_momentum_gev("PX0PY0PZ0", "S48T64", 0.0574) == 0.0
    unit = 2 * math.pi * HBAR_C_GEV_FM / (48 * 0.0574)
    assert physical_momentum_gev("PX1PY0PZ0", "S48T64", 0.0574) == pytest.approx(unit)
    assert physical_momentum_gev("PX-1PY0PZ0", "S48T64", 0.0574) == pytest.approx(unit)
    assert physical_momentum_gev("PX3PY3PZ3", "S48T64", 0.0574) == pytest.approx(unit * math.sqrt(27))
    assert physical_momentum_gev("PX5PY0PZ0", "S48T64", 0.0574) == pytest.approx(2.250003600391, rel=1e-12)


def test_partial_artifact_manifest_fallback_requires_complete_discrete_kinematics() -> None:
    valid = {
        "id": "rn",
        "stage": "renormalization",
        "path": "rn.nc",
        "momentum": "PX5PY0PZ0",
        "volume": "S48T64",
        "lattice_spacing_fm": 0.0574,
    }
    artifact = ArtifactInput.model_validate(valid)
    assert artifact.momentum_gev == pytest.approx(2.250003600391, rel=1e-12)
    with pytest.raises(ValueError, match="declared together"):
        ArtifactInput.model_validate({**valid, "volume": None})
    with pytest.raises(ValueError, match="derived"):
        ArtifactInput.model_validate({**valid, "momentum_gev": 2.15})


def _write_partial_artifact(path: Path, attrs: dict[str, str] | None = None) -> None:
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[[1.0 + 0.1j], [0.9 + 0.2j]],
        dims=("z",),
        coords={"z": [0.0]},
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    data.to_netcdf(path)


def _partial_fourier_payload(artifact: dict) -> dict:
    return {
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
        "inputs": {"correlators": [], "artifacts": [artifact], "kernels": []},
        "stages": {
            "fourier_transform": {
                "defaults": {
                    "Lambda0_gev": 0.0,
                    "method": "CG",
                    "order": "NLA",
                    "posterior_prior_error_scale": 3.0,
                    "sector": "valence",
                    "scheme_scan": {
                        "zmin_fm": [1.0], "zmax_fm": [2.0], "zmax_ext_fm": 3.0,
                        "smooth": "linear", "model_average": True,
                    },
                    "quasi_y_ls": {"start": -1.0, "stop": 1.0, "num": 4},
                },
                "jobs": [{"id": "ft", "inputs": {"input": artifact["id"]}}],
            }
        },
    }


def test_partial_artifact_uses_netcdf_attrs_without_materializing_manifest_fields(tmp_path: Path) -> None:
    artifact_path = tmp_path / "rn.nc"
    _write_partial_artifact(
        artifact_path,
        attrs={
            "momentum": "PX5PY0PZ0",
            "volume": "S48T64",
            "lattice_spacing_fm": "0.0574",
            "hadron": "pion",
            "gfix": "CG",
            "bz_direction": "X",
            "observable": "pion_quark_quasi_pdf",
            "parton": "quark",
            "current_operator": "gTg5_nonlocal",
            "polarization": "helicity",
        },
    )
    original_bytes = artifact_path.read_bytes()
    payload = _partial_fourier_payload(
        {"id": "rn", "stage": "renormalization", "path": artifact_path.name}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = validate_manifest_file(manifest_path)
    artifact = manifest.inputs.artifacts[0]
    job = manifest.stages["fourier_transform"].jobs[0]
    kinematics = derive_job_kinematics(manifest, job)

    assert artifact.momentum is None
    assert artifact.volume is None
    assert artifact.lattice_spacing_fm is None
    assert artifact.model_dump(exclude_none=True) == {
        "id": "rn",
        "stage": "renormalization",
        "path": str(artifact_path),
    }
    assert artifact.resolved_metadata["hadron"] == "pion"
    assert artifact.resolved_metadata["observable"] == "pion_quark_quasi_pdf"
    assert artifact.resolved_metadata["current_operator"] == "gTg5_nonlocal"
    assert artifact.resolved_metadata["polarization"] == "helicity"
    assert kinematics["momentum"] == "PX5PY0PZ0"
    assert kinematics["observable"] == "pion_quark_quasi_pdf"
    assert kinematics["momentum_gev"] == pytest.approx(2.250003600391, rel=1e-12)
    assert validate_stage_inputs("fourier_transform", manifest, job) == []
    assert artifact_path.read_bytes() == original_bytes


def test_partial_artifact_uses_complete_manifest_fallback_for_legacy_netcdf(tmp_path: Path) -> None:
    artifact_path = tmp_path / "legacy.nc"
    _write_partial_artifact(artifact_path)
    artifact = ArtifactInput.model_validate(
        {
            "id": "legacy",
            "stage": "renormalization",
            "path": str(artifact_path),
            "momentum": "PX5PY0PZ0",
            "volume": "S48T64",
            "lattice_spacing_fm": 0.0574,
            "hadron": "pion",
        }
    )

    metadata = resolve_artifact_metadata(artifact)

    assert metadata["momentum"] == "PX5PY0PZ0"
    assert metadata["hadron"] == "pion"
    assert artifact.momentum_gev == pytest.approx(2.250003600391, rel=1e-12)


@pytest.mark.parametrize(
    ("field", "conflicting_value"),
    [
        ("momentum", "PX4PY0PZ0"),
        ("volume", "S64T96"),
        ("lattice_spacing_fm", 0.06),
        ("hadron", "proton"),
        ("gfix", "GI"),
        ("bz_direction", "Z"),
        ("observable", "nucleon_quark_quasi_pdf"),
        ("parton", "gluon"),
        ("current_operator", "gT_nonlocal"),
        ("polarization", "transversity"),
    ],
)
def test_partial_artifact_rejects_manifest_netcdf_metadata_conflicts(
    tmp_path: Path,
    field: str,
    conflicting_value: object,
) -> None:
    artifact_path = tmp_path / f"conflict_{field}.nc"
    file_metadata = {
        "momentum": "PX5PY0PZ0",
        "volume": "S48T64",
        "lattice_spacing_fm": "0.0574",
        "hadron": "pion",
        "gfix": "CG",
        "bz_direction": "X",
        "observable": "pion_quark_quasi_pdf",
        "parton": "quark",
        "current_operator": "gTg5_nonlocal",
        "polarization": "helicity",
    }
    _write_partial_artifact(artifact_path, attrs=file_metadata)
    declared = {
        "id": "rn",
        "stage": "renormalization",
        "path": str(artifact_path),
        **file_metadata,
        field: conflicting_value,
    }
    artifact = ArtifactInput.model_validate(declared)

    with pytest.raises(ValueError, match=rf"artifact 'rn' metadata conflict for '{field}'"):
        resolve_artifact_metadata(artifact)


def test_partial_artifact_missing_kinematics_remains_a_stage_input_issue(tmp_path: Path) -> None:
    artifact_path = tmp_path / "incomplete.nc"
    _write_partial_artifact(artifact_path, attrs={"hadron": "pion"})
    manifest = AnalysisManifest.model_validate(
        _partial_fourier_payload(
            {"id": "rn", "stage": "renormalization", "path": str(artifact_path)}
        )
    )
    resolve_manifest_artifact_metadata(manifest)
    job = manifest.stages["fourier_transform"].jobs[0]

    diagnostics = validate_stage_diagnostics("fourier_transform", manifest, job)

    assert [item.code for item in diagnostics] == [
        "fourier.kinematics.momentum_required",
        "fourier.pdf_gpd.polarization_required",
    ]
    assert "Ioffe time" in diagnostics[0].physics
    assert "spin channel" in diagnostics[1].cause


def test_fourier_conflict_returns_structured_physics_diagnostic() -> None:
    payload = _partial_fourier_payload(
        {
            "id": "rn",
            "stage": "renormalization",
            "path": "rn.nc",
            "momentum": "PX1PY0PZ0",
            "volume": "S16T32",
            "lattice_spacing_fm": 0.1,
            "hadron": "pion",
            "polarization": "unpolarized",
        }
    )
    payload["stages"]["fourier_transform"]["defaults"]["part"] = "re"
    manifest = AnalysisManifest.model_validate(payload)
    job = manifest.stages["fourier_transform"].jobs[0]

    diagnostics = validate_stage_diagnostics("fourier_transform", manifest, job)

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "fourier.sector.manual_projection_conflict"
    assert "part" in diagnostics[0].cause
    assert "negative-x convention" in diagnostics[0].physics
    assert "remove the manual projection controls" in diagnostics[0].suggested_fix
