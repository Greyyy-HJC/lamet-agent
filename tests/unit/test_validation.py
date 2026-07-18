import json
import math
from pathlib import Path

import pytest

from lamet_agent.manifest import (
    HBAR_C_GEV_FM,
    AnalysisManifest,
    ArtifactInput,
    CorrelatorInput,
    derive_job_kinematics,
    parse_momentum,
    physical_momentum_gev,
    resolve_artifact_metadata,
    resolve_manifest_artifact_metadata,
    validate_manifest_file,
)
from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import validate_stage_inputs


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
                "current_operator": "gT_nonlocal", "bz_direction": "Z",
                "tsep": [8, 10, 12],
                "bT": [0],
                "bz": [0, 1],
            }
        )
    return payload


def test_validate_manifest_resolves_root_relative_source_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    examples = root / "examples"
    examples.mkdir(parents=True)
    payload = {
        "metadata": {
            "run_id": "demo", "root_directory": "..", "artifacts_directory": "runs/artifacts",
            "target_observable": "pdf", "parton": "quark", "resample_mode": "jk",
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


@pytest.mark.parametrize("direction", ["x", "YX", "XX", "XYZW", "longitudinal", ""])
def test_correlator_rejects_noncanonical_bz_directions(direction: str) -> None:
    payload = _correlator_payload("3pt")
    payload["bz_direction"] = direction
    with pytest.raises(ValueError):
        CorrelatorInput.model_validate(payload)


@pytest.mark.parametrize(
    "removed_field",
    ["kind", "source_sink", "src_gamma", "sink_gamma", "current_gamma", "a_fm", "pz_gev", "z_direction", "eta", "bt"],
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
            "random_seed": 1984,
            "stages": ["fourier_transform"],
        },
        "inputs": {"correlators": [], "artifacts": [artifact], "kernels": []},
        "stages": {
            "fourier_transform": {
                "defaults": {
                    "order": "NLA",
                    "part": "re",
                    "coord_unit": "lattice",
                    "y_grid": {"start": -1.0, "stop": 1.0, "num": 3},
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
    assert kinematics["momentum"] == "PX5PY0PZ0"
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

    assert validate_stage_inputs("fourier_transform", manifest, job) == [
        "Fourier job 'ft' is missing parameters: ['momentum_gev']"
    ]
