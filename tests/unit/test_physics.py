"""Focused physics boundary checks for the independent neo stages."""

from __future__ import annotations

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.parallel import fourier_transform
from lamet_agent.stages.fourier_transform.physics import complete_signed_z
from lamet_agent.stages.renormalization.physics import normalize_at_origin, physical_z_coordinates


def test_componentwise_signed_z_completion() -> None:
    data = EnsembleData(
        None, "bootstrap", [np.array([2.0 + 3.0j, 1.0 + 2.0j])], ["z"], {"z": [0.0, 1.0]}, attrs={"coord_unit": "fm"}
    )
    completed = complete_signed_z(data, {"real": "even", "imag": "odd"})
    assert completed.coords["z"] == [-1.0, 0.0, 1.0]
    assert np.allclose(completed.values[0], [1.0 - 2.0j, 2.0 + 3.0j, 1.0 + 2.0j])


def test_fourier_uses_declared_phase_and_prefactor() -> None:
    data = EnsembleData(
        None, "bootstrap", [np.array([1.0 + 0.0j, 1.0 + 0.0j])], ["z"], {"z": [-1.0, 1.0]}, attrs={"momentum_gev": 1.0}
    )
    result = fourier_transform(
        data, [-0.5, 0.5], momentum_gev=1.0, phase_sign=-1, x_shift=0.25, prefactor="one_over_2pi"
    )
    assert result.attrs["phase_sign"] == -1
    assert result.attrs["prefactor"] == "one_over_2pi"
    assert result.values.shape == (1, 2)


def test_lattice_z_is_converted_once() -> None:
    data = EnsembleData(
        None,
        "bootstrap",
        [np.array([1.0, 2.0])],
        ["z"],
        {"z": [0.0, 2.0]},
        attrs={"coord_unit": "lattice", "lattice_spacing_fm": 0.12},
    )
    converted = physical_z_coordinates(data)
    assert converted.coords["z"] == [0.0, 0.24]
    assert physical_z_coordinates(converted).coords["z"] == [0.0, 0.24]


def test_origin_normalization_uses_netcdf_safe_metadata(tmp_path) -> None:
    data = EnsembleData(
        None,
        "bootstrap",
        [np.array([2.0, 4.0]), np.array([3.0, 9.0])],
        ["z"],
        {"z": [0.0, 0.1]},
    )
    result = normalize_at_origin(data)
    result.to_netcdf(tmp_path / "normalized.nc")
    assert result.attrs["normalized_at_origin"] == 1
    assert np.allclose(result.values, [[1.0, 2.0], [1.0, 3.0]])
