import numpy as np

from lamet_agent.core.data import EnsembleInfo, EnsembleData


def test_netcdf_roundtrip_with_ensemble(tmp_path) -> None:
    ensemble = EnsembleInfo("S", "E", 0.12, 0.10, 24, 64, 0.14)
    data = EnsembleData(
        ensemble,
        "raw",
        [np.array([1 + 2j, 3 + 4j]), np.array([5 + 6j, 7 + 8j])],
        dims=("z",),
        coords={"z": [0, 1]},
    )

    path = tmp_path / "data.nc"
    data.to_netcdf(path)
    reload = EnsembleData.from_netcdf(path)

    assert data.ensemble == reload.ensemble
    assert data.resample == reload.resample
    assert (data.array == reload.array).all()


def test_netcdf_roundtrip_preserves_stage_attrs(tmp_path) -> None:
    data = EnsembleData(
        EnsembleInfo("", "E", 1.0, 1.0, 1, 1, 0.0),
        "jackknife",
        [np.array([1 + 2j, 3 + 4j]), np.array([5 + 6j, 7 + 8j])],
        dims=("z",),
        coords={"z": [0, 1]},
        attrs={"ensemble": "E", "momentum": "PX0PY0PZ0"},
        name="bare_matrix_element",
    )

    path = tmp_path / "bare.nc"
    data.to_netcdf(path)
    reload = EnsembleData.from_netcdf(path)

    assert reload.ensemble.id == "E"
    assert reload.resample == "jackknife"
    assert reload.attrs["momentum"] == "PX0PY0PZ0"
    assert (data.array == reload.array).all()
