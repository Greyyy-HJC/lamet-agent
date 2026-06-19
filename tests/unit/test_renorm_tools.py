from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.renorm.functions import (
    apply_ratio_scheme_renormalization,
    load_bare_matrix_element_grid,
    plot_renormalized_matrix_element,
)


def _write_bare_netcdf(base: Path, stem: str, values: np.ndarray, *, resample: str = "jackknife") -> Path:
    data = EnsembleData(
        ensemble=EnsembleInfo("", "E", 1.0, 1.0, 1, 1, 0.0),
        resample=resample,
        values=[values[idx] for idx in range(values.shape[0])],
        dims=("z",),
        coords={"z": [0, 1, 4, 5]},
        attrs={"ensemble": "E", "momentum": "PX0PY0PZ0"},
        name="bare_matrix_element",
    )
    path = base / f"{stem}.nc"
    data.to_netcdf(path)
    return path


def test_load_bare_matrix_element_grid_reads_correlator_netcdf(tmp_path: Path) -> None:
    samples = np.asarray([[1 + 0.1j, 2 + 0.2j, 3 + 0.3j, 4 + 0.4j], [2 + 0.2j, 4 + 0.4j, 6 + 0.6j, 8 + 0.8j]])
    artifact = _write_bare_netcdf(tmp_path, "target", samples)
    store = {}

    result = load_bare_matrix_element_grid(store, netcdf_path=str(artifact), out="target_bare_matrix_element")

    assert result["out"] == "target_bare_matrix_element"
    assert result["resample"] == "jackknife"
    data = store["target_bare_matrix_element"]
    assert isinstance(data, EnsembleData)
    assert data.dims == ["z"]
    assert data.values.shape == (2, 4)
    assert np.allclose(data.values, samples)


def test_ratio_scheme_preserves_samples_writes_netcdf_and_plot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = np.asarray([[2, 4, 8, 10], [4, 8, 16, 20]], dtype=complex)
    denom = np.asarray([[1, 2, 4, 5], [2, 4, 8, 10]], dtype=complex)
    target_artifact = _write_bare_netcdf(tmp_path, "target", target)
    denom_artifact = _write_bare_netcdf(tmp_path, "denom", denom)
    store = {}
    load_bare_matrix_element_grid(store, netcdf_path=str(target_artifact), out="target_bare_matrix_element")
    load_bare_matrix_element_grid(store, netcdf_path=str(denom_artifact), out="denominator_bare_matrix_element")

    result = apply_ratio_scheme_renormalization(
        store,
        zs=4,
        delta_m=0,
        m0=0,
        save_path="renorm",
    )

    assert Path(result["artifact"]).is_file()
    assert result["artifact"].endswith(".nc")
    data = store["matrix_element_data"]
    assert data.values.shape == (2, 4)
    assert np.allclose(data.values[:, :3], 1.0)
    assert np.allclose(data.values[:, 3], 1.25)

    saved = EnsembleData.from_netcdf(result["artifact"])
    assert saved.dims == ["z"]
    assert saved.values.shape == (2, 4)
    assert np.allclose(saved.coords["z"], [0, 1, 4, 5])

    plot = plot_renormalized_matrix_element(store, save_path="renorm")
    assert Path(plot["plot"]).is_file()
