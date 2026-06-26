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
        attrs={"ensemble": "E", "momentum": "PX0PY0PZ0", "a_fm": "0.1"},
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
        scheme="hybrid_ratio",
        scheme_parameters={"zs_fm": 0.4},
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


def test_hybrid_ratio_uses_physical_switch_and_nearest_grid_point(tmp_path: Path) -> None:
    z = list(range(6))
    target = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.full(6, 2.0), np.full(6, 4.0)], dims=("z",), coords={"z": z},
        attrs={"a_fm": "0.0574"}, name="target",
    )
    denominator_values = np.asarray([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]], dtype=complex)
    denominator = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=list(denominator_values), dims=("z",), coords={"z": z}, name="denominator",
    )
    store = {"target": target, "denominator": denominator}

    result = apply_ratio_scheme_renormalization(
        store, target="target", denominator="denominator",
        scheme_parameters={"zs_fm": 0.18}, save_path=str(tmp_path / "hybrid"),
    )

    assert result["zs_grid"] == 3.0
    assert result["zs_lattice"] == 0.18 / 0.0574
    # z=3 remains in the short-distance branch; z=4 uses h(z_s=3) in the denominator.
    assert np.allclose(store["output"].values[:, 3], [0.25, 0.25])
    assert np.allclose(store["output"].values[:, 4], [0.25, 0.25])


def test_hybrid_ratio_long_range_exponent_uses_physical_distance(tmp_path: Path) -> None:
    """Long-range exponent uses (m0_gev + delta_m_gev) * (z_fm - zs_fm) / GEV_FM."""
    from lamet_agent.stages.renorm.functions import GEV_FM

    z = [0, 1, 2, 3, 4, 5]
    a_fm = 0.1
    zs_fm = 0.3
    m0_gev = 0.2
    delta_m_gev = 0.1
    target = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.ones(6, dtype=complex), np.full(6, 2.0, dtype=complex)],
        dims=("z",), coords={"z": z}, attrs={"a_fm": str(a_fm)}, name="target",
    )
    denominator = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.ones(6, dtype=complex), np.full(6, 2.0, dtype=complex)],
        dims=("z",), coords={"z": z}, attrs={"a_fm": str(a_fm)}, name="denominator",
    )
    store = {"target": target, "denominator": denominator}

    apply_ratio_scheme_renormalization(
        store,
        target="target",
        denominator="denominator",
        scheme_parameters={"zs_fm": zs_fm, "m0_gev": m0_gev, "delta_m_gev": delta_m_gev},
        save_path=str(tmp_path / "exponent"),
    )

    z4_fm = 4 * a_fm
    expected_exp = np.exp((m0_gev + delta_m_gev) * (z4_fm - zs_fm) / GEV_FM)
    assert np.allclose(store["output"].values[:, 4], expected_exp)
