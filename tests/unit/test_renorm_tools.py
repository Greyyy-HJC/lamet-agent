from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.fourier.functions import load_renormalized_matrix_element_samples
from lamet_agent.stages.renorm.functions import (
    apply_ratio_scheme_renormalization,
    load_bare_matrix_element_grid,
    plot_renormalized_matrix_element,
)


def _write_bare_report(base: Path, stem: str, values: np.ndarray, *, resample_mode: str = "jk") -> Path:
    out_dir = base / stem
    out_dir.mkdir(parents=True)
    outputs = []
    for iz, z in enumerate([0, 1, 4, 5]):
        path = out_dir / f"E_T_free_X_PX0PY0PZ0_b0_z{z}.txt"
        complex_values = values[:, iz]
        np.savetxt(path, np.column_stack([np.real(complex_values), np.imag(complex_values)]))
        outputs.append({"z": z, "path": str(path), "n_samples": int(values.shape[0])})
    report = {
        "ensemble": "E",
        "momentum": "PX0PY0PZ0",
        "resample_mode": resample_mode,
        "outputs": outputs,
    }
    report_path = base / f"{stem}_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_load_bare_matrix_element_grid_reads_correlator_report(tmp_path: Path) -> None:
    samples = np.asarray([[1 + 0.1j, 2 + 0.2j, 3 + 0.3j, 4 + 0.4j], [2 + 0.2j, 4 + 0.4j, 6 + 0.6j, 8 + 0.8j]])
    report = _write_bare_report(tmp_path, "target", samples)
    store = {}

    result = load_bare_matrix_element_grid(store, report_json=str(report), out="target_bare_matrix_element")

    assert result["out"] == "target_bare_matrix_element"
    assert result["resample"] == "jackknife"
    data = store["target_bare_matrix_element"]
    assert isinstance(data, EnsembleData)
    assert data.dims == ["z"]
    assert data.values.shape == (2, 4)
    assert np.allclose(data.values, samples)


def test_ratio_scheme_preserves_samples_writes_npz_and_plot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = np.asarray([[2, 4, 8, 10], [4, 8, 16, 20]], dtype=complex)
    denom = np.asarray([[1, 2, 4, 5], [2, 4, 8, 10]], dtype=complex)
    target_report = _write_bare_report(tmp_path, "target", target)
    denom_report = _write_bare_report(tmp_path, "denom", denom)
    store = {}
    load_bare_matrix_element_grid(store, report_json=str(target_report), out="target_bare_matrix_element")
    load_bare_matrix_element_grid(store, report_json=str(denom_report), out="denominator_bare_matrix_element")

    result = apply_ratio_scheme_renormalization(
        store,
        zs=4,
        delta_m=0,
        m0=0,
        save_path="renorm",
    )

    assert Path(result["artifact"]).is_file()
    data = store["matrix_element_data"]
    assert data.values.shape == (2, 4)
    assert np.allclose(data.values[:, :3], 1.0)
    assert np.allclose(data.values[:, 3], 1.25)

    saved, extras = EnsembleData.load_npz(result["artifact"])
    assert saved.dims == ["z"]
    assert saved.values.shape == (2, 4)
    assert np.allclose(extras["coord"], [0, 1, 4, 5])
    assert extras["re_samples"].shape == (2, 4)

    fourier_store = {}
    loaded = load_renormalized_matrix_element_samples(fourier_store, path=result["artifact"])
    assert loaded["n_sample"] == 2
    assert "matrix_element_data" in fourier_store

    plot = plot_renormalized_matrix_element(store, save_path="renorm")
    assert Path(plot["plot"]).is_file()
