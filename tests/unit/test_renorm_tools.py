from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.renorm.functions import (
    apply_ratio_scheme_renormalization,
    apply_self_renormalization,
    load_bare_matrix_element_grid,
    normalize_bare_matrix_element_at_z0,
    plot_renormalized_matrix_element,
    plot_self_renormalization_diagnostics,
)
from lamet_agent.stages.renorm.reporting import build_renorm_stage_report_markdown


def _write_bare_netcdf(base: Path, stem: str, values: np.ndarray, *, resample: str = "jackknife") -> Path:
    data = EnsembleData(
        ensemble=EnsembleInfo("", "E", 1.0, 1.0, 1, 1, 0.0),
        resample=resample,
        values=[values[idx] for idx in range(values.shape[0])],
        dims=("z",),
        coords={"z": [0, 1, 4, 5]},
        attrs={"ensemble": "E", "momentum": "PX0PY0PZ0", "lattice_spacing_fm": "0.1"},
        name="bare_matrix_element",
    )
    path = base / f"{stem}.nc"
    data.to_netcdf(path)
    return path


def _prepare_renorm_inputs(store: dict[str, object], *, normalize: bool = True) -> None:
    for role in ("target", "denominator", "target_bare_matrix_element", "denominator_bare_matrix_element"):
        value = store.get(role)
        if isinstance(value, EnsembleData) and normalize:
            store[role] = normalize_bare_matrix_element_at_z0(value)


def test_normalize_bare_matrix_element_at_z0_scales_by_z0() -> None:
    samples = np.asarray([[2 + 0j, 4 + 0j, 8 + 0j], [4 + 0j, 8 + 0j, 16 + 0j]], dtype=complex)
    data = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[samples[0], samples[1]], dims=("z",), coords={"z": [0, 1, 2]}, name="bare",
    )

    normalized = normalize_bare_matrix_element_at_z0(data)

    assert normalized.attrs.get("normalized_at_z0") == "true"
    assert np.allclose(normalized.values[:, 0], 1.0)
    assert np.allclose(normalized.values[:, 1], 2.0)
    assert np.allclose(normalized.values[:, 2], 4.0)


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
    _prepare_renorm_inputs(store)

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


def test_ratio_scheme_without_normalization_uses_pure_ratio(tmp_path: Path) -> None:
    target = np.asarray([[2, 6, 20], [4, 8, 12]], dtype=complex)
    denom = np.asarray([[1, 2, 10], [2, 8, 3]], dtype=complex)
    store = {
        "target": EnsembleData(
            EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
            values=[target[0], target[1]], dims=("z",), coords={"z": [0, 1, 5]}, name="target",
        ),
        "denominator": EnsembleData(
            EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
            values=[denom[0], denom[1]], dims=("z",), coords={"z": [0, 1, 5]}, name="denominator",
        ),
    }

    result = apply_ratio_scheme_renormalization(
        store,
        target="target",
        denominator="denominator",
        scheme="ratio",
        scheme_parameters={"zs_fm": 0.1, "m0_gev": 9.0, "delta_m_gev": 8.0},
        save_path=str(tmp_path / "pure"),
    )

    assert np.allclose(store["output"].values, target / denom)
    assert result["scheme"] == "ratio"
    assert not {"zs_fm", "zs_lattice", "zs_grid", "m0_gev", "delta_m_gev"} & result.keys()
    assert not {"zs_fm", "zs_lattice", "zs_grid", "m0_gev", "delta_m_gev"} & store["output"].attrs.keys()


def test_ratio_scheme_uses_preprocessed_z0_normalization(tmp_path: Path) -> None:
    target = np.asarray([[2, 6, 20], [4, 8, 12]], dtype=complex)
    denom = np.asarray([[1, 2, 10], [2, 8, 3]], dtype=complex)
    store = {
        "target": EnsembleData(
            EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
            values=list(target), dims=("z",), coords={"z": [0, 1, 5]}, name="target",
        ),
        "denominator": EnsembleData(
            EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
            values=list(denom), dims=("z",), coords={"z": [0, 1, 5]}, name="denominator",
        ),
    }
    _prepare_renorm_inputs(store)

    apply_ratio_scheme_renormalization(
        store,
        target="target",
        denominator="denominator",
        scheme="ratio",
        save_path=str(tmp_path / "normalized"),
    )

    expected = (target / target[:, :1]) / (denom / denom[:, :1])
    assert np.allclose(store["output"].values, expected)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_ratio_report_omits_hybrid_parameters(language: str, tmp_path: Path) -> None:
    report = build_renorm_stage_report_markdown(
        jobs=[{
            "job_id": "rn_ratio",
            "result": {
                "scheme": "ratio",
                "n_sample": 2,
                "z_grid": [0, 1, 5],
                "zs_fm": 0.2,
                "m0_gev": 1.0,
                "delta_m_gev": 2.0,
            },
            "artifacts": {},
        }],
        base_dir=tmp_path,
        language=language,
    )

    assert "h^{\\rm tar}_s(z)" in report
    assert "h^{\\rm den}_s(z)" in report
    assert "$z_s$" not in report
    assert "delta m" not in report


def test_hybrid_ratio_uses_physical_switch_and_nearest_grid_point(tmp_path: Path) -> None:
    z = list(range(6))
    target = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.full(6, 2.0), np.full(6, 4.0)], dims=("z",), coords={"z": z},
        attrs={"lattice_spacing_fm": "0.0574"}, name="target",
    )
    denominator_values = np.asarray([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]], dtype=complex)
    denominator = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=list(denominator_values), dims=("z",), coords={"z": z}, name="denominator",
    )
    store = {"target": target, "denominator": denominator}
    _prepare_renorm_inputs(store)

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
    lattice_spacing_fm = 0.1
    zs_fm = 0.3
    m0_gev = 0.2
    delta_m_gev = 0.1
    target = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.ones(6, dtype=complex), np.full(6, 2.0, dtype=complex)],
        dims=("z",), coords={"z": z}, attrs={"lattice_spacing_fm": str(lattice_spacing_fm)}, name="target",
    )
    denominator = EnsembleData(
        EnsembleInfo("", "E", 1, 1, 1, 1, 0), "jackknife",
        values=[np.ones(6, dtype=complex), np.full(6, 2.0, dtype=complex)],
        dims=("z",), coords={"z": z}, attrs={"lattice_spacing_fm": str(lattice_spacing_fm)}, name="denominator",
    )
    store = {"target": target, "denominator": denominator}
    _prepare_renorm_inputs(store)

    apply_ratio_scheme_renormalization(
        store,
        target="target",
        denominator="denominator",
        scheme_parameters={"zs_fm": zs_fm, "m0_gev": m0_gev, "delta_m_gev": delta_m_gev},
        save_path=str(tmp_path / "exponent"),
    )

    z4_fm = 4 * lattice_spacing_fm
    expected_exp = np.exp((m0_gev + delta_m_gev) * (z4_fm - zs_fm) / GEV_FM)
    assert np.allclose(store["output"].values[:, 4], expected_exp)


@pytest.mark.parametrize("normalized", [True, False])
def test_fit_self_renormalization_respects_normalized_at_z0_attr(normalized: bool, tmp_path: Path) -> None:
    gv = pytest.importorskip("gvar")
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.0, 1.0, 2.0]
    samples = np.asarray([[2.0, 4.0, 8.0], [3.0, 6.0, 12.0]], dtype=complex)
    attrs = {"normalized_at_z0": "true"} if normalized else {}
    reference = EnsembleData(
        EnsembleInfo("", "E", 0.1, 0.1, 1, 1, 0), "jackknife",
        values=[samples[0], samples[1]], dims=("z",), coords={"z": z}, attrs=attrs, name="reference",
    )
    store = {"reference": reference}

    captured: dict[str, list[float]] = {"z": []}
    call_count = {"n": 0}

    def fake_nonlinear_fit(*, data, prior, fcn, **kwargs):
        z_x, _lnm = data
        call_count["n"] += 1
        if call_count["n"] == 1:
            assert isinstance(z_x, dict)
            captured["z"] = list(z_x["z"])
        fit = gv.BufferDict()
        for key in prior:
            fit[key] = gv.gvar(0.0, 0.1)
        fit.p = fit
        return fit

    pytest.importorskip("lsqfit")
    import lsqfit as lsf

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(lsf, "nonlinear_fit", fake_nonlinear_fit)
        fit_self_renormalization_factor(store, m0_gev=-0.094, d=0.19, save_path=str(tmp_path / "zR"))
    finally:
        monkeypatch.undo()

    if normalized:
        assert captured["z"] == [1.0, 2.0]
    else:
        assert captured["z"] == [0.0, 1.0, 2.0]


def test_fit_self_renormalization_requires_d(tmp_path: Path) -> None:
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.06, 0.12, 0.18]
    samples = np.asarray([[1.0, 0.8, 0.6], [1.1, 0.85, 0.65]], dtype=complex)
    reference = EnsembleData(
        EnsembleInfo("", "E", 0.0574, 0.0574, 1, 1, 0),
        "jackknife",
        values=[samples[0], samples[1]],
        dims=("z",),
        coords={"z": z},
        attrs={"normalized_at_z0": "true"},
        name="reference",
    )
    with pytest.raises(ValueError, match="requires d"):
        fit_self_renormalization_factor({"reference": reference}, save_path=str(tmp_path / "zR"))


def test_fit_self_renormalization_fits_m0_when_omitted(tmp_path: Path) -> None:
    gv = pytest.importorskip("gvar")
    pytest.importorskip("lsqfit")
    import lsqfit as lsf
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.06, 0.12, 0.18]
    samples = np.asarray([[1.0, 0.8, 0.6], [1.1, 0.85, 0.65]], dtype=complex)
    reference = EnsembleData(
        EnsembleInfo("", "E", 0.0574, 0.0574, 1, 1, 0),
        "jackknife",
        values=[samples[0], samples[1]],
        dims=("z",),
        coords={"z": z},
        attrs={"normalized_at_z0": "true"},
        name="reference",
    )
    store = {"reference": reference}
    call_count = {"n": 0}

    def fake_nonlinear_fit(*, data, prior, fcn, **kwargs):
        call_count["n"] += 1
        fit = gv.BufferDict()
        for key in prior:
            if key == "m0":
                fit[key] = gv.gvar(-0.1, 0.02)
            else:
                fit[key] = gv.gvar(0.0, 0.1)
        fit.p = fit
        return fit

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(lsf, "nonlinear_fit", fake_nonlinear_fit)
        result = fit_self_renormalization_factor(
            store,
            kernel_id="ZMSbar_da",
            d=0.19,
            save_path=str(tmp_path / "rn_zR_fit"),
        )
    finally:
        monkeypatch.undo()

    assert call_count["n"] == 2
    assert result["m0_source"] == "fit"
    assert result["m0"] == pytest.approx(-0.1)
    assert result["d"] == pytest.approx(0.19)
    assert store["self_renorm_fit"]["m0_source"] == "fit"
    assert store["zR"].attrs.get("m0_source") == "fit"
    assert store["zR"].attrs.get("d") == "0.19"


def test_fit_self_renormalization_uses_fixed_m0_and_d(tmp_path: Path) -> None:
    gv = pytest.importorskip("gvar")
    pytest.importorskip("lsqfit")
    import lsqfit as lsf
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.06, 0.12, 0.18]
    samples = np.asarray([[1.0, 0.8, 0.6], [1.1, 0.85, 0.65]], dtype=complex)
    reference = EnsembleData(
        EnsembleInfo("", "E", 0.0574, 0.0574, 1, 1, 0),
        "jackknife",
        values=[samples[0], samples[1]],
        dims=("z",),
        coords={"z": z},
        attrs={"normalized_at_z0": "true"},
        name="reference",
    )
    store = {"reference": reference}
    call_count = {"n": 0}
    captured_svdcut: list[float] = []

    def fake_nonlinear_fit(*, data, prior, fcn, **kwargs):
        call_count["n"] += 1
        captured_svdcut.append(kwargs.get("svdcut"))
        fit = gv.BufferDict()
        for key in prior:
            fit[key] = gv.gvar(0.0, 0.1)
        fit.p = fit
        return fit

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(lsf, "nonlinear_fit", fake_nonlinear_fit)
        result = fit_self_renormalization_factor(
            store,
            kernel_id="ZMSbar_da",
            m0_gev=-0.094,
            d=0.19,
            save_path=str(tmp_path / "rn_zR_fit"),
        )
    finally:
        monkeypatch.undo()

    assert call_count["n"] == 1
    assert captured_svdcut == [1e-12]
    assert result["m0"] == pytest.approx(-0.094)
    assert result["m0_source"] == "fixed"
    assert result["kernel_id"] == "ZMSbar_da"
    assert result["svdcut"] == pytest.approx(1e-12)
    assert result["d"] == pytest.approx(0.19)
    assert "d_fit" not in result
    assert result["n_sample"] == 1
    assert "zR" in store
    assert store["output"] is store["zR"]
    assert store["zR"].resample == "jackknife"
    assert store["zR"].values.shape[0] == 1
    assert store["zR"].attrs.get("sample_construction") == "mean_from_averaged_fit"
    assert store["zR"].attrs.get("m0_source") == "fixed"
    assert store["zR"].attrs.get("d") == "0.19"
    assert Path(result["artifact"]).is_file()
    assert "self_renorm_fit" in store
    assert store["self_renorm_fit"]["m0"] == pytest.approx(-0.094)
    assert store["self_renorm_fit"]["m0_source"] == "fixed"
    assert store["self_renorm_fit"]["svdcut"] == pytest.approx(1e-12)
    assert store["self_renorm_fit"]["d"] == pytest.approx(0.19)
    assert "d_fit" not in store["self_renorm_fit"]
    assert "n_m0" not in store["self_renorm_fit"]


def test_fit_self_renormalization_uses_d_in_gz_fit(tmp_path: Path) -> None:
    gv = pytest.importorskip("gvar")
    pytest.importorskip("lsqfit")
    import lsqfit as lsf
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.06, 0.12]
    a_vals = [0.0574, 0.0882]
    samples = [
        np.asarray([[1.0, 0.8], [1.05, 0.82]], dtype=complex),
        np.asarray([[1.1, 0.85], [1.15, 0.88]], dtype=complex),
    ]
    reference = EnsembleData(
        EnsembleInfo("", "E", a_vals[0], a_vals[0], 1, 1, 0),
        "bootstrap",
        values=samples,
        dims=("a", "z"),
        coords={"a": a_vals, "z": z},
        attrs={"normalized_at_z0": "true"},
        name="reference",
    )
    store = {"reference": reference}
    captured = {"fcn": None}

    def fake_nonlinear_fit(*, data, prior, fcn, **kwargs):
        captured["fcn"] = fcn
        fit = gv.BufferDict()
        for key in prior:
            fit[key] = gv.gvar(0.0, 0.1)
        fit.p = fit
        return fit

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(lsf, "nonlinear_fit", fake_nonlinear_fit)
        fit_self_renormalization_factor(
            store,
            kernel_id="ZMSbar_da",
            m0_gev=-0.094,
            d=0.19,
            save_path=str(tmp_path / "rn_zR_fit"),
        )
    finally:
        monkeypatch.undo()

    p = gv.BufferDict({f"g{z[0]}": gv.gvar(0.0, 0.0), f"f1{z[0]}": gv.gvar(0.0, 0.0)})
    out_fit = captured["fcn"]({"z": [z[0]], "x": [3.0]}, p)[0]
    assert np.isfinite(float(gv.mean(out_fit)))
    assert store["self_renorm_fit"]["d"] == pytest.approx(0.19)
    assert "d_fit" not in store["self_renorm_fit"]
    assert store["zR"].values.shape[0] == 1


def test_fit_self_renormalization_forwards_svdcut_override(tmp_path: Path) -> None:
    gv = pytest.importorskip("gvar")
    pytest.importorskip("lsqfit")
    import lsqfit as lsf
    from lamet_agent.stages.renorm.functions import fit_self_renormalization_factor

    z = [0.06, 0.12, 0.18]
    samples = np.asarray([[1.0, 0.8, 0.6], [1.1, 0.85, 0.65]], dtype=complex)
    reference = EnsembleData(
        EnsembleInfo("", "E", 0.0574, 0.0574, 1, 1, 0),
        "jackknife",
        values=[samples[0], samples[1]],
        dims=("z",),
        coords={"z": z},
        attrs={"normalized_at_z0": "true"},
        name="reference",
    )
    store = {"reference": reference}
    captured_svdcut: list[float] = []

    def fake_nonlinear_fit(*, data, prior, fcn, **kwargs):
        captured_svdcut.append(kwargs.get("svdcut"))
        fit = gv.BufferDict()
        for key in prior:
            fit[key] = gv.gvar(0.0, 0.1)
        fit.p = fit
        return fit

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(lsf, "nonlinear_fit", fake_nonlinear_fit)
        result = fit_self_renormalization_factor(
            store,
            kernel_id="ZMSbar_da",
            m0_gev=-0.094,
            d=0.19,
            svdcut=1e-8,
            save_path=str(tmp_path / "rn_zR_fit"),
        )
    finally:
        monkeypatch.undo()

    assert captured_svdcut == [1e-8]
    assert result["svdcut"] == pytest.approx(1e-8)
    assert store["self_renorm_fit"]["svdcut"] == pytest.approx(1e-8)


def test_apply_self_renormalization_divides_by_zr_times_zmsbar(tmp_path: Path) -> None:
    from lamet_agent import kernels

    z = np.asarray([0.06, 0.12, 0.18], dtype=float)
    lattice_spacing_fm = 0.0574
    zr_vals = np.asarray([0.5, 0.4, 0.3], dtype=float)
    zR = EnsembleData(
        EnsembleInfo("", "E", lattice_spacing_fm, lattice_spacing_fm, 1, 1, 0),
        "bootstrap",
        [np.asarray(zr_vals[None, :], dtype=complex)],
        dims=("a", "z"),
        coords={"a": [lattice_spacing_fm], "z": z.tolist()},
        attrs={
            "kernel_id": "ZMSbar_da",
            "m0_gev": "-0.094",
            "d": "-0.08183",
            "sample_construction": "mean_from_averaged_fit",
        },
        name="zR",
    )
    target_values = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], [2.0 + 0.0j, 4.0 + 0.0j, 6.0 + 0.0j]])
    target = EnsembleData(
        EnsembleInfo("", "E", lattice_spacing_fm, lattice_spacing_fm, 1, 1, 0),
        "jackknife",
        values=[target_values[0], target_values[1]],
        dims=("z",),
        coords={"z": z.tolist()},
        attrs={"lattice_spacing_fm": str(lattice_spacing_fm)},
        name="target",
    )
    store = {"target": target, "zR": zR}

    result = apply_self_renormalization(
        store,
        kernel_id="ZMSbar_da",
        mu=2.0,
        save_path=str(tmp_path / "self"),
    )

    zms = kernels.ZMSbar_da(z, mu=2.0)
    expected = target_values / (zr_vals[None, :] * zms[None, :])
    assert result["scheme"] == "self_renormalization"
    assert result["kernel_id"] == "ZMSbar_da"
    assert result["remapped"] is False
    assert Path(result["artifact"]).is_file()
    assert np.allclose(store["output"].values, expected)
    assert store["output"] is store["matrix_element_data"]


def test_apply_self_renormalization_remaps_d_and_m0(tmp_path: Path) -> None:
    from lamet_agent import kernels
    from lamet_agent.stages.renorm.functions import _remap_zr_values

    z = np.asarray([0.06, 0.12, 0.18], dtype=float)
    lattice_spacing_fm = 0.0574
    d_pdf = -0.08183
    d_da = 0.19
    m0_pdf = -0.05
    m0_da = -0.094
    zr_vals = np.asarray([0.5, 0.4, 0.3], dtype=float)
    zR = EnsembleData(
        EnsembleInfo("", "E", lattice_spacing_fm, lattice_spacing_fm, 1, 1, 0),
        "bootstrap",
        [np.asarray(zr_vals[None, :], dtype=complex)],
        dims=("a", "z"),
        coords={"a": [lattice_spacing_fm], "z": z.tolist()},
        attrs={
            "kernel_id": "ZMSbar_da",
            "m0_gev": str(m0_pdf),
            "d": str(d_pdf),
            "sample_construction": "mean_from_averaged_fit",
        },
        name="zR",
    )
    target_values = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], [2.0 + 0.0j, 4.0 + 0.0j, 6.0 + 0.0j]])
    target = EnsembleData(
        EnsembleInfo("", "E", lattice_spacing_fm, lattice_spacing_fm, 1, 1, 0),
        "jackknife",
        values=[target_values[0], target_values[1]],
        dims=("z",),
        coords={"z": z.tolist()},
        attrs={"lattice_spacing_fm": str(lattice_spacing_fm)},
        name="target",
    )
    store = {"target": target, "zR": zR}

    result = apply_self_renormalization(
        store,
        kernel_id="ZMSbar_da",
        mu=2.0,
        d=d_da,
        m0_gev=m0_da,
        save_path=str(tmp_path / "self_remap"),
    )

    zr_remapped = _remap_zr_values(
        zr_vals, z_vals=z, lattice_spacing_fm=lattice_spacing_fm, d_from=d_pdf, d_to=d_da, m0_from=m0_pdf, m0_to=m0_da
    )
    zms = kernels.ZMSbar_da(z, mu=2.0)
    expected = target_values / (zr_remapped[None, :] * zms[None, :])
    assert result["remapped"] is True
    assert result["d"] == pytest.approx(d_da)
    assert result["m0_gev"] == pytest.approx(m0_da)
    assert store["zR"].attrs["d"] == str(d_da)
    assert store["zR"].attrs["m0_gev"] == str(m0_da)
    assert np.allclose(store["output"].values, expected)


def test_plot_self_renormalization_diagnostics_fit_and_apply_modes(tmp_path: Path) -> None:
    z = np.asarray([0.06, 0.12, 0.18], dtype=float)
    a_vals = [0.0574, 0.0882]
    zr_mean = np.asarray([[0.5, 0.4, 0.3], [0.55, 0.45, 0.35]], dtype=float)
    zR = EnsembleData(
        EnsembleInfo("", "E", a_vals[0], a_vals[0], 1, 1, 0),
        "bootstrap",
        [np.asarray(zr_mean, dtype=complex)],
        dims=("a", "z"),
        coords={"a": a_vals, "z": z.tolist()},
        attrs={"kernel_id": "ZMSbar_da", "m0_gev": "-0.094", "mu": "2.0"},
        name="zR",
    )
    target_values = np.asarray([[1.0 + 0.1j, 0.8 + 0.2j, 0.5 + 0.1j], [1.1 + 0.1j, 0.85 + 0.2j, 0.55 + 0.1j]])
    target = EnsembleData(
        EnsembleInfo("", "E", a_vals[0], a_vals[0], 1, 1, 0),
        "jackknife",
        values=[target_values[0], target_values[1]],
        dims=("z",),
        coords={"z": z.tolist()},
        attrs={"lattice_spacing_fm": str(a_vals[0])},
        name="target",
    )
    sibling_values = target_values * 0.9
    sibling = EnsembleData(
        EnsembleInfo("", "E", a_vals[1], a_vals[1], 1, 1, 0),
        "jackknife",
        values=[sibling_values[0], sibling_values[1]],
        dims=("z",),
        coords={"z": z.tolist()},
        attrs={"lattice_spacing_fm": str(a_vals[1])},
        name="renormalized_matrix_element",
    )
    sibling_a = tmp_path / "rn_mom6_a06.nc"
    sibling_b = tmp_path / "rn_mom6_a09.nc"
    target_renorm = EnsembleData(
        EnsembleInfo("", "E", a_vals[0], a_vals[0], 1, 1, 0),
        "jackknife",
        values=[target_values[0], target_values[1]],
        dims=("z",),
        coords={"z": z.tolist()},
        attrs={"lattice_spacing_fm": str(a_vals[0])},
        name="renormalized_matrix_element",
    )
    target_renorm.to_netcdf(sibling_a)
    sibling.to_netcdf(sibling_b)

    fit = {
        "z": z.tolist(),
        "a": a_vals,
        "lnm_mean": np.asarray([[0.0, -0.2, -0.4], [0.1, -0.1, -0.3]], dtype=float),
        "lnm_sdev": np.full((2, 3), 0.05),
        "fit_lnm_mean": np.asarray([[0.0, -0.2, -0.4], [0.1, -0.1, -0.3]], dtype=float),
        "fit_lnm_sdev": np.full((2, 3), 0.05),
        "g_mean": np.asarray([0.1, 0.2, 0.3]),
        "g_sdev": np.asarray([0.01, 0.01, 0.01]),
        "f1_mean": np.asarray([0.0, 0.1, 0.2]),
        "f1_sdev": np.asarray([0.01, 0.01, 0.01]),
        "zR_mean": zr_mean,
        "mR": np.asarray([1.0, 0.9, 0.8]),
        "m0": -0.094,
        "m0_sdev": 0.0,
        "kernel_id": "ZMSbar_da",
        "mu": 2.0,
        "d": 0.19,
        "skip_z0": True,
    }
    store = {"target": target, "zR": zR, "self_renorm_fit": fit}

    fit_result = plot_self_renormalization_diagnostics(
        store,
        mode="fit",
        save_path=str(tmp_path / "rn_zR_fit"),
        artifacts_dir=tmp_path,
        kernel_id="ZMSbar_da",
        mu=2.0,
    )
    for key in ("fit_lnM_vs_inv_a", "fit_mR_zmsbar", "fit_m_over_zR", "fit_f1"):
        assert key in fit_result["plots"]
        assert Path(fit_result["plots"][key]).is_file()
        assert Path(fit_result["plots"][f"{key}_image"]).is_file()
    assert "fit_m0" not in fit_result["plots"]
    assert "fit_vs_data" not in fit_result["plots"]
    assert "zmsbar_compare" not in fit_result["plots"]

    apply_result = plot_self_renormalization_diagnostics(
        store,
        mode="apply",
        sibling_artifacts=[str(sibling_a), str(sibling_b)],
        include_discrete_effect=True,
        save_path=str(tmp_path / "rn_mom6_a12"),
        artifacts_dir=tmp_path,
        kernel_id="ZMSbar_da",
        mu=2.0,
    )
    assert "zmsbar_compare" in apply_result["plots"]
    assert "discrete_effect_re" in apply_result["plots"]
    assert "discrete_effect_im" in apply_result["plots"]
    assert "fit_m_over_zR" not in apply_result["plots"]
    assert "fit_vs_data" not in apply_result["plots"]
    for key in ("zmsbar_compare", "discrete_effect_re", "discrete_effect_im"):
        assert Path(apply_result["plots"][key]).is_file()
        assert Path(apply_result["plots"][f"{key}_image"]).is_file()
    assert Path(apply_result["plots"]["discrete_effect_re"]).name == "discrete_effect_re.pdf"
    assert Path(apply_result["plots"]["discrete_effect_im"]).name == "discrete_effect_im.pdf"
    assert "rn_mom6_a12" not in Path(apply_result["plots"]["discrete_effect_re"]).name
