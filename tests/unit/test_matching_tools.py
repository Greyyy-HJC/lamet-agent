from pathlib import Path
import tempfile

import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.matching.functions import (
    KERNEL_REGISTRY,
    apply_matching,
    build_matching_kernel,
    load_quasi_pdf,
    plot_matched_pdf,
    resolve_kernel_id,
)


def _quasi_on(x_grid: np.ndarray, *, n_sample: int = 4) -> EnsembleData:
    """A smooth quasi-PDF over ``x_grid``, so interpolation error stays measurable."""
    return EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.exp(-(x_grid**2)) + 0.001 * idx for idx in range(n_sample)],
        dims=("x",),
        coords={"x": x_grid.tolist()},
        name="quasi_pdf",
    )


def test_resolve_registered_hybrid_kernel() -> None:
    kernel_id = "CG_gt_quark_PDF_hybrid_NLO"
    assert resolve_kernel_id(kernel_id, "hybrid_ratio") == kernel_id


def test_kernel_registry_ids_match_kernels_module_function_names() -> None:
    assert all(kernel_id == builder.__name__ for kernel_id, builder in KERNEL_REGISTRY.items())


def test_omitted_grids_keep_the_fourier_grid() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}

    load_quasi_pdf(store, component="re")

    assert np.allclose(store["quasi_y_ls"], native)
    build_matching_kernel(store, kernel_id="CG_gt_quark_PDF_ratio_NLO", momentum_gev=1.5)
    apply_matching(store)
    # Rows and columns both on the Fourier grid, exactly as before the grids opened up.
    assert store["kernel_matrix"].shape == (100, 100)
    assert np.allclose(store["lc_x_ls"], native)


def test_quasi_y_ls_restating_the_fourier_grid_is_lossless() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}

    load_quasi_pdf(store, component="re", quasi_y_ls={"start": -2.0, "stop": 2.0, "num": 100})

    # Interpolating onto the points the samples already sit on returns them bit for
    # bit, which is why load_quasi_pdf needs no special case for this grid.
    assert np.array_equal(store["quasi_ed"].values, _quasi_on(native).values)
    assert np.array_equal(store["quasi_y_ls"], native)


def test_quasi_and_lc_grids_decouple_the_kernel_matrix() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}

    # The light-cone grid must stay no denser than the quasi grid it integrates over,
    # so the quasi grid is the fine one here.
    load_quasi_pdf(store, component="re", quasi_y_ls={"start": -1.5, "stop": 1.5, "num": 150})
    build_matching_kernel(
        store,
        kernel_id="CG_gt_quark_PDF_ratio_NLO",
        momentum_gev=1.5,
        lc_x_ls={"start": -1.0, "stop": 1.0, "num": 41},
    )
    apply_matching(store)

    assert store["kernel_matrix"].shape == (41, 150)  # rows light-cone, columns quasi
    assert store["lightcone_ed"].values.shape == (4, 41)
    assert np.allclose(store["lightcone_ed"].coords["x"], np.linspace(-1.0, 1.0, 41))
    # The light-cone grid is unconstrained, unlike the quasi one: 0 is allowed on it.
    assert 0.0 in store["lc_x_ls"]


def test_lc_grid_denser_than_quasi_is_rejected_rather_than_oscillating() -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}
    load_quasi_pdf(store, component="re")

    # The kernel's plus prescription lands each y column's subtraction on one nearest x
    # row, so a denser x grid leaves most rows unsubtracted and the matched curve
    # oscillates point to point. Nothing downstream notices, so this must raise.
    with pytest.raises(ValueError, match="oscillate"):
        build_matching_kernel(
            store,
            kernel_id="CG_gt_quark_PDF_ratio_NLO",
            momentum_gev=1.5,
            lc_x_ls={"start": -1.0, "stop": 2.0, "num": 300},
        )

    # A grid no denser than the quasi one is fine.
    build_matching_kernel(
        store,
        kernel_id="CG_gt_quark_PDF_ratio_NLO",
        momentum_gev=1.5,
        lc_x_ls={"start": -1.0, "stop": 1.0, "num": 25},
    )
    assert store["kernel_matrix"].shape == (25, 100)


def test_endpoint_cut_drops_the_da_divergent_window_only_for_da_kernels() -> None:
    # Fine enough that points land inside 0.01 of x = 0 and x = 1; on a coarser grid the
    # cut has nothing to remove and would pass vacuously. The resolution has to come from
    # the quasi grid itself -- a denser light-cone grid is rejected, not a workaround.
    native = np.linspace(-1.0, 2.0, 300)

    def matched(kernel_id: str, cut: float | None) -> tuple[int, np.ndarray]:
        store = {"quasi": _quasi_on(native)}
        load_quasi_pdf(store, component="re")
        build_matching_kernel(store, kernel_id=kernel_id, momentum_gev=2.4, zs_fm=0.2)
        result = apply_matching(store, save_path=str(Path(tempfile.mkdtemp()) / "mt"), endpoint_cut=cut)
        return result["endpoint_points_dropped"], np.asarray(store["lightcone_ed"].coords["x"])

    def in_window(x: np.ndarray) -> list[float]:
        return [float(v) for v in x if (0.0 < v < 0.01) or (0.99 < v < 1.0)]

    dropped, x = matched("GI_gzg5_DA_hybrid_NLO", 0.01)
    assert dropped == 2 and in_window(x) == []

    # Without the cut the divergent points ship, so the cut is what removes them.
    dropped, x = matched("GI_gzg5_DA_hybrid_NLO", None)
    assert dropped == 0 and len(in_window(x)) == 2

    # A PDF kernel has no endpoint divergence, so the cut must not touch its grid.
    dropped, x = matched("CG_gt_quark_PDF_hybrid_NLO", 0.01)
    assert dropped == 0 and len(in_window(x)) == 2


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"start": -2.0, "stop": 2.0, "num": 101}, "must not contain 0"),
        ({"start": -3.0, "stop": 3.0, "num": 100}, "extends beyond"),
        ([-1.0, -0.5, -0.1, 0.3, 1.2], "uniformly spaced"),
    ],
)
def test_quasi_y_ls_rejects_grids_the_kernels_cannot_integrate(spec, message: str) -> None:
    native = np.linspace(-2.0, 2.0, 100)
    store = {"quasi": _quasi_on(native)}

    with pytest.raises(ValueError, match=message):
        load_quasi_pdf(store, component="re", quasi_y_ls=spec)


def test_matching_consumes_in_memory_fourier_output_and_writes_primary_netcdf(tmp_path: Path) -> None:
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[np.array([1 + 0.1j, 2 + 0.2j]), np.array([1.2 + 0.1j, 2.2 + 0.2j])],
        dims=("x",),
        coords={"x": [-0.5, 0.5]},
        attrs={"bz_direction": "X"},
        name="fourier_transform",
    )
    store = {"quasi": data}
    loaded = load_quasi_pdf(store, component="re")
    store["kernel_matrix"] = np.eye(2)

    result = apply_matching(store, save_path=str(tmp_path / "mt_p5"))

    assert loaded["n_sample"] == 2
    assert store["output"] is store["lightcone_ed"]
    assert store["output"].attrs["bz_direction"] == "X"
    assert Path(result["artifact"]).is_file()
    saved = EnsembleData.from_netcdf(result["artifact"])
    assert saved.dims == ["x"]
    assert np.allclose(saved.values, [[1, 2], [1.2, 2.2]])


def test_plot_matched_pdf_writes_pdf_and_svg(tmp_path: Path) -> None:
    quasi = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0, 2.0]), np.array([1.1, 2.1])],
        dims=("x",),
        coords={"x": [0.0, 1.0]},
        attrs={"sector": "valence"},
        name="quasi_pdf",
    )
    lightcone = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([0.9, 1.8]), np.array([1.0, 1.9])],
        dims=("x",),
        coords={"x": [0.0, 1.0]},
        name="lightcone_pdf",
    )
    store = {"quasi_y_ls": np.array([0.0, 1.0]), "quasi_ed": quasi, "lightcone_ed": lightcone}

    result = plot_matched_pdf(store, save_path=str(tmp_path / "matched_pdf"))

    assert Path(result["path"]).is_file()
    assert Path(result["plot_image"]).is_file()
    assert Path(result["path"]).suffix == ".pdf"
    assert Path(result["plot_image"]).suffix == ".svg"
    assert result["xlim"] == [-0.01, 1.01]
    assert result["ylim"] == pytest.approx([0.6792893218813452, 3.1207106781186544])
    assert store["matching_plot"] == result


def test_plot_matched_pdf_honors_explicit_limits(tmp_path: Path) -> None:
    quasi = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([1.0, 2.0]), np.array([1.1, 2.1])],
        dims=("x",),
        coords={"x": [-1.0, 1.0]},
        attrs={"sector": "total"},
        name="quasi_pdf",
    )
    lightcone = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[np.array([0.9, 1.8]), np.array([1.0, 1.9])],
        dims=("x",),
        coords={"x": [-1.0, 1.0]},
        name="lightcone_pdf",
    )
    store = {"quasi_y_ls": np.array([-1.0, 1.0]), "quasi_ed": quasi, "lightcone_ed": lightcone}

    result = plot_matched_pdf(
        store,
        save_path=str(tmp_path / "matched_pdf"),
        ylim=[-0.2, 2.5],
    )

    assert result["xlim"] == [-1.01, 1.01]
    assert result["ylim"] == [-0.2, 2.5]
