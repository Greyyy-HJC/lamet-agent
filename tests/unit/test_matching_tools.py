from pathlib import Path

import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.matching.functions import apply_matching, load_quasi_pdf, plot_matched_pdf, resolve_kernel_id


def test_resolve_registered_hybrid_kernel() -> None:
    assert resolve_kernel_id("CG_gt_PDF_hybrid", "hybrid_ratio") == "CG_gt_PDF_hybrid"


def test_matching_consumes_in_memory_fourier_output_and_writes_primary_netcdf(tmp_path: Path) -> None:
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[np.array([1 + 0.1j, 2 + 0.2j]), np.array([1.2 + 0.1j, 2.2 + 0.2j])],
        dims=("x",),
        coords={"x": [-0.5, 0.5]},
        name="fourier_transform",
    )
    store = {"quasi": data}
    loaded = load_quasi_pdf(store, component="re")
    store["kernel_matrix"] = np.eye(2)

    result = apply_matching(store, save_path=str(tmp_path / "mt_p5"))

    assert loaded["n_sample"] == 2
    assert store["output"] is store["lightcone_ed"]
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
    store = {"x_ls": np.array([0.0, 1.0]), "quasi_ed": quasi, "lightcone_ed": lightcone}

    result = plot_matched_pdf(store, save_path=str(tmp_path / "matched_pdf"))

    assert Path(result["path"]).is_file()
    assert Path(result["plot_image"]).is_file()
    assert Path(result["path"]).suffix == ".pdf"
    assert Path(result["plot_image"]).suffix == ".svg"
    assert store["matching_plot"] == result
