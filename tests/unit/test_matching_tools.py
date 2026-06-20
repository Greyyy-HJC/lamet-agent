from pathlib import Path

import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.matching.functions import apply_matching, load_quasi_pdf, resolve_kernel_id


def test_resolve_logical_unpolarized_hybrid_kernel() -> None:
    assert resolve_kernel_id("unpolarized_gT", "hybrid_ratio") == "CG_gt_PDF_hybrid"


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
