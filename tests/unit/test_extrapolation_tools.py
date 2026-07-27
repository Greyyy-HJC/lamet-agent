from pathlib import Path

import numpy as np
import xarray as xr

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.extrapolation.functions import run_systematics_budget


def test_systematics_budget_combines_extrapolated_branches(tmp_path: Path) -> None:
    x = [0.0, 0.5, 1.0]
    main = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[[1.0, 2.0, 3.0], [1.2, 2.2, 3.2]],
        dims=("x",),
        coords={"x": x},
    )
    low = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[[0.8, 1.8, 2.8], [1.0, 2.0, 3.0]],
        dims=("x",),
        coords={"x": x},
    )
    high = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=[[1.3, 2.3, 3.3], [1.5, 2.5, 3.5]],
        dims=("x",),
        coords={"x": x},
    )
    store = {"main": main, "zs": [low, high]}

    result = run_systematics_budget(store, save_path=str(tmp_path / "ex_budget"))

    final = xr.load_dataset(result["final_artifact"])
    expected_systematic = np.full(3, 0.5)
    assert np.allclose(final["total_systematic_error"], expected_systematic)
    assert np.allclose(
        final["total_error"],
        np.sqrt(np.asarray(main.sdev) ** 2 + expected_systematic**2),
    )
    assert Path(result["plot"]).is_file()
    assert Path(result["plot_image"]).is_file()
    assert Path(result["final_plot"]).is_file()
    assert Path(result["final_plot_image"]).is_file()
