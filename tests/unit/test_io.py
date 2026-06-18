import numpy as np

from lamet_agent.core.data import EnsembleInfo, EnsembleData

ensemble = EnsembleInfo("S", "E", 0.12, 0.10, 24, 64, 0.14)
data = EnsembleData(
    ensemble,
    "raw",
    [np.array([1 + 2j, 3 + 4j]), np.array([5 + 6j, 7 + 8j])],
    dims=("z",),
    coords={"z": [0, 1]},
)

data.to_netcdf("data.nc")
reload = EnsembleData.from_netcdf("data.nc")
assert data.ensemble == reload.ensemble
assert data.resample == reload.resample
assert (data.array == reload.array).all()
