from pathlib import Path
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Union, get_args
import json
import warnings

import gvar
import numpy
from numpy.typing import NDArray
import xarray

DimType = str
DimsType = Sequence[DimType]
IndexType = Union[int, float, str]
CoordType = Sequence[IndexType]
CoordsType = Dict[DimType, CoordType]

ResampleType = Literal["raw", "jackknife", "bootstrap", "gvar"]
RESAMPLE_TYPE_VALUES = get_args(ResampleType)
RESAMPLE_DIM = "resample"

HBAR_C_GEV_FM = 0.197327  # hbar * c in GeV fm


class EnsembleInfo(NamedTuple):
    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float

    @property
    def k_s(self) -> float:
        return 2 * numpy.pi / self.L_s * HBAR_C_GEV_FM / self.a_s

    @property
    def k_t(self) -> float:
        return 2 * numpy.pi / self.L_t * HBAR_C_GEV_FM / self.a_t


def _is_gvar_values(values) -> bool:
    if isinstance(values, gvar.GVar):
        return True

    array = numpy.asarray(values)
    if array.ndim == 0:
        return isinstance(array.item(), gvar.GVar)
    if array.dtype != object:
        return False
    return all(isinstance(value, gvar.GVar) for value in array.flat)


class EnsembleData:
    def __init__(
        self,
        ensemble: EnsembleInfo,
        resample: ResampleType,
        values: Union[List[Union[int, float, complex, NDArray]], gvar.GVar, NDArray[gvar.GVar]],
        dims: DimsType,
        coords: CoordsType,
        attrs: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        if resample not in RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        self.resample: ResampleType = resample
        self.ensemble = ensemble
        self.array = self._build_xarray(resample, values, dims, coords, attrs, name)

    @staticmethod
    def _build_xarray(
        resample: ResampleType,
        values: Union[List[Union[int, float, complex, NDArray]], gvar.GVar, NDArray[gvar.GVar]],
        dims: DimsType,
        coords: CoordsType,
        attrs: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> xarray.DataArray:
        if RESAMPLE_DIM in dims:
            raise ValueError(f"Physical dimensions should not include resampling dimension '{RESAMPLE_DIM}'.")

        if isinstance(values, list):
            if resample == "gvar":
                raise TypeError(
                    "'gvar' does not support list of samples, use a gvar.GVar or an NDArray[gvar.GVar] instead."
                )
            if len(values) == 0:
                raise ValueError("Resampled values cannot be empty.")
            resample_values = numpy.stack(values, axis=0)
        else:
            if resample != "gvar":
                raise TypeError("raw/jackknife/bootstrap data must be initialized from a list of samples.")
            if not _is_gvar_values(values):
                raise TypeError("resample='gvar' requires a gvar.GVar or an NDArray[gvar.GVar].")
            resample_values = numpy.expand_dims(numpy.asarray(values, dtype=object), axis=0)

        if resample_values.ndim != len(dims) + 1:
            raise ValueError(
                "Resampled data must have one leading sample axis: "
                f"got ndim={resample_values.ndim}, physical dims={len(dims)}."
            )

        resample_dims: DimsType = (RESAMPLE_DIM, *dims)
        resample_coords: CoordsType = {RESAMPLE_DIM: list(range(resample_values.shape[0]))}
        for dim, size in zip(dims, resample_values.shape[1:]):
            if dim not in coords:
                raise ValueError(f"Missing dimension coordinate '{dim}'.")
            if len(coords[dim]) != size:
                raise ValueError(
                    f"Unmatched length of coordinates for dimension '{dim}' " f"{len(coords[dim])} != {size}."
                )
            resample_coords[dim] = list(coords[dim])

        return xarray.DataArray(resample_values, coords=resample_coords, dims=resample_dims, name=name, attrs=attrs)

    @classmethod
    def _from_xarray(
        cls, ensemble: Optional[EnsembleInfo], resample: ResampleType, array: xarray.DataArray
    ) -> "EnsembleData":
        if resample not in RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        if len(array.dims) == 0 or array.dims[0] != RESAMPLE_DIM:
            raise ValueError(f"resample='{resample}' requires the first xarray dimension to be named '{RESAMPLE_DIM}'.")
        if resample == "gvar":
            if array.sizes[RESAMPLE_DIM] != 1:
                raise ValueError("resample='gvar' requires a length-1 gvar dimension.")
            if not _is_gvar_values(array.values):
                raise TypeError("resample='gvar' requires values to be gvar.GVar objects.")

        obj = cls.__new__(cls)
        obj.ensemble = ensemble
        obj.resample = resample
        obj.array = array.copy(deep=False)
        return obj

    def to_netcdf(self, path: Union[str, Path]) -> None:
        array = self.array.copy(deep=False)
        array.attrs["ensemble"] = json.dumps(None if self.ensemble is None else self.ensemble._asdict())
        array.attrs["resample"] = self.resample
        array.to_netcdf(path, format="NETCDF4", auto_complex=True)

    @classmethod
    def from_netcdf(cls, path: Union[str, Path]) -> "EnsembleData":
        array = xarray.load_dataarray(path, auto_complex=True)
        ensemble_payload = json.loads(array.attrs.pop("ensemble"))
        ensemble = None if ensemble_payload is None else EnsembleInfo(**ensemble_payload)
        resample = array.attrs.pop("resample")
        return cls._from_xarray(ensemble, resample, array)

    def __repr__(self) -> str:
        return repr(self.array)

    @property
    def values(self):
        return self.array.values

    @property
    def dims(self) -> DimsType:
        dims_ = []
        for dim in self.array.dims[1:]:
            dims_.append(dim)
        return dims_

    @property
    def coords(self) -> CoordsType:
        coords_ = {}
        for dim in self.array.dims[1:]:
            coords_[dim] = self.array.coords[dim].values.tolist()
        return coords_

    @property
    def attrs(self) -> Dict[str, str]:
        return dict(self.array.attrs)

    @property
    def name(self) -> Optional[str]:
        assert self.array.name is None or isinstance(self.array.name, str)
        return self.array.name

    @property
    def real(self) -> "EnsembleData":
        return EnsembleData._from_xarray(self.ensemble, self.resample, self.array.real)

    @property
    def imag(self) -> "EnsembleData":
        return EnsembleData._from_xarray(self.ensemble, self.resample, self.array.imag)

    def copy(self, deep: bool = True) -> "EnsembleData":
        return EnsembleData._from_xarray(self.ensemble, self.resample, self.array.copy(deep=deep))

    def save_npz(self, path: Union[str, Path], **extra_arrays) -> None:
        """Write this EnsembleData plus optional diagnostic arrays to an NPZ file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        coords_payload = {dim: self.array.coords[dim].values.tolist() for dim in self.array.dims}
        metadata = {
            "format": "lamet_agent.EnsembleData",
            "name": self.name,
            "resample": self.resample,
            "dims": list(self.array.dims),
            "coords": coords_payload,
            "attrs": self.attrs,
            "ensemble": None if self.ensemble is None else self.ensemble._asdict(),
        }
        numpy.savez(
            output, __ensemble_data_metadata__=numpy.asarray(json.dumps(metadata)), values=self.values, **extra_arrays
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> tuple["EnsembleData", dict[str, NDArray]]:
        """Read an EnsembleData NPZ file and return ``(data, extra_arrays)``."""
        with numpy.load(path, allow_pickle=False) as npz:
            if "__ensemble_data_metadata__" not in npz or "values" not in npz:
                raise ValueError("NPZ file does not contain lamet-agent EnsembleData metadata")
            metadata = json.loads(str(npz["__ensemble_data_metadata__"]))
            if metadata.get("format") != "lamet_agent.EnsembleData":
                raise ValueError("NPZ file is not a lamet-agent EnsembleData file")
            values = numpy.asarray(npz["values"])
            dims_all = list(metadata["dims"])
            coords = metadata["coords"]
            attrs = metadata.get("attrs") or {}
            ensemble_payload = metadata.get("ensemble")
            ensemble = None if ensemble_payload is None else EnsembleInfo(**ensemble_payload)
            array = xarray.DataArray(
                values,
                coords={dim: coords[dim] for dim in dims_all},
                dims=dims_all,
                name=metadata.get("name"),
                attrs=attrs,
            )
            data = cls._from_xarray(ensemble, metadata["resample"], array)
            extras = {
                key: numpy.asarray(npz[key]) for key in npz.files if key not in {"__ensemble_data_metadata__", "values"}
            }
        return data, extras

    @property
    def n_sample(self) -> int:
        return self.array.sizes[RESAMPLE_DIM]

    def bin(self, bin_size: int) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled to other methods.")
        n_sample = self.n_sample
        if bin_size <= 0 or bin_size >= self.n_sample:
            raise ValueError("bin_size must be a positive integer smaller than the number of samples.")
        if n_sample % bin_size != 0:
            warnings.warn(
                "Number of samples is not divisible by bin_size, the last few samples will be dropped.", RuntimeWarning
            )
        n_resample = n_sample // bin_size
        values = [self.array.values[i * bin_size : (i + 1) * bin_size].mean(axis=0) for i in range(n_resample)]
        return EnsembleData(
            self.ensemble, "raw", values, dims=self.dims, coords=self.coords, attrs=self.attrs, name=self.name
        )

    def jackknife(self) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled to other methods.")
        values = []
        n_sample = self.n_sample
        values_sum = self.array.values.sum(axis=0)
        values = [(values_sum - self.array.values[i]) / (n_sample - 1) for i in range(n_sample)]
        return EnsembleData(
            self.ensemble, "jackknife", values, dims=self.dims, coords=self.coords, attrs=self.attrs, name=self.name
        )

    def bootstrap(self, n_resample: int) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled to other methods.")
        values = []
        n_sample = self.n_sample
        resample = numpy.random.randint(0, n_sample, (n_resample, n_sample))
        values = [self.array.values[resample[i]].mean(axis=0) for i in range(n_resample)]
        return EnsembleData(
            self.ensemble, "bootstrap", values, dims=self.dims, coords=self.coords, attrs=self.attrs, name=self.name
        )

    @classmethod
    def concat(
        cls, data_list: Sequence["EnsembleData"], dim: DimType, coord: Optional[CoordType] = None
    ) -> "EnsembleData":
        if len(data_list) == 0:
            raise ValueError("Cannot concatenate an empty list of EnsembleData.")
        if dim == RESAMPLE_DIM:
            raise ValueError(f"Cannot concatenate along reserved resampling dimension '{RESAMPLE_DIM}'.")
        ensemble = data_list[0].ensemble
        resample = data_list[0].resample
        dims = data_list[0].dims
        coords = data_list[0].coords
        coord_dims_to_match = [dim_ for dim_ in dims if dim_ != dim]
        if dim not in dims:
            coord_dims_to_match = list(dims)
        for data in data_list[1:]:
            if data.ensemble != ensemble:
                raise ValueError("Cannot concatenate EnsembleData from different ensembles.")
            if data.resample != resample:
                raise ValueError("Cannot concatenate EnsembleData with different resampling methods.")
            if data.dims != dims:
                raise ValueError("Cannot concatenate EnsembleData with different dimensions.")
            for dim_ in coord_dims_to_match:
                if data.coords[dim_] != coords[dim_]:
                    raise ValueError(
                        f"Cannot concatenate EnsembleData with different coordinates for dimension '{dim_}'."
                    )
        array = xarray.concat([data.array for data in data_list], dim)
        if dim in dims:
            if coord is not None:
                raise ValueError(f"Cannot specify coordinates for existing dimension '{dim}'.")
            dims_out = [RESAMPLE_DIM, *dims]
        else:
            if coord is None:
                raise ValueError(f"Must specify coordinates for new dimension '{dim}'.")
            dims_out = [RESAMPLE_DIM, dim, *dims]
            array = array.assign_coords({dim: coord})
        array = array.transpose(*dims_out).sortby(dim)
        return cls._from_xarray(ensemble, resample, array)

    def at(self, dim: DimType, coord: Union[IndexType, CoordType]) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return EnsembleData._from_xarray(self.ensemble, self.resample, self.array.sel({dim: coord}, drop=True))

    def near(self, dim: DimType, coord: Union[IndexType, CoordType], tolerance: float = 1e-8) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return EnsembleData._from_xarray(
            self.ensemble, self.resample, self.array.sel({dim: coord}, method="nearest", tolerance=tolerance, drop=True)
        )

    @property
    def gvar(self) -> Union[gvar.GVar, NDArray[gvar.GVar]]:
        if self.resample == "gvar":
            return self.array.values[0]
        else:
            values = numpy.ascontiguousarray(self.array.values)
            if self.resample == "jackknife":
                avg_data = gvar.dataset.avg_data(values, unbias=False, stderr=False)
                avg_data = gvar.gvar(gvar.mean(avg_data), gvar.evalcov(avg_data) * (self.n_sample - 1))
            elif self.resample == "bootstrap":
                avg_data = gvar.dataset.avg_data(values, unbias=True, stderr=False)
            else:
                avg_data = gvar.dataset.avg_data(values, unbias=True, stderr=True)
            return avg_data

    @property
    def mean(self) -> Union[float, complex, NDArray]:
        if self.resample == "gvar":
            return gvar.mean(self.array.values[0])
        else:
            return self.array.values.mean(0)

    @property
    def sdev(self) -> Union[float, complex, NDArray]:
        if self.resample == "gvar":
            return gvar.sdev(self.array.values[0])
        else:
            if self.resample == "jackknife":
                return self.array.values.std(0, ddof=0) * (self.n_sample - 1) ** 0.5
            elif self.resample == "bootstrap":
                return self.array.values.std(0, ddof=1)
            else:
                return self.array.values.std(0, ddof=1) / (self.n_sample) ** 0.5

    def avg_data(self) -> "EnsembleData":
        return EnsembleData(
            self.ensemble, "gvar", self.gvar, dims=self.dims, coords=self.coords, attrs=self.attrs, name=self.name
        )

    def update_dim(self, dim: DimType, dim_out: DimType, coord_out: Optional[CoordType] = None) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")

        if dim_out != dim and (dim_out == RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists in data dimensions.")

        array = self.array.rename({dim: dim_out})
        dim = dim_out

        if coord_out is not None:
            array = array.assign_coords({dim: coord_out})

        return EnsembleData._from_xarray(self.ensemble, self.resample, array)

    def sort_dim(self, dim: DimType, ascending: bool = True) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        array = self.array.sortby(dim, ascending=ascending)
        return EnsembleData._from_xarray(self.ensemble, self.resample, array)

    def aligned_ref_array(self, ref: "EnsembleData") -> xarray.DataArray:
        if not isinstance(ref, EnsembleData):
            raise TypeError("Reference data for renormalization must be an EnsembleData.")
        if self.resample == "gvar":
            ref_array = xarray.DataArray(ref.gvar, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif self.resample == "raw":
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        else:
            if ref.resample == "gvar" or ref.resample == "raw":
                ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
            elif ref.resample == self.resample:
                ref_array = ref.array
            else:
                raise ValueError(
                    f"Cannot apply renormalization with resample='{ref.resample}' reference data to "
                    f"resample='{self.resample}' target data."
                )

        for dim in ref_array.dims:
            if dim not in self.array.dims:
                raise ValueError(f"Reference dimension '{dim}' not found in data dimensions.")
            try:
                ref_array = ref_array.sel({dim: self.array.coords[dim]})
            except KeyError as exc:
                raise ValueError(f"Reference coordinates for dimension '{dim}' do not cover the target data.") from exc

        return ref_array.broadcast_like(self.array).transpose(*self.array.dims)

    def apply_renormalization(
        self,
        renorm_scheme: "EnsembleData",
        operator: Callable[[xarray.DataArray, xarray.DataArray], xarray.DataArray],
    ) -> "EnsembleData":
        values = operator(self.array, self.aligned_ref_array(renorm_scheme))
        values.attrs = self.attrs
        values.name = self.name
        return EnsembleData._from_xarray(self.ensemble, self.resample, values)

    def mul(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, rhs_value: value * rhs_value)

    def div(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, rhs_value: value / rhs_value)

    def add(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, rhs_value: value + rhs_value)

    def sub(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, rhs_value: value - rhs_value)

    def transform_dim(
        self,
        dim: DimType,
        dim_out: DimType,
        coord_out: CoordType,
        function: Callable[[NDArray, CoordType, CoordType, Dict[DimType, IndexType]], NDArray],
        dims_dispatch: Union[DimType, DimsType, None] = None,
    ) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if dim_out != dim and (dim_out == RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists in data dimensions.")

        if dims_dispatch is None:
            dims_dispatch = []
        elif isinstance(dims_dispatch, DimType):
            dims_dispatch = [dims_dispatch]
        for dim_ in dims_dispatch:
            if dim_ == dim or dim_ == dim_out:
                raise ValueError(
                    f"Dispatch dimension '{dim_}' cannot be the same as the transformed or output dimension."
                )
            if dim_ not in self.dims:
                raise ValueError(f"Dispatch dimension '{dim_}' not found in data dimensions.")

        dims_core = [dim] + [dim_ for dim_ in self.array.dims if dim_ != dim and dim_ not in dims_dispatch]
        dims_out_core = [dim_out] + [dim_ for dim_ in dims_core if dim_ != dim]
        dims_out = [dim_out if dim_ == dim else dim_ for dim_ in self.array.dims]

        def apply_function(value: NDArray, *index_list: IndexType) -> NDArray:
            return function(
                value,
                self.coords[dim],
                coord_out,
                {dim_: index_ for dim_, index_ in zip(dims_dispatch, index_list)},
            )

        array = xarray.apply_ufunc(
            apply_function,
            self.array,
            *[self.array.coords[dim_] for dim_ in dims_dispatch],
            input_core_dims=[dims_core] + [[]] * len(dims_dispatch),
            output_core_dims=[dims_out_core],
            exclude_dims={dim},
            vectorize=len(dims_dispatch) > 0,
            output_sizes={dim_out: len(coord_out)},
        )
        array = array.assign_coords({dim_out: coord_out}).transpose(*dims_out)
        array.attrs = self.attrs
        array.name = self.name
        return EnsembleData._from_xarray(self.ensemble, self.resample, array)
