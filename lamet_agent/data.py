"""The shared sample-bearing numerical exchange type for the neo pipeline.

This module preserves the approved ``EnsembleData`` wrapper and serialization
behavior.  Stage code adds stricter physical checks at its own boundary rather
than changing this numerical base.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Union, get_args

import gvar
import numpy
from numpy.typing import NDArray
import xarray

from .kernels import implementation as _kernel_implementation

_DimType = str
_DimsType = Sequence[_DimType]
_IndexType = Union[int, float, str]
_CoordType = Sequence[_IndexType]
_CoordsType = Dict[_DimType, _CoordType]
_ResampleType = Literal["raw", "jackknife", "bootstrap", "gvar"]
_RESAMPLE_TYPE_VALUES = get_args(_ResampleType)
_RESAMPLE_DIM = "resample"


class EnsembleInfo(NamedTuple):
    """Lattice ensemble metadata used for momentum conversion."""

    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float

    @property
    def k_s(self) -> float:
        return 2 * numpy.pi / self.L_s * _kernel_implementation.HBAR_C_GEV_FM / self.a_s

    @property
    def k_t(self) -> float:
        return 2 * numpy.pi / self.L_t * _kernel_implementation.HBAR_C_GEV_FM / self.a_t


def _is_gvar_values(values: object) -> bool:
    if isinstance(values, gvar.GVar):
        return True
    array = numpy.asarray(values)
    if array.ndim == 0:
        return isinstance(array.item(), gvar.GVar)
    if array.dtype != object:
        return False
    return all(isinstance(value, gvar.GVar) for value in array.flat)


class EnsembleData:
    """An xarray array with a mandatory leading resampling dimension."""

    def __init__(
        self,
        ensemble: Optional[EnsembleInfo],
        resample: _ResampleType,
        values: Union[List[Union[int, float, complex, NDArray]], gvar.GVar, NDArray[gvar.GVar]],
        dims: _DimsType,
        coords: _CoordsType,
        attrs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        if resample not in _RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        self.resample = resample
        self.ensemble = ensemble
        self.array = self._build_xarray(resample, values, dims, coords, attrs, name)

    @staticmethod
    def _build_xarray(resample, values, dims, coords, attrs=None, name=None) -> xarray.DataArray:
        if _RESAMPLE_DIM in dims:
            raise ValueError(f"Physical dimensions should not include resampling dimension '{_RESAMPLE_DIM}'.")
        if isinstance(values, list):
            if resample == "gvar":
                raise TypeError("'gvar' does not support list of samples")
            if len(values) == 0:
                raise ValueError("Resampled values cannot be empty.")
            resample_values = numpy.stack(values, axis=0)
        else:
            if resample != "gvar":
                raise TypeError("raw/jackknife/bootstrap data must be initialized from a list of samples.")
            if not _is_gvar_values(values):
                raise TypeError("resample='gvar' requires a gvar.GVar or an array of gvar.GVar.")
            resample_values = numpy.expand_dims(numpy.asarray(values, dtype=object), axis=0)
        if resample_values.ndim != len(dims) + 1:
            raise ValueError("Resampled data must have one leading sample axis")
        resample_coords: dict[str, Any] = {_RESAMPLE_DIM: list(range(resample_values.shape[0]))}
        for dim, size in zip(dims, resample_values.shape[1:]):
            if dim not in coords:
                raise ValueError(f"Missing dimension coordinate '{dim}'.")
            if len(coords[dim]) != size:
                raise ValueError(f"Unmatched length of coordinates for dimension '{dim}'")
            resample_coords[dim] = list(coords[dim])
        return xarray.DataArray(
            resample_values, coords=resample_coords, dims=(_RESAMPLE_DIM, *dims), name=name, attrs=attrs
        )

    @classmethod
    def _from_xarray(
        cls, ensemble: Optional[EnsembleInfo], resample: _ResampleType, array: xarray.DataArray
    ) -> "EnsembleData":
        if resample not in _RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        if len(array.dims) == 0 or array.dims[0] != _RESAMPLE_DIM:
            raise ValueError(f"The first xarray dimension must be '{_RESAMPLE_DIM}'.")
        if resample == "gvar":
            if array.sizes[_RESAMPLE_DIM] != 1:
                raise ValueError("resample='gvar' requires a length-1 dimension.")
            if not _is_gvar_values(array.values):
                raise TypeError("resample='gvar' requires gvar values.")
        obj = cls.__new__(cls)
        obj.ensemble = ensemble
        obj.resample = resample
        obj.array = array.copy(deep=False)
        return obj

    def to_netcdf(self, path: Union[str, Path]) -> None:
        """Persist the current sample representation as NetCDF4."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        array = self.array.copy(deep=False)
        array.attrs["ensemble"] = json.dumps(None if self.ensemble is None else self.ensemble._asdict())
        array.attrs["resample"] = self.resample
        if self.resample == "gvar":
            mean = numpy.asarray(gvar.mean(self.array.values), dtype=float)
            sdev = numpy.asarray(gvar.sdev(self.array.values), dtype=float)
            mean_da = xarray.DataArray(
                mean, coords=self.array.coords, dims=self.array.dims, name=self.array.name, attrs=dict(array.attrs)
            )
            mean_da.attrs["gvar_encoding"] = "mean_sdev"
            dataset = mean_da.to_dataset(name=mean_da.name or "data")
            dataset["sdev"] = (self.array.dims, sdev)
            dataset.to_netcdf(output, format="NETCDF4")
            return
        array.to_netcdf(output, format="NETCDF4", auto_complex=True)

    @classmethod
    def from_netcdf(cls, path: Union[str, Path]) -> "EnsembleData":
        try:
            dataset = xarray.open_dataset(path)
            if "gvar_encoding" in dataset.attrs or (
                len(dataset.data_vars) and "gvar_encoding" in next(iter(dataset.data_vars.values())).attrs
            ):
                data_name = next(name for name in dataset.data_vars if name != "sdev")
                mean_da = dataset[data_name]
                sdev = numpy.asarray(dataset["sdev"].values, dtype=float)
                gvar_values = gvar.gvar(numpy.asarray(mean_da.values, dtype=float), sdev)
                attrs = dict(mean_da.attrs)
                attrs.pop("gvar_encoding", None)
                ensemble_payload = json.loads(attrs.pop("ensemble"))
                ensemble = None if ensemble_payload is None else EnsembleInfo(**ensemble_payload)
                resample = attrs.pop("resample")
                rebuilt = xarray.DataArray(
                    gvar_values,
                    coords=mean_da.coords,
                    dims=mean_da.dims,
                    name=None if mean_da.name == "data" else mean_da.name,
                    attrs=attrs,
                )
                dataset.close()
                return cls._from_xarray(ensemble, resample, rebuilt)
            dataset.close()
        except (OSError, ValueError, KeyError, StopIteration):
            try:
                dataset.close()
            except UnboundLocalError:
                pass
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
    def dims(self) -> list[str]:
        return list(self.array.dims[1:])

    @property
    def coords(self) -> _CoordsType:
        return {dim: self.array.coords[dim].values.tolist() for dim in self.array.dims[1:]}

    @property
    def attrs(self) -> Dict[str, Any]:
        return dict(self.array.attrs)

    @property
    def name(self) -> Optional[str]:
        return self.array.name

    @property
    def real(self) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.real)

    @property
    def imag(self) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.imag)

    def copy(self, deep: bool = True) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.copy(deep=deep))

    @property
    def n_sample(self) -> int:
        return self.array.sizes[_RESAMPLE_DIM]

    def bin(self, bin_size: int) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        if bin_size <= 0 or bin_size >= self.n_sample:
            raise ValueError("bin_size must be positive and smaller than the sample count.")
        if self.n_sample % bin_size:
            warnings.warn("The final incomplete bin is dropped.", RuntimeWarning)
        n_bins = self.n_sample // bin_size
        values = [self.array.values[index * bin_size : (index + 1) * bin_size].mean(axis=0) for index in range(n_bins)]
        return EnsembleData(self.ensemble, "raw", values, self.dims, self.coords, self.attrs, self.name)

    def jackknife(self) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        total = self.array.values.sum(axis=0)
        values = [(total - self.array.values[index]) / (self.n_sample - 1) for index in range(self.n_sample)]
        return EnsembleData(self.ensemble, "jackknife", values, self.dims, self.coords, self.attrs, self.name)

    def bootstrap(self, n_resample: int, *, seed: int | None = None) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        if n_resample < 1:
            raise ValueError("n_resample must be positive.")
        indices = numpy.random.default_rng(seed).integers(0, self.n_sample, (n_resample, self.n_sample))
        values = [self.array.values[indices[index]].mean(axis=0) for index in range(n_resample)]
        return EnsembleData(self.ensemble, "bootstrap", values, self.dims, self.coords, self.attrs, self.name)

    @classmethod
    def concat(
        cls, data_list: Sequence["EnsembleData"], dim: _DimType, coord: Optional[_CoordType] = None
    ) -> "EnsembleData":
        if not data_list:
            raise ValueError("Cannot concatenate an empty list of EnsembleData.")
        if dim == _RESAMPLE_DIM:
            raise ValueError(f"Cannot concatenate along '{_RESAMPLE_DIM}'.")
        first = data_list[0]
        for data in data_list[1:]:
            if data.ensemble != first.ensemble or data.resample != first.resample or data.dims != first.dims:
                raise ValueError("EnsembleData metadata must match for concatenation.")
            for other_dim in first.dims:
                if other_dim != dim and data.coords[other_dim] != first.coords[other_dim]:
                    raise ValueError(f"Coordinates differ for dimension '{other_dim}'.")
        array = xarray.concat([data.array for data in data_list], dim)
        if dim in first.dims:
            if coord is not None:
                raise ValueError("Coordinates cannot be supplied for an existing dimension.")
            dims_out = [_RESAMPLE_DIM, *first.dims]
        else:
            if coord is None:
                raise ValueError("Coordinates are required for a new dimension.")
            array = array.assign_coords({dim: coord})
            dims_out = [_RESAMPLE_DIM, dim, *first.dims]
        return cls._from_xarray(first.ensemble, first.resample, array.transpose(*dims_out).sortby(dim))

    def at(self, dim: _DimType, coord: Union[_IndexType, _CoordType]) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(self.ensemble, self.resample, self.array.sel({dim: coord}, drop=True))

    def near(self, dim: _DimType, coord: Union[_IndexType, _CoordType], tolerance: float = 1e-8) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(
            self.ensemble, self.resample, self.array.sel({dim: coord}, method="nearest", tolerance=tolerance, drop=True)
        )

    @property
    def gvar(self):
        if self.resample == "gvar":
            return self.array.values[0]
        else:
            if numpy.iscomplexobj(self.array.values):
                raise TypeError("gvar conversion requires real data; select .real or .imag first.")
            n_sample = self.array.values.shape[0]
            shape = self.array.values.shape[1:]
            values = self.array.values.reshape(n_sample, -1)
            mean = numpy.mean(values, axis=0)
            if n_sample == 1:
                cov = numpy.zeros((mean.size, mean.size), mean.dtype)
            elif self.resample == "raw":
                cov = numpy.atleast_2d(numpy.cov(values, rowvar=False, ddof=1) / n_sample)
            elif self.resample == "jackknife":
                cov = numpy.atleast_2d(numpy.cov(values, rowvar=False, ddof=0) * (n_sample - 1))
            elif self.resample == "bootstrap":
                cov = numpy.atleast_2d(numpy.cov(values, rowvar=False, ddof=1))
            else:
                raise ValueError(f"Unknown resampling method '{self.resample}'.")
            return gvar.gvar(mean.reshape(shape), cov.reshape(shape + shape))

    @property
    def gvar_median(self):
        if self.resample == "gvar":
            return self.array.values[0]
        else:
            if numpy.iscomplexobj(self.array.values):
                raise TypeError("gvar conversion requires real data; select .real or .imag first.")
            n_sample = self.array.values.shape[0]
            values = self.array.values
            meanm, mean, meanp = numpy.percentile(values, q=[50 - 34.1344746, 50, 50 + 34.1344746], axis=0)
            std = numpy.maximum(meanp - mean, mean - meanm)
            if self.resample == "raw":
                std /= n_sample**0.5
            elif self.resample == "jackknife":
                std *= (n_sample - 1) ** 0.5
            elif self.resample == "bootstrap":
                pass
            else:
                raise ValueError(f"Unknown resampling method '{self.resample}'.")
            return gvar.gvar(mean, std)

    def average(self, mode: Literal["covariance", "mean", "median"] = "covariance"):
        """Return the selected center and uncertainty representation as gvars."""
        if mode == "covariance":
            return self.gvar
        if mode == "median":
            return self.gvar_median
        if mode == "mean":
            average = self.gvar
            return gvar.gvar(gvar.mean(average), gvar.sdev(average))
        raise ValueError("average mode must be 'covariance', 'mean', or 'median'")

    @property
    def mean(self):
        return gvar.mean(self.array.values[0]) if self.resample == "gvar" else gvar.mean(self.gvar)

    @property
    def sdev(self):
        return gvar.sdev(self.array.values[0]) if self.resample == "gvar" else gvar.sdev(self.gvar)

    def avg_data(self) -> "EnsembleData":
        return EnsembleData(self.ensemble, "gvar", self.gvar, self.dims, self.coords, self.attrs, self.name)

    def update_dim(self, dim: _DimType, dim_out: _DimType, coord_out: Optional[_CoordType] = None) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if dim_out != dim and (dim_out == _RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists.")
        array = self.array.rename({dim: dim_out})
        if coord_out is not None:
            array = array.assign_coords({dim_out: coord_out})
        return self._from_xarray(self.ensemble, self.resample, array)

    def sort_dim(self, dim: _DimType, ascending: bool = True) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(self.ensemble, self.resample, self.array.sortby(dim, ascending=ascending))

    def aligned_ref_array(self, ref: "EnsembleData") -> xarray.DataArray:
        if not isinstance(ref, EnsembleData):
            raise TypeError("Reference data must be EnsembleData.")
        if self.resample == "gvar":
            ref_array = xarray.DataArray(ref.gvar, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif self.resample == "raw":
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif ref.resample in {"gvar", "raw"}:
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif ref.resample == self.resample:
            ref_array = ref.array
        else:
            raise ValueError("Target and reference resampling methods are incompatible.")
        for dim in ref_array.dims:
            if dim not in self.array.dims:
                raise ValueError(f"Reference dimension '{dim}' not found in target data.")
            try:
                ref_array = ref_array.sel({dim: self.array.coords[dim]})
            except KeyError as exc:
                raise ValueError(f"Reference coordinates for '{dim}' do not cover target data.") from exc
        return ref_array.broadcast_like(self.array).transpose(*self.array.dims)

    def apply_renormalization(
        self, renorm_scheme: "EnsembleData", operator: Callable[[xarray.DataArray, xarray.DataArray], xarray.DataArray]
    ) -> "EnsembleData":
        values = operator(self.array, self.aligned_ref_array(renorm_scheme))
        values.attrs = self.attrs
        values.name = self.name
        return self._from_xarray(self.ensemble, self.resample, values)

    def mul(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value * other)

    def div(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value / other)

    def add(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value + other)

    def sub(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value - other)

    def transform_dim(
        self,
        dim: _DimType,
        dim_out: _DimType,
        coord_out: _CoordType,
        function: Callable[[NDArray, _CoordType, _CoordType, Dict[_DimType, _IndexType]], NDArray],
        dims_dispatch: Union[_DimType, _DimsType, None] = None,
    ) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if dim_out != dim and (dim_out == _RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists.")
        if dims_dispatch is None:
            dispatch: list[str] = []
        elif isinstance(dims_dispatch, str):
            dispatch = [dims_dispatch]
        else:
            dispatch = list(dims_dispatch)
        for dispatch_dim in dispatch:
            if dispatch_dim in {dim, dim_out} or dispatch_dim not in self.dims:
                raise ValueError(f"Invalid dispatch dimension '{dispatch_dim}'.")
        dims_core = [dim] + [
            candidate for candidate in self.array.dims if candidate != dim and candidate not in dispatch
        ]
        dims_out_core = [dim_out] + [candidate for candidate in dims_core if candidate != dim]
        dims_out = [dim_out if candidate == dim else candidate for candidate in self.array.dims]

        def apply_function(value: NDArray, *index_list: _IndexType) -> NDArray:
            return function(
                value, self.coords[dim], coord_out, {name: index for name, index in zip(dispatch, index_list)}
            )

        array = xarray.apply_ufunc(
            apply_function,
            self.array,
            *[self.array.coords[name] for name in dispatch],
            input_core_dims=[dims_core] + [[]] * len(dispatch),
            output_core_dims=[dims_out_core],
            exclude_dims={dim},
            vectorize=bool(dispatch),
            output_sizes={dim_out: len(coord_out)},
        )
        array = array.assign_coords({dim_out: coord_out}).transpose(*dims_out)
        array.attrs = self.attrs
        array.name = self.name
        return self._from_xarray(self.ensemble, self.resample, array)


__all__ = [
    "EnsembleInfo",
    "EnsembleData",
]
