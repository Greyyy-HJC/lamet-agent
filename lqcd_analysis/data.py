from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Union, get_args

import gvar
import numpy
from numpy.typing import NDArray
import xarray

DimsType = Sequence[str]
CoordType = Union[int, float]
CoordsType = Dict[str, Sequence[CoordType]]

ResampleType = Literal["none", "jackknife", "bootstrap", "gvar"]
RESAMPLE_TYPE_VALUES = get_args(ResampleType)


class EnsembleInfo(NamedTuple):
    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float


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
        ensemble: Optional[EnsembleInfo],
        resample: ResampleType,
        values: Union[List[Union[int, float, complex, NDArray]], NDArray, gvar.GVar],
        dims: Sequence[str],
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
        values: Union[List[Union[int, float, complex, NDArray]], gvar.GVar, NDArray],
        dims: DimsType,
        coords: CoordsType,
        attrs: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> xarray.DataArray:
        if resample in dims:
            raise ValueError(f"Physical dimensions should not include resampling dimension '{resample}'.")

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
                raise TypeError("none/jackknife/bootstrap data must be initialized from a list of samples.")
            if not _is_gvar_values(values):
                raise TypeError("resample='gvar' requires a gvar.GVar or an NDArray[gvar.GVar].")
            resample_values = numpy.expand_dims(numpy.asarray(values, dtype=object), axis=0)

        if resample_values.ndim != len(dims) + 1:
            raise ValueError(
                "Resampled data must have one leading sample axis: "
                f"got ndim={resample_values.ndim}, physical dims={len(dims)}."
            )

        resample_dims: DimsType = (resample, *dims)
        resample_coords: CoordsType = {resample: list(range(resample_values.shape[0]))}
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
        if len(array.dims) == 0 or array.dims[0] != resample:
            raise ValueError(f"resample='{resample}' requires the first xarray dimension to be named '{resample}'.")
        if resample == "gvar":
            if array.sizes["gvar"] != 1:
                raise ValueError("resample='gvar' requires a length-1 gvar dimension.")
            if not _is_gvar_values(array.values):
                raise TypeError("resample='gvar' requires values to be gvar.GVar objects.")

        obj = cls.__new__(cls)
        obj.ensemble = ensemble
        obj.resample = resample
        obj.array = array.copy(deep=False)
        return obj

    def __repr__(self) -> str:
        return repr(self.array)

    @property
    def values(self):
        return self.array.values

    @property
    def dims(self) -> DimsType:
        dims_ = []
        for dim in self.array.dims[1:]:
            assert isinstance(dim, str)
            dims_.append(dim)
        return dims_

    @property
    def coords(self) -> CoordsType:
        coords_ = {}
        for dim in self.array.dims[1:]:
            assert isinstance(dim, str)
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
    def n_sample(self) -> int:
        return int(self.array.sizes[self.resample])

    def sel(self, indexers: Dict[str, Union[CoordType, Sequence[CoordType]]]) -> "EnsembleData":
        for dim in indexers:
            if dim not in self.dims:
                raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return EnsembleData._from_xarray(self.ensemble, self.resample, self.array.sel(indexers, drop=True))

    def at(self, dim: str, coord: Union[CoordType, Sequence[CoordType]]) -> "EnsembleData":
        return self.sel({dim: coord})

    @property
    def gvar(self) -> Union[gvar.GVar, NDArray[gvar.GVar]]:
        if self.resample == "gvar":
            return self.array.values[0]
        else:
            return gvar.dataset.avg_data(self.array.values)

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
                return self.array.values.std(0, ddof=1) * numpy.sqrt(self.n_sample)
            elif self.resample == "bootstrap":
                return self.array.values.std(0, ddof=0)
            else:
                return self.array.values.std(0, ddof=1) / numpy.sqrt(self.n_sample)

    def to_gvar_data(self) -> "EnsembleData":
        return EnsembleData(
            self.ensemble, "gvar", self.gvar, dims=self.dims, coords=self.coords, attrs=self.attrs, name=self.name
        )

    def update_dim(self, dim: str, operator: Callable[[CoordType], CoordType], dim_out: Optional[str] = None):
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")

        self.array = self.array.assign_coords({dim: [operator(coord) for coord in self.coords[dim]]})

        if dim_out is not None and dim_out != dim:
            if dim_out in self.dims:
                raise ValueError(f"Output dimension '{dim_out}' already exists in data dimensions.")
            self.array = self.array.rename({dim: dim_out})

    def update_value(self, dim: str, operator: Callable[[NDArray], NDArray]):
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")

        for coord in self.coords[dim]:
            indexer = {dim: coord}
            self.array.loc[indexer] = operator(self.array.sel(indexer).values)

    def aligned_ref_array(self, ref: "EnsembleData") -> xarray.DataArray:
        if not isinstance(ref, EnsembleData):
            raise TypeError("Reference data for renormalization must be an EnsembleData.")
        if ref.resample != "gvar":
            raise TypeError("Reference data for renormalization must use resample='gvar'.")
        if self.resample == "gvar":
            ref_array = xarray.DataArray(ref.gvar, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        else:
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)

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

    def fast_fourier_transform(self, dim: str, n: int, dim_out: str) -> "EnsembleData":
        if n <= 0:
            raise ValueError(f"Padding length must be positive, got {n}.")
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if n < self.array.sizes[dim]:
            raise ValueError(
                f"Padding length {n} is smaller than dimension '{dim}' length " f"{self.array.sizes[dim]}."
            )
        if dim_out != dim and dim_out in self.dims:
            raise ValueError(f"Output dimension '{dim_out}' already exists in data dimensions.")

        axis = self.array.get_axis_num(dim)
        assert isinstance(axis, int)
        transformed_values = numpy.fft.ifft(self.array.values, n=n, axis=axis)

        dims = []
        coords = {}
        for dim_ in self.array.dims:
            if dim_ == dim:
                dims.append(dim_out)
                coords[dim_out] = numpy.fft.fftfreq(n, d=1 / n).tolist()
            else:
                dims.append(dim_)
                coords[dim_] = self.array.coords[dim_].values.tolist()

        array = xarray.DataArray(transformed_values, dims=tuple(dims), coords=coords, attrs=self.attrs, name=self.name)
        return EnsembleData._from_xarray(self.ensemble, self.resample, array)

    def spatial_fourier_transform(self, dim: str, dim_out: str) -> "EnsembleData":
        if self.ensemble is None:
            raise ValueError("spatial_fourier_transform requires ensemble metadata with L_s.")
        return self.fast_fourier_transform(dim, self.ensemble.L_s, dim_out)
