from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy
from numpy.typing import NDArray
import gvar


class EnsembleInfo(NamedTuple):
    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float


class GlobalData:
    def __init__(
        self,
        values: Union[gvar.GVar, NDArray[gvar.GVar]],
        dims: List[str],
        coords: Dict[str, Union[List[int], List[float]]],
        attrs: Optional[Dict[str, str]] = None,
    ) -> None:
        if isinstance(values, list):
            raise TypeError("Use EnsembleData for a list of ensemble resampled values.")

        dims, coords = self.validate_values_dims_coords(values, dims, coords)
        self.values = values
        self.dims = dims
        self.coords = coords
        self.attrs = attrs

    @classmethod
    def validate_values_dims_coords(
        cls,
        values: Union[gvar.GVar, NDArray[gvar.GVar]],
        dims: List[str],
        coords: Dict[str, Union[List[int], List[float]]],
    ) -> Tuple[List[str], Dict[str, Union[List[int], List[float]]]]:
        if isinstance(values, gvar.GVar):
            return [], {}
        else:
            if len(dims) != values.ndim:
                raise ValueError("Unmatched number of dimensions " f"{len(dims)} != {values.ndim}.")
            for axis, dim in enumerate(dims):
                if dim not in coords:
                    raise ValueError("Missing dimension " f"'{dim}'.")
                if len(coords[dim]) != values.shape[axis]:
                    raise ValueError(
                        f"Unmatched length of coordinates for dimension '{dim}' "
                        f"{len(coords[dim])} != {values.shape[axis]}."
                    )
            return list(dims), {dim: list(coord) for dim, coord in coords.items()}

    def aligned_ref_values(self, ref: "GlobalData"):
        for dim in ref.dims:
            if dim not in self.dims:
                raise ValueError(f"Reference dimension '{dim}' not found in data dimensions.")
            for coord in self.coords[dim]:
                if coord not in ref.coords[dim]:
                    raise ValueError(
                        f"Data coordinate '{coord}' for dimension '{dim}' not found in reference coordinates."
                    )

        ref_values = gvar.mean(ref.values)

        if isinstance(ref_values, gvar.GVar):
            return ref_values
        else:
            for axis, dim in enumerate(ref.dims):
                coord_to_index = {coord: index for index, coord in enumerate(ref.coords[dim])}
                indices = [coord_to_index[coord] for coord in self.coords[dim]]
                ref_values = ref_values.take(indices, axis=axis)

            dim_to_axis = {dim: axis for axis, dim in enumerate(ref.dims)}
            axes = [dim_to_axis[dim] for dim in self.dims if dim in ref.dims]
            if len(axes) > 1 and axes != list(range(len(axes))):
                ref_values = ref_values.transpose(axes)

            shape = tuple(len(self.coords[dim]) if dim in ref.dims else 1 for dim in self.dims)
            return ref_values.reshape(shape)


class EnsembleData:
    def __init__(
        self,
        ensemble: EnsembleInfo,
        values: Union[List[Union[int, float, complex]], List[NDArray]],
        dims: List[str],
        coords: Dict[str, Union[List[int], List[float]]],
        attrs: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(values, list):
            raise TypeError("EnsembleData values must be a list.")
        if len(values) == 0:
            raise ValueError("Resampled values cannot be empty.")

        dims, coords = GlobalData.validate_values_dims_coords(values[0], dims, coords)

        self.ensemble = ensemble
        self.values: Union[List[Union[int, float, complex]], List[NDArray]] = list(values)
        self.dims = dims
        self.coords = coords
        self.attrs = attrs

    @classmethod
    def validate_values_dims_coords(
        cls,
        values: Union[List[Union[int, float, complex]], List[NDArray]],
        dims: List[str],
        coords: Dict[str, Union[List[int], List[float]]],
    ) -> Tuple[List[str], Dict[str, Union[List[int], List[float]]]]:
        if isinstance(values[0], (int, float, complex)):
            return [], {}
        if len(dims) != values[0].ndim:
            raise ValueError("Unmatched number of dimensions " f"{len(dims)} != {values[0].ndim}.")
        for axis, dim in enumerate(dims):
            if dim not in coords:
                raise ValueError("Missing dimension " f"'{dim}'.")
            if len(coords[dim]) != values[0].shape[axis]:
                raise ValueError(
                    f"Unmatched length of coordinates for dimension '{dim}' "
                    f"{len(coords[dim])} != {values[0].shape[axis]}."
                )
        return list(dims), {dim: list(coord) for dim, coord in coords.items()}

    def at(self, dim: str, coord: Union[int, float, List[Union[int, float]]]):
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")

        axis = self.dims.index(dim)
        coord_to_index = {coord: index for index, coord in enumerate(self.coords[dim])}
        if isinstance(coord, list):
            indices = [coord_to_index[c] for c in coord]
            dims = list(self.dims)
            coords: Dict[str, Union[List[int], List[float]]] = {
                dim_: list(coord_) if dim_ != dim else list(coord) for dim_, coord_ in self.coords.items()
            }
        else:
            indices = coord_to_index[coord]
            dims = [dim_ for dim_ in self.dims if dim_ != dim]
            coords: Dict[str, Union[List[int], List[float]]] = {
                dim_: list(coord_) for dim_, coord_ in self.coords.items() if dim_ != dim
            }
        values = [value.take(indices, axis=axis) for value in self.values if isinstance(value, numpy.ndarray)]
        return EnsembleData(
            ensemble=self.ensemble,
            values=values,
            dims=dims,
            coords=coords,
            attrs=self.attrs,
        )

    def gvar(self) -> Union[gvar.GVar, NDArray[gvar.GVar]]:
        return gvar.dataset.avg_data(self.values)

    def gvar_at(self, dim: str, coord: Union[int, float, List[Union[int, float]]]):
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")

        axis = self.dims.index(dim)
        coord_to_index = {coord: index for index, coord in enumerate(self.coords[dim])}
        if isinstance(coord, list):
            indices = [coord_to_index[c] for c in coord]
        else:
            indices = [coord_to_index[coord]]
        values = [value.take(indices, axis=axis) for value in self.values if isinstance(value, numpy.ndarray)]
        return gvar.dataset.avg_data(values)

    def update_dim(self, dim_in: str, dim_out: str, operator: Callable[[Union[int, float]], Union[int, float]]):
        if dim_in not in self.dims:
            raise ValueError(f"Dimension '{dim_in}' not found in data dimensions.")
        axis = self.dims.index(dim_in)
        coord = [operator(coord_) for coord_ in self.coords[dim_in]]
        self.dims[axis] = dim_out
        self.coords.update({dim_out: coord})

    def replace_dim(self, dim_in: str, dim_out: str, coord_out: Union[List[int], List[float]]):
        if dim_in not in self.dims:
            raise ValueError(f"Input dimension '{dim_in}' not found in data dimensions.")
        if dim_out != dim_in and dim_out in self.dims:
            raise ValueError(f"Output dimension '{dim_out}' already exists in data dimensions.")

        dims = list(self.dims)
        axis = dims.index(dim_in)
        dims[axis] = dim_out
        coords: Dict[str, Union[List[int], List[float]]] = {
            dim: list(coord_out) if dim == dim_out else list(self.coords[dim]) for dim in dims
        }
        return axis, dims, coords

    def _apply_renormalization(self, renorm_scheme: GlobalData, operator):
        if not isinstance(renorm_scheme, GlobalData):
            raise TypeError("Renormalization scheme must be a GlobalData.")

        values = []
        for value in self.values:
            data = GlobalData(value, self.dims, self.coords, self.attrs)
            values.append(operator(value, data.aligned_ref_values(renorm_scheme)))

        return EnsembleData(
            ensemble=self.ensemble,
            values=values,
            dims=self.dims,
            coords=self.coords,
            attrs=self.attrs,
        )

    def multiplicative_renormalization(self, renorm_scheme: GlobalData) -> "EnsembleData":
        return self._apply_renormalization(renorm_scheme, lambda value, ref_value: value * ref_value)

    def additive_renormalization(self, renorm_scheme: GlobalData) -> "EnsembleData":
        return self._apply_renormalization(renorm_scheme, lambda value, ref_value: value + ref_value)

    def fast_fourier_transform(self, dim_in: str, dim_out: str, n: int) -> "EnsembleData":
        if n <= 0:
            raise ValueError(f"Padding length must be positive, got {n}.")
        if dim_in not in self.dims:
            raise ValueError(f"Input dimension '{dim_in}' not found in data dimensions.")
        if n < len(self.coords[dim_in]):
            raise ValueError(
                f"Padding length {n} is smaller than dimension '{dim_in}' length {len(self.coords[dim_in])}."
            )

        axis, dims, coords = self.replace_dim(
            dim_in,
            dim_out,
            (2 * numpy.pi * numpy.fft.fftfreq(n)).tolist(),
        )
        values = []
        for value in self.values:
            values.append(numpy.fft.irfft(value, n=n, axis=axis))

        return EnsembleData(
            ensemble=self.ensemble,
            values=values,
            dims=dims,
            coords=coords,
            attrs=self.attrs,
        )

    def spatial_fourier_transform(self, dim_in: str, dim_out: str) -> "EnsembleData":
        return self.fast_fourier_transform(dim_in, dim_out, self.ensemble.L_s)
