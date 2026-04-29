from typing import Dict, List, NamedTuple, Optional, Union

import numpy
import gvar


class EnsembleInfo(NamedTuple):
    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float


class EnsembleData:
    def __init__(
        self,
        ensemble: EnsembleInfo,
        values: Union[int, float, gvar.GVar, numpy.ndarray],
        dims: List[str],
        coords: Dict[str, List[int]],
        attrs: Optional[Dict[str, str]] = None,
    ) -> None:
        if isinstance(values, (int, float)):
            values = gvar.gvar(values)
            dims = []
            coords = {}
        elif isinstance(values, gvar.GVar):
            dims = []
            coords = {}
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
            dims = list(dims)
            coords = {dim: list(coord) for dim, coord in coords.items()}
        self.ensemble = ensemble
        self.values: Union[gvar.GVar, numpy.ndarray] = values
        self.dims = dims
        self.coords = coords
        self.attrs = attrs

    def aligned_ref_values(self, ref: "EnsembleData"):
        if self.ensemble != ref.ensemble:
            raise ValueError("Cannot multiply EnsembleData with different ensembles.")
        for dim in ref.dims:
            if dim not in self.dims:
                raise ValueError(f"Reference dimension '{dim}' not found in data dimensions.")
            for coord in self.coords[dim]:
                if coord not in ref.coords[dim]:
                    raise ValueError(
                        f"Data coordinate '{coord}' for dimension '{dim}' not found in reference coordinates."
                    )

        ref_values = ref.values

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
