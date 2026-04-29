from .data import EnsembleData


def multiplicative_renormalization(bare_data: EnsembleData, renorm_scheme: EnsembleData) -> EnsembleData:
    renorm_scheme_values = bare_data.aligned_ref_values(renorm_scheme)

    return EnsembleData(
        ensemble=bare_data.ensemble,
        values=bare_data.values * renorm_scheme_values,
        dims=bare_data.dims,
        coords=bare_data.coords,
        attrs=bare_data.attrs,
    )


def additive_renormalization(bare_data: EnsembleData, renorm_scheme: EnsembleData) -> EnsembleData:
    renorm_scheme_values = bare_data.aligned_ref_values(renorm_scheme)

    return EnsembleData(
        ensemble=bare_data.ensemble,
        values=bare_data.values + renorm_scheme_values,
        dims=bare_data.dims,
        coords=bare_data.coords,
        attrs=bare_data.attrs,
    )
