"""Shared lattice/QCD constants and running-coupling helpers."""

from __future__ import annotations

import numpy as np

GEV_FM = 0.1973269631
CF = 4.0 / 3.0
NF = 3
CA = 3.0
TF = 0.5


def lattice_unit_to_physical(value: float | np.ndarray, a_fm: float, spatial_extent: int, dimension: str):
    """Convert a lattice-unit quantity to a physical unit.

    Supported dimensions:
    - ``"P"``: integer lattice momentum to GeV
    - ``"M"``: lattice mass/energy to GeV
    """

    if dimension == "P":
        return np.asarray(value) * 2.0 * np.pi * GEV_FM / float(spatial_extent) / float(a_fm)
    if dimension == "M":
        return np.asarray(value) / float(a_fm) * GEV_FM
    raise ValueError(f"Unsupported lattice-unit dimension: {dimension!r}")


def qcd_beta(order: int = 0, n_f: int = NF) -> float:
    """Return the QCD beta-function coefficient at the requested loop order."""

    if order == 0:
        return 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * n_f
    if order == 1:
        return 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * n_f
    if order == 2:
        return (
            2857.0 / 54.0 * CA**3
            + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * n_f
            + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * n_f**2
        )
    raise ValueError(f"Unsupported beta-function loop order: {order}")


def alphas_nloop(mu_gev: float | np.ndarray, order: int = 0, n_f: int = NF):
    """Return ``alpha_s(mu)`` using the reference n-loop running convention from LaMETLat."""

    mu = np.asarray(mu_gev, dtype=float)
    a_s_reference = 0.293 / (4.0 * np.pi)
    temp = 1.0 + a_s_reference * qcd_beta(0, n_f) * np.log((mu / 2.0) ** 2)

    if order == 0:
        return a_s_reference * 4.0 * np.pi / temp
    if order == 1:
        return a_s_reference * 4.0 * np.pi / (
            temp + a_s_reference * qcd_beta(1, n_f) / qcd_beta(0, n_f) * np.log(temp)
        )
    if order == 2:
        return a_s_reference * 4.0 * np.pi / (
            temp
            + a_s_reference * qcd_beta(1, n_f) / qcd_beta(0, n_f) * np.log(temp)
            + a_s_reference**2
            * (
                qcd_beta(2, n_f) / qcd_beta(0, n_f) * (1.0 - 1.0 / temp)
                + qcd_beta(1, n_f) ** 2 / qcd_beta(0, n_f) ** 2 * (np.log(temp) / temp + 1.0 / temp - 1.0)
            )
        )
    raise ValueError(f"Unsupported alpha_s loop order: {order}")


def lat_unit_convert(value: float | np.ndarray, a: float, Ls: int, dimension: str):
    """Compatibility alias matching the legacy LaMETLat helper name."""

    return lattice_unit_to_physical(value, a_fm=a, spatial_extent=Ls, dimension=dimension)


def beta(order: int = 0, Nf: int = NF) -> float:
    """Compatibility alias matching the legacy LaMETLat helper name."""

    return qcd_beta(order=order, n_f=Nf)
