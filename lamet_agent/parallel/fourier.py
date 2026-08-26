"""Parallel discrete Fourier transforms over independent resample chunks."""

from __future__ import annotations

import math

import numpy as np

from ..data import EnsembleData
from ..kernels.implementation import HBAR_C_GEV_FM
from ._pool import _ParallelPool


def _fourier_chunk(task: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    values, operator = task
    return values @ operator.T


def fourier_transform(
    data: EnsembleData,
    x_grid: list[float],
    *,
    momentum_gev: float,
    phase_sign: int = 1,
    x_shift: float = 0.0,
    prefactor: str = "none",
    workers: int = 1,
    _parallel: _ParallelPool | None = None,
) -> EnsembleData:
    """Compute the authored discrete Fourier integral with physical units.

    The phase is ``exp(i * phase_sign * (x-x_shift) P z / hbar*c)`` and the
    finite integration interval uses trapezoidal endpoint weights.
    ``pz_over_2pi``, ``one_over_2pi``, and ``none`` are the only normalizations.
    """
    if "z" not in data.dims:
        raise ValueError("Fourier transformation requires z")
    z = np.asarray(data.coords["z"], dtype=float)
    if z.size < 2 or np.any(np.diff(z) <= 0):
        raise ValueError("z grid must be strictly increasing")
    if not math.isfinite(momentum_gev) or momentum_gev <= 0:
        raise ValueError("momentum_gev must be finite and positive")
    if phase_sign not in {-1, 1} or not math.isfinite(x_shift):
        raise ValueError("phase_sign must be +/-1 and x_shift must be finite")
    if prefactor not in {"pz_over_2pi", "one_over_2pi", "none"}:
        raise ValueError("unsupported Fourier prefactor")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    x = np.asarray(x_grid, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("x_grid must be finite, nonempty, and strictly increasing")
    differences = np.diff(z)
    if np.allclose(differences, differences[0], rtol=1e-7, atol=1e-12):
        weights = np.full_like(z, abs(float(differences[0])))
        quadrature = "reference_uniform_rectangle"
    else:
        weights = np.empty_like(z)
        weights[0] = abs(float(z[1] - z[0])) / 2.0
        weights[-1] = abs(float(z[-1] - z[-2])) / 2.0
        weights[1:-1] = np.abs(z[2:] - z[:-2]) / 2.0
        quadrature = "trapezoid_nonuniform"
    phase = np.exp(1j * phase_sign * (x[:, None] - x_shift) * momentum_gev * z[None, :] / HBAR_C_GEV_FM)
    normalization = (
        momentum_gev / (2.0 * math.pi * HBAR_C_GEV_FM)
        if prefactor == "pz_over_2pi"
        else 1.0 / (2.0 * math.pi)
        if prefactor == "one_over_2pi"
        else 1.0
    )
    operator = normalization * phase * weights[None, :]
    chunk_count = min(workers, data.n_sample)
    chunks = [chunk for chunk in np.array_split(np.asarray(data.values), chunk_count) if len(chunk)]
    tasks = [(chunk, operator) for chunk in chunks]
    if _parallel is None:
        with _ParallelPool(chunk_count) as parallel:
            transformed = parallel.map(
                _fourier_chunk,
                tasks,
                description="Fourier transforms",
                unit="chunk",
            )
    else:
        transformed = _parallel.map(
            _fourier_chunk,
            tasks,
            description="Fourier transforms",
            unit="chunk",
        )
    values = np.concatenate(transformed, axis=0)
    attrs = data.attrs
    attrs.update(
        {
            "fourier_convention": f"exp({phase_sign:+d}i*(x-{x_shift})*P*z)",
            "phase_sign": int(phase_sign),
            "x_shift": float(x_shift),
            "prefactor": prefactor,
            "momentum_gev": float(momentum_gev),
            "quadrature": quadrature,
            "workers": workers,
            "units": '{"values":"dimensionless","x":"dimensionless"}',
        }
    )
    return EnsembleData(
        data.ensemble,
        data.resample,
        list(values),
        ["x"],
        {"x": x.tolist()},
        attrs=attrs,
        name="quasi_distribution",
    )


__all__ = ["fourier_transform"]
