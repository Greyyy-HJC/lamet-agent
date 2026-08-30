"""Lanczos/Krylov extraction for real two- and three-point correlators.

The implementation follows arXiv:2406.20009 and arXiv:2407.21777. It keeps
the resample-wise numerical recurrence separate from stage data loading and
artifact publication.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, NamedTuple

import numpy as np

from ._pool import _ParallelPool


def _plan_tsep_tau_conversion(
    tseps: list[int] | tuple[int, ...],
    *,
    source_times: int,
    sink_times: int,
    requested_iterations: int | None = None,
    t0: int | None = None,
    time_step: int | None = None,
) -> dict[str, Any]:
    """Choose a feasible ``(t0, T**n, m)`` map from standard ``(t_f, tau)`` data."""
    available_tseps = sorted({int(value) for value in tseps})
    if not available_tseps or any(value < 0 for value in available_tseps):
        raise ValueError("Lanczos conversion requires nonnegative integer tsep values")
    if t0 is not None and (type(t0) is not int or t0 < 0):
        raise ValueError("lanczos_t0 must be a nonnegative integer")
    if time_step is not None and (type(time_step) is not int or time_step < 1):
        raise ValueError("lanczos_time_step must be a positive integer")
    if requested_iterations is not None and (type(requested_iterations) is not int or requested_iterations < 1):
        raise ValueError("lanczos_iterations must be a positive integer")

    tsep_set = set(available_tseps)
    starts = [2 * t0] if t0 is not None else [value for value in available_tseps if value % 2 == 0]
    inferred_steps = {
        later - earlier
        for index, earlier in enumerate(available_tseps)
        for later in available_tseps[index + 1 :]
        if later > earlier
    }
    steps = [time_step] if time_step is not None else sorted({1, *inferred_steps})
    candidates: list[dict[str, int]] = []
    for start in starts:
        if start not in tsep_set:
            continue
        candidate_t0 = start // 2
        for step in steps:
            progression_length = 0
            while start + step * progression_length in tsep_set:
                progression_length += 1
            c3_capacity = (progression_length + 1) // 2
            source_points = max(0, (source_times - 1 - start) // step + 1)
            sink_points = max(0, (sink_times - 1 - start) // step + 1)
            max_iterations = min(c3_capacity, source_points // 2, sink_points // 2)
            if max_iterations < 1:
                continue
            if requested_iterations is not None and max_iterations < requested_iterations:
                continue
            candidates.append(
                {
                    "t0": candidate_t0,
                    "time_step": step,
                    "max_iterations": max_iterations,
                }
            )
    if not candidates:
        requested = "automatic order" if requested_iterations is None else f"m={requested_iterations}"
        raise ValueError(
            "standard 3pt tsep values cannot form a Lanczos grid for "
            f"{requested} with t0={t0!r}, time_step={time_step!r}; "
            "need tsep=2*t0+n*k for every k=0..2m-2 and C2 through 2*t0+n*(2m-1)"
        )
    chosen = max(
        candidates,
        key=lambda item: (
            item["max_iterations"],
            -item["t0"],
            -item["time_step"],
        ),
    )
    iterations = requested_iterations or chosen["max_iterations"]
    selected_tseps = [2 * chosen["t0"] + chosen["time_step"] * k for k in range(2 * iterations - 1)]
    used_points = []
    used_tau_by_tsep: dict[int, set[int]] = {value: set() for value in selected_tseps}
    for sigma_index in range(iterations):
        for tau_index in range(iterations):
            sigma = chosen["t0"] + chosen["time_step"] * sigma_index
            tau = chosen["t0"] + chosen["time_step"] * tau_index
            tsep = sigma + tau
            used_tau_by_tsep[tsep].add(tau)
            used_points.append(
                {
                    "sigma_index": sigma_index,
                    "tau_index": tau_index,
                    "sigma": sigma,
                    "tau": tau,
                    "tsep": tsep,
                }
            )
    discarded_by_tsep = {
        str(tsep): [tau for tau in range(tsep + 1) if tau not in used_tau_by_tsep.get(tsep, set())]
        for tsep in available_tseps
    }
    total_points = sum(tsep + 1 for tsep in available_tseps)
    used_count = iterations**2
    return {
        **chosen,
        "iterations": iterations,
        "available_tseps": available_tseps,
        "selected_tseps": selected_tseps,
        "used_points": used_points,
        "used_point_count": used_count,
        "discarded_point_count": total_points - used_count,
        "total_point_count": total_points,
        "discarded_tau_by_tsep": discarded_by_tsep,
        "warning": (
            f"Lanczos uses {used_count} of {total_points} standard 3pt (tsep,tau) points per z; "
            f"{total_points - used_count} points are discarded by t0={chosen['t0']}, "
            f"T**{chosen['time_step']}, and the complete m={iterations} square requirement."
        ),
    }


def _plan_twopt_grid(
    *,
    source_times: int,
    sink_times: int,
    requested_iterations: int | None = None,
    t0: int | None = None,
    time_step: int | None = None,
) -> dict[str, Any]:
    """Plan ``C2(2*t0 + n*r)`` sampling for a 2pt-only Lanczos job."""
    selected_t0 = 0 if t0 is None else t0
    selected_step = 1 if time_step is None else time_step
    if type(selected_t0) is not int or selected_t0 < 0:
        raise ValueError("lanczos_t0 must be a nonnegative integer")
    if type(selected_step) is not int or selected_step < 1:
        raise ValueError("lanczos_time_step must be a positive integer")
    c2_start = 2 * selected_t0
    source_points = max(0, (source_times - 1 - c2_start) // selected_step + 1)
    sink_points = max(0, (sink_times - 1 - c2_start) // selected_step + 1)
    maximum = min(source_points // 2, sink_points // 2)
    iterations = maximum if requested_iterations is None else requested_iterations
    if type(iterations) is not int or iterations < 1 or iterations > maximum:
        raise ValueError(
            f"lanczos_iterations must be in [1, {maximum}] after t0={selected_t0}, "
            f"time_step={selected_step}; got {iterations}"
        )
    return {
        "t0": selected_t0,
        "time_step": selected_step,
        "iterations": iterations,
        "max_iterations": maximum,
    }


class _Ritz(NamedTuple):
    """Ritz values and eigenvectors in the oblique-Lanczos convention."""

    values: np.ndarray
    right_vectors: np.ndarray
    inverse_vectors: np.ndarray
    cullum_willoughby_distance: np.ndarray

    def physical_order(self) -> np.ndarray:
        """Return indices of ``0 < lambda < 1`` ordered from low to high energy."""
        physical = (self.values > 0.0) & (self.values < 1.0)
        return np.flatnonzero(physical)[np.argsort(self.values[physical])[::-1]]

    def physical_value(self, state: int) -> float:
        """Return one ordered physical Ritz value or NaN when it is absent."""
        order = self.physical_order()
        return float(self.values[order[state]]) if state < len(order) else float("nan")

    def filter_spurious(self, epsilon: float) -> "_Ritz":
        """Apply a Cullum-Willoughby distance threshold."""
        keep = (self.cullum_willoughby_distance > epsilon) & (self.values < 1.0)
        return _Ritz(
            self.values[keep],
            self.right_vectors[:, keep],
            self.inverse_vectors[keep, :],
            np.empty(0, dtype=float),
        )


def _transfer_matrix_numpy(c2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the real oblique-Lanczos tridiagonal matrix in double precision."""
    k = len(c2)
    m = k // 2
    a = np.zeros((k, m), dtype=float)
    b = np.zeros((k, m), dtype=float)
    g = np.zeros((k, m), dtype=float)
    matrix = np.zeros((m, m), dtype=float)
    a[:, 0] = c2 / c2[0]

    for j in range(m - 1):
        alpha = a[1, j]
        beta = b[1, j]
        gamma = g[1, j]
        product = a[2, j] - alpha**2 - beta * gamma
        if product == 0.0:
            matrix[j, j] = alpha
            stop = j + 1
            return matrix[:stop, :stop], a[1, :stop], b[1, :stop], g[1, :stop]
        gamma_next = np.sqrt(abs(product))
        beta_next = product / gamma_next

        k -= 2
        a[:k, j + 1] = (
            a[2 : k + 2, j]
            - 2.0 * alpha * a[1 : k + 1, j]
            + alpha**2 * a[:k, j]
            + alpha * (beta * g[:k, j] + gamma * b[:k, j])
            - (beta * g[1 : k + 1, j] + gamma * b[1 : k + 1, j])
            + gamma * beta * a[:k, j - 1]
        ) / product
        g[:k, j + 1] = (a[1 : k + 1, j] - alpha * a[:k, j] - gamma * b[:k, j]) / beta_next
        b[:k, j + 1] = (a[1 : k + 1, j] - alpha * a[:k, j] - beta * g[:k, j]) / gamma_next
        matrix[j, j] = alpha
        matrix[j, j + 1] = beta_next
        matrix[j + 1, j] = gamma_next

    matrix[m - 1, m - 1] = a[1, m - 1]
    return matrix, a[1], b[1], g[1]


def _transfer_matrix_gmpy2(c2: np.ndarray, precision: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the recurrence with ``precision`` decimal digits using gmpy2."""
    from gmpy2 import context as gmpy2_context
    from gmpy2 import get_context as gmpy2_get_context
    from gmpy2 import mpfr, sqrt

    k = len(c2)
    m = k // 2
    bits = int(np.ceil(precision * np.log2(10)))
    with gmpy2_context(gmpy2_get_context(), precision=bits):
        zero = mpfr(0)
        one = mpfr(1)
        a = [[zero for _ in range(m)] for _ in range(k)]
        b = [[zero for _ in range(m)] for _ in range(k)]
        g = [[zero for _ in range(m)] for _ in range(k)]
        matrix = [[zero for _ in range(m)] for _ in range(m)]
        c2_zero = mpfr(float(c2[0]))
        for t, value in enumerate(c2):
            a[t][0] = mpfr(float(value)) / c2_zero

        def arrays(stop: int):
            return (
                np.asarray([[float(value) for value in row[:stop]] for row in matrix[:stop]]),
                np.asarray([float(value) for value in a[1][:stop]]),
                np.asarray([float(value) for value in b[1][:stop]]),
                np.asarray([float(value) for value in g[1][:stop]]),
            )

        for j in range(m - 1):
            alpha = a[1][j]
            beta = b[1][j]
            gamma = g[1][j]
            product = a[2][j] - alpha**2 - beta * gamma
            if product == 0:
                matrix[j][j] = alpha
                return arrays(j + 1)
            gamma_next = sqrt(abs(product))
            beta_next = product / gamma_next
            k -= 2
            for t in range(k):
                a[t][j + 1] = (one / product) * (
                    a[t + 2][j]
                    - 2 * alpha * a[t + 1][j]
                    + alpha**2 * a[t][j]
                    + alpha * (beta * g[t][j] + gamma * b[t][j])
                    - (beta * g[t + 1][j] + gamma * b[t + 1][j])
                    + gamma * beta * a[t][j - 1]
                )
                g[t][j + 1] = (one / beta_next) * (a[t + 1][j] - alpha * a[t][j] - gamma * b[t][j])
                b[t][j + 1] = (one / gamma_next) * (a[t + 1][j] - alpha * a[t][j] - beta * g[t][j])
            matrix[j][j] = alpha
            matrix[j][j + 1] = beta_next
            matrix[j + 1][j] = gamma_next
        matrix[m - 1][m - 1] = a[1][m - 1]
        return arrays(m)


def _transfer_matrix(c2: np.ndarray, precision: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the transfer-matrix projection from an averaged real 2pt signal."""
    values = np.asarray(c2, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Lanczos 2pt input must be a one-dimensional array with at least two times")
    if not np.all(np.isfinite(values)):
        raise ValueError("Lanczos 2pt input contains NaN or Inf")
    if values[0] <= 0.0:
        raise ValueError("Lanczos normalization requires the ensemble-average C(0) to be positive")
    if precision < 0:
        raise ValueError("Lanczos precision must be nonnegative")
    return _transfer_matrix_numpy(values) if precision == 0 else _transfer_matrix_gmpy2(values, precision)


def _ritz_spectrum(matrix: np.ndarray, epsilon_float: float = 1e-12) -> _Ritz:
    """Compute real Ritz values and their Cullum-Willoughby distances."""
    m = len(matrix)
    if m == 0:
        empty = np.empty(0, dtype=float)
        return _Ritz(empty, np.empty((0, 0)), np.empty((0, 0)), empty)
    if m == 1:
        value = np.asarray([matrix[0, 0]], dtype=float)
        return _Ritz(value, np.ones((1, 1)), np.ones((1, 1)), np.abs(value))
    values = np.linalg.eigvals(matrix)
    real = (np.abs(np.angle(values)) <= epsilon_float) & (values != 0)
    values = values[real].real
    reduced = np.linalg.eigvals(matrix[1:, 1:])
    distances = np.min(np.abs(values[:, None] - reduced[None, :]), axis=1) if len(values) else np.empty(0, dtype=float)
    return _Ritz(values, np.identity(len(values)), np.identity(len(values)), distances)


def _ritz_hermitian(matrix: np.ndarray, a_cw: float = 10.0, b_cw: float = 1.0, epsilon_float: float = 1e-8) -> _Ritz:
    """Keep the Hermitian Ritz subspace and apply the iteration-local CW cut."""
    m = len(matrix)
    if m == 0:
        empty = np.empty(0, dtype=float)
        return _Ritz(empty, np.empty((0, 0)), np.empty((0, 0)), empty)
    values, right = np.linalg.eig(matrix)
    right = right * np.exp(-1j * np.angle(right[0, :]))[None, :]
    inverse = np.linalg.inv(right)
    norms_squared = inverse[:, 0].conj() / right[0, :]
    keep = (
        (np.abs(values.imag) <= epsilon_float * np.abs(values))
        & (np.abs(norms_squared.imag) <= epsilon_float * np.abs(norms_squared))
        & (norms_squared.real > 0.0)
        & (values != 0)
    )
    values = values[keep].real
    right = right[:, keep]
    inverse = inverse[keep, :]
    if not len(values):
        return _Ritz(values, right, inverse, np.empty(0, dtype=float))

    reduced = np.linalg.eigvals(matrix[1:, 1:])
    reduced = reduced[np.abs(reduced.imag) <= epsilon_float * np.abs(reduced)].real
    if len(reduced):
        distances = np.min(np.abs(values[:, None] - reduced[None, :]), axis=1)
        epsilon_cw = (np.max(distances) - np.min(distances)) / (a_cw * len(values) + b_cw)
        keep = distances > epsilon_cw
        values = values[keep]
        right = right[:, keep]
        inverse = inverse[keep, :]
    return _Ritz(values, right, inverse, np.empty(0, dtype=float))


def _filter_twopt_cw(results: list[list[_Ritz]]) -> list[list[_Ritz]]:
    """Apply the bootstrap-histogram CW prescription to a nested result."""
    n_boot = len(results)
    n_iterations = max((len(result) for result in results), default=0)
    distances = [
        value
        for result in results
        for ritz in result
        for value in ritz.cullum_willoughby_distance
        if np.isfinite(value) and value > 0.0
    ]
    if not distances or n_boot == 0 or n_iterations == 0:
        return results
    n_lambda = max(1, round(len(distances) / n_boot / n_iterations))
    delta = n_boot * max(n_iterations - n_lambda, 0) * 3 / 4
    hist, edges = np.histogram(np.log(distances), bins=max(1, 4 * n_lambda))
    crossing = next((index for index, count in enumerate(hist) if count > delta), len(hist))
    epsilon = np.exp(edges[min(crossing, len(edges) - 1)]) / 50
    return [[ritz.filter_spurious(epsilon) for ritz in result] for result in results]


def _analyze_twopt(
    c2_configurations: np.ndarray,
    n_bootstrap: int,
    *,
    seed: Any = None,
    precision: int = 0,
    max_iterations: int | None = None,
) -> list[list[_Ritz]]:
    """Return CW-filtered inner-bootstrap Ritz spectra for every iteration."""
    c2 = np.asarray(c2_configurations, dtype=float)
    if c2.ndim != 2:
        raise ValueError("Lanczos 2pt input must have shape (configuration, time)")
    available = c2.shape[1] // 2
    requested = available if max_iterations is None else int(max_iterations)
    if requested < 1 or requested > available:
        raise ValueError(f"Lanczos iterations must be in [1, {available}], got {requested}")
    if n_bootstrap < 1:
        raise ValueError("Lanczos inner bootstrap count must be positive")
    rng = np.random.default_rng(seed)
    results: list[list[_Ritz]] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, c2.shape[0], c2.shape[0])
        matrix, _alpha, _beta, _gamma = _transfer_matrix(c2[indices].mean(axis=0)[: 2 * requested], precision=precision)
        results.append([_ritz_spectrum(matrix[:m, :m]) for m in range(1, len(matrix) + 1)])
    return _filter_twopt_cw(results)


def _krylov_polynomial(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct right and left Krylov-polynomial coefficients."""
    m = len(alpha)
    right = np.zeros((m, m), dtype=float)
    left = np.zeros((m, m), dtype=float)
    right[0, 0] = left[0, 0] = 1.0
    if m == 1:
        return right, left
    right[1, 0], right[1, 1] = -alpha[0] / gamma[1], 1.0 / gamma[1]
    left[1, 0], left[1, 1] = -alpha[0] / beta[1], 1.0 / beta[1]
    for j in range(1, m - 1):
        for t in range(j + 2):
            right[j + 1, t] = (right[j, t - 1] - alpha[j] * right[j, t] - beta[j] * right[j - 1, t]) / gamma[j + 1]
            left[j + 1, t] = (left[j, t - 1] - alpha[j] * left[j, t] - gamma[j] * left[j - 1, t]) / beta[j + 1]
    return right, left


def _ritz_rotator(ritz: _Ritz, right_krylov: np.ndarray, left_krylov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct normalized right and left Ritz rotators."""
    norms = np.sqrt((ritz.inverse_vectors[:, 0].conj() / ritz.right_vectors[0, :]).real)
    right = np.einsum("k,ik,it->kt", norms, ritz.right_vectors, right_krylov)
    left = np.einsum("k,ki,it->kt", 1.0 / norms, ritz.inverse_vectors, left_krylov)
    return right, left


def _analyze_threept(
    c3_configurations: np.ndarray,
    c2_sink_configurations: np.ndarray,
    c2_source_configurations: np.ndarray,
    n_bootstrap: int,
    *,
    seed: Any = None,
    precision: int = 0,
    max_iterations: int | None = None,
) -> list[list[np.ndarray]]:
    """Return inner-bootstrap source-to-sink Ritz-basis matrix elements."""
    c3 = np.asarray(c3_configurations, dtype=float)
    c2_sink = np.asarray(c2_sink_configurations, dtype=float)
    c2_source = np.asarray(c2_source_configurations, dtype=float)
    if c3.ndim != 3 or c3.shape[1] != c3.shape[2]:
        raise ValueError("Lanczos 3pt input must have shape (configuration, sigma, tau) with a square time grid")
    if c2_sink.ndim != 2 or c2_source.ndim != 2:
        raise ValueError("Lanczos source and sink 2pt inputs must have shape (configuration, time)")
    if c3.shape[0] != c2_sink.shape[0] or c3.shape[0] != c2_source.shape[0]:
        raise ValueError("Lanczos 2pt and 3pt configuration counts must match")
    available = min(c3.shape[1], c2_sink.shape[1] // 2, c2_source.shape[1] // 2)
    requested = available if max_iterations is None else int(max_iterations)
    if requested < 1 or requested > available:
        raise ValueError(f"Lanczos iterations must be in [1, {available}], got {requested}")
    if n_bootstrap < 1:
        raise ValueError("Lanczos inner bootstrap count must be positive")

    rng = np.random.default_rng(seed)
    results: list[list[np.ndarray]] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, c3.shape[0], c3.shape[0])
        c2_sink_mean = c2_sink[indices].mean(axis=0)[: 2 * requested]
        c2_source_mean = c2_source[indices].mean(axis=0)[: 2 * requested]
        c3_mean = c3[indices].mean(axis=0) / np.sqrt(c2_sink_mean[0] * c2_source_mean[0])

        sink_matrix, sink_alpha, sink_beta, sink_gamma = _transfer_matrix(c2_sink_mean, precision=precision)
        source_matrix, source_alpha, source_beta, source_gamma = _transfer_matrix(c2_source_mean, precision=precision)
        sink_krylov = _krylov_polynomial(sink_alpha, sink_beta, sink_gamma)
        source_krylov = _krylov_polynomial(source_alpha, source_beta, source_gamma)
        usable = min(requested, len(sink_matrix), len(source_matrix))
        matrices: list[np.ndarray] = []
        for m in range(1, usable + 1):
            sink_ritz = _ritz_hermitian(sink_matrix[:m, :m])
            source_ritz = _ritz_hermitian(source_matrix[:m, :m])
            sink_left = _ritz_rotator(sink_ritz, sink_krylov[0][:m, :m], sink_krylov[1][:m, :m])[1]
            source_right = _ritz_rotator(source_ritz, source_krylov[0][:m, :m], source_krylov[1][:m, :m])[0]
            matrices.append(
                np.einsum(
                    "fs,st,it->fi",
                    sink_left[sink_ritz.physical_order()],
                    c3_mean[:m, :m],
                    source_right[source_ritz.physical_order()],
                )
            )
        results.append(matrices)
    return results


def _median_twopt_energies(results: list[list[_Ritz]], *, max_states: int, time_step: int = 1) -> np.ndarray:
    """Aggregate inner bootstraps into ``(iteration, state)`` median energies."""
    if time_step < 1:
        raise ValueError("Lanczos time_step must be a positive integer")
    n_iterations = max((len(result) for result in results), default=0)
    energies = np.full((n_iterations, max_states), np.nan, dtype=float)
    for m in range(n_iterations):
        for state in range(max_states):
            values = np.asarray(
                [result[m].physical_value(state) for result in results if m < len(result)],
                dtype=float,
            )
            values = values[np.isfinite(values) & (values > 0.0) & (values < 1.0)]
            if values.size:
                energies[m, state] = -np.log(np.median(values)) / time_step
    return energies


def _median_threept_matrix(results: list[list[np.ndarray]], *, iteration: int, max_states: int) -> np.ndarray:
    """Aggregate one iteration into a fixed, NaN-padded state matrix."""
    samples = np.full((len(results), max_states, max_states), np.nan, dtype=float)
    for index, result in enumerate(results):
        if iteration > len(result):
            continue
        matrix = np.real_if_close(result[iteration - 1]).real
        n_final = min(max_states, matrix.shape[0])
        n_initial = min(max_states, matrix.shape[1])
        samples[index, :n_final, :n_initial] = matrix[:n_final, :n_initial]
    out = np.full((max_states, max_states), np.nan, dtype=float)
    for final in range(max_states):
        for initial in range(max_states):
            values = samples[:, final, initial]
            values = values[np.isfinite(values)]
            if values.size:
                out[final, initial] = np.median(values)
    return out


def _momentum(data: Any, name: str) -> tuple[int, int, int]:
    value = data.attrs.get(name)
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"correlator {name} is not a momentum triple") from exc
    if not isinstance(decoded, list) or len(decoded) != 3 or any(type(component) is not int for component in decoded):
        raise ValueError(f"correlator {name} is not a momentum triple")
    return tuple(decoded)


def _ordered_integer_coords(data: Any, dim: str) -> list[int]:
    if dim not in data.dims:
        raise ValueError(f"Lanczos input is missing the {dim!r} dimension")
    values = data.coords[dim]
    if any(type(value) is not int for value in values):
        raise ValueError(f"Lanczos {dim} coordinates must be integers")
    return [int(value) for value in values]


def prepare_lanczos_data(
    correlators: Mapping[str, Any],
    *,
    scope: str,
    t0: int | None = None,
    time_step: int | None = None,
) -> dict[str, Any]:
    """Select raw channels and build the original effective Lanczos grids."""
    if scope not in {"2pt_spectrum", "3pt_matrix"}:
        raise ValueError("Lanczos scope must be 2pt_spectrum or 3pt_matrix")
    if not correlators:
        raise ValueError("Lanczos requires selected correlators")
    if any(getattr(data, "resample", None) != "raw" for data in correlators.values()):
        raise ValueError("Lanczos requires configuration-level raw correlators")
    sample_counts = {data.n_sample for data in correlators.values()}
    if len(sample_counts) != 1:
        raise ValueError("Lanczos correlators must share configuration counts")
    two_points = [
        (correlator_id, data)
        for correlator_id, data in correlators.items()
        if data.attrs.get("correlator_type") == "two_point"
    ]
    three_points = [
        (correlator_id, data)
        for correlator_id, data in correlators.items()
        if data.attrs.get("correlator_type") == "three_point"
    ]
    for _correlator_id, data in two_points:
        if data.dims != ["t"]:
            raise ValueError("Lanczos two-point inputs must have only the t dimension")
        times = _ordered_integer_coords(data, "t")
        if times != list(range(len(times))):
            raise ValueError("Lanczos two-point times must be consecutive from zero")

    if scope == "2pt_spectrum":
        if not 1 <= len(two_points) <= 2 or three_points:
            raise ValueError("Lanczos 2pt_spectrum requires one or two two-point correlators only")
        channels = sorted(two_points, key=lambda item: item[0])
        source_id, source_data = channels[0]
        sink_id, sink_data = channels[-1]
        plan = _plan_twopt_grid(
            source_times=len(source_data.coords["t"]),
            sink_times=len(sink_data.coords["t"]),
            t0=t0,
            time_step=time_step,
        )
        three_point_id = None
        three_point = None
    else:
        if len(three_points) != 1:
            raise ValueError("Lanczos 3pt_matrix requires exactly one three-point correlator")
        three_point_id, three_point = three_points[0]
        if three_point.dims != ["tsep", "tau", "z"]:
            raise ValueError("Lanczos three-point input must have tsep, tau, and z dimensions")
        source_momentum = _momentum(three_point, "source_momentum")
        sink_momentum = _momentum(three_point, "sink_momentum")

        def matching(momentum: tuple[int, int, int]) -> list[tuple[str, Any]]:
            return [
                (correlator_id, data)
                for correlator_id, data in two_points
                if _momentum(data, "source_momentum") == momentum and _momentum(data, "sink_momentum") == momentum
            ]

        source_matches = matching(source_momentum)
        sink_matches = matching(sink_momentum)
        if len(source_matches) != 1 or len(sink_matches) != 1:
            raise ValueError("Lanczos 3pt_matrix requires exactly one matching source and sink two-point channel")
        source_id, source_data = source_matches[0]
        sink_id, sink_data = sink_matches[0]
        tseps = _ordered_integer_coords(three_point, "tsep")
        taus = _ordered_integer_coords(three_point, "tau")
        if any(any(tau not in taus for tau in range(tsep + 1)) for tsep in tseps):
            raise ValueError("Lanczos three-point tau coordinates must cover 0 through every tsep")
        plan = _plan_tsep_tau_conversion(
            tseps,
            source_times=len(source_data.coords["t"]),
            sink_times=len(sink_data.coords["t"]),
            t0=t0,
            time_step=time_step,
        )

    iterations = int(plan["iterations"])
    c2_times = [2 * int(plan["t0"]) + int(plan["time_step"]) * index for index in range(2 * iterations)]
    source = np.real(np.asarray(source_data.values)[:, c2_times])
    sink = np.real(np.asarray(sink_data.values)[:, c2_times])
    for label, values in (("source", source), ("sink", sink)):
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"Lanczos effective {label} two-point data are invalid")
        if float(np.mean(values[:, 0])) <= 0.0:
            raise ValueError(f"Lanczos effective {label} normalization requires C2(0)>0")

    c3_by_z: list[np.ndarray] = []
    z_values: list[int | float] = []
    if scope == "3pt_matrix":
        tseps = [int(value) for value in three_point.coords["tsep"]]
        taus = [int(value) for value in three_point.coords["tau"]]
        z_values = list(three_point.coords["z"])
        raw_c3 = np.asarray(three_point.values)
        for z_index, _z in enumerate(z_values):
            effective = np.empty((three_point.n_sample, iterations, iterations), dtype=raw_c3.dtype)
            for point in plan["used_points"]:
                effective[:, point["sigma_index"], point["tau_index"]] = raw_c3[
                    :, tseps.index(point["tsep"]), taus.index(point["tau"]), z_index
                ]
            if not np.all(np.isfinite(effective)):
                raise ValueError(f"effective Lanczos three-point data contain NaN or Inf for z={_z}")
            c3_by_z.append(effective)

    inspection = {
        "status": ("valid_with_discarded_points" if int(plan.get("discarded_point_count", 0)) > 0 else "valid"),
        "scope": scope,
        "configuration_count": int(source.shape[0]),
        "source_correlator_id": source_id,
        "sink_correlator_id": sink_id,
        "three_point_correlator_id": three_point_id,
        "source_2pt_times": c2_times,
        "sink_2pt_times": c2_times,
        "iterations": iterations,
        "max_iterations": int(plan["max_iterations"]),
        "lanczos_t0": int(plan["t0"]),
        "lanczos_time_step": int(plan["time_step"]),
        "sampling_plan": plan,
    }
    if scope == "3pt_matrix":
        inspection.update(
            {
                "z_values": z_values,
                "point_usage_warning": str(plan["warning"]),
                "point_usage": {
                    "used_per_z": int(plan["used_point_count"]),
                    "discarded_per_z": int(plan["discarded_point_count"]),
                    "used_all_z": int(plan["used_point_count"]) * len(z_values),
                    "discarded_all_z": int(plan["discarded_point_count"]) * len(z_values),
                },
            }
        )
    return {
        "inspection": inspection,
        "source": source,
        "sink": sink,
        "source_data": source_data,
        "sink_data": sink_data,
        "three_point": three_point,
        "c3_by_z": c3_by_z,
        "z_values": z_values,
    }


def _bin_configurations(values: np.ndarray, bin_size: int) -> np.ndarray:
    if type(bin_size) is not int or bin_size < 1:
        raise ValueError("Lanczos bin_size must be a positive integer")
    n_bins = len(values) // bin_size
    if n_bins < 2:
        raise ValueError("Lanczos binning must leave at least two configurations")
    return values[: n_bins * bin_size].reshape(n_bins, bin_size, *values.shape[1:]).mean(axis=1)


def _outer_indices(
    n_configurations: int,
    *,
    resampling: str,
    bootstrap_samples: int | None,
    seed: int,
) -> list[np.ndarray]:
    if resampling == "jackknife":
        base = np.arange(n_configurations)
        return [np.delete(base, index) for index in range(n_configurations)]
    if resampling == "bootstrap":
        if type(bootstrap_samples) is not int or bootstrap_samples < 1:
            raise ValueError("Lanczos bootstrap requires bootstrap_samples")
        rng = np.random.default_rng(seed)
        return [rng.integers(0, n_configurations, n_configurations) for _ in range(bootstrap_samples)]
    raise ValueError("Lanczos resampling must be jackknife or bootstrap")


def _twopt_outer_result(task: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    (
        outer,
        indices,
        channels,
        inner_samples,
        seed,
        precision,
        iterations,
        max_states,
        time_step,
    ) = task
    values = np.full((len(channels), iterations, max_states), np.nan, dtype=float)
    for channel, data in enumerate(channels):
        inner = _analyze_twopt(
            data[indices],
            inner_samples,
            seed=np.random.SeedSequence([seed, outer, channel]),
            precision=precision,
            max_iterations=iterations,
        )
        energies = _median_twopt_energies(inner, max_states=max_states, time_step=time_step)
        values[channel, : len(energies)] = energies
    return outer, values


def _threept_outer_result(task: tuple[Any, ...]) -> tuple[int, np.ndarray]:
    (
        outer,
        indices,
        c3_by_z,
        sink,
        source,
        components,
        inner_samples,
        seed,
        precision,
        iterations,
        max_states,
    ) = task
    values = np.full(
        (len(c3_by_z), len(components), max_states, max_states),
        np.nan,
        dtype=float,
    )
    for z_index, c3 in enumerate(c3_by_z):
        for component_index, component in enumerate(components):
            signal = np.real(c3) if component == "real" else np.imag(c3)
            inner = _analyze_threept(
                signal[indices],
                sink[indices],
                source[indices],
                inner_samples,
                seed=np.random.SeedSequence([seed, outer, z_index, component_index]),
                precision=precision,
                max_iterations=iterations,
            )
            values[z_index, component_index] = _median_threept_matrix(
                inner,
                iteration=iterations,
                max_states=max_states,
            )
    return outer, values


def analyze_prepared_lanczos(
    prepared: Mapping[str, Any],
    *,
    components: str,
    max_states: int,
    resampling: str,
    bootstrap_samples: int | None,
    bin_size: int,
    inner_samples: int,
    precision: int,
    seed: int,
    workers: int,
    _parallel: _ParallelPool | None = None,
) -> dict[str, Any]:
    """Run the original nested outer/inner Lanczos resampling procedure."""
    if type(max_states) is not int or max_states < 1:
        raise ValueError("Lanczos nstate must contain one positive integer")
    if type(inner_samples) is not int or inner_samples < 1:
        raise ValueError("Lanczos inner_samples must be positive")
    if type(precision) is not int or precision < 0:
        raise ValueError("Lanczos precision must be nonnegative")
    if type(workers) is not int or workers < 1:
        raise ValueError("Lanczos workers must be positive")
    inspection = prepared["inspection"]
    source = _bin_configurations(np.asarray(prepared["source"]), bin_size)
    sink = _bin_configurations(np.asarray(prepared["sink"]), bin_size)
    c3_by_z = [_bin_configurations(np.asarray(values), bin_size) for values in prepared["c3_by_z"]]
    outer = _outer_indices(
        len(source),
        resampling=resampling,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    iterations = int(inspection["iterations"])
    time_step = int(inspection["lanczos_time_step"])
    if inspection["scope"] == "2pt_spectrum":
        channels = [source] if np.array_equal(source, sink) else [source, sink]
        labels = ["source"] if len(channels) == 1 else ["source", "sink"]
        values = np.full(
            (len(outer), len(channels), iterations, max_states),
            np.nan,
            dtype=float,
        )

        tasks = [
            (
                index,
                indices,
                channels,
                inner_samples,
                seed,
                precision,
                iterations,
                max_states,
                time_step,
            )
            for index, indices in enumerate(outer)
        ]
        if _parallel is None:
            with _ParallelPool(min(workers, len(tasks))) as parallel:
                results = parallel.map(
                    _twopt_outer_result,
                    tasks,
                )
        else:
            results = _parallel.map(
                _twopt_outer_result,
                tasks,
            )
        for index, result in results:
            values[index] = result
        return {"values": values, "channels": labels, "outer_samples": len(outer)}

    selected_components = {
        "real": ["real"],
        "imag": ["imag"],
        "both": ["real", "imag"],
    }[components]
    matrices = np.full(
        (len(outer), len(c3_by_z), len(selected_components), max_states, max_states),
        np.nan,
        dtype=float,
    )

    tasks = [
        (
            index,
            indices,
            c3_by_z,
            sink,
            source,
            selected_components,
            inner_samples,
            seed,
            precision,
            iterations,
            max_states,
        )
        for index, indices in enumerate(outer)
    ]
    if _parallel is None:
        with _ParallelPool(min(workers, len(tasks))) as parallel:
            results = parallel.map(
                _threept_outer_result,
                tasks,
            )
    else:
        results = _parallel.map(
            _threept_outer_result,
            tasks,
        )
    for index, result in results:
        matrices[index] = result
    real = (
        matrices[:, :, selected_components.index("real"), 0, 0]
        if "real" in selected_components
        else np.zeros((len(outer), len(c3_by_z)))
    )
    imag = (
        matrices[:, :, selected_components.index("imag"), 0, 0]
        if "imag" in selected_components
        else np.zeros((len(outer), len(c3_by_z)))
    )
    return {
        "values": real + 1j * imag,
        "matrices": matrices,
        "components": selected_components,
        "outer_samples": len(outer),
    }


__all__ = ["prepare_lanczos_data", "analyze_prepared_lanczos"]
