"""Lanczos/Krylov extraction for real two- and three-point correlators.

The implementation follows arXiv:2406.20009 and arXiv:2407.21777.  It keeps
the numerical recurrence separate from the correlator-stage HDF5 and artifact
contracts in :mod:`lamet_agent.stages.correlator.functions`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


def plan_tsep_tau_conversion(
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
    if requested_iterations is not None and (
        type(requested_iterations) is not int or requested_iterations < 1
    ):
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
    selected_tseps = [
        2 * chosen["t0"] + chosen["time_step"] * k
        for k in range(2 * iterations - 1)
    ]
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


def plan_twopt_grid(
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


class Ritz(NamedTuple):
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

    def filter_spurious(self, epsilon: float) -> "Ritz":
        """Apply a Cullum-Willoughby distance threshold."""
        keep = (self.cullum_willoughby_distance > epsilon) & (self.values < 1.0)
        return Ritz(
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
        g[:k, j + 1] = (
            a[1 : k + 1, j] - alpha * a[:k, j] - gamma * b[:k, j]
        ) / beta_next
        b[:k, j + 1] = (
            a[1 : k + 1, j] - alpha * a[:k, j] - beta * g[:k, j]
        ) / gamma_next
        matrix[j, j] = alpha
        matrix[j, j + 1] = beta_next
        matrix[j + 1, j] = gamma_next

    matrix[m - 1, m - 1] = a[1, m - 1]
    return matrix, a[1], b[1], g[1]


def _transfer_matrix_mpmath(
    c2: np.ndarray, precision: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the recurrence with ``precision`` decimal digits using mpmath."""
    import mpmath as mp

    k = len(c2)
    m = k // 2
    with mp.workdps(precision):
        zero = mp.mpf("0")
        a = [[zero for _ in range(m)] for _ in range(k)]
        b = [[zero for _ in range(m)] for _ in range(k)]
        g = [[zero for _ in range(m)] for _ in range(k)]
        matrix = [[zero for _ in range(m)] for _ in range(m)]
        c2_zero = mp.mpf(float(c2[0]))
        for t, value in enumerate(c2):
            a[t][0] = mp.mpf(float(value)) / c2_zero

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
            gamma_next = mp.sqrt(abs(product))
            beta_next = product / gamma_next
            k -= 2
            for t in range(k):
                a[t][j + 1] = (
                    a[t + 2][j]
                    - 2 * alpha * a[t + 1][j]
                    + alpha**2 * a[t][j]
                    + alpha * (beta * g[t][j] + gamma * b[t][j])
                    - (beta * g[t + 1][j] + gamma * b[t + 1][j])
                    + gamma * beta * a[t][j - 1]
                ) / product
                g[t][j + 1] = (
                    a[t + 1][j] - alpha * a[t][j] - gamma * b[t][j]
                ) / beta_next
                b[t][j + 1] = (
                    a[t + 1][j] - alpha * a[t][j] - beta * g[t][j]
                ) / gamma_next
            matrix[j][j] = alpha
            matrix[j][j + 1] = beta_next
            matrix[j + 1][j] = gamma_next
        matrix[m - 1][m - 1] = a[1][m - 1]
        return arrays(m)


def transfer_matrix(
    c2: np.ndarray, precision: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        _transfer_matrix_numpy(values)
        if precision == 0
        else _transfer_matrix_mpmath(values, precision)
    )


def ritz_spectrum(matrix: np.ndarray, epsilon_float: float = 1e-12) -> Ritz:
    """Compute real Ritz values and their Cullum-Willoughby distances."""
    m = len(matrix)
    if m == 0:
        empty = np.empty(0, dtype=float)
        return Ritz(empty, np.empty((0, 0)), np.empty((0, 0)), empty)
    if m == 1:
        value = np.asarray([matrix[0, 0]], dtype=float)
        return Ritz(value, np.ones((1, 1)), np.ones((1, 1)), np.abs(value))
    values = np.linalg.eigvals(matrix)
    real = (np.abs(np.angle(values)) <= epsilon_float) & (values != 0)
    values = values[real].real
    reduced = np.linalg.eigvals(matrix[1:, 1:])
    distances = (
        np.min(np.abs(values[:, None] - reduced[None, :]), axis=1)
        if len(values)
        else np.empty(0, dtype=float)
    )
    return Ritz(values, np.identity(len(values)), np.identity(len(values)), distances)


def ritz_hermitian(matrix: np.ndarray, a_cw: float = 10.0, b_cw: float = 1.0, epsilon_float: float = 1e-8) -> Ritz:
    """Keep the Hermitian Ritz subspace and apply the iteration-local CW cut."""
    m = len(matrix)
    if m == 0:
        empty = np.empty(0, dtype=float)
        return Ritz(empty, np.empty((0, 0)), np.empty((0, 0)), empty)
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
        return Ritz(values, right, inverse, np.empty(0, dtype=float))

    reduced = np.linalg.eigvals(matrix[1:, 1:])
    reduced = reduced[
        np.abs(reduced.imag) <= epsilon_float * np.abs(reduced)
    ].real
    if len(reduced):
        distances = np.min(np.abs(values[:, None] - reduced[None, :]), axis=1)
        epsilon_cw = (np.max(distances) - np.min(distances)) / (a_cw * len(values) + b_cw)
        keep = distances > epsilon_cw
        values = values[keep]
        right = right[:, keep]
        inverse = inverse[keep, :]
    return Ritz(values, right, inverse, np.empty(0, dtype=float))


def _filter_twopt_cw(results: list[list[Ritz]]) -> list[list[Ritz]]:
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


def analyze_twopt(
    c2_configurations: np.ndarray,
    n_bootstrap: int,
    *,
    seed: Any = None,
    precision: int = 0,
    max_iterations: int | None = None,
) -> list[list[Ritz]]:
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
    results: list[list[Ritz]] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, c2.shape[0], c2.shape[0])
        matrix, _alpha, _beta, _gamma = transfer_matrix(
            c2[indices].mean(axis=0)[: 2 * requested], precision=precision
        )
        results.append([ritz_spectrum(matrix[:m, :m]) for m in range(1, len(matrix) + 1)])
    return _filter_twopt_cw(results)


def krylov_polynomial(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
            right[j + 1, t] = (
                right[j, t - 1] - alpha[j] * right[j, t] - beta[j] * right[j - 1, t]
            ) / gamma[j + 1]
            left[j + 1, t] = (
                left[j, t - 1] - alpha[j] * left[j, t] - gamma[j] * left[j - 1, t]
            ) / beta[j + 1]
    return right, left


def ritz_rotator(ritz: Ritz, right_krylov: np.ndarray, left_krylov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct normalized right and left Ritz rotators."""
    norms = np.sqrt((ritz.inverse_vectors[:, 0].conj() / ritz.right_vectors[0, :]).real)
    right = np.einsum("k,ik,it->kt", norms, ritz.right_vectors, right_krylov)
    left = np.einsum("k,ki,it->kt", 1.0 / norms, ritz.inverse_vectors, left_krylov)
    return right, left


def analyze_threept(
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

        sink_matrix, sink_alpha, sink_beta, sink_gamma = transfer_matrix(c2_sink_mean, precision=precision)
        source_matrix, source_alpha, source_beta, source_gamma = transfer_matrix(c2_source_mean, precision=precision)
        sink_krylov = krylov_polynomial(sink_alpha, sink_beta, sink_gamma)
        source_krylov = krylov_polynomial(source_alpha, source_beta, source_gamma)
        usable = min(requested, len(sink_matrix), len(source_matrix))
        matrices: list[np.ndarray] = []
        for m in range(1, usable + 1):
            sink_ritz = ritz_hermitian(sink_matrix[:m, :m])
            source_ritz = ritz_hermitian(source_matrix[:m, :m])
            sink_left = ritz_rotator(
                sink_ritz, sink_krylov[0][:m, :m], sink_krylov[1][:m, :m]
            )[1]
            source_right = ritz_rotator(
                source_ritz, source_krylov[0][:m, :m], source_krylov[1][:m, :m]
            )[0]
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


def median_twopt_energies(
    results: list[list[Ritz]], *, max_states: int, time_step: int = 1
) -> np.ndarray:
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


def median_threept_matrix(
    results: list[list[np.ndarray]], *, iteration: int, max_states: int
) -> np.ndarray:
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
