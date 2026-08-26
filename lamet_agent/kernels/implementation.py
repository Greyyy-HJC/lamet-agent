from __future__ import annotations
import contextlib
import functools
from typing import Callable, Final
import numpy as np

HBAR_C_GEV_FM = 0.1973269804
GEV_FM = HBAR_C_GEV_FM
CF: Final[float] = 4.0 / 3.0
NF: Final[int] = 3
CA: Final[float] = 3.0
TF: Final[float] = 1.0 / 2.0


def beta(order: int = 0, Nf: int = 3) -> float:
    if order == 0:
        return 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * Nf
    if order == 1:
        return 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * Nf
    if order == 2:
        return (
            2857.0 / 54.0 * CA**3
            + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * Nf
            + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * Nf**2
        )
    raise NotImplementedError(f"beta coefficient at order={order} is not implemented.")


def alphas_nloop(mu: float, order: int = 0, Nf: int = 3) -> float:
    a_s_ref = 0.293 / (4.0 * np.pi)
    b0 = beta(0, Nf)
    temp = 1.0 + a_s_ref * b0 * np.log((mu / 2.0) ** 2)
    if order == 0:
        return a_s_ref * 4.0 * np.pi / temp
    if order == 1:
        b1 = beta(1, Nf)
        return a_s_ref * 4.0 * np.pi / (temp + a_s_ref * b1 / b0 * np.log(temp))
    if order == 2:
        b1 = beta(1, Nf)
        b2 = beta(2, Nf)
        correction = (
            temp
            + a_s_ref * b1 / b0 * np.log(temp)
            + a_s_ref**2 * (b2 / b0 * (1.0 - 1.0 / temp) + b1**2 / b0**2 * (np.log(temp) / temp + 1.0 / temp - 1.0))
        )
        return a_s_ref * 4.0 * np.pi / correction
    raise NotImplementedError(f"alpha_s at order={order} is not implemented.")


def ZMSbar(z_fm: np.ndarray | float, *, mu: float = 2.0, offset: float, order: int = 0, Nf: int = 3) -> np.ndarray:
    z_arr = np.asarray(z_fm, dtype=float)
    alphas = alphas_nloop(mu, order=order, Nf=Nf)
    log_term = np.log(mu**2 * (z_arr / GEV_FM) ** 2 * np.exp(2.0 * np.euler_gamma) / 4.0)
    return 1.0 + alphas * CF / (2.0 * np.pi) * (1.5 * log_term + offset)


def ZMSbar_pdf(z_fm: np.ndarray | float, mu: float = 2.0, order: int = 0, Nf: int = 3) -> np.ndarray:
    return ZMSbar(z_fm, mu=mu, offset=2.5, order=order, Nf=Nf)


def ZMSbar_da(z_fm: np.ndarray | float, mu: float = 2.0, order: int = 0, Nf: int = 3) -> np.ndarray:
    return ZMSbar(z_fm, mu=mu, offset=3.5, order=order, Nf=Nf)


def _sine_integral(value: float) -> float:
    try:
        from scipy.special import sici

        return float(sici(value)[0])
    except ModuleNotFoundError:
        pass
    if np.isclose(value, 0.0, atol=1e-14, rtol=0.0):
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    upper = abs(value)
    n_steps = max(256, int(128 * upper))
    if n_steps % 2:
        n_steps += 1
    grid = np.linspace(0.0, upper, n_steps + 1)
    integrand = np.ones_like(grid)
    integrand[1:] = np.sin(grid[1:]) / grid[1:]
    h = upper / n_steps
    integral = (
        h / 3.0 * (integrand[0] + integrand[-1] + 4.0 * np.sum(integrand[1:-1:2]) + 2.0 * np.sum(integrand[2:-2:2]))
    )
    return sign * float(integral)


def _atan_piece(ksi: float, eps: float) -> float:
    if ksi < 0.5 - eps:
        sqrt_term = np.sqrt(1.0 - 2.0 * ksi)
        piece = (3.0 * ksi - 1.0) / (ksi - 1.0 + eps)
        return piece * np.arctan(sqrt_term / (np.abs(ksi) + eps)) / (sqrt_term + eps)
    if ksi > 0.5 + eps:
        sqrt_term = np.sqrt(2.0 * ksi - 1.0)
        piece = (3.0 * ksi - 1.0) / (ksi - 1.0 + eps)
        return piece * np.arctanh(sqrt_term / (np.abs(ksi) + eps)) / (sqrt_term + eps)
    return (3.0 * ksi - 1.0) / (ksi - 1.0) / (np.abs(ksi) + eps)


def C_ratio(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    one_minus_ksi = 1.0 - ksi
    entry = 0.0
    if eps < ksi < 1.0 - eps:
        entry += (1.0 + ksi**2) / one_minus_ksi * log_scale + ksi - 1.0
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    signed_logs = np.sign(ksi) * np.log(np.abs(ksi) + eps) + np.sign(one_minus_ksi) * np.log(
        np.abs(one_minus_ksi) + eps
    )
    entry += (1.0 + ksi**2) / sign_safe_denominator * signed_logs
    entry += np.sign(ksi) + _atan_piece(ksi, eps) - 1.5 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def C_ratio_perp(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    one_minus_ksi = 1.0 - ksi
    entry = 0.0
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * ksi / one_minus_ksi * log_scale
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    signed_logs = np.sign(ksi) * np.log(np.abs(ksi) + eps) + np.sign(one_minus_ksi) * np.log(
        np.abs(one_minus_ksi) + eps
    )
    entry += 2.0 * ksi / sign_safe_denominator * signed_logs
    entry += _atan_piece(ksi, eps) - 1.0 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def C_msbar(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    return C_ratio(ksi, log_scale, eps) + 0.5 / (np.abs(1.0 - ksi) + eps)


def C_msbar_plus(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    entry = C_ratio(ksi, log_scale, eps)
    if 0.0 <= ksi <= 2.0:
        entry += 0.5 / (np.abs(1.0 - ksi) + eps)
    return entry


def C_msbar_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    entry = C_msbar(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_msbar_gz_plus(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    entry = C_msbar_plus(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_ratio_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    entry = C_ratio(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_hybrid(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    wilson_scale = np.abs(y) * zspz
    delta = 0.5 * (
        1.0 / (np.abs(one_minus_ksi) + eps)
        - 2.0 * _sine_integral(one_minus_ksi * wilson_scale) / (np.pi * sign_safe_denominator)
    )
    return C_ratio(ksi, log_scale, eps) + delta


def C_hybrid_gz(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    return C_ratio_gz(ksi, log_scale, eps) + (C_hybrid(ksi, log_scale, y, zspz, eps) - C_ratio(ksi, log_scale, eps))


DensityFn = Callable[[float, float], float]
_PROGRESS_SILENCED = False


@contextlib.contextmanager
def _quiet_progress():
    global _PROGRESS_SILENCED
    previous = _PROGRESS_SILENCED
    _PROGRESS_SILENCED = True
    try:
        yield
    finally:
        _PROGRESS_SILENCED = previous


def _progress(iterable, *, desc: str, leave: bool = True):
    if _PROGRESS_SILENCED:
        return iterable
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)


class _NoOpBar:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _progress_bar(*, total: int, desc: str):
    if _PROGRESS_SILENCED:
        return _NoOpBar()
    try:
        from tqdm import tqdm
    except Exception:
        return _NoOpBar()
    return tqdm(total=total, desc=desc)


def _lo_interp_matrix(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(y_grid)
    ys = y_grid[order]
    lo_sorted = np.column_stack([np.interp(x_grid, ys, unit, left=0.0, right=0.0) for unit in np.eye(len(y_grid))])
    lo = np.empty_like(lo_sorted)
    lo[:, order] = lo_sorted
    return lo


def build_matching_matrix(
    lc_x_ls: np.ndarray,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    *,
    density: DensityFn,
    color_factor: float = CF,
    diagonal_extra: Callable[[float], float] | None = None,
) -> np.ndarray:
    x_grid = np.asarray(lc_x_ls, dtype=float)
    y_grid = np.asarray(x_grid if quasi_y_ls is None else quasi_y_ls, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("`lc_x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`quasi_y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`quasi_y_ls` must avoid values too close to 0.")
    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))
    if dy <= eps:
        raise ValueError("`quasi_y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`quasi_y_ls` must be uniformly spaced.")
    alpha_s = _alpha_s(float(mu))
    nx, ny = (len(x_grid), len(y_grid))
    nlo_matrix = np.zeros((nx, ny))
    identity = _lo_interp_matrix(x_grid, y_grid)
    offsets = np.abs(x_grid[:, None] - y_grid[None, :])
    diag_rows = offsets.argmin(axis=0)
    has_diag = offsets[diag_rows, np.arange(ny)] <= eps * np.maximum(np.abs(y_grid), 1.0)
    for idx, x_val in enumerate(_progress(x_grid, desc="matching kernel")):
        for idy, y_val in enumerate(y_grid):
            if np.abs(x_val - y_val) <= eps * np.abs(y_val):
                continue
            nlo_matrix[idx, idy] = density(x_val, y_val)
    column_totals = _column_plus_totals(y_grid, density, eps)
    for idy, diag_row in enumerate(diag_rows):
        if not has_diag[idy]:
            continue
        nlo_matrix[int(diag_row), idy] -= column_totals[idy]
        if diagonal_extra is not None:
            nlo_matrix[int(diag_row), idy] += diagonal_extra(float(y_grid[idy])) / dy
    return identity - alpha_s * color_factor / (2.0 * np.pi) * nlo_matrix * dy


CoeffFn = Callable[[float, float, float], float]


def _pdf_log_scale(y: float, momentum_gev: float, mu: float) -> float:
    return float(np.log(4.0 * y**2 * momentum_gev**2 / mu**2))


def _pdf_density(coeff: CoeffFn, momentum_gev: float, mu: float) -> DensityFn:

    def density(x: float, y: float) -> float:
        return coeff(x / y, _pdf_log_scale(y, momentum_gev, mu), y) / np.abs(y)

    return density


_TAIL_KSI_CUTOFF: Final = 10000.0


def _asymptotic_tail(func: Callable[[float], float], edge: float) -> float:
    return float(func(edge) * edge**2 / abs(edge))


def _integrate(func: Callable[[float], float], lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    if not np.isfinite(hi):
        edge = max(lo, _TAIL_KSI_CUTOFF)
        return _integrate(func, lo, edge) + _asymptotic_tail(func, edge)
    if not np.isfinite(lo):
        edge = min(hi, -_TAIL_KSI_CUTOFF)
        return _asymptotic_tail(func, edge) + _integrate(func, edge, hi)
    try:
        import warnings
        from scipy import integrate as _si

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", _si.IntegrationWarning)
            value, _ = _si.quad(func, lo, hi, limit=400)
        return float(value)
    except ModuleNotFoundError:
        pass

    def simpson(f: Callable[[float], float], a: float, b: float, n: int = 2000) -> float:
        grid = np.linspace(a, b, n + 1)
        vals = np.array([f(t) for t in grid])
        h = (b - a) / n
        return float(h / 3.0 * (vals[0] + vals[-1] + 4.0 * vals[1:-1:2].sum() + 2.0 * vals[2:-2:2].sum()))

    return simpson(func, lo, hi)


def _uncovered_ksi_intervals(x_val: float, y_grid: np.ndarray, dy: float, eps: float) -> list[tuple[float, float]]:
    covered: list[tuple[float, float]] = []
    for sign in (1.0, -1.0):
        ys = y_grid[y_grid * sign > eps]
        if ys.size:
            lo, hi = (x_val / (sign * np.max(sign * ys)), x_val / (sign * np.min(sign * ys)))
            covered.append((min(lo, hi), max(lo, hi)))
    covered.sort()
    gaps: list[tuple[float, float]] = []
    edge = -np.inf
    for lo, hi in covered:
        gaps.append((edge, lo))
        edge = hi
    gaps.append((edge, np.inf))
    guard = 0.5 * dy / np.abs(x_val)
    clipped = []
    for lo, hi in gaps:
        for piece_lo, piece_hi in ((lo, min(hi, 1.0 - guard)), (max(lo, 1.0 + guard), hi)):
            if piece_hi > piece_lo:
                clipped.append((piece_lo, piece_hi))
    return clipped


def _build_pdf_matrix(
    x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    y_ls: np.ndarray | None,
    eps: float,
    *,
    coeff: CoeffFn,
    plus_coeff: CoeffFn | None = None,
    color_factor: float = CF,
    diagonal_extra: Callable[[float], float] | None = None,
) -> np.ndarray:
    x_grid = np.asarray(x_ls, dtype=float)
    y_grid = np.asarray(x_grid if y_ls is None else y_ls, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")
    if y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("`quasi_y_ls` must be a 1D array with at least 2 points.")
    if np.any(np.abs(y_grid) <= eps):
        raise ValueError("`quasi_y_ls` must avoid values too close to 0 to keep ksi=x/y finite.")
    y_step = np.diff(y_grid)
    dy = float(np.abs(y_step[0]))
    if dy <= eps:
        raise ValueError("`quasi_y_ls` spacing must be non-zero.")
    if not np.allclose(y_step, y_step[0], rtol=0.0, atol=eps):
        raise ValueError("`quasi_y_ls` must be uniformly spaced.")
    if np.any(x_grid < np.min(y_grid) - 0.5 * dy) or np.any(x_grid > np.max(y_grid) + 0.5 * dy):
        raise ValueError("every x must lie inside the y grid to place its delta term.")
    if plus_coeff is None:
        plus_coeff = coeff
    alpha_s = _alpha_s(float(mu))
    nx, ny = (len(x_grid), len(y_grid))
    identity = _lo_interp_matrix(x_grid, y_grid)
    nlo_matrix = np.zeros((nx, ny))
    with _progress_bar(total=2 * nx, desc="matching kernel") as bar:
        for idx, x_val in enumerate(x_grid):
            for idy, y_val in enumerate(y_grid):
                ksi = x_val / y_val
                if np.abs(1.0 - ksi) <= eps:
                    continue
                log_scale = _pdf_log_scale(y_val, momentum_gev, mu)
                nlo_matrix[idx, idy] = coeff(ksi, log_scale, y_val) / np.abs(y_val)
            bar.update(1)
        ascending = y_grid[1] > y_grid[0]
        y_sorted = y_grid if ascending else y_grid[::-1]
        for idx, x_val in enumerate(x_grid):
            if np.abs(x_val) <= eps:
                bar.update(1)
                continue
            pos = int(np.searchsorted(y_sorted, x_val))
            pos = min(max(pos, 1), ny - 1)
            w_hi = np.clip((x_val - y_sorted[pos - 1]) / (y_sorted[pos] - y_sorted[pos - 1]), 0.0, 1.0)
            cols_weights = (
                (pos - 1 if ascending else ny - pos, 1.0 - w_hi),
                (pos if ascending else ny - 1 - pos, w_hi),
            )
            log_scale = _pdf_log_scale(x_val, momentum_gev, mu)
            subtraction = 0.0
            for y_val in y_grid:
                ksi = x_val / y_val
                if np.abs(1.0 - ksi) <= eps:
                    continue
                subtraction += plus_coeff(ksi, log_scale, x_val) * np.abs(x_val) * dy / y_val**2
            for lo, hi in _uncovered_ksi_intervals(x_val, y_grid, dy, eps):
                subtraction += _integrate(lambda ksi: plus_coeff(ksi, log_scale, x_val), lo, hi)
            delta_entry = -subtraction
            if diagonal_extra is not None:
                delta_entry += diagonal_extra(log_scale)
            for col, weight in cols_weights:
                nlo_matrix[idx, int(col)] += weight * delta_entry / dy
            bar.update(1)
    return identity - alpha_s * color_factor / (2.0 * np.pi) * nlo_matrix * dy


def C_ratio_gi(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    splitting = (1.0 + ksi**2) / sign_safe_denominator
    if eps < ksi < 1.0 - eps:
        lamet_log = log_scale - np.log(4.0)
        entry = splitting * (lamet_log + np.log(4.0 * ksi * one_minus_ksi + eps) - 1.0) + 1.0
    else:
        log_ratio = np.log((np.abs(ksi) + eps) / (np.abs(ksi - 1.0) + eps))
        entry = np.sign(ksi) * (splitting * log_ratio + 1.0)
    entry += 1.5 / (np.abs(one_minus_ksi) + eps)
    return float(entry)


def _hybrid_gi_delta(ksi: float, y: float, zspz: float, eps: float, strength: float) -> float:
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    wilson_scale = np.abs(y) * zspz
    return strength * (
        -1.0 / (np.abs(one_minus_ksi) + eps)
        + 2.0 * _sine_integral(one_minus_ksi * wilson_scale) / (np.pi * sign_safe_denominator)
    )


def C_hybrid_gi(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    return C_ratio_gi(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=1.5)


def C_ratio_gi_gz(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    entry = C_ratio_gi(ksi, log_scale, eps)
    if eps < ksi < 1.0 - eps:
        entry += 2.0 * (1.0 - ksi)
    return entry


def C_hybrid_gi_gz(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    return C_ratio_gi_gz(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=1.5)


def C_ratio_gi_perp(ksi: float, log_scale: float, eps: float = 1e-12) -> float:
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    splitting = 2.0 * ksi / sign_safe_denominator
    log_ratio = np.log((np.abs(ksi) + eps) / (np.abs(ksi - 1.0) + eps))
    if ksi > 1.0 + eps:
        entry = splitting * log_ratio - 2.0 / sign_safe_denominator
    elif eps < ksi < 1.0 - eps:
        entry = splitting * (log_scale + np.log(ksi * one_minus_ksi + eps)) + 2.0
    elif ksi < -eps:
        entry = -splitting * log_ratio + 2.0 / sign_safe_denominator
    else:
        entry = 0.0
    return float(entry)


def C_hybrid_gi_perp(ksi: float, log_scale: float, y: float, zspz: float, eps: float = 1e-12) -> float:
    return C_ratio_gi_perp(ksi, log_scale, eps) + _hybrid_gi_delta(ksi, y, zspz, eps, strength=2.0)


def _da_log(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    return float(np.log(4.0 * momentum_gev**2 * value**2 / mu**2 + eps))


def V_qq_t(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    l_x = _da_log(x, momentum_gev, mu, eps)
    l_xbar = _da_log(1.0 - x, momentum_gev, mu, eps)
    l_xy = _da_log(x - y, momentum_gev, mu, eps)
    return (
        np.abs(x) / (y * (y - x)) * (l_x - 1.0)
        + np.abs(1.0 - x) / ((1.0 - y) * (x - y)) * (l_xbar - 1.0)
        + (x + y - 2.0 * x * y) / (np.abs(x - y) * y * (1.0 - y)) * (l_xy - 1.0)
    )


def V_qq_h(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    l_x = _da_log(x, momentum_gev, mu, eps)
    l_xbar = _da_log(1.0 - x, momentum_gev, mu, eps)
    l_xy = _da_log(x - y, momentum_gev, mu, eps)
    brace = (
        np.abs(x) / y * (l_x - 1.0)
        + np.abs(1.0 - x) / (1.0 - y) * (l_xbar - 1.0)
        + np.abs(x - y) / (y * (y - 1.0)) * (l_xy - 1.0)
    )
    return float(brace + V_qq_t(x, y, momentum_gev, mu, eps))


def V_qq_p(x: float, y: float, momentum_gev: float, mu: float, eps: float = 1e-12) -> float:
    extra = np.abs(x) / y + np.abs(1.0 - x) / (1.0 - y) + np.abs(x - y) / ((y - 1.0) * y)
    return float(V_qq_h(x, y, momentum_gev, mu, eps) + 2.0 * extra)


def V_qq_rto(x: float, y: float) -> float:
    return float(1.5 / np.abs(x - y))


def _da_matrix(
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    *,
    coefficient: Callable[[float, float, float, float, float], float],
    wilson_line: Callable[[float, float], float],
) -> np.ndarray:

    def density(x: float, y: float) -> float:
        if not eps < y < 1.0 - eps:
            return 0.0
        return 0.5 * coefficient(x, y, momentum_gev, mu, eps) + wilson_line(x, y)

    return build_matching_matrix(lc_x_ls, mu, quasi_y_ls, eps, density=density)


def _da_wilson_line(scheme: str, zspz: float | None, eps: float) -> Callable[[float, float], float]:
    if scheme == "ratio":
        return V_qq_rto
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)

    def hybrid(x: float, y: float) -> float:
        return V_qq_rto(x, y) + _hybrid_gi_delta(x / y, y, z, eps, strength=1.5) / np.abs(y)

    return hybrid


_NM_RENORMALON: Final[dict[int, float]] = {3: 0.5749687262865643, 4: 0.5522713118193284, 5: 0.5235323457364502}


def _renormalon_params(nf: int) -> tuple[float, float, float, float]:
    beta0 = 11.0 - 2.0 * nf / 3.0
    beta1 = 102.0 - 38.0 * nf / 3.0
    beta2 = 2857.0 / 2.0 - 5033.0 * nf / 18.0 + 325.0 * nf**2 / 54.0
    beta3 = 29243.0 - 6946.3 * nf + 405.089 * nf**2 + 1.49931 * nf**3
    b = beta1 / (2.0 * beta0**2)
    c1 = 1.0 / (4.0 * b * beta0**3) * (beta1**2 / beta0 - beta2)
    c2 = (
        beta1**4
        + 4.0 * beta0**3 * beta1 * beta2
        - 2.0 * beta0 * beta1**2 * beta2
        + beta0**2 * (-2.0 * beta1**3 + beta2**2)
        - 2.0 * beta0**4 * beta3
    ) / (32.0 * (b - 1.0) * b * beta0**8)
    return (beta0, b, c1, c2)


def rnasym(n: int, z: float, mu: float, nf: int = 3) -> float:
    from math import gamma

    beta0, b, c1, c2 = _renormalon_params(nf)
    tail = 1.0 + b * c1 / (n + b) + b * (b - 1.0) * c2 / ((n + b) * (n + b - 1.0))
    return float(
        _NM_RENORMALON[nf] * abs(z * mu) * (beta0 / (2.0 * np.pi)) ** n * gamma(n + 1.0 + b) / gamma(1.0 + b) * tail
    )


def dPVasym(z: float, mu: float, nf: int, alphas: float) -> float:
    try:
        import mpmath
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The LRR matching kernels need mpmath for the exponential-integral E_nu; install the 'analysis' extra (pip install -e '.[analysis]')."
        ) from exc
    beta0, b, c1, c2 = _renormalon_params(nf)
    w = -2.0 * np.pi / (alphas * beta0)
    borel = mpmath.expint(1.0 + b, w) + c1 * mpmath.expint(b, w) + c2 * mpmath.expint(-1.0 + b, w)
    value = _NM_RENORMALON[nf] * abs(z * mu) * mpmath.e**w * (-2.0 * np.pi / beta0) * mpmath.re(borel)
    return float(value)


def C_z_lrr(ksi: float, y: float, momentum_gev: float, zspz: float, eps_m: float = 0.005, eps: float = 1e-12) -> float:
    M = eps_m
    pz = abs(y) * momentum_gev
    zs = zspz / momentum_gev
    emz = np.exp(-M * zs)
    om = ksi - 1.0
    if abs(om) <= eps:
        return float(emz * pz * (1.0 + M * zs + M**2 * zs**2) / (M**2 * np.pi))
    phi = pz * zs * (1.0 - ksi)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    denom = (M**2 + pz**2 * om**2) ** 2
    term1 = -(emz * zs * sin_phi) / (np.pi * om)
    term2 = (
        emz
        * pz
        / (np.pi * denom)
        * (
            (M**2 - pz**2 * om**2 + M**3 * zs + M * pz**2 * om**2 * zs) * cos_phi
            + pz * om * (2.0 * M + M**2 * zs + pz**2 * om**2 * zs) * sin_phi
        )
    )
    return float(term1 + term2)


def _column_plus_totals(y_grid: np.ndarray, density: DensityFn, eps: float) -> np.ndarray:
    y = np.asarray(y_grid, dtype=float)
    totals = np.zeros(y.size)
    for idy, y_val in enumerate(y):
        column = 0.0
        for y_row in y:
            if np.abs(y_row - y_val) <= eps * np.abs(y_val):
                continue
            column += density(float(y_row), float(y_val))
        totals[idy] = column
    return totals


def _plus_prescription_matrix(x_grid: np.ndarray, y_grid: np.ndarray, density: DensityFn, eps: float) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    dy = float(np.abs(np.diff(y)[0]))
    matrix = np.zeros((len(x), len(y)))
    offsets = np.abs(x[:, None] - y[None, :])
    diag_rows = offsets.argmin(axis=0)
    has_diag = offsets[diag_rows, np.arange(len(y))] <= eps * np.maximum(np.abs(y), 1.0)
    for idx, x_val in enumerate(_progress(x, desc="matching LRR kernel")):
        for idy, y_val in enumerate(y):
            if np.abs(x_val - y_val) <= eps * np.abs(y_val):
                continue
            matrix[idx, idy] = density(x_val, y_val)
    column_totals = _column_plus_totals(y, density, eps)
    for idy, diag_row in enumerate(diag_rows):
        if has_diag[idy]:
            matrix[int(diag_row), idy] -= column_totals[idy]
    return matrix * dy


def _lrr_improve(
    fixed_order_matrix: np.ndarray,
    x_ls: np.ndarray,
    y_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    zspz: float,
    eps: float,
    *,
    nf: int = 3,
    eps_m: float = 0.005,
    restrict_unit_interval: bool = False,
) -> np.ndarray:
    from scipy.linalg import expm

    def density_cz(x: float, y: float) -> float:
        if restrict_unit_interval and (not eps < y < 1.0 - eps):
            return 0.0
        return C_z_lrr(x / y, y, momentum_gev, zspz, eps_m, eps) / np.abs(y)

    m_cz_sum = _plus_prescription_matrix(x_ls, y_ls, density_cz, eps)
    m_cz_exp = (
        m_cz_sum
        if x_ls.shape == y_ls.shape and np.allclose(x_ls, y_ls, rtol=0.0, atol=1e-12)
        else _plus_prescription_matrix(y_ls, y_ls, density_cz, eps)
    )
    alpha_s = _alpha_s(float(mu))
    rsum_pv = dPVasym(1.0, mu, nf, alpha_s)
    r0 = rnasym(0, 1.0, mu, nf) * alpha_s
    return (fixed_order_matrix + r0 * m_cz_sum) @ expm(-m_cz_exp * rsum_pv)


def _lrr_from_fixed_order(
    fixed_order_builder: Callable[..., np.ndarray],
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    zspz: float | None,
    *,
    restrict_unit_interval: bool = False,
) -> np.ndarray:
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid (LRR) matching kernel.")
    x = np.asarray(lc_x_ls, dtype=float)
    y = x if quasi_y_ls is None else np.asarray(quasi_y_ls, dtype=float)
    m_fix = fixed_order_builder(x, momentum_gev=momentum_gev, mu=mu, quasi_y_ls=y, eps=eps, zspz=zspz)
    return _lrr_improve(m_fix, x, y, momentum_gev, mu, float(zspz), eps, restrict_unit_interval=restrict_unit_interval)


_ALPHAS_MZ: Final = 0.1179
_M_Z: Final = 91.1876
_M_B: Final = 4.18
_M_C: Final = 1.27
_RUNNING_STEPS: Final = 100
_BETA_ORDER: Final = 2
_SPLIT_CF: Final = 4.0 / 3.0
_SPLIT_CA: Final = 3.0
_SPLIT_NF: Final = 4.0


def _beta_coefficient(index: int, nf: float) -> float:
    if index == 0:
        return (33.0 - 2.0 * nf) / (12.0 * np.pi)
    if index == 1:
        return (153.0 - 19.0 * nf) / (24.0 * np.pi**2)
    if index == 2:
        return (2857.0 - 5033.0 / 9.0 * nf + 325.0 / 27.0 * nf**2) / (128.0 * np.pi**3)
    if index == 3:
        return (29243.0 - 6946.3 * nf + 405.089 * nf**2 + 1.49931 * nf**3) / (4.0 * np.pi) ** 4
    raise ValueError(f"beta coefficient index {index} is not tabulated.")


def _beta(nf: float, alpha: float, order: int) -> float:
    return -sum((_beta_coefficient(i - 1, nf) * alpha ** (i + 1) for i in range(1, order + 1)))


def _run_alpha(nf: float, mu_from: float, mu_to: float, alpha_from: float, order: int) -> float:
    step = (np.log(mu_to**2) - np.log(mu_from**2)) / _RUNNING_STEPS
    alpha = float(alpha_from)
    for _ in range(_RUNNING_STEPS):
        k1 = _beta(nf, alpha, order)
        k2 = _beta(nf, alpha + step * k1 / 2.0, order)
        k3 = _beta(nf, alpha + step * k2 / 2.0, order)
        k4 = _beta(nf, alpha + step * k3, order)
        alpha = alpha + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return alpha


@functools.lru_cache(maxsize=4096)
def _alpha_s(mu: float) -> float:
    if mu <= 0.0:
        raise ValueError("alpha_s needs a positive scale.")
    if mu >= _M_B:
        return _run_alpha(5.0, _M_Z, mu, _ALPHAS_MZ, _BETA_ORDER)
    alpha_mb = _run_alpha(5.0, _M_Z, _M_B, _ALPHAS_MZ, _BETA_ORDER)
    if mu >= _M_C:
        return _run_alpha(4.0, _M_B, mu, alpha_mb, _BETA_ORDER)
    alpha_mc = _run_alpha(4.0, _M_B, _M_C, alpha_mb, _BETA_ORDER)
    return _run_alpha(3.0, _M_C, mu, alpha_mc, _BETA_ORDER)


def _dilog(z: np.ndarray) -> np.ndarray:
    from scipy.special import spence

    return spence(1.0 - np.asarray(z, dtype=float))


def _p_qq_lo(nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    inside = (nu >= 0.0) & (nu < 1.0)
    safe = np.where(inside, nu, 0.0)
    value = 2.0 * _SPLIT_CF * (1.0 + safe**2) / np.where(inside, 1.0 - safe, 1.0)
    return np.where(inside, value, 0.0)


def _p_nlo_full_unpolarized(nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    inside = (nu >= 0.0) & (nu < 1.0)
    v = np.where(inside, np.clip(nu, 1e-300, None), 0.5)
    ln, ln1m, ln1p = (np.log(v), np.log1p(-v), np.log1p(v))
    z2 = np.pi**2 / 6.0
    a = (1.0 + v**2) / (1.0 - v)
    b = (1.0 + v**2) / (1.0 + v)
    cf, ca, nf = (_SPLIT_CF, _SPLIT_CA, _SPLIT_NF)
    value = (
        4.0
        * ca
        * cf
        * (
            a * (67.0 / 18.0 - z2 + 11.0 / 6.0 * ln + ln**2 / 2.0)
            + b * (z2 + 2.0 * (ln * ln1p + _dilog(-v)) - ln**2 / 2.0)
            + 14.0 / 3.0 * (1.0 - v)
        )
        - 4.0 * cf * nf * (a * (5.0 / 9.0 + 1.0 / 3.0 * ln) + 2.0 / 3.0 * (1.0 - v))
        + 4.0
        * cf**2
        * (
            2.0 * a * (-ln * ln1m - _dilog(v) - 0.75 * ln + _dilog(v))
            - 2.0 * b * (z2 + 2.0 * (ln * ln1p + _dilog(-v)) - ln**2 / 2.0)
            - (1.0 - v) * (1.0 - 1.5 * ln)
            - ln
            - (1.0 + v) * ln**2 / 2.0
        )
    )
    return np.where(inside, value, 0.0)


def _c_parity_term(nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    inside = (nu >= 0.0) & (nu < 1.0)
    v = np.where(inside, np.clip(nu, 1e-300, None), 0.5)
    ln, ln1p = (np.log(v), np.log1p(v))
    z2 = np.pi**2 / 6.0
    b = (1.0 + v**2) / (1.0 + v)
    cf, ca = (_SPLIT_CF, _SPLIT_CA)
    value = (
        16.0
        * cf
        * (cf - ca / 2.0)
        * (b * (z2 + 2.0 * (ln * ln1p + _dilog(-v)) - ln**2 / 2.0) - 2.0 * (1.0 - v) - (1.0 + v) * ln)
    )
    return np.where(inside, value, 0.0)


def _p_nlo_valence(nu: np.ndarray) -> np.ndarray:
    return _p_nlo_full_unpolarized(nu) + _c_parity_term(nu)


def _p_nlo_full_helicity(nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    inside = (nu >= 0.0) & (nu < 1.0)
    v = np.where(inside, np.clip(nu, 1e-300, None), 0.5)
    ln = np.log(v)
    extra = 4.0 * _SPLIT_CF * _SPLIT_NF * (-(1.0 - 3.0 * v) * ln + 1.0 - v - 2.0 * (1.0 + v) * ln**2 / 2.0)
    return _p_nlo_valence(nu) + np.where(inside, extra, 0.0)


def _p_nlo_transversity(nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    inside = (nu >= 0.0) & (nu < 1.0)
    v = np.where(inside, np.clip(nu, 1e-300, None), 0.5)
    ln, ln1m, ln1p = (np.log(v), np.log1p(-v), np.log1p(v))
    cf, ca, nf = (_SPLIT_CF, _SPLIT_CA, _SPLIT_NF)
    q = 4.0 * v / (1.0 - v)
    value = (
        ca * cf * (-2.0 * (1.0 - v) + q * (ln**2 + 11.0 / 3.0 * ln + 67.0 / 9.0 - np.pi**2 / 3.0))
        - cf * (nf / 2.0) * (4.0 / 3.0 * q * (ln + 5.0 / 3.0))
        + cf**2 * (4.0 * (1.0 - v) - q * (3.0 * ln + 4.0 * ln * ln1m))
        + 4.0
        * (cf**2 - ca * cf / 2.0)
        * (-(1.0 - v) + 4.0 * v / (1.0 + v) * (ln**2 / 2.0 - np.pi**2 / 6.0 - 2.0 * _dilog(-v) - 2.0 * ln * ln1p))
    )
    return np.where(inside, value, 0.0)


_ZPSI_REF_GEV: Final = 2.0


def _zpsi_msbar_ratio(mu: float, mu_ref: float = _ZPSI_REF_GEV, nf: float = 3.0) -> float:
    b0 = _beta_coefficient(0, nf)
    b1 = _beta_coefficient(1, nf)
    alpha, alpha_ref = (_alpha_s(float(mu)), _alpha_s(float(mu_ref)))
    ratio = alpha / (b0 + b1 * alpha) / (alpha_ref / (b0 + b1 * alpha_ref))
    return float(ratio ** (1.0 / (3.0 * np.pi * b0)))


def _dglap_evolution_matrices(
    x_grid: np.ndarray, p_nlo: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_grid, dtype=float)
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = x[:, None] / np.where(np.abs(x[None, :]) > 0, x[None, :], np.nan)
        weight = 1.0 / np.abs(np.where(np.abs(x[None, :]) > 0, x[None, :], np.nan))
    lo = np.nan_to_num(_p_qq_lo(ratio) * weight)
    nlo = np.nan_to_num(p_nlo(ratio) * weight)
    return (dx * (lo - np.diag(lo.sum(axis=0))), dx * (nlo - np.diag(nlo.sum(axis=0))))


def _evolution_operator(
    mu_initial: float, mu_final: float, evo_lo: np.ndarray, evo_nlo: np.ndarray, steps: int
) -> np.ndarray:
    from scipy.linalg import expm

    t0, t1 = (np.log(mu_initial**2), np.log(mu_final**2))
    dt = (t1 - t0) / steps
    operator = np.eye(evo_lo.shape[0])
    for index in range(steps):
        mu_mid = float(np.exp((t0 + dt * (index + 0.5)) / 2.0))
        a = _alpha_s(mu_mid) / (4.0 * np.pi)
        operator = operator @ expm((a * evo_lo + a**2 * evo_nlo) * dt)
    return operator


def _rgr_from_fixed_order(
    fixed_order_builder: Callable[..., np.ndarray],
    p_nlo: Callable[[np.ndarray], np.ndarray],
    lc_x_ls: np.ndarray,
    momentum_gev: float,
    mu: float,
    quasi_y_ls: np.ndarray | None,
    eps: float,
    zspz: float | None,
    *,
    needs_zspz: bool,
    takes_zspz: bool,
    needs_zpsi: bool,
    kappa: float = 1.0,
    mu_min: float = 0.6,
    steps: int = 20,
) -> np.ndarray:
    if needs_zspz and zspz is None:
        raise ValueError("`zspz` is required for the hybrid (RGR) matching kernel.")
    x = np.asarray(lc_x_ls, dtype=float)
    y = x if quasi_y_ls is None else np.asarray(quasi_y_ls, dtype=float)
    evo_lo, evo_nlo = _dglap_evolution_matrices(x, p_nlo)
    matrix = np.zeros((x.size, y.size), dtype=float)
    with _quiet_progress():
        for index, x_value in enumerate(_progress(range(x.size), desc="RGR matching kernel")):
            mu0 = 2.0 * kappa * float(x[x_value]) * float(momentum_gev)
            if not np.isfinite(mu0) or mu0 < mu_min:
                continue
            extra = {"zspz": zspz} if takes_zspz else {}
            fixed = fixed_order_builder(x, momentum_gev=momentum_gev, mu=mu0, quasi_y_ls=y, eps=eps, **extra)
            evolution = _evolution_operator(mu0, mu, evo_lo, evo_nlo, steps)
            row = (evolution @ fixed)[x_value, :]
            matrix[x_value, :] = row * _zpsi_msbar_ratio(mu0) if needs_zpsi else row
    return matrix
