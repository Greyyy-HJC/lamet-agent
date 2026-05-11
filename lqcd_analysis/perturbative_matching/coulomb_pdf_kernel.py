"""CG PDF matching kernels in the MSbar scheme.

References use arXiv identifiers in docstrings where applicable.
"""
import numpy as np

from .coulomb_tmd_kernel import _alpha_s_1, C_F

def _validate_uniform_y_grid(y_ls_mat: np.ndarray, eps: float = 1e-12) -> float:
    """Validate the integration grid and return ``dy``."""
    if y_ls_mat.ndim != 1 or y_ls_mat.size < 2:
        raise ValueError("`y_ls` must be a 1D array with at least 2 points.")

    if np.any(np.abs(y_ls_mat) <= eps):
        raise ValueError("`y_ls` must avoid values too close to 0 to keep xi=x/y finite.")

    y_diff = np.diff(y_ls_mat)
    dy = float(np.abs(y_diff[0]))
    if dy <= eps:
        raise ValueError("`y_ls` spacing must be non-zero.")

    if not np.allclose(y_diff, y_diff[0], rtol=0.0, atol=eps):
        raise ValueError("`y_ls` must be uniformly spaced.")

    return dy


def _identity_kernel(x_ls_mat: np.ndarray, y_ls_mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Discrete LO identity on the ``(x, y)`` grid."""
    identity = np.zeros((len(x_ls_mat), len(y_ls_mat)))

    for idx, x_val in enumerate(x_ls_mat):
        matches = np.where(np.isclose(y_ls_mat, x_val, atol=eps, rtol=0.0))[0]
        if matches.size > 0:
            identity[idx, matches[0]] = 1.0

    return identity


def _closest_row_indices(x_ls_mat: np.ndarray, y_ls_mat: np.ndarray) -> np.ndarray:
    """Rows where discrete ``delta(1 - xi)`` terms live."""
    return np.abs(x_ls_mat[:, None] - y_ls_mat[None, :]).argmin(axis=0)


def _apply_plus_prescription(matrix: np.ndarray, diag_rows: np.ndarray) -> None:
    """Apply the discrete plus prescription column by column."""
    for idy in range(matrix.shape[1]):
        diag_row = int(diag_rows[idy])
        matrix[diag_row, idy] -= np.sum(matrix[:, idy])


def _delta_column_matrix(
    diag_rows: np.ndarray,
    coeff_per_y: np.ndarray,
    nx: int,
    ny: int,
    dy: float,
) -> np.ndarray:
    """Discrete ``delta(1 - xi)`` matrix from per-column coefficients."""
    matrix = np.zeros((nx, ny))

    for idy in range(ny):
        matrix[int(diag_rows[idy]), idy] = coeff_per_y[idy] / dy

    return matrix


def _is_regular_xi(xi: float, eps: float = 1e-12) -> bool:
    """Return whether ``xi`` is away from the ``xi = 1`` pole."""
    return bool(np.abs(1.0 - xi) > eps)


def _is_unit_interval_xi(xi: float, eps: float = 1e-12) -> bool:
    """Return whether ``xi`` is inside the open ``[0, 1]`` support."""
    return bool(eps < xi < 1.0 - eps)


def _signed_log_sum(xi: float, eps: float = 1e-12) -> float:
    """Signed logarithm combination appearing in the full-domain pieces."""
    one_minus_xi = 1.0 - xi
    xi_log = np.sign(xi) * np.log(np.abs(xi) + eps)
    one_minus_xi_log = np.sign(one_minus_xi) * np.log(np.abs(one_minus_xi) + eps)

    return xi_log + one_minus_xi_log


def arctan_term(xi: float | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Analytic arctan/arctanh term in the CG quasi-PDF kernels."""
    xi_arr = np.asarray(xi, dtype=np.float64)
    result = np.zeros_like(xi_arr, dtype=np.float64)

    below_half = xi_arr < 0.5 - eps
    above_half = xi_arr > 0.5 + eps
    near_half = ~(below_half | above_half)

    if np.any(below_half):
        x_val = xi_arr[below_half]
        sqrt_term = np.sqrt(1.0 - 2.0 * x_val)
        atan_piece = np.arctan(sqrt_term / (np.abs(x_val) + eps)) / (sqrt_term + eps)
        prefactor = (3.0 * x_val - 1.0) / (x_val - 1.0 + eps)
        result[below_half] = prefactor * atan_piece

    if np.any(above_half):
        x_val = xi_arr[above_half]
        sqrt_term = np.sqrt(2.0 * x_val - 1.0)
        atanh_piece = np.arctanh(sqrt_term / (np.abs(x_val) + eps)) / (sqrt_term + eps)
        prefactor = (3.0 * x_val - 1.0) / (x_val - 1.0 + eps)
        result[above_half] = prefactor * atanh_piece

    if np.any(near_half):
        x_val = xi_arr[near_half]
        result[near_half] = (3.0 * x_val - 1.0) / (x_val - 1.0)

    return result


def _transversity_full_domain_piece(xi: float, eps: float = 1e-12) -> float:
    """Full-domain NLO term in Eq. (2.18) of arXiv:2602.11283."""
    one_minus_xi = 1.0 - xi

    log_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
    log_piece = 2.0 * xi / log_denominator * _signed_log_sum(xi, eps=eps)
    pole_piece = -1.0 / (np.abs(one_minus_xi) + eps)
    atan_piece = float(arctan_term(xi, eps=eps))

    return log_piece + pole_piece + atan_piece


def _unpolarized_full_domain_piece(xi: float, eps: float = 1e-12) -> float:
    """Full-domain NLO term in Eq. (2.14) of arXiv:2602.11283."""
    one_minus_xi = 1.0 - xi

    ratio_denominator = one_minus_xi + np.sign(one_minus_xi) * eps
    splitting_ratio = (1.0 + xi**2) / ratio_denominator
    log_piece = splitting_ratio * _signed_log_sum(xi, eps=eps)
    sign_piece = np.sign(xi)
    atan_piece = float(arctan_term(xi, eps=eps))
    pole_piece = -1.5 / (np.abs(one_minus_xi) + eps)

    return log_piece + sign_piece + atan_piece + pole_piece


def _kernel_setup(
    x_ls: np.ndarray,
    y_ls: np.ndarray | None,
    mu: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Common grid data for matching-kernel matrix builders."""
    x_ls_mat = np.asarray(x_ls, dtype=float)
    y_ls_mat = np.asarray(x_ls if y_ls is None else y_ls, dtype=float).copy()

    if x_ls_mat.ndim != 1:
        raise ValueError("`x_ls` must be a 1D array.")

    dy = _validate_uniform_y_grid(y_ls_mat, eps=eps)
    alpha_s = _alpha_s_1(mu, n_f=3)
    identity = _identity_kernel(x_ls_mat, y_ls_mat, eps=eps)
    diag_rows = _closest_row_indices(x_ls_mat, y_ls_mat)

    return x_ls_mat, y_ls_mat, dy, alpha_s, identity, diag_rows


def _kernel_from_nlo_matrix(
    identity: np.ndarray,
    nlo_matrix: np.ndarray,
    alpha_s: float,
    dy: float,
) -> np.ndarray:
    """Eq. (2.13): ``C = 1 - alpha_s CF / (2 pi) C^(1)``."""
    prefactor = alpha_s * C_F / (2.0 * np.pi)

    return identity - prefactor * nlo_matrix * dy


def _zero_nlo_piece(x_ls_mat: np.ndarray, y_ls_mat: np.ndarray) -> np.ndarray:
    """Allocate one NLO matrix piece on the matching grid."""
    return np.zeros((len(x_ls_mat), len(y_ls_mat)))


def _eq215_shift_nlo_matrix(
    x_ls_mat: np.ndarray,
    y_ls_mat: np.ndarray,
    diag_rows: np.ndarray,
    dy: float,
    eps: float,
) -> np.ndarray:
    """Eq. (2.15) shift from ``gamma^t`` to ``gamma^z``."""
    nx = len(x_ls_mat)
    ny = len(y_ls_mat)
    plus_shift = np.zeros((nx, ny))

    for idx, x_val in enumerate(x_ls_mat):
        for idy, y_val in enumerate(y_ls_mat):
            xi = x_val / y_val
            if _is_unit_interval_xi(xi, eps=eps):
                plus_shift[idx, idy] = 2.0 * (1.0 - xi) / np.abs(y_val)

    _apply_plus_prescription(plus_shift, diag_rows)

    delta_piece = _delta_column_matrix(diag_rows, np.ones(ny), nx, ny, dy)

    return plus_shift + delta_piece


def transversity_matching_kernel_nlo(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO matching matrix for the Coulomb-gauge transversity quasi-PDF.

    The returned matrix has shape ``(len(x_ls), len(y_ls))`` and maps
    ``quasi_pdf(y)`` to ``lc_pdf(x)`` via ``np.dot(kernel, quasi_pdf)``.
    For samples with shape ``(N_samp, len(y_ls))``, use ``np.dot(samples, kernel.T)``.
    """
    x_ls_mat, y_ls_mat, dy, alpha_s, identity, diag_rows = _kernel_setup(x_ls, y_ls, mu=mu, eps=eps)

    plus_01 = _zero_nlo_piece(x_ls_mat, y_ls_mat)
    plus_full = _zero_nlo_piece(x_ls_mat, y_ls_mat)

    for idx, x_val in enumerate(x_ls_mat):
        for idy, y_val in enumerate(y_ls_mat):
            xi = x_val / y_val
            if not _is_regular_xi(xi, eps=eps):
                continue

            y_norm = np.abs(y_val)
            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)

            if _is_unit_interval_xi(xi, eps=eps):
                plus_01[idx, idy] = 2.0 * xi / (1.0 - xi) * log_scale / y_norm

            plus_full[idx, idy] = _transversity_full_domain_piece(xi, eps=eps) / y_norm

    _apply_plus_prescription(plus_01, diag_rows)
    _apply_plus_prescription(plus_full, diag_rows)

    nlo_matrix = plus_01 + plus_full

    return _kernel_from_nlo_matrix(identity, nlo_matrix, alpha_s, dy)


def unpolarized_matching_kernel_nlo_gT(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO unpolarized kernel for ``gamma^t`` in MSbar.

    Eq. (2.14) of arXiv:2602.11283.
    """
    x_ls_mat, y_ls_mat, dy, alpha_s, identity, diag_rows = _kernel_setup(x_ls, y_ls, mu=mu, eps=eps)
    nx = len(x_ls_mat)
    ny = len(y_ls_mat)

    plus_01 = _zero_nlo_piece(x_ls_mat, y_ls_mat)
    plus_full = _zero_nlo_piece(x_ls_mat, y_ls_mat)
    half_pole = _zero_nlo_piece(x_ls_mat, y_ls_mat)

    for idx, x_val in enumerate(x_ls_mat):
        for idy, y_val in enumerate(y_ls_mat):
            xi = x_val / y_val
            if not _is_regular_xi(xi, eps=eps):
                continue

            y_norm = np.abs(y_val)
            log_scale = np.log(4.0 * y_val**2 * pz_gev**2 / mu**2)

            if _is_unit_interval_xi(xi, eps=eps):
                splitting = (1.0 + xi**2) / (1.0 - xi)
                plus_01[idx, idy] = (splitting * log_scale + xi - 1.0) / y_norm

            plus_full[idx, idy] = _unpolarized_full_domain_piece(xi, eps=eps) / y_norm
            half_pole[idx, idy] = 0.5 / (np.abs(1.0 - xi) + eps) / y_norm

    _apply_plus_prescription(plus_01, diag_rows)
    _apply_plus_prescription(plus_full, diag_rows)
    _apply_plus_prescription(half_pole, diag_rows)

    delta_coeff = 0.5 * (1.0 + np.log(4.0 * y_ls_mat**2 * pz_gev**2 / mu**2))
    delta_piece = _delta_column_matrix(diag_rows, delta_coeff, nx, ny, dy)
    nlo_matrix = plus_01 + plus_full + half_pole + delta_piece

    return _kernel_from_nlo_matrix(identity, nlo_matrix, alpha_s, dy)


def unpolarized_matching_kernel_nlo_gZ(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO unpolarized kernel for ``gamma^z`` in MSbar.

    Eq. (2.15) of arXiv:2602.11283.
    """
    x_ls_mat, y_ls_mat, dy, alpha_s, _identity, diag_rows = _kernel_setup(x_ls, y_ls, mu=mu, eps=eps)

    kernel_gt = unpolarized_matching_kernel_nlo_gT(x_ls=x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps)
    nlo_shift = _eq215_shift_nlo_matrix(
        x_ls_mat=x_ls_mat,
        y_ls_mat=y_ls_mat,
        diag_rows=diag_rows,
        dy=dy,
        eps=eps,
    )

    prefactor = alpha_s * C_F / (2.0 * np.pi)

    return kernel_gt - prefactor * nlo_shift * dy


def helicity_matching_kernel_nlo_gTg5(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO helicity kernel for ``gamma^t gamma5`` in MSbar."""
    return unpolarized_matching_kernel_nlo_gT(x_ls=x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps)


def helicity_matching_kernel_nlo_gZg5(
    x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    y_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """NLO helicity kernel for ``gamma^z gamma5`` in MSbar."""
    return unpolarized_matching_kernel_nlo_gZ(x_ls=x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls, eps=eps)
