"""Every matching kernel must stay on the plus prescription it was built for.

The two prescriptions are not interchangeable: the column-sum one is second order
and smooth, but cannot reproduce the MSbar counterterm whose ``int_0^2`` tail leaves
any finite y grid; the row-wise ksi integral does reproduce it, but is first order and
puts grid-scale noise on the hybrid Wilson-line term. Moving a kernel to the other
builder is silent -- the matrix stays finite and the report reads normally -- so the
assignment is pinned here.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import lamet_agent.kernels.implementation as implementation
from lamet_agent.kernels import list_kernel_ids, load_kernel

COLUMN_PLUS = frozenset(
    {
        "da_gi_gtg5_hybrid_lrr_nlo",
        "da_gi_gtg5_hybrid_nlo",
        "da_gi_gtg5_ratio_nlo",
        "da_gi_gzg5_hybrid_lrr_nlo",
        "da_gi_gzg5_hybrid_nlo",
        "da_gi_gzg5_ratio_nlo",
        "quark_pdf_cg_gt_hybrid_nlo",
        "quark_pdf_cg_gt_hybrid_rgr_nlo_re",
        "quark_pdf_cg_gt_ratio_nlo",
        "quark_pdf_cg_gtg5_hybrid_nlo",
        "quark_pdf_cg_gtg5_hybrid_rgr_nlo_re",
        "quark_pdf_cg_gtg5_ratio_nlo",
        "quark_pdf_cg_gtgpg5_hybrid_nlo",
        "quark_pdf_cg_gtgpg5_hybrid_rgr_nlo_re",
        "quark_pdf_cg_gtgpg5_msbar_nlo",
        "quark_pdf_cg_gtgpg5_msbar_rgr_nlo_im",
        "quark_pdf_cg_gtgpg5_ratio_nlo",
        "quark_pdf_cg_gz_hybrid_nlo",
        "quark_pdf_cg_gz_hybrid_rgr_nlo_re",
        "quark_pdf_cg_gz_ratio_nlo",
        "quark_pdf_cg_gzg5_hybrid_nlo",
        "quark_pdf_cg_gzg5_hybrid_rgr_nlo_re",
        "quark_pdf_cg_gzg5_ratio_nlo",
        "quark_pdf_gi_gt_hybrid_lrr_nlo",
        "quark_pdf_gi_gt_hybrid_nlo",
        "quark_pdf_gi_gt_ratio_nlo",
        "quark_pdf_gi_gtg5_hybrid_lrr_nlo",
        "quark_pdf_gi_gtg5_hybrid_nlo",
        "quark_pdf_gi_gtg5_ratio_nlo",
        "quark_pdf_gi_gtgpg5_hybrid_lrr_nlo",
        "quark_pdf_gi_gtgpg5_hybrid_nlo",
        "quark_pdf_gi_gtgpg5_ratio_nlo",
        "quark_pdf_gi_gz_hybrid_lrr_nlo",
        "quark_pdf_gi_gz_hybrid_nlo",
        "quark_pdf_gi_gz_ratio_nlo",
        "quark_pdf_gi_gzg5_hybrid_lrr_nlo",
        "quark_pdf_gi_gzg5_hybrid_nlo",
        "quark_pdf_gi_gzg5_ratio_nlo",
    }
)

# MSbar only: Eq. (2.14)'s 1/(2|1-ksi|) is plus-prescribed over ksi in [0, 2] alone, and
# that tail leaves the y grid, so the subtraction has to be the ksi integral rather than
# a column sum. CG transversity is absent because MSbar equals the ratio kernel there.
ROW_PLUS = frozenset(
    {
        "quark_pdf_cg_gt_msbar_nlo",
        "quark_pdf_cg_gt_msbar_rgr_nlo_im",
        "quark_pdf_cg_gtg5_msbar_nlo",
        "quark_pdf_cg_gtg5_msbar_rgr_nlo_im",
        "quark_pdf_cg_gz_msbar_nlo",
        "quark_pdf_cg_gz_msbar_rgr_nlo_im",
        "quark_pdf_cg_gzg5_msbar_nlo",
        "quark_pdf_cg_gzg5_msbar_rgr_nlo_im",
    }
)

BUILDERS = ("build_matching_matrix_column_plus", "build_matching_matrix_row_plus")


def _builders_used(kernel_id: str, grid: np.ndarray) -> set[str]:
    used: set[str] = set()
    originals = {name: getattr(implementation, name) for name in BUILDERS}

    def spy(name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            used.add(name)
            return originals[name](*args, **kwargs)

        return record

    for name in BUILDERS:
        setattr(implementation, name, spy(name))
    try:
        kernel = load_kernel(kernel_id)
        arguments: dict[str, Any] = {"momentum_gev": 2.0, "scale_gev": 2.0}
        if "zs_fm" in inspect.signature(kernel).parameters:
            arguments["zs_fm"] = 0.3
        kernel(grid, grid, **arguments)
    finally:
        for name, original in originals.items():
            setattr(implementation, name, original)
    return used


def test_every_kernel_declares_one_plus_prescription() -> None:
    assert not COLUMN_PLUS & ROW_PLUS
    assert set(list_kernel_ids()) == COLUMN_PLUS | ROW_PLUS


@pytest.mark.parametrize("kernel_id", sorted(COLUMN_PLUS | ROW_PLUS))
def test_kernel_stays_on_its_plus_prescription(kernel_id: str) -> None:
    expected = "build_matching_matrix_row_plus" if kernel_id in ROW_PLUS else "build_matching_matrix_column_plus"
    grid = np.linspace(-1.0, 1.0, 9) + 1.0 / 8
    assert _builders_used(kernel_id, grid) == {expected}
