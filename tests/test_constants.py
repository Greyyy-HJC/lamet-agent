"""Tests for shared lattice/QCD constants and helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.constants import GEV_FM, alphas_nloop, lat_unit_convert, lattice_unit_to_physical, qcd_beta


class ConstantsTests(unittest.TestCase):
    def test_lattice_unit_conversion_matches_formula(self) -> None:
        converted = lattice_unit_to_physical(4, a_fm=0.09, spatial_extent=64, dimension="P")
        expected = 4 * 2 * np.pi * GEV_FM / 64 / 0.09
        self.assertAlmostEqual(float(converted), float(expected))
        self.assertAlmostEqual(float(lat_unit_convert(4, a=0.09, Ls=64, dimension="P")), float(expected))

    def test_qcd_beta_matches_known_sign(self) -> None:
        self.assertGreater(qcd_beta(0, 3), 0.0)
        self.assertGreater(qcd_beta(1, 3), 0.0)

    def test_alphas_nloop_is_positive_and_decreases_with_scale(self) -> None:
        alpha_low = float(alphas_nloop(1.0, order=1))
        alpha_high = float(alphas_nloop(3.0, order=1))
        self.assertGreater(alpha_low, 0.0)
        self.assertGreater(alpha_high, 0.0)
        self.assertGreater(alpha_low, alpha_high)


if __name__ == "__main__":
    unittest.main()
