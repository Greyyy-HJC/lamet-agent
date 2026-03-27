"""Tests for inline hard-kernel loading and validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.errors import KernelLoadError
from lamet_agent.kernel import load_kernel
from lamet_agent.schemas import KernelSpec


class KernelTests(unittest.TestCase):
    """Cover inline kernel compilation and interface validation."""

    def test_kernel_loads_successfully(self) -> None:
        kernel = load_kernel(
            KernelSpec(
                source="def demo_kernel(axis, values, metadata):\n    return values\n",
                callable_name="demo_kernel",
            )
        )
        self.assertEqual(kernel([0.0], [1.0], {}), [1.0])

    def test_kernel_missing_callable_raises(self) -> None:
        with self.assertRaises(KernelLoadError):
            load_kernel(KernelSpec(source="x = 1\n", callable_name="demo_kernel"))

    def test_kernel_signature_validation_raises(self) -> None:
        with self.assertRaises(KernelLoadError):
            load_kernel(KernelSpec(source="def bad_kernel():\n    return 1\n", callable_name="bad_kernel"))

    def test_kernel_syntax_error_raises(self) -> None:
        with self.assertRaises(KernelLoadError):
            load_kernel(KernelSpec(source="def broken(:\n    pass\n", callable_name="broken"))


if __name__ == "__main__":
    unittest.main()
