"""Lexical discovery of matching kernels.

Public kernel ids are immediate non-private Python file stems.  There is no
registry or metadata object; ``load_kernel`` imports exactly the requested file.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _root(root: str | Path | None) -> Path:
    return Path(root).expanduser().resolve() if root is not None else Path(__file__).parent.resolve()


def list_kernel_ids(root: str | Path | None = None) -> list[str]:
    """Return paired public kernel implementation/document stems."""
    directory = _root(root)
    return sorted(
        path.stem
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix == ".py"
        and path.name != "__init__.py"
        and path.stem.isidentifier()
        and (directory / f"{path.stem}.md").is_file()
    )


def _load_module(path: Path) -> ModuleType:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    name = f"_lamet_agent_neo_kernel_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load kernel '{path.stem}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_kernel(kernel_id: str, root: str | Path | None = None) -> Any:
    """Load the exact public kernel stem and return its ``kernel`` callable."""
    if not isinstance(kernel_id, str) or not kernel_id or not kernel_id.isidentifier() or kernel_id.startswith("_"):
        raise ValueError("kernel_id must be an exact public filename stem")
    directory = _root(root)
    path = directory / f"{kernel_id}.py"
    if not path.is_file() or not (directory / f"{kernel_id}.md").is_file():
        raise ValueError(f"kernel '{kernel_id}' is not available")
    function = getattr(_load_module(path), "kernel", None)
    if not callable(function):
        raise TypeError(f"kernel file '{kernel_id}.py' must export kernel()")
    return function


def load_renormalization_kernel(kernel_id: str, root: str | Path | None = None) -> Any:
    """Load one explicit renormalization formula callable by filename stem."""
    if not isinstance(kernel_id, str) or not kernel_id or not kernel_id.isidentifier() or kernel_id.startswith("_"):
        raise ValueError("renormalization kernel_id must be an exact public filename stem")
    path = _root(root) / f"{kernel_id}.py"
    if not path.is_file():
        raise ValueError(f"renormalization kernel '{kernel_id}' is not available")
    module = importlib.import_module(f"{__package__}.{kernel_id}") if root is None else _load_module(path)
    function = getattr(module, "kernel", None)
    if not callable(function):
        raise TypeError(f"renormalization kernel file '{kernel_id}.py' must export kernel()")
    return function


def load_kernel_document(kernel_id: str, root: str | Path | None = None) -> str:
    """Read the formula document paired with an exact kernel filename stem."""
    if not isinstance(kernel_id, str) or not kernel_id or not kernel_id.isidentifier() or kernel_id.startswith("_"):
        raise ValueError("kernel_id must be an exact public filename stem")
    path = _root(root) / f"{kernel_id}.md"
    if not path.is_file():
        raise ValueError(f"kernel '{kernel_id}' has no formula document")
    document = path.read_text(encoding="utf-8").strip()
    if not document:
        raise ValueError(f"kernel '{kernel_id}' has an empty formula document")
    return document


__all__ = [
    "list_kernel_ids",
    "load_kernel",
    "load_renormalization_kernel",
    "load_kernel_document",
]
