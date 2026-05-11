"""
Logging helpers for analysis scripts.

Example:
```python
setup_logger("filename.log")

logger = logging.getLogger("my_logger")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
```
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def log_nonlinear_fit_quality(
    fit_res: object,
    *,
    kind: str,
    label: str | None = None,
    logger_name: str = "my_logger",
    q_threshold: float = 0.05,
) -> None:
    """Log INFO/WARNING for an ``lsqfit.nonlinear_fit`` result using Q and chi²/dof."""
    log = logging.getLogger(logger_name)
    fit_label = f" {label}" if label else ""
    fit_quality = (
        f"Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2 / fit_res.dof:.3f}"
    )
    if fit_res.Q < q_threshold:
        log.warning(">>> Bad %s%s fit with %s", kind, fit_label, fit_quality)
    else:
        log.info(">>> Good %s%s fit with %s", kind, fit_label, fit_quality)


def setup_logger(
    log_file: str | Path,
    console_output: bool = False,
    mode: str = "w",
) -> logging.Logger:
    """Create and configure a logger with file and optional console handlers."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(path, mode=mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

