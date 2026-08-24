"""Independent implementation of the LaMET analysis refactor.

The ``lamet_agent_neo`` package intentionally lives beside the historical
``lamet_agent`` package while the refactor is being evaluated.  It exposes the
new manifest, contract, data, and ordered-agent APIs without compatibility
aliases to the old framework.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
