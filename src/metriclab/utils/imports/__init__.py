r"""Helpers for optional dependencies used by metriclab.

The utilities in this package let callers:

- check whether an optional package is installed,
- fail early with a clear error when a package is required,
- gate function execution behind package availability.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_rich",
    "colorlog_available",
    "is_colorlog_available",
    "is_rich_available",
    "raise_colorlog_missing_error",
    "raise_rich_missing_error",
    "rich_available",
]

from metriclab.utils.imports.colorlog import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)
from metriclab.utils.imports.rich import (
    check_rich,
    is_rich_available,
    raise_rich_missing_error,
    rich_available,
)
