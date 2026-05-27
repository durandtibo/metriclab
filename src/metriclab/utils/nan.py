r"""Utilities for NaNs."""

from __future__ import annotations

__all__ = ["is_nan"]

import math
from typing import Any


def is_nan(value: Any) -> bool:
    """Return whether a value is a floating-point NaN.

    Args:
        value: The value to check.

    Returns:
        True if ``value`` is a float and is NaN, otherwise False.

    Example:
        ```pycon
        >>> from metriclab.utils.nan import is_nan
        >>> is_nan(2)
        False
        >>> is_nan(float("nan"))
        True

        ```
    """
    return isinstance(value, float) and math.isnan(value)
