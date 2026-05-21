r"""Contain equality utilities."""

from __future__ import annotations

__all__ = ["equal_to"]

import math
from typing import Any

import numpy as np


def equal_to(arr: np.ndarray, value: Any) -> np.ndarray:
    r"""Return a boolean mask indicating which elements equal a given
    value.

    Works correctly with special values: ``nan`` (using
    :func:`numpy.isnan` since ``nan != nan``), and ``None``
    (using identity check in object arrays).

    Args:
        arr: The input array.
        value: The value to compare against.

    Returns:
        A boolean array of the same shape as ``arr`` where ``True``
        indicates the element equals ``value``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import equal_to
        >>> equal_to(np.array([1, 2, 3, 2]), 2)
        array([False,  True, False,  True])
        >>> equal_to(np.array([1.0, float("nan"), 3.0, float("nan")]), float("nan"))
        array([False,  True, False,  True])
        >>> equal_to(np.array([1, None, 3, None], dtype=object), None)
        array([False,  True, False,  True])

        ```
    """
    if isinstance(value, float) and math.isnan(value):
        if arr.dtype == object:
            return np.array(
                [isinstance(x, float) and math.isnan(x) for x in arr.flat],
                dtype=bool,
            ).reshape(arr.shape)
        try:
            return np.isnan(arr)
        except TypeError:
            return np.zeros(arr.shape, dtype=bool)

    if value is None:
        if arr.dtype == object:
            return np.array([x is None for x in arr.flat], dtype=bool).reshape(arr.shape)
        return np.zeros(arr.shape, dtype=bool)

    return arr == value
