r"""Utilities to search for values in NumPy arrays."""

from __future__ import annotations

__all__ = ["contains_value"]

import math
from typing import TYPE_CHECKING, Any

from metriclab.utils.array.nan import contains_nan

if TYPE_CHECKING:
    import numpy as np


def contains_value(arr: np.ndarray, value: Any) -> bool:
    r"""Check if a value is present in a numpy array.

    Works correctly with special values: ``nan`` (using
    :func:`numpy.isnan` since ``nan != nan``), ``inf`` (using
    :func:`numpy.isinf` to distinguish ``+inf`` from ``-inf``),
    and ``None`` (using identity check in object arrays).

    Args:
        arr: The array to search.
        value: The value to search for.

    Returns:
        ``True`` if ``value`` is present in ``arr``, ``False``
            otherwise.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import contains_value
        >>> contains_value(np.array([1, 2, 3]), 2)
        True
        >>> contains_value(np.array([1, 2, 3]), 4)
        False
        >>> contains_value(np.array([1.0, float("nan"), 3.0]), float("nan"))
        True
        >>> contains_value(np.array([1.0, float("inf"), 3.0]), float("inf"))
        True
        >>> contains_value(np.array([1.0, float("-inf"), 3.0]), float("-inf"))
        True
        >>> contains_value(np.array([1, None, 3], dtype=object), None)
        True
        >>> contains_value(np.array([1, 2, 3]), None)
        False

        ```
    """
    if isinstance(value, float) and math.isnan(value):
        return contains_nan(arr)

    if arr.dtype == object:
        if value is None:
            return any(x is None for x in arr.flat)
        # Vectorized equality is faster than element-wise iteration
        # for object arrays with non-special values
        return bool((arr == value).any())

    return bool((arr == value).any())
