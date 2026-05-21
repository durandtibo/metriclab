r"""Contain equality utilities."""

from __future__ import annotations

__all__ = ["equal_to"]

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from metriclab.utils.array.nan import is_nan

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        return is_nan(arr)

    if value is None:
        if arr.dtype == object:
            return np.array([x is None for x in arr.flat], dtype=bool).reshape(arr.shape)
        return np.zeros(arr.shape, dtype=bool)

    return arr == value


def multi_equal_to(arrays: Sequence[np.ndarray], value: Any) -> np.ndarray:
    r"""Test element-wise equality against a value for all input arrays
    and return result as a boolean array.

    Returns ``True`` at positions where **any** array has an element
    equal to ``value``. Works correctly with special values such as
    ``nan``, ``inf``, and ``None``.

    Args:
        arrays: The input arrays to test. All arrays must have the
            same shape.
        value: The value to test equality against.

    Returns:
        A boolean array of the same shape as the input arrays.
        ``True`` where any array has an element equal to ``value``,
        ``False`` otherwise.

    Raises:
        ValueError: if ``arrays`` is empty.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import multi_equal_to
        >>> multi_equal_to([np.array([1, 0, 2, 1]), np.array([1, 2, 0, 1])], value=2)
        array([False,  True,  True, False])
        >>> multi_equal_to(
        ...     [np.array([1.0, 0.0, float("nan")]), np.array([1.0, float("nan"), 0.0])],
        ...     value=float("nan"),
        ... )
        array([False,  True,  True])
        >>> multi_equal_to(
        ...     [np.array([1, None, 3], dtype=object), np.array([None, 2, 3], dtype=object)],
        ...     value=None,
        ... )
        array([ True,  True, False])

        ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise ValueError(msg)
    return np.logical_or.reduce([equal_to(a, value) for a in arrays])
