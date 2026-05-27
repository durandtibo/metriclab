r"""Utilities for removing values from an array."""

from __future__ import annotations

__all__ = ["remove_values"]

import math
from typing import TYPE_CHECKING, Any

from metriclab.utils.array.nan import remove_nans
from metriclab.utils.array.shape import validate_array_ndim

if TYPE_CHECKING:
    import numpy as np


def remove_values(arr: np.ndarray, value: Any) -> np.ndarray:
    """Return *arr* with all occurrences of *value* removed.

    Args:
        arr: A 1-D array to remove values from.
        value: The value to remove from *arr*. If *value* is NaN,
            all NaN values are removed regardless of sign or dtype.

    Returns:
        A 1-D array with all occurrences of *value* removed. Returns
            an empty array when all values match or the input is empty.

    Raises:
        ValueError: if ``arr`` is not 1-D.

    Example:
        ```pycon
        >>> import numpy as np
        >>> remove_values(np.array([1.0, 2.0, 1.0, 3.0]), 1.0)
        array([2., 3.])
        >>> remove_values(np.array([1.0, float("nan"), 2.0, float("nan")]), float("nan"))
        array([1., 2.])
        >>> remove_values(np.array(["cat", "dog", "cat"], dtype=object), "cat")
        array(['dog'], dtype=object)
        >>> remove_values(np.array([1.0, 2.0, 3.0]), 4.0)
        array([1., 2., 3.])
        >>> remove_values(np.array([]), 1.0)
        array([], dtype=float64)

        ```
    """
    validate_array_ndim(arr, ndim=1)

    try:
        if math.isnan(value):
            return remove_nans(arr)
    except TypeError:
        # isnan raises TypeError for non-float values such as
        # strings or integers; fall through to equality comparison.
        pass

    return arr[arr != value]
