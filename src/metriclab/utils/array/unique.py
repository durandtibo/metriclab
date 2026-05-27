r"""Utilities to search for values in NumPy arrays."""

from __future__ import annotations

__all__ = ["count_unique_non_missing"]

from typing import TYPE_CHECKING, Any

import numpy as np

from metriclab.utils.array import contains_value
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike


def count_unique_non_missing(
    arr: ArrayLike,
    missing_values: Any = NOT_SET,
) -> int:
    r"""Count the number of unique non-missing values.

    Args:
        arr: Input array-like object.
        missing_values: Value representing missing data. If ``NOT_SET``,
            all unique values are counted.

    Returns:
        The number of unique non-missing values.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import count_unique_non_missing
        >>> count_unique_non_missing(np.array([1, 0, 0, 1, 1, 2]))
        3
        >>> count_unique_non_missing(np.array([1, 0, 0, 1, 1, 2, float("nan")]))
        4
        >>> count_unique_non_missing(
        ...     np.array([1, 0, 0, 1, 1, 2, float("nan")]),
        ...     missing_values=float("nan"),
        ... )
        3

        ```
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        return 0

    num_unique = _count_unique(arr)

    if missing_values is NOT_SET:
        return num_unique

    return num_unique - int(contains_value(arr, missing_values))


def _count_unique(arr: np.ndarray) -> int:
    r"""Count the number of unique values in an array.

    Uses ``np.unique`` for typed arrays and object arrays where
    sorting is supported. Falls back to a set-based approach for
    object arrays with heterogeneous types where ``np.unique`` raises
    ``TypeError``, and correctly deduplicates ``nan`` values which
    ``np.unique`` may not deduplicate in object arrays.

    Args:
        arr: The input array.

    Returns:
        The number of unique values.
    """
    if arr.size == 0:
        return 0
    if arr.dtype == object:
        # np.unique fails on object arrays for two reasons:
        # 1. nan != nan so multiple NaN values are not deduplicated
        # 2. sorting raises TypeError for mixed types (e.g. int + str)
        seen = set()
        for x in arr.flat:
            if isinstance(x, float) and np.isnan(x):
                seen.add((float, "nan"))
            else:
                seen.add((type(x), x))
        return len(seen)
    return int(np.unique(arr).size)
