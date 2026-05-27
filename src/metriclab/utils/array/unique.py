r"""Utilities to search for values in NumPy arrays."""

from __future__ import annotations

__all__ = ["count_unique_non_missing", "multi_count_unique_non_missing"]

from typing import TYPE_CHECKING, Any

import numpy as np

from metriclab.utils.array import contains_value
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    num_unique = _count_unique(arr)
    if missing_values is NOT_SET:
        return num_unique

    return num_unique - int(contains_value(arr, missing_values))


def multi_count_unique_non_missing(
    arrays: Sequence[np.ndarray], missing_values: Any = NOT_SET
) -> int:
    r"""Count the number of unique non-missing values across all arrays.

        Counts unique values across all input arrays combined, as if all
        arrays were concatenated into one. Works correctly with special
        values such as ``nan``, ``inf``, and ``None``.

    Args:
        arrays: The input arrays to search.
        missing_values: The value representing missing data. If not
            set, all unique values are counted including ``nan``,
            ``inf``, and ``None``.

    Returns:
        The number of unique non-missing values across all arrays.
            Returns ``0`` when ``arrays`` is empty or all arrays are
            empty.

    Example:
    ```pycon
    >>> import numpy as np
    >>> from metriclab.utils.array import multi_count_unique_non_missing
    >>> # two arrays with overlapping values
    >>> multi_count_unique_non_missing([np.array([1, 0, 2]), np.array([2, 3, 0])])
    4
    >>> # nan excluded when missing_values=float("nan")
    >>> multi_count_unique_non_missing(
    ...     [np.array([1.0, float("nan")]), np.array([2.0, float("nan")])],
    ...     missing_values=float("nan"),
    ... )
    2
    >>> # empty list of arrays
    >>> multi_count_unique_non_missing([])
    0

    ```
    """
    if not arrays:
        return 0
    return count_unique_non_missing(
        np.concatenate([np.asarray(a).ravel() for a in arrays]),
        missing_values=missing_values,
    )


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
