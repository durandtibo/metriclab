r"""Utilities to search for values in NumPy arrays."""

from __future__ import annotations

__all__ = ["count_unique_non_missing", "multi_count_unique_non_missing", "multi_unique", "unique"]

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

from metriclab.utils.array import validate_array_ndim
from metriclab.utils.array.nan import remove_duplicate_nans
from metriclab.utils.array.remove import remove_values
from metriclab.utils.array.search import contains_value
from metriclab.utils.nan import is_nan
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
            if is_nan(x):
                seen.add((float, "nan"))
            else:
                seen.add((type(x), x))
        return len(seen)
    return int(np.unique(arr).size)


def unique(arr: np.ndarray) -> np.ndarray:
    """Return the unique values in a 1-D array.

    Unlike :func:`numpy.unique`, this function handles object arrays
    containing NaN, None, and mixed types that cannot be sorted.
    Uniqueness is determined by equality for hashable values and by
    identity for unhashable ones. The order of first appearance is
    preserved in the result.

    Args:
        arr: A 1-D array to find unique values in.

    Returns:
        A 1-D array of unique values in order of first appearance.
            NaN values are collapsed to a single NaN. Returns an empty
            array when the input is empty.

    Raises:
        ValueError: if ``arr`` is not 1-D.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import unique
        >>> unique(np.array([3, 1, 2, 1, 3]))
        array([1, 2, 3])
        >>> unique(np.array([1.0, float("nan"), 2.0, float("nan")]))
        array([ 1.,  2., nan])
        >>> unique(np.array([1, None, 2, None], dtype=object))
        array([1, None, 2], dtype=object)
        >>> unique(np.array(["cat", 1, "cat", float("nan"), 1], dtype=object))
        array(['cat', 1, nan], dtype=object)
        >>> unique(np.array([]))
        array([], dtype=float64)

        ```
    """
    validate_array_ndim(arr, ndim=1)

    # Fast path: np.unique handles all numeric and string dtypes
    # correctly, including sorting. Only fall through to the custom
    # logic when arr is an object array that cannot be sorted.
    with contextlib.suppress(TypeError):
        return remove_duplicate_nans(np.unique(arr))

    seen = set()
    non_nan_result = []
    seen_nan = False
    nan_value = None  # holds the actual NaN object to re-append later

    # arr.tolist() converts NumPy scalars to native Python objects,
    # which are cheaper to hash and compare than their NumPy wrappers.
    for value in arr.tolist():
        if is_nan(value):
            if not seen_nan:
                seen_nan = True
                nan_value = value
            continue

        if value not in seen:
            seen.add(value)
            non_nan_result.append(value)

    # Attempt to sort the unique values. NaN is excluded from sorting
    # and appended at the end (consistent with np.unique's behaviour
    # for numeric types). Fall back to insertion order when the
    # elements are not mutually comparable (e.g. mixed str and int).
    with contextlib.suppress(TypeError):
        non_nan_result = sorted(non_nan_result)

    if seen_nan:
        non_nan_result.append(nan_value)

    return np.array(non_nan_result, dtype=object)


def multi_unique(arrays: Sequence[np.ndarray], missing_values: Any = NOT_SET) -> np.ndarray:
    """Return the sorted unique values across all arrays.

    Args:
        arrays: A list of arrays to compute unique values across.
            Arrays may have different dtypes and lengths. Empty arrays
            contribute no values to the result. For arrays that can
            contain NaN (including object arrays holding
            ``float('nan')``), multiple NaN values are collapsed to a
            single NaN at the end of the result.
        missing_values: Value representing missing data. If ``NOT_SET``,
            all unique values are included. If set, occurrences of
            ``missing_values`` are excluded from the result.

    Returns:
        A 1-D sorted array of unique values. Returns an empty array
            when ``arrays`` is empty or all values are filtered out.

    Example:
        ```pycon
        >>> import numpy as np
        >>> multi_unique([np.array([1, 3, 4, 3]), np.array([3, 1, 2, 1]), np.array([6, 3, 4, 2])])
        array([1, 2, 3, 4, 6])
        >>> multi_unique([np.array([1.0, float("nan")]), np.array([2.0, float("nan")])])
        array([ 1.,  2., nan])
        >>> multi_unique(
        ...     [np.array(["cat", "dog"], dtype=object), np.array(["dog", "bear"], dtype=object)]
        ... )
        array(['bear', 'cat', 'dog'], dtype=object)
        >>> multi_unique(
        ...     [
        ...         np.array([float("nan"), 1.0], dtype=object),
        ...         np.array([float("nan"), 2.0], dtype=object),
        ...     ]
        ... )
        array([1.0, 2.0, nan], dtype=object)
        >>> multi_unique(
        ...     [np.array([1.0, 2.0, float("nan")]), np.array([2.0, 3.0])],
        ...     missing_values=float("nan"),
        ... )
        array([1., 2., 3.])
        >>> # all values filtered out
        >>> multi_unique([np.array([float("nan")])], missing_values=float("nan"))
        array([], dtype=float64)
        >>> multi_unique([])
        array([], dtype=float64)

        ```
    """
    if not arrays:
        return np.array([], dtype=np.float64)

    if missing_values is not NOT_SET:
        arrays = [remove_values(arr, missing_values) for arr in arrays]

    return unique(np.concatenate(arrays) if arrays else np.array([], dtype=np.float64))
