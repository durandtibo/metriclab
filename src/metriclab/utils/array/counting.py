r"""Element-wise equality utilities for NumPy arrays."""

from __future__ import annotations

__all__ = ["count_values"]

from typing import TYPE_CHECKING, Any

from metriclab.utils.array.equal import equal_to

if TYPE_CHECKING:
    import numpy as np


def count_values(arr: np.ndarray, value: Any) -> int:
    """Count the number of occurrences of *value* in *arr*.

    Args:
        arr: An array to count values in. Arrays of any shape are
            accepted.
        value: The value to count. If *value* is NaN, all NaN values
            are counted regardless of sign or dtype.

    Returns:
        The number of occurrences of *value* in *arr*.

    Example:
        ```pycon
        >>> import numpy as np
        >>> count_values(np.array([1, 2, 1, 3, 1]), 1)
        3
        >>> count_values(np.array([1.0, float("nan"), 2.0, float("nan")]), float("nan"))
        2
        >>> count_values(np.array(["cat", "dog", "cat"], dtype=object), "cat")
        2
        >>> count_values(np.array([[1, 2], [1, 3]]), 1)
        2
        >>> count_values(np.array([1, 2, 3]), 4)
        0
        >>> count_values(np.array([]), 1.0)
        0

        ```
    """
    return int(equal_to(arr, value).sum())
