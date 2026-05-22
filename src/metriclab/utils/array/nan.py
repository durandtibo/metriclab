r"""Utilities to validate NaN policies and inspect arrays for NaNs."""

from __future__ import annotations

__all__ = [
    "NAN_POLICIES",
    "NanPolicy",
    "check_nan_policy",
    "contains_nan",
    "is_nan",
    "multi_is_nan",
    "validate_nan_policy",
]

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

NAN_POLICIES = ["omit", "propagate", "raise"]
NanPolicy = Literal["omit", "propagate", "raise"]


def validate_nan_policy(nan_policy: str) -> None:
    r"""Validate a NaN handling policy value.

    Args:
        nan_policy: The policy name to validate.

    Raises:
        ValueError: if ``nan_policy`` is not ``'omit'``,
            ``'propagate'``, or ``'raise'``.

    Example:
        ```pycon
        >>> from metriclab.utils.array import validate_nan_policy
        >>> validate_nan_policy("omit")

        ```
    """
    if nan_policy not in set(NAN_POLICIES):
        msg = (
            f"Incorrect 'nan_policy': {nan_policy}. The valid values are: "
            f"'omit', 'propagate', 'raise'"
        )
        raise ValueError(msg)


def contains_nan(arr: np.ndarray) -> bool:
    r"""Indicate if the given array contains at least one NaN value.

    Args:
        arr: The array to check.

    Returns:
        ``True`` if the array contains at least one NaN value.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import contains_nan
        >>> contains_nan(np.array([1, 2, 3]))
        False
        >>> contains_nan(np.array([1, 2, np.nan]))
        True
        >>> contains_nan(np.array([1, 2, float("nan")], dtype=object))
        True
        >>> contains_nan(np.array(["a", "b", "c"]))
        False

        ```
    """
    if arr.dtype == object:
        return any(isinstance(x, float) and np.isnan(x) for x in arr.flat)
    try:
        return bool(np.isnan(arr).any())
    except TypeError:
        # Non-numeric dtypes (e.g. datetime64, str) cannot contain NaN
        return False


def check_nan_policy(
    arr: np.ndarray, nan_policy: NanPolicy = "propagate", name: str = "input"
) -> bool:
    r"""Indicate if the given array contains at least one NaN value.

    Args:
        arr: The array to check.
        nan_policy: The NaN policy. The valid values are ``'omit'``,
            ``'propagate'``, or ``'raise'``.
        name: An optional name to be more precise about the array when
            the exception is raised.

    Returns:
        ``True`` if the array contains at least one NaN value.

    Raises:
        ValueError: if the array contains at least one NaN value and
            ``nan_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import check_nan_policy
        >>> check_nan_policy(np.array([1, 2, 3]))
        False
        >>> check_nan_policy(np.array([1, 2, np.nan]))
        True
        >>> check_nan_policy(np.array([1, 2, float("nan")], dtype=object))
        True
        >>> check_nan_policy(np.array(["a", "b", "c"]))
        False

        ```
    """
    validate_nan_policy(nan_policy)
    isnan = contains_nan(arr)
    if isnan and nan_policy == "raise":
        msg = f"{name} contains at least one NaN value"
        raise ValueError(msg)
    return isnan


def is_nan(arr: np.ndarray) -> np.ndarray:
    r"""Test element-wise for NaN values and return result as a boolean
    array.

    A value is considered missing if it is ``NaN``.

    Args:
        arr: The input array to test.

    Returns:
        A boolean array. ``True`` where the value is NaN, ``False`` otherwise.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import is_nan
        >>> is_nan(np.array([1.0, 0.0, float("nan"), 1.0]))
        array([False, False,  True, False])
        >>> is_nan(np.array([1.0, 0.0, float("nan"), 1.0, None], dtype=object))
        array([False, False,  True, False,  False])

        ```
    """
    if arr.dtype == object:
        return np.array(
            [isinstance(x, float) and np.isnan(x) for x in arr.flat],
            dtype=bool,
        ).reshape(arr.shape)
    try:
        return np.isnan(arr)
    except TypeError:
        # Non-numeric dtypes (e.g. datetime64, str) cannot be missing
        return np.zeros(arr.shape, dtype=bool)


def multi_is_nan(arrays: Sequence[np.ndarray]) -> np.ndarray:
    r"""Test element-wise for missing values for all input arrays and
    return result as a boolean array.

    A value is considered missing if it is ``NaN`` or ``None``.

    Args:
        arrays: The input arrays to test. All the arrays must have the
            same shape.

    Returns:
        A boolean array. ``True`` where any array has a missing value,
            ``False`` otherwise.

    Raises:
        ValueError: if ``arrays`` is empty.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import multi_is_nan
        >>> mask = multi_is_nan([np.array([1, 0, 0, 1, np.nan]), np.array([1, np.nan, 0, 1, 1])])
        >>> mask
        array([False,  True, False, False,  True])
        >>> mask = multi_is_nan(
        ...     [np.array([1, np.nan, 0], dtype=object), np.array([None, 2, 0], dtype=object)]
        ... )
        >>> mask
        array([False,  True, False])

        ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise ValueError(msg)
    return np.logical_or.reduce([is_nan(a) for a in arrays])
