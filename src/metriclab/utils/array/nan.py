r"""Utilities to validate NaN policies and inspect arrays for NaNs."""

from __future__ import annotations

__all__ = ["NAN_POLICIES", "NanPolicy", "check_nan_policy", "contains_nan", "validate_nan_policy"]

from typing import Literal

import numpy as np

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
