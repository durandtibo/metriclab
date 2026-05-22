r"""Utilities to preprocess ``numpy.ndarray`` with missing values."""

from __future__ import annotations

__all__ = ["preprocess_1d"]


from typing import TYPE_CHECKING, Any

import numpy as np

from metriclab.utils.array.equal import multi_equal_to
from metriclab.utils.array.search import multi_contains_value
from metriclab.utils.array.shape import validate_array_ndim, validate_same_shape
from metriclab.utils.missing import validate_missing_policy
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from collections.abc import Sequence

    from metriclab.utils.missing import MissingPolicy


def preprocess_1d(
    arrays: Sequence[np.ndarray],
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
) -> list[np.ndarray]:
    r"""Preprocess a sequence of 1-dimensional arrays by optionally
    removing rows with a specific missing value.

    Args:
        arrays: The arrays to preprocess. All arrays must be
            1-dimensional and have the same shape.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``. Only ``'omit'`` has an effect — it removes
            rows where any array equals ``missing_values``. ``'raise'``
            raises a ``ValueError`` if any array contains
            ``missing_values``. ``'propagate'`` leaves the arrays
            unchanged.
        missing_values: The value to treat as missing. Only used when
            ``missing_policy`` is ``'omit'`` or ``'raise'``. If not
            set, no rows are removed regardless of ``missing_policy``.

    Returns:
        A list of preprocessed arrays with the same order as the
            input. Returns an empty list when ``arrays`` is empty.

    Raises:
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if the arrays do not all have the same shape.
        ValueError: if any array is not 1-dimensional.
        ValueError: if any array contains ``missing_values`` and
            ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import preprocess_1d
        >>> arrays = [np.array([1, 0, 0, 1, 1, np.nan]), np.array([0, 1, 0, 1, np.nan, 1])]
        >>> preprocess_1d(arrays)
        [array([ 1.,  0.,  0.,  1.,  1., nan]), array([ 0.,  1.,  0.,  1., nan,  1.])]
        >>> preprocess_1d(arrays, missing_policy="omit", missing_values=float("nan"))
        [array([1., 0., 0., 1.]), array([0., 1., 0., 1.])]

        ```
    """
    if not arrays:
        return []
    validate_missing_policy(missing_policy)
    validate_same_shape(arrays)
    # Only check first array since validate_same_shape guarantees
    # all arrays have the same shape.
    validate_array_ndim(arrays[0], ndim=1)
    if missing_values is NOT_SET or missing_policy == "propagate":
        return list(arrays)
    if missing_policy == "raise":
        if multi_contains_value(arrays, value=missing_values):
            msg = f"arrays contain at least one missing value ({missing_values!r})"
            raise ValueError(msg)
        return list(arrays)
    # missing_policy == "omit"
    mask = np.logical_not(multi_equal_to(arrays, value=missing_values))
    return [arr[mask] for arr in arrays]
