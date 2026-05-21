r"""Utilities to convert array-like objects to NumPy arrays."""

from __future__ import annotations

__all__ = ["to_numpy", "to_numpy_1d"]

from typing import Any

import numpy as np

from metriclab.utils.array.shape import validate_array_ndim


def to_numpy(x: Any, name: str = "input") -> np.ndarray:
    r"""Convert an array-like object to a NumPy array.

    Supported input types are :class:`numpy.ndarray`,
    :class:`polars.Series`, :class:`list`, and :class:`tuple`.

    Args:
        x: The array-like object to convert.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Returns:
        A NumPy array representation of ``x``. If ``x`` is already a
            :class:`numpy.ndarray`, it is returned as-is without copying.

    Raises:
        TypeError: If ``x`` is not a supported array-like type.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import to_numpy
        >>> to_numpy([1, 2, 3])
        array([1, 2, 3])
        >>> to_numpy((1, 2, 3))
        array([1, 2, 3])

        ```
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    msg = f"{name}: unsupported type {type(x)}"
    raise TypeError(msg)


def to_numpy_1d(x: Any, name: str = "input") -> np.ndarray:
    r"""Convert an array-like object to a 1D NumPy array.

    Supported input types are the same as :func:`to_numpy`.

    Args:
        x: The array-like object to convert. Must be 1-dimensional
            after conversion.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Returns:
        A 1D NumPy array representation of ``x``.

    Raises:
        TypeError: If ``x`` is not a supported array-like type.
        ValueError: If ``x`` does not have exactly one dimension after
            conversion.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import to_numpy_1d
        >>> to_numpy_1d([1, 2, 3])
        array([1, 2, 3])

        ```
    """
    arr = to_numpy(x, name=name)
    validate_array_ndim(arr, ndim=1, name=name)
    return arr
