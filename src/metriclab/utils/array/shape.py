r"""Utilities to validate array shapes and dimensions."""

from __future__ import annotations

__all__ = ["validate_array_ndim", "validate_same_shape"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


def validate_array_ndim(arr: np.ndarray, ndim: int, name: str = "input") -> None:
    r"""Validate that an array has the expected number of dimensions.

    Args:
        arr: The array to validate.
        ndim: The expected number of dimensions.
        name: The name of the input, used in error messages.
            Defaults to ``"input"``.

    Raises:
        ValueError: If ``arr.ndim`` is different from ``ndim``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import validate_array_ndim
        >>> validate_array_ndim(np.ones((2, 3)), ndim=2)

        ```
    """
    if arr.ndim != ndim:
        msg = f"{name}: expected {ndim}D array, got shape {arr.shape}"
        raise ValueError(msg)


def validate_same_shape(arrays: Iterable[np.ndarray]) -> None:
    r"""Check if arrays have the same shape.

    Args:
        arrays: The arrays to check.

    Raises:
        ValueError: if the arrays have different shapes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.array import validate_same_shape
        >>> validate_same_shape([np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1])])

        ```
    """
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        msg = f"arrays have different shapes: {shapes}"
        raise ValueError(msg)
