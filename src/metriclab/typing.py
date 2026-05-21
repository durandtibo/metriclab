r"""Shared typing aliases used by the public API.

Example:
    ```pycon
    >>> from metriclab.typing import ArrayLike
    >>> values: ArrayLike = [1, 0, 1]
    >>> values
    [1, 0, 1]

    ```
"""

from __future__ import annotations

__all__ = ["ArrayLike"]

from typing import Any

import numpy as np

ArrayLike = np.ndarray | list[Any] | tuple[Any, ...]
