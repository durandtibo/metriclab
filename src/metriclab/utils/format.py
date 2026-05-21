r"""Helpers to build compact string representations for metric outputs."""

from __future__ import annotations

__all__ = ["make_robust_bar"]

import math

from coola.utils.format import make_bar


def make_robust_bar(value: float, length: int = 10) -> str:
    r"""Create a progress bar string for a given value.

    Finite values in ``[0, 1]`` produce a filled progress bar.
    NaN and infinite values produce a bar filled with ``'?'`` to
    indicate an invalid or undefined metric.

    Args:
        value: The metric value. Must be in ``[0, 1]``, or ``nan``/``inf``.
        length: The number of characters in the bar body. Must be a
            positive integer.

    Returns:
        A string of the form ``'[{bar}]'`` where ``{bar}`` is
        ``length`` characters wide. Finite values produce a mix of
        ``'█'`` and ``'░'`` characters. NaN and infinite values
        produce ``'?'`` characters.

    Raises:
        ValueError: if ``length`` is not a positive integer.
        ValueError: if ``value`` is finite but outside ``[0, 1]``.

    Example:
        ```pycon
        >>> from metriclab.utils.format import make_robust_bar
        >>> make_robust_bar(0.6)
        '[██████░░░░]'
        >>> make_robust_bar(0.1)
        '[█░░░░░░░░░]'
        >>> make_robust_bar(1.0)
        '[██████████]'
        >>> make_robust_bar(0.0)
        '[░░░░░░░░░░]'
        >>> make_robust_bar(float("nan"))
        '[??????????]'
        >>> make_robust_bar(float("inf"))
        '[??????????]'
        >>> make_robust_bar(0.6, length=20)
        '[████████████░░░░░░░░]'

        ```
    """
    if length <= 0:
        msg = f"length must be a positive integer, got {length}"
        raise ValueError(msg)
    if math.isnan(value) or math.isinf(value):
        return "[" + "?" * length + "]"
    return make_bar(value=value, length=length)
