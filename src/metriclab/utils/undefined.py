r"""Validation helpers for undefined-value handling policies."""

from __future__ import annotations

__all__ = ["UndefinedPolicy", "resolve_fill_value", "validate_undefined_policy"]

import warnings
from typing import Any, Literal

import numpy as np

UndefinedPolicy = float | Literal["nan", "raise", "warn"]


def validate_undefined_policy(undefined_policy: Any) -> None:
    r"""Validate an ``undefined_policy`` value.

    Args:
        undefined_policy: The policy value to validate. Accepted values:

            - ``float``: substitute the undefined entry with this
              value (e.g. ``0.0``, ``1.0``, ``float('nan')``).
            - ``"nan"``: string alias for ``float('nan')``.
            - ``"raise"``: raise an exception when an undefined value
              is encountered.
            - ``"warn"``: emit a ``UserWarning`` and substitute
              ``0.0``.

    Raises:
        ValueError: if ``undefined_policy`` is not one of the accepted
            values.

    Example:
        ```pycon
        >>> from metriclab.utils.undefined import validate_undefined_policy
        >>> validate_undefined_policy("warn")
        >>> validate_undefined_policy(0.0)
        >>> validate_undefined_policy(float("nan"))

        ```
    """
    if isinstance(undefined_policy, float) or (
        isinstance(undefined_policy, str) and undefined_policy in {"nan", "raise", "warn"}
    ):
        return
    msg = (
        f"Invalid 'undefined_policy' value {undefined_policy!r}. "
        "Expected a float (e.g. 0.0, 1.0, float('nan')) or one of the "
        "string aliases 'nan', 'raise', or 'warn'."
    )
    raise ValueError(msg)


def resolve_fill_value(
    undefined_mask: np.ndarray,
    undefined_policy: UndefinedPolicy,
) -> float:
    """Return the fill value for undefined metric entries.

    Args:
        undefined_mask: A 1-D boolean array where ``True`` indicates
            that the metric is undefined for the corresponding class.
        undefined_policy: Controls the behaviour when undefined entries
            are detected. See :class:`UndefinedPolicy` for accepted
            values.

    Returns:
        The fill value to substitute for undefined entries. Returns
            ``0.0`` immediately when no entries are undefined.

    Raises:
        ValueError: if ``undefined_policy='raise'`` and at least one
            entry is undefined.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.utils.undefined import resolve_fill_value
        >>> resolve_fill_value(
        ...     undefined_mask=np.array([True, False, True, False]),
        ...     undefined_policy=0.0,
        ... )
        0.0

        ```
    """
    if not undefined_mask.any():
        return 0.0  # fill is irrelevant; no undefined entries

    n_undefined = int(undefined_mask.sum())
    indices = np.where(undefined_mask)[0].tolist()

    if undefined_policy == "nan":
        return float("nan")

    if undefined_policy == "raise":
        msg = (
            f"The metric is undefined for {n_undefined} element(s) at "
            f"indices {indices}. Use 'undefined_policy=\"warn\"' to emit a "
            "warning and substitute 0.0, or "
            "'undefined_policy=float(\"nan\")' to propagate NaN instead."
        )
        raise ValueError(msg)

    if undefined_policy == "warn":
        warnings.warn(
            f"The metric is undefined for {n_undefined} element(s) at "
            f"indices {indices}. Substituting 0.0. Use "
            "'undefined_policy=0.0' to silence this warning, or "
            "'undefined_policy=float(\"nan\")' to propagate NaN instead.",
            UserWarning,
            stacklevel=4,
        )
        return 0.0

    # Covers float values including float('nan') passed directly.
    return float(undefined_policy)
