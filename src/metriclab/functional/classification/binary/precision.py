r"""Compute binary classification precision from array-like inputs.

This module powers :func:`metriclab.functional.binary_precision` and supports
NumPy arrays, and Python sequences.
"""

from __future__ import annotations

__all__ = ["binary_precision"]

from typing import TYPE_CHECKING, Any

from sklearn.metrics import precision_score

from metriclab.exceptions import EmptyMetricError
from metriclab.results import BinaryPrecisionResult
from metriclab.utils.array import multi_contains_value, preprocess_1d, to_numpy_1d
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike
    from metriclab.utils.missing import MissingPolicy


def binary_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = 1,
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
    raise_empty: bool = True,
) -> BinaryPrecisionResult:
    r"""Compute the binary precision score.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        pos_label: The label of the positive class. Defaults to ``1``.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``. ``'omit'`` removes rows where any array
            equals ``missing_values`` before computing the metric.
            ``'propagate'`` keeps the rows but returns
            ``precision=nan`` when missing values are detected.
            ``'raise'`` raises a ``ValueError`` if any array contains
            ``missing_values``.
        missing_values: The value to treat as missing. If not set,
            missing value handling is disabled regardless of
            ``missing_policy``.
        raise_empty: If ``True``, raises ``EmptyMetricError`` when
            there are no valid predictions. If ``False``, returns a
            ``BinaryPrecisionResult`` with ``num_predictions=0`` and
            ``precision=nan``.

    Returns:
        A ``BinaryPrecisionResult``. When missing values are present
        and ``missing_policy='propagate'``, the result keeps
        ``num_predictions`` and sets ``precision`` to ``nan``.

    Raises:
        EmptyMetricError: if there are no valid predictions and
            ``raise_empty`` is ``True``.
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` or ``y_pred`` contains
            ``missing_values`` and ``missing_policy`` is ``'raise'``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from metriclab.functional import binary_precision
        >>> # with numpy arrays
        >>> binary_precision(
        ...     y_true=np.array([1, 0, 0, 1, 1]),
        ...     y_pred=np.array([1, 0, 0, 1, 1]),
        ... )
        BinaryPrecisionResult(precision=1.0, num_predictions=5)
        >>> # with lists
        >>> binary_precision(y_true=[1, 0, 0, 1, 1], y_pred=[1, 0, 1, 1, 1])
        BinaryPrecisionResult(precision=0.75, num_predictions=5)
        >>> # with string labels
        >>> binary_precision(
        ...     y_true=["cat", "dog", "cat", "dog", "dog"],
        ...     y_pred=["cat", "dog", "dog", "dog", "dog"],
        ...     pos_label="dog",
        ... )
        BinaryPrecisionResult(precision=0.75, num_predictions=5)
        >>> # with missing values and missing_policy='propagate' (default)
        >>> binary_precision(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_values=float("nan"),
        ... )
        BinaryPrecisionResult(precision=nan, num_predictions=5)
        >>> # with missing values and missing_policy='omit'
        >>> binary_precision(
        ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        ...     missing_policy="omit",
        ...     missing_values=float("nan"),
        ... )
        BinaryPrecisionResult(precision=1.0, num_predictions=4)
        >>> # allow empty result instead of raising
        >>> binary_precision(y_true=[], y_pred=[], raise_empty=False)
        BinaryPrecisionResult(precision=nan, num_predictions=0)

        ```
    """
    y_true, y_pred = preprocess_1d(
        arrays=[to_numpy_1d(y_true), to_numpy_1d(y_pred)],
        missing_policy=missing_policy,
        missing_values=missing_values,
    )

    num_predictions = y_true.size
    if num_predictions == 0:
        if raise_empty:
            msg = (
                "Cannot compute precision because 'y_true' and 'y_pred' are empty. "
                "Use 'raise_empty=False' to return a result with 'num_predictions=0' instead."
            )
            raise EmptyMetricError(msg)
        return BinaryPrecisionResult(
            precision=float("nan"),
            num_predictions=0,
        )

    # When missing_policy is 'propagate', check for missing values in
    # the arrays. When 'omit', rows have already been dropped so this
    # check always returns False. When 'raise', preprocess_1d already
    # raised if missing values were present.
    has_missing = (
        missing_values is not NOT_SET
        and missing_policy == "propagate"
        and multi_contains_value([y_true, y_pred], value=missing_values)
    )
    if has_missing:
        return BinaryPrecisionResult(
            precision=float("nan"),
            num_predictions=num_predictions,
        )

    return BinaryPrecisionResult(
        precision=float(
            precision_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average="binary",
                zero_division=0.0,
            )
        ),
        num_predictions=num_predictions,
    )
