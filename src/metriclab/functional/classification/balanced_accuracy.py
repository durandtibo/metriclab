r"""Compute classification accuracy from array-like inputs.

This module powers :func:`metriclab.functional.balanced_accuracy` and
supports NumPy arrays, and Python sequences.
"""

from __future__ import annotations

__all__ = ["balanced_accuracy"]

from typing import TYPE_CHECKING, Any

from sklearn.metrics import balanced_accuracy_score

from metriclab.exceptions import EmptyMetricError
from metriclab.results import BalancedAccuracyResult
from metriclab.utils.array import multi_contains_value, preprocess_1d, to_numpy_1d
from metriclab.utils.sentinel import NOT_SET

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike
    from metriclab.utils.missing import MissingPolicy


def balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    missing_policy: MissingPolicy = "propagate",
    missing_values: Any = NOT_SET,
    raise_empty: bool = True,
) -> BalancedAccuracyResult:
    r"""Compute the accuracy score.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        missing_policy: The policy for handling missing values.
            Valid values are ``'omit'``, ``'propagate'``, or
            ``'raise'``. ``'omit'`` removes rows where any array
            equals ``missing_values`` before computing the metric.
            ``'propagate'`` keeps the rows but returns
            ``num_correct_predictions=nan`` when missing values are
            detected. ``'raise'`` raises a ``ValueError`` if any
            array contains ``missing_values``.
        missing_values: The value to treat as missing. If not set,
            missing value handling is disabled regardless of
            ``missing_policy``.
        raise_empty: If ``True``, raises ``EmptyMetricError`` when
            there are no valid predictions. If ``False``, returns an
            ``BalancedAccuracyResult`` with ``num_predictions=0``.

    Returns:
        The accuracy result. When missing values are present and
            ``missing_policy='propagate'``, the result keeps
            ``num_predictions`` and sets ``num_correct_predictions`` to
            ``nan``.

    Raises:
        EmptyMetricError: if there are no valid predictions and
            ``raise_empty`` is ``True``.
        ValueError: if ``missing_policy`` is invalid.
        ValueError: if ``y_true`` or ``y_pred`` contains
            ``missing_values`` and ``missing_policy`` is ``'raise'``.

    Example:
    ```pycon
    >>> import numpy as np
    >>> from metriclab.functional import balanced_accuracy
    >>> # with numpy arrays
    >>> balanced_accuracy(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ... )
    BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=5)
    >>> # with lists
    >>> balanced_accuracy(y_true=[1, 0, 0, 1, 1], y_pred=[1, 0, 1, 1, 1])
    BalancedAccuracyResult(balanced_accuracy=0.75, num_predictions=5)
    >>> # with string labels
    >>> balanced_accuracy(
    ...     y_true=["cat", "dog", "cat", "dog"],
    ...     y_pred=["cat", "dog", "dog", "dog"],
    ... )
    BalancedAccuracyResult(balanced_accuracy=0.75, num_predictions=4)
    >>> # with missing values and missing_policy='propagate' (default)
    >>> balanced_accuracy(
    ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
    ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
    ...     missing_values=float("nan"),
    ... )
    BalancedAccuracyResult(balanced_accuracy=nan, num_predictions=5)
    >>> # with missing values and missing_policy='omit'
    >>> balanced_accuracy(
    ...     y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
    ...     y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
    ...     missing_policy="omit",
    ...     missing_values=float("nan"),
    ... )
    BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4)
    >>> # allow empty result instead of raising
    >>> balanced_accuracy(y_true=[], y_pred=[], raise_empty=False)
    BalancedAccuracyResult(balanced_accuracy=nan, num_predictions=0)

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
                "Cannot compute accuracy because 'y_true' and 'y_pred' are empty. "
                "Use 'raise_empty=False' to return a result with 'num_predictions=0' instead."
            )
            raise EmptyMetricError(msg)
        return BalancedAccuracyResult(
            balanced_accuracy=float("nan"),
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
        return BalancedAccuracyResult(
            balanced_accuracy=float("nan"),
            num_predictions=num_predictions,
        )

    return BalancedAccuracyResult(
        balanced_accuracy=float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred)),
        num_predictions=num_predictions,
    )
