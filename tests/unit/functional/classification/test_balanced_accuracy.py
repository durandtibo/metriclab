from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from metriclab.exceptions import EmptyMetricError
from metriclab.functional import balanced_accuracy
from metriclab.results import BalancedAccuracyResult

##################################
#  Tests for balanced_accuracy   #
##################################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=5),
            id="all-correct-int",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            BalancedAccuracyResult(balanced_accuracy=0.0, num_predictions=5),
            id="all-incorrect-int",
        ),
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 0, 1]),
            BalancedAccuracyResult(balanced_accuracy=0.5, num_predictions=4),
            id="partial-correct-int",
        ),
        pytest.param(
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4),
            id="imbalanced-all-correct",
        ),
        pytest.param(
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 0]),
            BalancedAccuracyResult(balanced_accuracy=0.5, num_predictions=4),
            id="imbalanced-miss-minority",
        ),
        pytest.param(
            np.array([1]),
            np.array([1]),
            BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=1),
            id="single-correct",
        ),
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=6),
            id="multiclass-all-correct",
        ),
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 2, 1, 0, 2, 1]),
            BalancedAccuracyResult(balanced_accuracy=1 / 3, num_predictions=6),
            id="multiclass-partial",
        ),
    ],
)
def test_balanced_accuracy_basic(
    y_true: np.ndarray, y_pred: np.ndarray, expected: BalancedAccuracyResult
) -> None:
    assert balanced_accuracy(y_true=y_true, y_pred=y_pred).allclose(expected)


# --- input types ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), id="numpy"),
        pytest.param([1, 0, 1, 0], [1, 0, 1, 0], id="list"),
        pytest.param((1, 0, 1, 0), (1, 0, 1, 0), id="tuple"),
    ],
)
def test_balanced_accuracy_input_types(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    assert balanced_accuracy(y_true=y_true, y_pred=y_pred).allclose(
        BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4)
    )


# --- string labels ---


def test_balanced_accuracy_str_labels_all_correct() -> None:
    assert balanced_accuracy(
        y_true=["cat", "dog", "cat", "dog"],
        y_pred=["cat", "dog", "cat", "dog"],
    ).allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4))


def test_balanced_accuracy_str_labels_partial() -> None:
    result = balanced_accuracy(
        y_true=["cat", "dog", "cat", "dog"],
        y_pred=["cat", "dog", "dog", "dog"],
    )
    assert result.num_predictions == 4
    assert result.balanced_accuracy == 0.75


# --- sklearn alignment ---


def test_balanced_accuracy_matches_sklearn() -> None:
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 2, 2, 0])
    expected = float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
    result = balanced_accuracy(y_true=y_true, y_pred=y_pred)
    assert result.allclose(
        BalancedAccuracyResult(balanced_accuracy=expected, num_predictions=len(y_true))
    )


def test_balanced_accuracy_imbalanced_matches_sklearn() -> None:
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.array([0, 0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    expected = float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
    result = balanced_accuracy(y_true=y_true, y_pred=y_pred)
    assert result.allclose(
        BalancedAccuracyResult(balanced_accuracy=expected, num_predictions=len(y_true))
    )


# --- missing_policy='propagate' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "missing_values"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("nan"),
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, float("nan"), 0.0]),
            float("nan"),
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 0.0]),
            np.array([1.0, float("nan"), 0.0]),
            float("nan"),
            id="nan-in-both",
        ),
        pytest.param(
            np.array([1, None, 0], dtype=object),
            np.array([1, 0, 0], dtype=object),
            None,
            id="none-in-y_true",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("inf"),
            id="inf-in-y_true",
        ),
        pytest.param(
            np.array([1, 99, 0]),
            np.array([1, 0, 0]),
            99,
            id="int-sentinel-in-y_true",
        ),
    ],
)
def test_balanced_accuracy_propagate_returns_nan(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any
) -> None:
    result = balanced_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="propagate",
        missing_values=missing_values,
    )
    assert math.isnan(result.balanced_accuracy)
    assert result.num_predictions == len(y_true)


def test_balanced_accuracy_propagate_keeps_num_predictions() -> None:
    result = balanced_accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.num_predictions == 5
    assert math.isnan(result.balanced_accuracy)


def test_balanced_accuracy_propagate_missing_values_not_set() -> None:
    result = balanced_accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
    )
    assert result.allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4))


def test_balanced_accuracy_propagate_no_missing_computes_correctly() -> None:
    result = balanced_accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4))


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "missing_values", "num_predictions"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("nan"),
            3,
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            3,
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            3,
            id="nan-in-both",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("inf"),
            3,
            id="inf-in-y_true",
        ),
        pytest.param(
            np.array([1, 99, 0, 1]),
            np.array([1, 0, 0, 1]),
            99,
            3,
            id="int-sentinel",
        ),
    ],
)
def test_balanced_accuracy_omit(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any, num_predictions: int
) -> None:
    result = balanced_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="omit",
        missing_values=missing_values,
        raise_empty=False,
    )
    assert result.num_predictions == num_predictions
    assert not math.isnan(result.balanced_accuracy)


def test_balanced_accuracy_omit_all_correct_after_drop() -> None:
    result = balanced_accuracy(
        y_true=np.array([1.0, float("nan"), 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="omit",
        missing_values=float("nan"),
    )
    assert result.allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=3))


def test_balanced_accuracy_omit_missing_values_not_set() -> None:
    with pytest.raises(ValueError, match="Input y_true contains NaN"):
        balanced_accuracy(
            y_true=np.array([1.0, float("nan"), 0.0]),
            y_pred=np.array([1.0, 0.0, 0.0]),
            missing_policy="omit",
        )


def test_balanced_accuracy_omit_all_missing_returns_empty() -> None:
    result = balanced_accuracy(
        y_true=np.array([float("nan"), float("nan")]),
        y_pred=np.array([1.0, 0.0]),
        missing_policy="omit",
        missing_values=float("nan"),
        raise_empty=False,
    )
    assert result.allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=0),
        equal_nan=True,
    )


# --- missing_policy='raise' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "missing_values"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("nan"),
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, float("nan"), 0.0]),
            float("nan"),
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1, None, 0], dtype=object),
            np.array([1, 0, 0], dtype=object),
            None,
            id="none-in-y_true",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("inf"),
            id="inf-in-y_true",
        ),
    ],
)
def test_balanced_accuracy_raise_with_missing_raises(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any
) -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        balanced_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            missing_policy="raise",
            missing_values=missing_values,
        )


def test_balanced_accuracy_raise_no_missing_computes_correctly() -> None:
    result = balanced_accuracy(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([1, 0, 0, 1]),
        missing_policy="raise",
        missing_values=float("nan"),
    )
    assert result.allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=4))


# --- raise_empty ---


def test_balanced_accuracy_raise_empty_true_raises() -> None:
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute accuracy because 'y_true' and 'y_pred' are empty"
    ):
        balanced_accuracy(y_true=np.array([]), y_pred=np.array([]), raise_empty=True)


def test_balanced_accuracy_raise_empty_false_returns_result() -> None:
    result = balanced_accuracy(
        y_true=np.array([], dtype=float),
        y_pred=np.array([], dtype=float),
        raise_empty=False,
    )
    assert result.allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=0),
        equal_nan=True,
    )


def test_balanced_accuracy_raise_empty_after_omit_raises() -> None:
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute accuracy because 'y_true' and 'y_pred' are empty"
    ):
        balanced_accuracy(
            y_true=np.array([float("nan"), float("nan")]),
            y_pred=np.array([1.0, 0.0]),
            missing_policy="omit",
            missing_values=float("nan"),
            raise_empty=True,
        )


def test_balanced_accuracy_raise_empty_after_omit_false_returns_result() -> None:
    result = balanced_accuracy(
        y_true=np.array([float("nan"), float("nan")]),
        y_pred=np.array([1.0, 0.0]),
        missing_policy="omit",
        missing_values=float("nan"),
        raise_empty=False,
    )
    assert result.allclose(
        BalancedAccuracyResult(balanced_accuracy=float("nan"), num_predictions=0),
        equal_nan=True,
    )


# --- invalid missing_policy ---


def test_balanced_accuracy_invalid_missing_policy_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"Incorrect 'missing_policy': invalid. The valid values are: 'omit', 'propagate', 'raise'",
    ):
        balanced_accuracy(
            y_true=np.array([1, 0, 1]),
            y_pred=np.array([1, 0, 1]),
            missing_policy="invalid",
        )


# --- num_predictions ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_num_predictions"),
    [
        pytest.param(np.array([1, 0, 1, 0, 1]), np.array([1, 0, 1, 0, 1]), 5, id="five"),
        pytest.param(np.array([1]), np.array([1]), 1, id="one"),
        pytest.param(np.array([1, 0]), np.array([1, 0]), 2, id="two"),
    ],
)
def test_balanced_accuracy_num_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, expected_num_predictions: int
) -> None:
    assert (
        balanced_accuracy(y_true=y_true, y_pred=y_pred).num_predictions == expected_num_predictions
    )


# --- edge cases ---


def test_balanced_accuracy_large_arrays() -> None:
    n = 100_000
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    result = balanced_accuracy(y_true=y_true, y_pred=y_pred)
    assert result.allclose(BalancedAccuracyResult(balanced_accuracy=1.0, num_predictions=n))


def test_balanced_accuracy_all_same_label() -> None:
    result = balanced_accuracy(
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
    )
    assert result.num_predictions == 4
    assert not math.isnan(result.balanced_accuracy)
