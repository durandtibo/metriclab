from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from metriclab.exceptions import EmptyMetricError
from metriclab.functional import accuracy
from metriclab.results import AccuracyResult

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike

##############################
#     Tests for accuracy     #
##############################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            AccuracyResult(num_correct_predictions=5, num_predictions=5),
            id="all-correct-int",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=5),
            id="all-incorrect-int",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 1, 1, 1]),
            AccuracyResult(num_correct_predictions=4, num_predictions=5),
            id="partial-correct-int",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            AccuracyResult(num_correct_predictions=4, num_predictions=4),
            id="all-correct-float",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "dog"]),
            np.array(["cat", "dog", "dog", "dog"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=4),
            id="str-partial",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat"]),
            np.array(["cat", "dog", "cat"]),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="str-all-correct",
        ),
        pytest.param(
            np.array([True, False, True, False]),
            np.array([True, False, False, True]),
            AccuracyResult(num_correct_predictions=2, num_predictions=4),
            id="bool-partial",
        ),
        pytest.param(
            np.array([1]),
            np.array([1]),
            AccuracyResult(num_correct_predictions=1, num_predictions=1),
            id="single-correct",
        ),
        pytest.param(
            np.array([1]),
            np.array([0]),
            AccuracyResult(num_correct_predictions=0, num_predictions=1),
            id="single-incorrect",
        ),
    ],
)
def test_accuracy_basic(y_true: np.ndarray, y_pred: np.ndarray, expected: AccuracyResult) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(expected)


# --- input types ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 0, 1]),
            id="numpy",
        ),
        pytest.param([1, 0, 1, 0], [1, 0, 0, 1], id="list"),
        pytest.param((1, 0, 1, 0), (1, 0, 0, 1), id="tuple"),
    ],
)
def test_accuracy_input_types(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).equal(
        AccuracyResult(num_correct_predictions=2, num_predictions=4)
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
            np.array([1, 0, 0], dtype=object),
            np.array([1, None, 0], dtype=object),
            None,
            id="none-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("inf"),
            id="inf-in-y_true",
        ),
        pytest.param(
            np.array([1.0, float("-inf"), 0.0]),
            np.array([1.0, 0.0, 0.0]),
            float("-inf"),
            id="neg-inf-in-y_true",
        ),
        pytest.param(
            np.array([1, 99, 0]),
            np.array([1, 0, 0]),
            99,
            id="int-sentinel-in-y_true",
        ),
    ],
)
def test_accuracy_propagate_returns_nan(
    y_true: ArrayLike, y_pred: ArrayLike, missing_values: Any
) -> None:
    result = accuracy(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="propagate",
        missing_values=missing_values,
    )
    assert math.isnan(result.num_correct_predictions)
    assert result.num_predictions == len(y_true)


def test_accuracy_propagate_keeps_num_predictions() -> None:
    result = accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.num_predictions == 5
    assert math.isnan(result.num_correct_predictions)


def test_accuracy_propagate_missing_values_not_set() -> None:
    # missing_values not set — nan rows treated as regular values
    result = accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
    )
    assert result.equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


def test_accuracy_propagate_no_missing_computes_correctly() -> None:
    result = accuracy(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "missing_values", "expected"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("nan"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="nan-in-both",
        ),
        pytest.param(
            np.array([1, None, 0, 1], dtype=object),
            np.array([1, 0, 0, 1], dtype=object),
            None,
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="none-in-y_true",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("inf"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="inf-in-y_true",
        ),
        pytest.param(
            np.array([1.0, float("-inf"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("-inf"),
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="neg-inf-in-y_true",
        ),
        pytest.param(
            np.array([1, 99, 0, 1]),
            np.array([1, 0, 0, 1]),
            99,
            AccuracyResult(num_correct_predictions=3, num_predictions=3),
            id="int-sentinel-in-y_true",
        ),
        pytest.param(
            np.array([float("nan"), float("nan"), float("nan")]),
            np.array([1.0, 0.0, 1.0]),
            float("nan"),
            AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
            id="all-missing-omit-empty",
        ),
    ],
)
def test_accuracy_omit(
    y_true: ArrayLike, y_pred: ArrayLike, missing_values: Any, expected: AccuracyResult
) -> None:
    assert accuracy(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="omit",
        missing_values=missing_values,
        raise_empty=False,
    ).equal(expected, equal_nan=True)


def test_accuracy_omit_missing_values_not_set() -> None:
    # missing_values not set — nothing is omitted
    result = accuracy(
        y_true=np.array([1.0, float("nan"), 0.0]),
        y_pred=np.array([1.0, 0.0, 0.0]),
        missing_policy="omit",
    )
    assert result.num_predictions == 3


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
def test_accuracy_raise_with_missing_raises(
    y_true: ArrayLike, y_pred: ArrayLike, missing_values: Any
) -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        accuracy(
            y_true=y_true,
            y_pred=y_pred,
            missing_policy="raise",
            missing_values=missing_values,
        )


def test_accuracy_raise_no_missing_computes_correctly() -> None:
    result = accuracy(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([1, 0, 0, 1]),
        missing_policy="raise",
        missing_values=float("nan"),
    )
    assert result.equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


# --- raise_empty ---


def test_accuracy_raise_empty_true_raises() -> None:
    with pytest.raises(EmptyMetricError):
        accuracy(y_true=np.array([]), y_pred=np.array([]), raise_empty=True)


def test_accuracy_raise_empty_false_returns_result() -> None:
    result = accuracy(
        y_true=np.array([], dtype=float),
        y_pred=np.array([], dtype=float),
        raise_empty=False,
    )
    assert result.equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
        equal_nan=True,
    )


def test_accuracy_raise_empty_after_omit_raises() -> None:
    # All rows omitted — should raise EmptyMetricError
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute accuracy because 'y_true' and 'y_pred' are empty."
    ):
        accuracy(
            y_true=np.array([float("nan"), float("nan")]),
            y_pred=np.array([1.0, 0.0]),
            missing_policy="omit",
            missing_values=float("nan"),
            raise_empty=True,
        )


def test_accuracy_raise_empty_after_omit_false_returns_result() -> None:
    result = accuracy(
        y_true=np.array([float("nan"), float("nan")]),
        y_pred=np.array([1.0, 0.0]),
        missing_policy="omit",
        missing_values=float("nan"),
        raise_empty=False,
    )
    assert result.equal(
        AccuracyResult(num_correct_predictions=float("nan"), num_predictions=0),
        equal_nan=True,
    )


# --- invalid missing_policy ---


def test_accuracy_invalid_missing_policy_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"Incorrect 'missing_policy': invalid. The valid values are: 'omit', 'propagate', 'raise'",
    ):
        accuracy(
            y_true=np.array([1, 0, 1]),
            y_pred=np.array([1, 0, 1]),
            missing_policy="invalid",
        )


# --- accuracy property ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_accuracy"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            1.0,
            id="perfect",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 1, 0]),
            0.0,
            id="zero",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([1, 0, 1, 0]),
            0.5,
            id="half",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 1, 1, 1]),
            0.8,
            id="partial",
        ),
    ],
)
def test_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, expected_accuracy: float) -> None:
    assert accuracy(y_true=y_true, y_pred=y_pred).accuracy == expected_accuracy


# --- edge cases ---


def test_accuracy_large_arrays() -> None:
    n = 100_000
    y_true = np.ones(n, dtype=int)
    y_pred = np.ones(n, dtype=int)
    result = accuracy(y_true=y_true, y_pred=y_pred)
    assert result.equal(AccuracyResult(num_correct_predictions=n, num_predictions=n))


def test_accuracy_all_same_label() -> None:
    result = accuracy(
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
    )
    assert result.equal(AccuracyResult(num_correct_predictions=4, num_predictions=4))


def test_accuracy_preserves_order_of_result() -> None:
    result = accuracy(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 0, 1]),
    )
    assert result.num_correct_predictions == 2
    assert result.num_predictions == 4
