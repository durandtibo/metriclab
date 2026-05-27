from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from metriclab.exceptions import EmptyMetricError
from metriclab.functional import binary_precision
from metriclab.results import BinaryPrecisionResult

##################################
#   Tests for binary_precision   #
##################################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 0, 1, 1]),
            BinaryPrecisionResult(precision=1.0, num_predictions=5),
            id="all-correct",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([0, 1, 1, 0, 0]),
            BinaryPrecisionResult(precision=0.0, num_predictions=5),
            id="all-incorrect",
        ),
        pytest.param(
            np.array([1, 0, 0, 1, 1]),
            np.array([1, 0, 1, 1, 1]),
            BinaryPrecisionResult(precision=0.75, num_predictions=5),
            id="partial",
        ),
        pytest.param(
            np.array([1, 0, 0, 1]),
            np.array([0, 0, 0, 0]),
            BinaryPrecisionResult(precision=0.0, num_predictions=4),
            id="no-positive-predictions-zero-division",
        ),
        pytest.param(
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            BinaryPrecisionResult(precision=0.0, num_predictions=4),
            id="all-negative",
        ),
        pytest.param(
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            BinaryPrecisionResult(precision=1.0, num_predictions=4),
            id="all-positive-all-correct",
        ),
        pytest.param(
            np.array([1]),
            np.array([1]),
            BinaryPrecisionResult(precision=1.0, num_predictions=1),
            id="single-correct",
        ),
        pytest.param(
            np.array([1]),
            np.array([0]),
            BinaryPrecisionResult(precision=0.0, num_predictions=1),
            id="single-incorrect",
        ),
    ],
)
def test_binary_precision_basic(
    y_true: np.ndarray, y_pred: np.ndarray, expected: BinaryPrecisionResult
) -> None:
    assert binary_precision(y_true=y_true, y_pred=y_pred).allclose(expected)


# --- input types ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), id="numpy"),
        pytest.param([1, 0, 1, 0], [1, 0, 1, 0], id="list"),
        pytest.param((1, 0, 1, 0), (1, 0, 1, 0), id="tuple"),
    ],
)
def test_binary_precision_input_types(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    assert binary_precision(y_true=y_true, y_pred=y_pred).allclose(
        BinaryPrecisionResult(precision=1.0, num_predictions=4)
    )


# --- pos_label ---


def test_binary_precision_pos_label_default() -> None:
    assert binary_precision(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 1, 0]),
    ).allclose(BinaryPrecisionResult(precision=1.0, num_predictions=4))


def test_binary_precision_pos_label_zero() -> None:
    # swap positive class to 0
    assert binary_precision(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 1, 0]),
        pos_label=0,
    ).allclose(BinaryPrecisionResult(precision=1.0, num_predictions=4))


def test_binary_precision_pos_label_string() -> None:
    assert binary_precision(
        y_true=["cat", "dog", "cat", "dog", "dog"],
        y_pred=["cat", "dog", "dog", "dog", "dog"],
        pos_label="dog",
    ).allclose(BinaryPrecisionResult(precision=0.75, num_predictions=5))


def test_binary_precision_pos_label_affects_result() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])
    # pos_label=1: TP=2, FP=1 → precision=2/3
    result_1 = binary_precision(y_true=y_true, y_pred=y_pred, pos_label=1)
    # pos_label=0: TP=1, FP=0 → precision=1/1
    result_0 = binary_precision(y_true=y_true, y_pred=y_pred, pos_label=0)
    assert not result_1.allclose(result_0)


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
def test_binary_precision_propagate_returns_nan(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any
) -> None:
    result = binary_precision(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="propagate",
        missing_values=missing_values,
    )
    assert math.isnan(result.precision)
    assert result.num_predictions == len(y_true)


def test_binary_precision_propagate_keeps_num_predictions() -> None:
    result = binary_precision(
        y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.num_predictions == 5
    assert math.isnan(result.precision)


def test_binary_precision_propagate_missing_values_not_set() -> None:
    result = binary_precision(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
    )
    assert result.allclose(BinaryPrecisionResult(precision=1.0, num_predictions=4))


def test_binary_precision_propagate_no_missing_computes_correctly() -> None:
    result = binary_precision(
        y_true=np.array([1.0, 0.0, 0.0, 1.0]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0]),
        missing_policy="propagate",
        missing_values=float("nan"),
    )
    assert result.allclose(BinaryPrecisionResult(precision=1.0, num_predictions=4))


# --- missing_policy='omit' ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "missing_values", "expected"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("nan"),
            BinaryPrecisionResult(precision=1.0, num_predictions=3),
            id="nan-in-y_true",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            BinaryPrecisionResult(precision=1.0, num_predictions=3),
            id="nan-in-y_pred",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 0.0, 1.0]),
            np.array([1.0, float("nan"), 0.0, 1.0]),
            float("nan"),
            BinaryPrecisionResult(precision=1.0, num_predictions=3),
            id="nan-in-both",
        ),
        pytest.param(
            np.array([1.0, float("inf"), 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            float("inf"),
            BinaryPrecisionResult(precision=1.0, num_predictions=3),
            id="inf-in-y_true",
        ),
        pytest.param(
            np.array([1, 99, 0, 1]),
            np.array([1, 0, 0, 1]),
            99,
            BinaryPrecisionResult(precision=1.0, num_predictions=3),
            id="int-sentinel",
        ),
    ],
)
def test_binary_precision_omit(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any, expected: BinaryPrecisionResult
) -> None:
    assert binary_precision(
        y_true=y_true,
        y_pred=y_pred,
        missing_policy="omit",
        missing_values=missing_values,
    ).allclose(expected)


def test_binary_precision_omit_all_missing_returns_empty() -> None:
    result = binary_precision(
        y_true=np.array([float("nan"), float("nan")]),
        y_pred=np.array([1.0, 0.0]),
        missing_policy="omit",
        missing_values=float("nan"),
        raise_empty=False,
    )
    assert result.allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=0),
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
def test_binary_precision_raise_with_missing_raises(
    y_true: np.ndarray, y_pred: np.ndarray, missing_values: Any
) -> None:
    with pytest.raises(ValueError, match=r"arrays contain at least one missing value"):
        binary_precision(
            y_true=y_true,
            y_pred=y_pred,
            missing_policy="raise",
            missing_values=missing_values,
        )


def test_binary_precision_raise_no_missing_computes_correctly() -> None:
    result = binary_precision(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([1, 0, 0, 1]),
        missing_policy="raise",
        missing_values=float("nan"),
    )
    assert result.allclose(BinaryPrecisionResult(precision=1.0, num_predictions=4))


# --- raise_empty ---


def test_binary_precision_raise_empty_true_raises() -> None:
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute precision because 'y_true' and 'y_pred' are empty"
    ):
        binary_precision(y_true=np.array([]), y_pred=np.array([]), raise_empty=True)


def test_binary_precision_raise_empty_error_message() -> None:
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute precision because 'y_true' and 'y_pred' are empty"
    ):
        binary_precision(y_true=np.array([]), y_pred=np.array([]), raise_empty=True)


def test_binary_precision_raise_empty_false_returns_result() -> None:
    result = binary_precision(
        y_true=np.array([], dtype=float),
        y_pred=np.array([], dtype=float),
        raise_empty=False,
    )
    assert result.allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=0),
        equal_nan=True,
    )


def test_binary_precision_raise_empty_after_omit_raises() -> None:
    with pytest.raises(
        EmptyMetricError, match=r"Cannot compute precision because 'y_true' and 'y_pred' are empty"
    ):
        binary_precision(
            y_true=np.array([float("nan"), float("nan")]),
            y_pred=np.array([1.0, 0.0]),
            missing_policy="omit",
            missing_values=float("nan"),
            raise_empty=True,
        )


def test_binary_precision_raise_empty_after_omit_false_returns_result() -> None:
    result = binary_precision(
        y_true=np.array([float("nan"), float("nan")]),
        y_pred=np.array([1.0, 0.0]),
        missing_policy="omit",
        missing_values=float("nan"),
        raise_empty=False,
    )
    assert result.allclose(
        BinaryPrecisionResult(precision=float("nan"), num_predictions=0),
        equal_nan=True,
    )


# --- invalid missing_policy ---


def test_binary_precision_invalid_missing_policy_raises() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'missing_policy': invalid"):
        binary_precision(
            y_true=np.array([1, 0, 1]),
            y_pred=np.array([1, 0, 1]),
            missing_policy="invalid",
        )


# --- precision score ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_precision"),
    [
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            1.0,
            id="perfect",
        ),
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
            0.5,
            id="half",
        ),
        pytest.param(
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 0, 0]),
            0.0,
            id="zero-no-positive-predictions",
        ),
    ],
)
def test_binary_precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, expected_precision: float
) -> None:
    assert binary_precision(y_true=y_true, y_pred=y_pred).precision == expected_precision


# --- num_predictions ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_n"),
    [
        pytest.param(np.array([1, 0, 1, 0, 1]), np.array([1, 0, 1, 0, 1]), 5, id="five"),
        pytest.param(np.array([1]), np.array([1]), 1, id="one"),
        pytest.param(np.array([1, 0]), np.array([1, 0]), 2, id="two"),
    ],
)
def test_binary_precision_num_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, expected_n: int
) -> None:
    assert binary_precision(y_true=y_true, y_pred=y_pred).num_predictions == expected_n


# --- edge cases ---


def test_binary_precision_large_arrays() -> None:
    n = 100_000
    y_true = np.ones(n, dtype=int)
    y_pred = np.ones(n, dtype=int)
    result = binary_precision(y_true=y_true, y_pred=y_pred)
    assert result.allclose(BinaryPrecisionResult(precision=1.0, num_predictions=n))


def test_binary_precision_returns_float_precision() -> None:
    # precision should be a plain Python float, not np.float64
    result = binary_precision(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 0, 1, 0]),
    )
    assert isinstance(result.precision, float)
