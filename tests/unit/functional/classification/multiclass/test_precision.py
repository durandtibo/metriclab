from __future__ import annotations

import numpy as np
import pytest

from metriclab.functional.classification.multiclass.precision import (
    compute_multiclass_precision,
)
from metriclab.results import MulticlassPrecisionResult

##################################################
#     Tests for compute_multiclass_precision     #
##################################################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            MulticlassPrecisionResult(
                macro_precision=1.0,
                micro_precision=1.0,
                weighted_precision=1.0,
                per_class_precision=np.array([1.0, 1.0, 1.0]),
                support=np.array([2, 2, 2]),
                num_predictions=6,
            ),
            id="all-correct",
        ),
        pytest.param(
            np.array([0, 1, 2]),
            np.array([1, 2, 0]),
            MulticlassPrecisionResult(
                macro_precision=0.0,
                micro_precision=0.0,
                weighted_precision=0.0,
                per_class_precision=np.array([0.0, 0.0, 0.0]),
                support=np.array([1, 1, 1]),
                num_predictions=3,
            ),
            id="all-incorrect",
        ),
        pytest.param(
            np.array([0, 0, 1, 2]),
            np.array([0, 1, 2, 2]),
            MulticlassPrecisionResult(
                macro_precision=0.5,
                micro_precision=0.5,
                weighted_precision=0.625,
                per_class_precision=np.array([1.0, 0.0, 0.5]),
                support=np.array([2, 1, 1]),
                num_predictions=4,
            ),
            id="partial-correctness-3-classes",
        ),
        pytest.param(
            np.array([0, 1, 2, 3]),
            np.array([0, 2, 2, 3]),
            MulticlassPrecisionResult(
                macro_precision=0.625,
                micro_precision=0.75,
                weighted_precision=0.625,
                per_class_precision=np.array([1.0, 0.0, 0.5, 1.0]),
                support=np.array([1, 1, 1, 1]),
                num_predictions=4,
            ),
            id="partial-correctness-4-classes",
        ),
        pytest.param(
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 0, 0]),
            MulticlassPrecisionResult(
                macro_precision=0.25,
                micro_precision=0.5,
                weighted_precision=0.25,
                per_class_precision=np.array([0.5, 0.0]),
                support=np.array([2, 2]),
                num_predictions=4,
            ),
            id="zero-division-unpredicted-class",
        ),
        pytest.param(
            np.array([0]),
            np.array([0]),
            MulticlassPrecisionResult(
                macro_precision=1.0,
                micro_precision=1.0,
                weighted_precision=1.0,
                per_class_precision=np.array([1.0]),
                support=np.array([1]),
                num_predictions=1,
            ),
            id="single-sample-correct",
        ),
    ],
)
def test_compute_multiclass_precision_basic(
    y_true: np.ndarray, y_pred: np.ndarray, expected: MulticlassPrecisionResult
) -> None:
    assert compute_multiclass_precision(y_true=y_true, y_pred=y_pred).allclose(expected)


# --- input types ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), id="numpy"),
        pytest.param([0, 1, 0, 1], [0, 1, 0, 1], id="list"),
        pytest.param((0, 1, 0, 1), (0, 1, 0, 1), id="tuple"),
    ],
)
def test_compute_multiclass_precision_input_types(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    assert compute_multiclass_precision(y_true=y_true, y_pred=y_pred).allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0]),
            support=np.array([2, 2]),
            num_predictions=4,
        )
    )


# --- empty array handling ---


def test_compute_multiclass_precision_empty_arrays() -> None:
    result = compute_multiclass_precision(y_true=np.array([]), y_pred=np.array([]))
    expected = MulticlassPrecisionResult(
        macro_precision=0.0,
        micro_precision=0.0,
        weighted_precision=0.0,
        per_class_precision=np.array([], dtype=np.float64),
        support=np.array([], dtype=np.int64),
        num_predictions=0,
    )
    assert result.allclose(expected)


# --- type checking for elements ---


def test_compute_multiclass_precision_scalar_types() -> None:
    result = compute_multiclass_precision(
        y_true=np.array([0, 1, 2]),
        y_pred=np.array([0, 1, 2]),
    )
    assert result.equal(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            support=np.array([1, 1, 1], dtype=np.int64),
            num_predictions=3,
        )
    )
    assert isinstance(result.macro_precision, float)
    assert isinstance(result.micro_precision, float)
    assert isinstance(result.weighted_precision, float)
    assert isinstance(result.num_predictions, int)
    assert isinstance(result.per_class_precision, np.ndarray)
    assert isinstance(result.support, np.ndarray)


# --- edge cases ---


def test_compute_multiclass_precision_large_arrays() -> None:
    n = 30_000
    # 3 classes evenly distributed
    y_true = np.repeat([0, 1, 2], n // 3)
    y_pred = np.repeat([0, 1, 2], n // 3)
    result = compute_multiclass_precision(y_true=y_true, y_pred=y_pred)
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0]),
            support=np.array([10000, 10000, 10000]),
            num_predictions=n,
        )
    )


def test_compute_multiclass_precision_string_labels() -> None:
    y_true = np.array(["apple", "banana", "apple", "cherry"])
    y_pred = np.array(["apple", "cherry", "apple", "cherry"])
    result = compute_multiclass_precision(y_true=y_true, y_pred=y_pred)

    # Classes are alphabetically sorted by confusion_matrix: apple, banana, cherry
    # apple: TP=2, FP=0 -> prec=1.0, support=2
    # banana: TP=0, FP=0 -> prec=0.0, support=1
    # cherry: TP=1, FP=1 -> prec=0.5, support=1
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.5,
            micro_precision=0.75,
            weighted_precision=0.625,
            per_class_precision=np.array([1.0, 0.0, 0.5]),
            support=np.array([2, 1, 1]),
            num_predictions=4,
        )
    )
