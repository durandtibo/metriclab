from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from metriclab.exceptions import EmptyMetricError
from metriclab.functional import multiclass_precision
from metriclab.functional.classification.multiclass.precision import (
    UndefinedPrecisionValue,
    _resolve_fill_value,
    _validate_undefined_precision,
    compute_multiclass_precision,
)
from metriclab.results import MulticlassPrecisionResult

if TYPE_CHECKING:
    from metriclab.typing import ArrayLike

##################################################
#        Tests for multiclass_precision          #
##################################################


# --- basic correctness ---


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]),
            id="numpy",
        ),
        pytest.param(
            [0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2],
            id="list",
        ),
        pytest.param(
            (0, 1, 2, 0, 1, 2),
            (0, 1, 2, 0, 1, 2),
            id="tuple",
        ),
    ],
)
def test_multiclass_precision_all_correct(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    result = multiclass_precision(y_true=y_true, y_pred=y_pred)
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0]),
            support=np.array([2, 2, 2]),
            num_predictions=6,
        )
    )


def test_multiclass_precision_with_string_labels() -> None:
    result = multiclass_precision(
        y_true=["cat", "bear", "cat", "cat", "bear", "dog"],
        y_pred=["cat", "bear", "cat", "cat", "bear", "dog"],
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0]),
            support=np.array([2, 3, 1]),
            num_predictions=6,
        )
    )


# --- raise_empty ---


def test_multiclass_precision_raise_empty_true() -> None:
    with pytest.raises(
        EmptyMetricError, match="Cannot compute precision because 'y_true' and 'y_pred' are empty"
    ):
        multiclass_precision(y_true=[], y_pred=[])


def test_multiclass_precision_raise_empty_false() -> None:
    result = multiclass_precision(y_true=[], y_pred=[], raise_empty=False)
    assert result.equal(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            weighted_precision=float("nan"),
            per_class_precision=np.array([], dtype=np.float64),
            support=np.array([], dtype=np.int64),
            num_predictions=0,
        ),
        equal_nan=True,
    )


# --- missing values ---


def test_multiclass_precision_missing_propagate_returns_nan() -> None:
    # Two classes (0, 1) are present in the non-missing rows.
    result = multiclass_precision(
        y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        missing_values=float("nan"),
    )
    assert result.equal(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            weighted_precision=float("nan"),
            per_class_precision=np.array([float("nan"), float("nan")]),
            support=np.array([float("nan"), float("nan")]),
            num_predictions=5,
        ),
        equal_nan=True,
    )


def test_multiclass_precision_missing_omit_drops_rows() -> None:
    result = multiclass_precision(
        y_true=np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
        y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        missing_policy="omit",
        missing_values=float("nan"),
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0]),
            support=np.array([2, 2]),
            num_predictions=4,
        )
    )


def test_multiclass_precision_missing_raise_raises() -> None:
    with pytest.raises(ValueError, match="arrays contain at least one missing value"):
        multiclass_precision(
            y_true=np.array([1.0, float("nan")]),
            y_pred=np.array([1.0, 1.0]),
            missing_policy="raise",
            missing_values=float("nan"),
        )


def test_multiclass_precision_no_missing_values_set_ignores_policy() -> None:
    # missing_values not set: missing_policy is irrelevant, no rows dropped.
    result = multiclass_precision(
        y_true=np.array([0, 1, 2]),
        y_pred=np.array([0, 1, 2]),
        missing_policy="omit",
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0]),
            support=np.array([1, 1, 1]),
            num_predictions=3,
        )
    )


# --- undefined_precision forwarded correctly ---


def test_multiclass_precision_undefined_precision_warn_forwarded() -> None:
    with pytest.warns(UserWarning, match="Precision is undefined"):
        result = multiclass_precision(
            y_true=[0, 0, 1, 2],
            y_pred=[0, 0, 1, 1],
            undefined_precision="warn",
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.5,
            micro_precision=0.75,
            weighted_precision=0.625,
            per_class_precision=np.array([1.0, 0.5, 0.0]),
            support=np.array([2, 1, 1]),
            num_predictions=4,
        )
    )


def test_multiclass_precision_undefined_precision_zero_no_warning() -> None:
    result = multiclass_precision(
        y_true=[0, 0, 1, 2],
        y_pred=[0, 0, 1, 1],
        undefined_precision=0.0,
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.5,
            micro_precision=0.75,
            weighted_precision=0.625,
            per_class_precision=np.array([1.0, 0.5, 0.0]),
            support=np.array([2, 1, 1]),
            num_predictions=4,
        )
    )


def test_multiclass_precision_undefined_precision_nan_propagates() -> None:
    result = multiclass_precision(
        y_true=[0, 0, 1, 2],
        y_pred=[0, 0, 1, 1],
        undefined_precision=float("nan"),
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=0.75,
            weighted_precision=float("nan"),
            per_class_precision=np.array([1.0, 0.5, float("nan")]),
            support=np.array([2, 1, 1]),
            num_predictions=4,
        ),
        equal_nan=True,
    )


def test_multiclass_precision_invalid_undefined_precision_raises() -> None:
    with pytest.raises(ValueError, match="Invalid 'undefined_precision' value"):
        multiclass_precision(
            y_true=[0, 1],
            y_pred=[0, 1],
            undefined_precision="bad",
        )


# --- validation is eagerly applied before preprocessing ---


def test_multiclass_precision_validates_undefined_precision_before_preprocessing() -> None:
    # Even with valid arrays, an invalid undefined_precision should raise immediately.
    with pytest.raises(ValueError, match="Invalid 'undefined_precision' value"):
        multiclass_precision(
            y_true=[0, 1, 2],
            y_pred=[0, 1, 2],
            undefined_precision=99,
        )


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


# --- undefined_precision parameter ---


def test_compute_multiclass_precision_undefined_warn_emits_warning() -> None:
    # Class 1 is never predicted, so precision is undefined for it.
    with pytest.warns(UserWarning, match="Precision is undefined"):
        result = compute_multiclass_precision(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 0, 0, 0]),
            undefined_precision="warn",
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.25,
            micro_precision=0.5,
            weighted_precision=0.25,
            per_class_precision=np.array([0.5, 0.0]),
            support=np.array([2, 2]),
            num_predictions=4,
        )
    )


def test_compute_multiclass_precision_undefined_warn_lists_affected_classes() -> None:
    with pytest.warns(UserWarning, match=r"\[2\]"):
        compute_multiclass_precision(
            y_true=np.array([0, 1, 2]),
            y_pred=np.array([0, 1, 1]),
            undefined_precision="warn",
        )


def test_compute_multiclass_precision_undefined_zero_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_multiclass_precision(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 0, 0, 0]),
            undefined_precision=0.0,
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.25,
            micro_precision=0.5,
            weighted_precision=0.25,
            per_class_precision=np.array([0.5, 0.0]),
            support=np.array([2, 2]),
            num_predictions=4,
        )
    )


def test_compute_multiclass_precision_undefined_one_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_multiclass_precision(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 0, 0, 0]),
            undefined_precision=1.0,
        )
    # Class 1: undefined -> filled with 1.0; macro = (0.5 + 1.0) / 2 = 0.75
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=0.75,
            micro_precision=0.5,
            weighted_precision=0.75,
            per_class_precision=np.array([0.5, 1.0]),
            support=np.array([2, 2]),
            num_predictions=4,
        )
    )


def test_compute_multiclass_precision_undefined_nan_propagates() -> None:
    # micro is derived from global TP/PP counts and is unaffected by the
    # per-class fill; macro and weighted receive NaN via np.mean/np.average.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_multiclass_precision(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 0, 0, 0]),
            undefined_precision=float("nan"),
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=0.5,
            weighted_precision=float("nan"),
            per_class_precision=np.array([0.5, float("nan")]),
            support=np.array([2, 2]),
            num_predictions=4,
        ),
        equal_nan=True,
    )


def test_compute_multiclass_precision_undefined_nan_string_alias() -> None:
    # "nan" string alias must produce identical output to float("nan").
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_multiclass_precision(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 0, 0, 0]),
            undefined_precision="nan",
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=0.5,
            weighted_precision=float("nan"),
            per_class_precision=np.array([0.5, float("nan")]),
            support=np.array([2, 2]),
            num_predictions=4,
        ),
        equal_nan=True,
    )


def test_compute_multiclass_precision_no_undefined_no_warning() -> None:
    # All classes are predicted; no undefined precision; no warning expected.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_multiclass_precision(
            y_true=np.array([0, 1, 2]),
            y_pred=np.array([0, 1, 2]),
        )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0, 1.0, 1.0]),
            support=np.array([1, 1, 1]),
            num_predictions=3,
        )
    )


# --- invalid undefined_precision ---


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("zero", id="string-zero"),
        pytest.param("raise", id="string-raise"),
        pytest.param(None, id="none"),
        pytest.param(1, id="int"),
        pytest.param([0.0], id="list"),
    ],
)
def test_compute_multiclass_precision_invalid_undefined_precision(bad_value: object) -> None:
    with pytest.raises(ValueError, match="Invalid 'undefined_precision' value"):
        compute_multiclass_precision(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            undefined_precision=bad_value,
        )


# --- empty array handling ---


def test_compute_multiclass_precision_empty_arrays() -> None:
    result = compute_multiclass_precision(y_true=np.array([]), y_pred=np.array([]))
    assert result.equal(
        MulticlassPrecisionResult(
            macro_precision=float("nan"),
            micro_precision=float("nan"),
            weighted_precision=float("nan"),
            per_class_precision=np.array([], dtype=np.float64),
            support=np.array([], dtype=np.int64),
            num_predictions=0,
        ),
        equal_nan=True,
    )


# --- return type checking ---


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
    # Classes are alphabetically sorted by confusion_matrix: apple, banana, cherry.
    # apple:  TP=2, FP=0 -> prec=1.0, support=2
    # banana: TP=0, FP=0 -> undefined (filled with 0.0), support=1
    # cherry: TP=1, FP=1 -> prec=0.5, support=1
    y_true = np.array(["apple", "banana", "apple", "cherry"])
    y_pred = np.array(["apple", "cherry", "apple", "cherry"])
    with pytest.warns(UserWarning, match="banana"):
        result = compute_multiclass_precision(y_true=y_true, y_pred=y_pred)
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


def test_compute_multiclass_precision_single_class() -> None:
    result = compute_multiclass_precision(
        y_true=np.array([0, 0, 0]),
        y_pred=np.array([0, 0, 0]),
    )
    assert result.allclose(
        MulticlassPrecisionResult(
            macro_precision=1.0,
            micro_precision=1.0,
            weighted_precision=1.0,
            per_class_precision=np.array([1.0]),
            support=np.array([3]),
            num_predictions=3,
        )
    )


##################################################
#      Tests for _validate_undefined_precision   #
##################################################


@pytest.mark.parametrize(
    "valid_value",
    [
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param(float("nan"), id="float-nan"),
        pytest.param(0.5, id="float-half"),
        pytest.param("nan", id="string-nan"),
        pytest.param("warn", id="string-warn"),
    ],
)
def test_validate_undefined_precision_valid(valid_value: UndefinedPrecisionValue) -> None:
    # Should not raise.
    _validate_undefined_precision(valid_value)


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param("zero", id="string-zero"),
        pytest.param("raise", id="string-raise"),
        pytest.param("omit", id="string-omit"),
        pytest.param(None, id="none"),
        pytest.param(1, id="int-one"),
        pytest.param(0, id="int-zero"),
        pytest.param([0.0], id="list"),
        pytest.param({"nan"}, id="set"),
    ],
)
def test_validate_undefined_precision_invalid(invalid_value: object) -> None:
    with pytest.raises(ValueError, match=r"Invalid 'undefined_precision' value"):
        _validate_undefined_precision(invalid_value)


##################################################
#        Tests for _resolve_fill_value           #
##################################################


def _make_mask(*flags: bool) -> np.ndarray:
    return np.array(flags, dtype=bool)


def _make_labels(*values: int) -> np.ndarray:
    return np.array(values)


def test_resolve_fill_value_no_undefined_returns_zero() -> None:
    # No undefined classes: fill is irrelevant, should return 0.0 cheaply.
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(False, False),
        labels=_make_labels(0, 1),
        undefined_precision="warn",
    )
    assert fill == 0.0


def test_resolve_fill_value_nan_string() -> None:
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(True, False),
        labels=_make_labels(0, 1),
        undefined_precision="nan",
    )
    assert fill != fill  # NaN identity check: NaN != NaN


def test_resolve_fill_value_warn_emits_warning_and_returns_zero() -> None:
    with pytest.warns(UserWarning, match="Precision is undefined"):
        fill = _resolve_fill_value(
            undefined_mask=_make_mask(False, True),
            labels=_make_labels(0, 1),
            undefined_precision="warn",
        )
    assert fill == 0.0


def test_resolve_fill_value_warn_lists_affected_labels() -> None:
    with pytest.warns(UserWarning, match=r"\[2, 5\]"):
        _resolve_fill_value(
            undefined_mask=_make_mask(False, True, False, True),
            labels=_make_labels(0, 2, 3, 5),
            undefined_precision="warn",
        )


def test_resolve_fill_value_warn_counts_affected_classes() -> None:
    with pytest.warns(UserWarning, match="2 class"):
        _resolve_fill_value(
            undefined_mask=_make_mask(True, True),
            labels=_make_labels(0, 1),
            undefined_precision="warn",
        )


def test_resolve_fill_value_float_zero() -> None:
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(True),
        labels=_make_labels(0),
        undefined_precision=0.0,
    )
    assert fill == 0.0


def test_resolve_fill_value_float_one() -> None:
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(True),
        labels=_make_labels(0),
        undefined_precision=1.0,
    )
    assert fill == 1.0


def test_resolve_fill_value_float_nan() -> None:
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(True),
        labels=_make_labels(0),
        undefined_precision=float("nan"),
    )
    assert fill != fill  # NaN identity check: NaN != NaN


def test_resolve_fill_value_arbitrary_float() -> None:
    fill = _resolve_fill_value(
        undefined_mask=_make_mask(True),
        labels=_make_labels(0),
        undefined_precision=0.5,
    )
    assert fill == 0.5
