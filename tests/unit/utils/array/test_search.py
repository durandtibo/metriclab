from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from metriclab.utils.array import contains_value, multi_contains_value

##################################
#   Tests for contains_value     #
##################################


# --- integer arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1, 2, 3]), 2, True, id="int-present"),
        pytest.param(np.array([1, 2, 3]), 4, False, id="int-absent"),
        pytest.param(np.array([1, 2, 3]), 1, True, id="int-first"),
        pytest.param(np.array([1, 2, 3]), 3, True, id="int-last"),
        pytest.param(np.array([1, 1, 1]), 1, True, id="int-all-same-present"),
        pytest.param(np.array([1, 1, 1]), 2, False, id="int-all-same-absent"),
        pytest.param(np.array([1, 2, 3]), 2.0, True, id="int-array-float-value"),
        pytest.param(np.array([1, 2, 3]), 4.0, False, id="int-array-float-value-absent"),
    ],
)
def test_contains_value_int_array(arr: np.ndarray, value: Any, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- float arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), 2.0, True, id="float-present"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 4.0, False, id="float-absent"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 1.0, True, id="float-first"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 3.0, True, id="float-last"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 2, True, id="float-array-int-value"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 4, False, id="float-array-int-value-absent"),
        pytest.param(np.array([0.0, 1.0, 2.0]), 0.0, True, id="float-zero-present"),
        pytest.param(np.array([-1.0, 0.0, 1.0]), -1.0, True, id="float-negative-present"),
    ],
)
def test_contains_value_float_array(arr: np.ndarray, value: Any, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- NaN ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, float("nan"), 3.0]), True, id="nan-in-float-array"),
        pytest.param(np.array([1.0, 2.0, 3.0]), False, id="nan-absent-in-float-array"),
        pytest.param(np.array([float("nan"), float("nan")]), True, id="nan-all-nan-array"),
        pytest.param(np.array([float("nan")]), True, id="nan-single-nan-array"),
        pytest.param(np.array([float("nan"), 1.0, 2.0]), True, id="nan-at-start"),
        pytest.param(np.array([1.0, 2.0, float("nan")]), True, id="nan-at-end"),
    ],
)
def test_contains_value_nan_float_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_value(arr, float("nan")) == expected


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, float("nan"), 3], dtype=object), True, id="nan-in-object-array"),
        pytest.param(np.array([1, 2, 3], dtype=object), False, id="nan-absent-in-object-array"),
        pytest.param(
            np.array([None, float("nan"), "x"], dtype=object),
            True,
            id="nan-mixed-object-array",
        ),
        pytest.param(np.array([None, None], dtype=object), False, id="nan-only-none-object-array"),
    ],
)
def test_contains_value_nan_object_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_value(arr, float("nan")) == expected


def test_contains_value_nan_in_int_array() -> None:
    # int arrays cannot contain nan — always False
    assert not contains_value(np.array([1, 2, 3]), float("nan"))


def test_contains_value_nan_not_confused_with_none() -> None:
    assert not contains_value(np.array([None, None], dtype=object), float("nan"))


def test_contains_value_nan_not_confused_with_inf() -> None:
    assert not contains_value(np.array([float("inf"), float("-inf")]), float("nan"))


# --- inf ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1.0, float("inf"), 3.0]), float("inf"), True, id="pos-inf-present"),
        pytest.param(np.array([1.0, 2.0, 3.0]), float("inf"), False, id="pos-inf-absent"),
        pytest.param(
            np.array([1.0, float("-inf"), 3.0]), float("-inf"), True, id="neg-inf-present"
        ),
        pytest.param(np.array([1.0, 2.0, 3.0]), float("-inf"), False, id="neg-inf-absent"),
        pytest.param(np.full(10, float("inf")), float("inf"), True, id="all-pos-inf-present"),
        pytest.param(np.full(10, float("-inf")), float("-inf"), True, id="all-neg-inf-present"),
        pytest.param(np.array([float("inf"), 1.0, 2.0]), float("inf"), True, id="pos-inf-at-start"),
        pytest.param(np.array([1.0, 2.0, float("inf")]), float("inf"), True, id="pos-inf-at-end"),
    ],
)
def test_contains_value_inf_float_array(arr: np.ndarray, value: float, expected: bool) -> None:
    assert contains_value(arr, value) == expected


def test_contains_value_pos_inf_not_confused_with_neg_inf() -> None:
    assert not contains_value(np.array([float("-inf"), 1.0, 2.0]), float("inf"))


def test_contains_value_neg_inf_not_confused_with_pos_inf() -> None:
    assert not contains_value(np.array([float("inf"), 1.0, 2.0]), float("-inf"))


def test_contains_value_inf_not_confused_with_nan() -> None:
    assert not contains_value(np.array([float("nan"), 1.0, 2.0]), float("inf"))


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1, float("inf"), 3], dtype=object),
            float("inf"),
            True,
            id="pos-inf-in-object-array",
        ),
        pytest.param(
            np.array([1, float("-inf"), 3], dtype=object),
            float("-inf"),
            True,
            id="neg-inf-in-object-array",
        ),
        pytest.param(
            np.array([1, float("inf"), 3], dtype=object),
            float("-inf"),
            False,
            id="pos-inf-not-confused-with-neg-inf-object",
        ),
    ],
)
def test_contains_value_inf_object_array(arr: np.ndarray, value: float, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- None ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, None, 3], dtype=object), True, id="none-present"),
        pytest.param(np.array([1, 2, 3], dtype=object), False, id="none-absent"),
        pytest.param(np.array([None, None, None], dtype=object), True, id="all-none"),
        pytest.param(np.array([None], dtype=object), True, id="single-none"),
        pytest.param(np.array([None, 1, 2], dtype=object), True, id="none-at-start"),
        pytest.param(np.array([1, 2, None], dtype=object), True, id="none-at-end"),
        pytest.param(
            np.array([None, float("nan"), "x"], dtype=object), True, id="none-mixed-object"
        ),
    ],
)
def test_contains_value_none_object_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_value(arr, None) == expected


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1, 2, 3]), id="int-array"),
        pytest.param(np.array([1.0, 2.0, 3.0]), id="float-array"),
        pytest.param(np.array(["a", "b", "c"]), id="str-array"),
    ],
)
def test_contains_value_none_in_non_object_array(arr: np.ndarray) -> None:
    assert not contains_value(arr, None)


def test_contains_value_none_not_confused_with_zero() -> None:
    assert not contains_value(np.array([0, 0, 0], dtype=object), None)


def test_contains_value_none_not_confused_with_nan() -> None:
    assert not contains_value(np.array([float("nan")]), None)


def test_contains_value_none_not_confused_with_false() -> None:
    assert not contains_value(np.array([False, False], dtype=object), None)


# --- string arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array(["cat", "dog", "bird"]), "cat", True, id="str-first"),
        pytest.param(np.array(["cat", "dog", "bird"]), "bird", True, id="str-last"),
        pytest.param(np.array(["cat", "dog", "bird"]), "dog", True, id="str-middle"),
        pytest.param(np.array(["cat", "dog", "bird"]), "wolf", False, id="str-absent"),
        pytest.param(np.array(["cat", "cat", "cat"]), "cat", True, id="str-all-same-present"),
        pytest.param(np.array(["cat", "cat", "cat"]), "dog", False, id="str-all-same-absent"),
        pytest.param(np.array(["cat", "dog"]), "CAT", False, id="str-case-sensitive"),
        pytest.param(np.array(["", "dog"]), "", True, id="str-empty-string"),
    ],
)
def test_contains_value_str_array(arr: np.ndarray, value: str, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- bool arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([True, False, True]), True, True, id="bool-true-present"),
        pytest.param(np.array([True, False, True]), False, True, id="bool-false-present"),
        pytest.param(np.array([True, True, True]), False, False, id="bool-false-absent"),
        pytest.param(np.array([False, False, False]), True, False, id="bool-true-absent"),
    ],
)
def test_contains_value_bool_array(arr: np.ndarray, value: bool, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- object arrays with mixed types ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1, "cat", 3.0], dtype=object), "cat", True, id="object-str-present"),
        pytest.param(np.array([1, "cat", 3.0], dtype=object), "dog", False, id="object-str-absent"),
        pytest.param(np.array([1, "cat", 3.0], dtype=object), 3.0, True, id="object-float-present"),
        pytest.param(np.array([1, "cat", 3.0], dtype=object), 2.0, False, id="object-float-absent"),
        pytest.param(np.array([1, "cat", 3.0], dtype=object), 1, True, id="object-int-present"),
        pytest.param(np.array([1, "cat", 3.0], dtype=object), 99, False, id="object-int-absent"),
    ],
)
def test_contains_value_object_array_mixed_types(
    arr: np.ndarray, value: Any, expected: bool
) -> None:
    assert contains_value(arr, value) == expected


# --- multidimensional arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([[1, 2], [3, 4]]), 3, True, id="2d-int-present"),
        pytest.param(np.array([[1, 2], [3, 4]]), 5, False, id="2d-int-absent"),
        pytest.param(
            np.array([[1.0, float("nan")], [3.0, 4.0]]), float("nan"), True, id="2d-nan-present"
        ),
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), float("nan"), False, id="2d-nan-absent"),
        pytest.param(np.array([[1, None], [3, 4]], dtype=object), None, True, id="2d-none-present"),
        pytest.param(np.array([[1, 2], [3, 4]], dtype=object), None, False, id="2d-none-absent"),
        pytest.param(
            np.array([[1.0, float("inf")], [3.0, 4.0]]),
            float("inf"),
            True,
            id="2d-inf-present",
        ),
    ],
)
def test_contains_value_multidimensional(arr: np.ndarray, value: Any, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- edge cases ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([], dtype=float), 1.0, id="empty-float"),
        pytest.param(np.array([], dtype=int), 1, id="empty-int"),
        pytest.param(np.array([], dtype=object), None, id="empty-object-none"),
        pytest.param(np.array([], dtype=object), float("nan"), id="empty-object-nan"),
        pytest.param(np.array([]), float("nan"), id="empty-default-float"),
    ],
)
def test_contains_value_empty_array(arr: np.ndarray, value: Any) -> None:
    assert not contains_value(arr, value)


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([42]), 42, True, id="single-int-present"),
        pytest.param(np.array([42]), 43, False, id="single-int-absent"),
        pytest.param(np.array([float("nan")]), float("nan"), True, id="single-nan-present"),
        pytest.param(np.array([None], dtype=object), None, True, id="single-none-present"),
    ],
)
def test_contains_value_single_element(arr: np.ndarray, value: Any, expected: bool) -> None:
    assert contains_value(arr, value) == expected


# --- NaN on non-numeric typed arrays (TypeError fallback) ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            id="datetime64",
        ),
        pytest.param(
            np.array(["a", "b", "c"]),
            id="str",
        ),
        pytest.param(
            np.array([True, False, True]),
            id="bool",
        ),
    ],
)
def test_contains_value_nan_non_numeric_typed_array(arr: np.ndarray) -> None:
    # np.isnan raises TypeError on non-numeric dtypes — should return False
    assert not contains_value(arr, float("nan"))


# --- TypeError fallback: incompatible types on typed arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            np.datetime64("2021-01-01"),
            True,
            id="datetime-present",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            np.datetime64("2021-01-03"),
            False,
            id="datetime-absent",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            "not-a-date",
            False,
            id="datetime64-vs-invalid-str",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            42,
            False,
            id="datetime64-vs-int",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            None,
            False,
            id="datetime64-vs-none",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            float("nan"),
            False,
            id="datetime64-vs-nan",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            float("inf"),
            False,
            id="datetime64-vs-inf",
        ),
    ],
)
def test_contains_value_datetime_array(arr: np.ndarray, value: Any, expected: bool) -> None:
    assert contains_value(arr, value) == expected


############################################
#     Tests for multi_contains_value       #
############################################


# --- single array ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param([np.array([1, 2, 3])], 2, True, id="single-int-present"),
        pytest.param([np.array([1, 2, 3])], 4, False, id="single-int-absent"),
        pytest.param([np.array([1.0, 2.0, 3.0])], 2.0, True, id="single-float-present"),
        pytest.param([np.array([1.0, 2.0, 3.0])], 4.0, False, id="single-float-absent"),
        pytest.param([np.array(["cat", "dog"])], "cat", True, id="single-str-present"),
        pytest.param([np.array(["cat", "dog"])], "wolf", False, id="single-str-absent"),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0])], float("nan"), True, id="single-nan-present"
        ),
        pytest.param([np.array([1.0, 2.0, 3.0])], float("nan"), False, id="single-nan-absent"),
        pytest.param([np.array([1, None, 3], dtype=object)], None, True, id="single-none-present"),
        pytest.param([np.array([1, 2, 3], dtype=object)], None, False, id="single-none-absent"),
        pytest.param(
            [np.array([1.0, float("inf"), 3.0])], float("inf"), True, id="single-pos-inf-present"
        ),
        pytest.param(
            [np.array([1.0, float("-inf"), 3.0])], float("-inf"), True, id="single-neg-inf-present"
        ),
        pytest.param([np.array([1.0, 2.0, 3.0])], float("inf"), False, id="single-inf-absent"),
    ],
)
def test_multi_contains_value_single_array(
    arrays: list[np.ndarray], value: Any, expected: bool
) -> None:
    assert multi_contains_value(arrays, value=value) == expected


# --- integer values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1, 0, 0]), np.array([1, 2, 0])],
            2,
            True,
            id="two-arrays-int-second-has-match",
        ),
        pytest.param(
            [np.array([1, 2, 0]), np.array([1, 0, 0])],
            2,
            True,
            id="two-arrays-int-first-has-match",
        ),
        pytest.param(
            [np.array([1, 2, 0]), np.array([1, 0, 2])],
            2,
            True,
            id="two-arrays-int-both-have-match",
        ),
        pytest.param(
            [np.array([1, 0, 0]), np.array([1, 0, 0])],
            2,
            False,
            id="two-arrays-int-absent",
        ),
        pytest.param(
            [np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 2])],
            2,
            True,
            id="three-arrays-int-third-has-match",
        ),
        pytest.param(
            [np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])],
            2,
            False,
            id="three-arrays-int-absent",
        ),
    ],
)
def test_multi_contains_value_int(arrays: list[np.ndarray], value: int, expected: bool) -> None:
    assert multi_contains_value(arrays, value=value) == expected


# --- float values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0])],
            2.0,
            True,
            id="two-arrays-float-second-has-match",
        ),
        pytest.param(
            [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            2.0,
            False,
            id="two-arrays-float-absent",
        ),
    ],
)
def test_multi_contains_value_float(arrays: list[np.ndarray], value: float, expected: bool) -> None:
    assert multi_contains_value(arrays, value=value) == expected


# --- string values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array(["cat", "bird"]), np.array(["fish", "dog"])],
            "dog",
            True,
            id="two-str-arrays-second-has-match",
        ),
        pytest.param(
            [np.array(["cat", "dog"]), np.array(["fish", "bird"])],
            "dog",
            True,
            id="two-str-arrays-first-has-match",
        ),
        pytest.param(
            [np.array(["cat", "bird"]), np.array(["fish", "fox"])],
            "dog",
            False,
            id="two-str-arrays-absent",
        ),
        pytest.param(
            [np.array(["cat", "dog"]), np.array(["CAT", "DOG"])],
            "cat",
            True,
            id="str-case-sensitive-lower-match",
        ),
        pytest.param(
            [np.array(["cat", "dog"]), np.array(["CAT", "DOG"])],
            "CAT",
            True,
            id="str-case-sensitive-upper-match",
        ),
        pytest.param(
            [np.array(["cat", "dog"]), np.array(["fish", "fox"])],
            "Cat",
            False,
            id="str-case-sensitive-no-match",
        ),
    ],
)
def test_multi_contains_value_str(arrays: list[np.ndarray], value: str, expected: bool) -> None:
    assert multi_contains_value(arrays, value=value) == expected


# --- NaN ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])],
            True,
            id="float-nan-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("nan"), 6.0])],
            True,
            id="float-nan-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([float("nan"), 5.0, 6.0])],
            True,
            id="float-nan-in-both",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            False,
            id="float-nan-absent",
        ),
        pytest.param(
            [
                np.array([1, float("nan"), 0], dtype=object),
                np.array([1, 2, 0], dtype=object),
            ],
            True,
            id="object-nan-in-first",
        ),
        pytest.param(
            [
                np.array([1, 2, 3], dtype=object),
                np.array([1, 2, 0], dtype=object),
            ],
            False,
            id="object-nan-absent",
        ),
        pytest.param(
            [np.array([None, None], dtype=object)],
            False,
            id="nan-not-confused-with-none",
        ),
        pytest.param(
            [np.array([float("inf"), float("-inf")])],
            False,
            id="nan-not-confused-with-inf",
        ),
    ],
)
def test_multi_contains_value_nan(arrays: list[np.ndarray], expected: bool) -> None:
    assert multi_contains_value(arrays, value=float("nan")) == expected


# --- None ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
        pytest.param(
            [np.array([1, None, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            True,
            id="none-in-first",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            True,
            id="none-in-second",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            False,
            id="none-absent",
        ),
        pytest.param(
            [np.array([0, False, ""], dtype=object)],
            False,
            id="none-not-confused-with-falsy",
        ),
        pytest.param(
            [np.array([1.0, float("nan")]), np.array([2.0, 3.0])],
            False,
            id="none-not-confused-with-nan",
        ),
        pytest.param(
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            False,
            id="none-non-object-arrays",
        ),
    ],
)
def test_multi_contains_value_none(arrays: list[np.ndarray], expected: bool) -> None:
    assert multi_contains_value(arrays, value=None) == expected


# --- inf ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1.0, float("inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("inf"),
            True,
            id="pos-inf-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("inf"), 6.0])],
            float("inf"),
            True,
            id="pos-inf-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("-inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("-inf"),
            True,
            id="neg-inf-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            float("inf"),
            False,
            id="inf-absent",
        ),
        pytest.param(
            [np.array([float("inf"), 2.0]), np.array([3.0, float("-inf")])],
            float("inf"),
            True,
            id="pos-inf-not-confused-with-neg-inf",
        ),
        pytest.param(
            [np.array([float("inf"), 2.0]), np.array([3.0, float("-inf")])],
            float("-inf"),
            True,
            id="neg-inf-not-confused-with-pos-inf",
        ),
    ],
)
def test_multi_contains_value_inf(arrays: list[np.ndarray], value: float, expected: bool) -> None:
    assert multi_contains_value(arrays, value=value) == expected


# --- short-circuit behavior ---


def test_multi_contains_value_short_circuits_on_first_match() -> None:
    # value is in first array — second array is never searched
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    assert multi_contains_value(arrays, value=2)


# --- arrays of different shapes ---


def test_multi_contains_value_different_shapes() -> None:
    # shape constraint is not required for this function
    assert multi_contains_value([np.array([1, 2, 3]), np.array([4, 5, 6, 7])], value=5)


# --- edge cases ---


def test_multi_contains_value_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="'arrays' cannot be empty"):
        multi_contains_value([], value=2)


def test_multi_contains_value_empty_arrays() -> None:
    assert not multi_contains_value(
        [np.array([], dtype=float), np.array([], dtype=float)], value=1.0
    )


def test_multi_contains_value_single_element_match() -> None:
    assert multi_contains_value([np.array([2.0]), np.array([1.0])], value=2.0)


def test_multi_contains_value_single_element_no_match() -> None:
    assert not multi_contains_value([np.array([1.0]), np.array([3.0])], value=2.0)


def test_multi_contains_value_all_arrays_match() -> None:
    assert multi_contains_value([np.array([2, 2, 2]), np.array([2, 2, 2])], value=2)


def test_multi_contains_value_2d_arrays() -> None:
    assert multi_contains_value([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])], value=6)


def test_multi_contains_value_bool_array() -> None:
    assert multi_contains_value([np.array([True, False])], value=False)
    assert not multi_contains_value([np.array([True, True])], value=False)
