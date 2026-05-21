from typing import Any

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.utils.array import equal_to

##############################
#     Tests for equal_to     #
##############################


# --- integer arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1, 2, 3, 2]),
            2,
            np.array([False, True, False, True]),
            id="int-present-multiple",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            4,
            np.array([False, False, False]),
            id="int-absent",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            1,
            np.array([True, False, False]),
            id="int-first",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            3,
            np.array([False, False, True]),
            id="int-last",
        ),
        pytest.param(
            np.array([2, 2, 2]),
            2,
            np.array([True, True, True]),
            id="int-all-match",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            2.0,
            np.array([False, True, False]),
            id="int-array-float-value",
        ),
    ],
)
def test_equal_to_int_array(arr: np.ndarray, value: Any, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- float arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1.0, 2.0, 3.0, 2.0]),
            2.0,
            np.array([False, True, False, True]),
            id="float-present-multiple",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            4.0,
            np.array([False, False, False]),
            id="float-absent",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            2,
            np.array([False, True, False]),
            id="float-array-int-value",
        ),
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            0.0,
            np.array([True, False, False]),
            id="float-zero",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            -1.0,
            np.array([True, False, False]),
            id="float-negative",
        ),
    ],
)
def test_equal_to_float_array(arr: np.ndarray, value: Any, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- NaN ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 3.0, float("nan")]),
            np.array([False, True, False, True]),
            id="float-nan-multiple",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            np.array([False, False, False]),
            id="float-no-nan",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")]),
            np.array([True, True]),
            id="float-all-nan",
        ),
        pytest.param(
            np.array([float("nan")]),
            np.array([True]),
            id="float-single-nan",
        ),
        pytest.param(
            np.array([float("nan"), 1.0, 2.0]),
            np.array([True, False, False]),
            id="float-nan-at-start",
        ),
        pytest.param(
            np.array([1.0, 2.0, float("nan")]),
            np.array([False, False, True]),
            id="float-nan-at-end",
        ),
    ],
)
def test_equal_to_nan_float_array(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, float("nan")), expected)


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1, float("nan"), 3, float("nan")], dtype=object),
            np.array([False, True, False, True]),
            id="object-nan-multiple",
        ),
        pytest.param(
            np.array([1, 2, 3], dtype=object),
            np.array([False, False, False]),
            id="object-no-nan",
        ),
        pytest.param(
            np.array([None, float("nan"), "x"], dtype=object),
            np.array([False, True, False]),
            id="object-mixed-nan-not-none",
        ),
    ],
)
def test_equal_to_nan_object_array(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, float("nan")), expected)


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1, 2, 3]), id="int"),
        pytest.param(np.array(["a", "b", "c"]), id="str"),
        pytest.param(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), id="datetime64"),
        pytest.param(np.array([True, False]), id="bool"),
    ],
)
def test_equal_to_nan_non_numeric_array(arr: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, float("nan")), np.zeros(arr.shape, dtype=bool))


def test_equal_to_nan_not_confused_with_none() -> None:
    arr = np.array([None, None], dtype=object)
    assert objects_are_equal(equal_to(arr, float("nan")), np.array([False, False]))


def test_equal_to_nan_not_confused_with_inf() -> None:
    arr = np.array([float("inf"), float("-inf")])
    assert objects_are_equal(equal_to(arr, float("nan")), np.array([False, False]))


# --- inf ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1.0, float("inf"), 3.0, float("inf")]),
            float("inf"),
            np.array([False, True, False, True]),
            id="pos-inf-multiple",
        ),
        pytest.param(
            np.array([1.0, float("-inf"), 3.0, float("-inf")]),
            float("-inf"),
            np.array([False, True, False, True]),
            id="neg-inf-multiple",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            float("inf"),
            np.array([False, False, False]),
            id="pos-inf-absent",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            float("-inf"),
            np.array([False, False, False]),
            id="neg-inf-absent",
        ),
        pytest.param(
            np.array([float("inf"), float("-inf"), 1.0]),
            float("inf"),
            np.array([True, False, False]),
            id="pos-inf-not-confused-with-neg-inf",
        ),
        pytest.param(
            np.array([float("inf"), float("-inf"), 1.0]),
            float("-inf"),
            np.array([False, True, False]),
            id="neg-inf-not-confused-with-pos-inf",
        ),
    ],
)
def test_equal_to_inf_float_array(arr: np.ndarray, value: float, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- None ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1, None, 3, None], dtype=object),
            np.array([False, True, False, True]),
            id="none-multiple",
        ),
        pytest.param(
            np.array([1, 2, 3], dtype=object),
            np.array([False, False, False]),
            id="none-absent",
        ),
        pytest.param(
            np.array([None, None, None], dtype=object),
            np.array([True, True, True]),
            id="all-none",
        ),
        pytest.param(
            np.array([None, 1, 2], dtype=object),
            np.array([True, False, False]),
            id="none-at-start",
        ),
        pytest.param(
            np.array([1, 2, None], dtype=object),
            np.array([False, False, True]),
            id="none-at-end",
        ),
        pytest.param(
            np.array([None, float("nan"), "x"], dtype=object),
            np.array([True, False, False]),
            id="none-not-confused-with-nan",
        ),
    ],
)
def test_equal_to_none_object_array(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, None), expected)


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1, 2, 3]), id="int"),
        pytest.param(np.array([1.0, 2.0, 3.0]), id="float"),
        pytest.param(np.array(["a", "b", "c"]), id="str"),
    ],
)
def test_equal_to_none_non_object_array(arr: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, None), np.zeros(arr.shape, dtype=bool))


def test_equal_to_none_not_confused_with_zero() -> None:
    arr = np.array([0, 0, 0], dtype=object)
    assert objects_are_equal(equal_to(arr, None), np.array([False, False, False]))


def test_equal_to_none_not_confused_with_false() -> None:
    arr = np.array([False, False], dtype=object)
    assert objects_are_equal(equal_to(arr, None), np.array([False, False]))


# --- string arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array(["cat", "dog", "cat"]),
            "cat",
            np.array([True, False, True]),
            id="str-multiple",
        ),
        pytest.param(
            np.array(["cat", "dog", "bird"]),
            "wolf",
            np.array([False, False, False]),
            id="str-absent",
        ),
        pytest.param(
            np.array(["cat", "dog", "bird"]),
            "CAT",
            np.array([False, False, False]),
            id="str-case-sensitive",
        ),
        pytest.param(
            np.array(["", "dog", ""]),
            "",
            np.array([True, False, True]),
            id="str-empty-string",
        ),
    ],
)
def test_equal_to_str_array(arr: np.ndarray, value: str, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- bool arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([True, False, True]),
            True,
            np.array([True, False, True]),
            id="bool-true",
        ),
        pytest.param(
            np.array([True, False, True]),
            False,
            np.array([False, True, False]),
            id="bool-false",
        ),
    ],
)
def test_equal_to_bool_array(arr: np.ndarray, value: bool, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- object arrays with mixed types ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1, "cat", 3.0, "cat"], dtype=object),
            "cat",
            np.array([False, True, False, True]),
            id="object-str",
        ),
        pytest.param(
            np.array([1, "cat", 3.0], dtype=object),
            3.0,
            np.array([False, False, True]),
            id="object-float",
        ),
        pytest.param(
            np.array([1, "cat", 3.0], dtype=object),
            1,
            np.array([True, False, False]),
            id="object-int",
        ),
    ],
)
def test_equal_to_object_array_mixed_types(
    arr: np.ndarray, value: Any, expected: np.ndarray
) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)


# --- incompatible types (TypeError/ValueError fallback) ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            "not-a-date",
            id="datetime64-vs-invalid-str",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            42,
            id="datetime64-vs-int",
        ),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"),
            None,
            id="datetime64-vs-none",
        ),
    ],
)
def test_equal_to_incompatible_type_returns_all_false(arr: np.ndarray, value: Any) -> None:
    assert objects_are_equal(equal_to(arr, value), np.zeros(arr.shape, dtype=bool))


# --- output shape ---


def test_equal_to_preserves_shape_1d() -> None:
    arr = np.array([1, 2, 3])
    assert equal_to(arr, 2).shape == arr.shape


def test_equal_to_preserves_shape_2d() -> None:
    arr = np.array([[1, 2], [3, 2]])
    assert objects_are_equal(equal_to(arr, 2), np.array([[False, True], [False, True]]))


def test_equal_to_preserves_shape_2d_nan() -> None:
    arr = np.array([[1.0, float("nan")], [float("nan"), 4.0]])
    assert objects_are_equal(
        equal_to(arr, float("nan")),
        np.array([[False, True], [True, False]]),
    )


def test_equal_to_preserves_shape_2d_none() -> None:
    arr = np.array([[1, None], [None, 4]], dtype=object)
    assert objects_are_equal(
        equal_to(arr, None),
        np.array([[False, True], [True, False]]),
    )


def test_equal_to_returns_bool_dtype() -> None:
    assert equal_to(np.array([1, 2, 3]), 2).dtype == bool


def test_equal_to_nan_returns_bool_dtype() -> None:
    assert equal_to(np.array([1.0, float("nan")]), float("nan")).dtype == bool


def test_equal_to_none_returns_bool_dtype() -> None:
    assert equal_to(np.array([1, None], dtype=object), None).dtype == bool


# --- edge cases ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([], dtype=float), 1.0, id="empty-float"),
        pytest.param(np.array([], dtype=int), 1, id="empty-int"),
        pytest.param(np.array([], dtype=object), None, id="empty-object-none"),
        pytest.param(np.array([], dtype=object), float("nan"), id="empty-object-nan"),
        pytest.param(np.array([]), float("nan"), id="empty-default"),
    ],
)
def test_equal_to_empty_array(arr: np.ndarray, value: Any) -> None:
    result = equal_to(arr, value)
    assert objects_are_equal(result, np.array([], dtype=bool))


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([42]), 42, np.array([True]), id="single-match"),
        pytest.param(np.array([42]), 43, np.array([False]), id="single-no-match"),
        pytest.param(np.array([float("nan")]), float("nan"), np.array([True]), id="single-nan"),
        pytest.param(np.array([None], dtype=object), None, np.array([True]), id="single-none"),
    ],
)
def test_equal_to_single_element(arr: np.ndarray, value: Any, expected: np.ndarray) -> None:
    assert objects_are_equal(equal_to(arr, value), expected)
