from typing import Any

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.utils.array import equal_to, multi_equal_to

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


##################################
#    Tests for multi_equal_to    #
##################################


# --- single array ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1, 2, 3, 2])],
            2,
            np.array([False, True, False, True]),
            id="single-int-present",
        ),
        pytest.param(
            [np.array([1, 2, 3])],
            4,
            np.array([False, False, False]),
            id="single-int-absent",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0])],
            float("nan"),
            np.array([False, True, False]),
            id="single-nan",
        ),
        pytest.param(
            [np.array([1, None, 3], dtype=object)],
            None,
            np.array([False, True, False]),
            id="single-none",
        ),
        pytest.param(
            [np.array([1.0, float("inf"), 3.0])],
            float("inf"),
            np.array([False, True, False]),
            id="single-pos-inf",
        ),
        pytest.param(
            [np.array([1.0, float("-inf"), 3.0])],
            float("-inf"),
            np.array([False, True, False]),
            id="single-neg-inf",
        ),
    ],
)
def test_multi_equal_to_single_array(
    arrays: list[np.ndarray], value: Any, expected: np.ndarray
) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=value), expected)


# --- integer values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1, 0, 0, 1]), np.array([1, 2, 0, 1])],
            2,
            np.array([False, True, False, False]),
            id="two-arrays-int-second-has-match",
        ),
        pytest.param(
            [np.array([1, 2, 0, 1]), np.array([1, 0, 0, 2])],
            2,
            np.array([False, True, False, True]),
            id="two-arrays-int-both-have-match",
        ),
        pytest.param(
            [np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])],
            2,
            np.array([False, False, False, False]),
            id="two-arrays-int-absent",
        ),
        pytest.param(
            [np.array([2, 2, 2]), np.array([2, 2, 2])],
            2,
            np.array([True, True, True]),
            id="two-arrays-int-all-match",
        ),
        pytest.param(
            [
                np.array([1, 0, 0]),
                np.array([0, 2, 0]),
                np.array([0, 0, 2]),
            ],
            2,
            np.array([False, True, True]),
            id="three-arrays-int",
        ),
    ],
)
def test_multi_equal_to_int_value(
    arrays: list[np.ndarray], value: int, expected: np.ndarray
) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=value), expected)


# --- float values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0])],
            2.0,
            np.array([False, True, False]),
            id="two-arrays-float-second-has-match",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 0.0]), np.array([1.0, 0.0, 2.0])],
            2.0,
            np.array([False, True, True]),
            id="two-arrays-float-both-have-match",
        ),
        pytest.param(
            [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            2.0,
            np.array([False, False, False]),
            id="two-arrays-float-absent",
        ),
    ],
)
def test_multi_equal_to_float_value(
    arrays: list[np.ndarray], value: float, expected: np.ndarray
) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=value), expected)


# --- string values ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array(["cat", "dog", "cat"]), np.array(["bird", "dog", "fish"])],
            "dog",
            np.array([False, True, False]),
            id="two-str-arrays-match-in-both",
        ),
        pytest.param(
            [np.array(["cat", "bird", "cat"]), np.array(["bird", "cat", "fish"])],
            "dog",
            np.array([False, False, False]),
            id="two-str-arrays-absent",
        ),
        pytest.param(
            [np.array(["cat", "dog", "cat"]), np.array(["cat", "cat", "dog"])],
            "dog",
            np.array([False, True, True]),
            id="two-str-arrays-different-positions",
        ),
        pytest.param(
            [np.array(["cat", "dog"]), np.array(["cat", "dog"])],
            "CAT",
            np.array([False, False]),
            id="str-case-sensitive",
        ),
    ],
)
def test_multi_equal_to_str_value(
    arrays: list[np.ndarray], value: str, expected: np.ndarray
) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=value), expected)


# --- NaN ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])],
            np.array([False, True, False]),
            id="float-nan-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("nan"), 6.0])],
            np.array([False, True, False]),
            id="float-nan-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([float("nan"), 5.0, 6.0])],
            np.array([True, True, False]),
            id="float-nan-in-both",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, float("nan"), 6.0])],
            np.array([False, True, False]),
            id="float-nan-overlap",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            np.array([False, False, False]),
            id="float-no-nan",
        ),
        pytest.param(
            [
                np.array([1, float("nan"), 0], dtype=object),
                np.array([float("nan"), 2, 0], dtype=object),
            ],
            np.array([True, True, False]),
            id="object-nan-in-both",
        ),
        pytest.param(
            [
                np.array([1, None, 3], dtype=object),
                np.array([4, 5, 6], dtype=object),
            ],
            np.array([False, False, False]),
            id="object-none-not-nan",
        ),
    ],
)
def test_multi_equal_to_nan(arrays: list[np.ndarray], expected: np.ndarray) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=float("nan")), expected)


# --- None ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
        pytest.param(
            [np.array([1, None, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            np.array([False, True, False]),
            id="none-in-first",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            np.array([False, True, False]),
            id="none-in-second",
        ),
        pytest.param(
            [np.array([None, 2, 3], dtype=object), np.array([4, None, 6], dtype=object)],
            np.array([True, True, False]),
            id="none-in-both",
        ),
        pytest.param(
            [np.array([1, 2, 3], dtype=object), np.array([4, 5, 6], dtype=object)],
            np.array([False, False, False]),
            id="none-absent",
        ),
        pytest.param(
            [np.array([0, False, ""], dtype=object), np.array([1, 2, 3], dtype=object)],
            np.array([False, False, False]),
            id="none-not-confused-with-falsy",
        ),
        pytest.param(
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            np.array([False, False, False]),
            id="none-non-object-arrays",
        ),
    ],
)
def test_multi_equal_to_none(arrays: list[np.ndarray], expected: np.ndarray) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=None), expected)


# --- inf ---


@pytest.mark.parametrize(
    ("arrays", "value", "expected"),
    [
        pytest.param(
            [np.array([1.0, float("inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("inf"),
            np.array([False, True, False]),
            id="pos-inf-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("inf"), 6.0])],
            float("inf"),
            np.array([False, True, False]),
            id="pos-inf-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("-inf"), 3.0]), np.array([4.0, 5.0, 6.0])],
            float("-inf"),
            np.array([False, True, False]),
            id="neg-inf-in-first",
        ),
        pytest.param(
            [np.array([float("inf"), 2.0, 3.0]), np.array([4.0, float("-inf"), 6.0])],
            float("inf"),
            np.array([True, False, False]),
            id="pos-inf-not-confused-with-neg-inf",
        ),
        pytest.param(
            [np.array([float("inf"), 2.0, 3.0]), np.array([4.0, float("-inf"), 6.0])],
            float("-inf"),
            np.array([False, True, False]),
            id="neg-inf-not-confused-with-pos-inf",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            float("inf"),
            np.array([False, False, False]),
            id="inf-absent",
        ),
    ],
)
def test_multi_equal_to_inf(arrays: list[np.ndarray], value: float, expected: np.ndarray) -> None:
    assert objects_are_equal(multi_equal_to(arrays, value=value), expected)


# --- output properties ---


def test_multi_equal_to_returns_bool_dtype() -> None:
    result = multi_equal_to([np.array([1.0, 2.0])], value=2.0)
    assert result.dtype == bool


def test_multi_equal_to_preserves_shape() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert multi_equal_to(arrays, value=2.0).shape == arrays[0].shape


def test_multi_equal_to_2d_arrays() -> None:
    assert objects_are_equal(
        multi_equal_to(
            [np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])],
            value=2,
        ),
        np.array([[False, True], [True, False]]),
    )


def test_multi_equal_to_output_length_matches_input_length() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert len(multi_equal_to(arrays, value=2.0)) == len(arrays[0])


# --- empty input raises ---


def test_multi_equal_to_empty_input_raises() -> None:
    with pytest.raises(ValueError, match=r"'arrays' cannot be empty"):
        multi_equal_to([], value=2)


# --- edge cases ---


def test_multi_equal_to_empty_arrays() -> None:
    assert objects_are_equal(
        multi_equal_to([np.array([], dtype=float), np.array([], dtype=float)], value=1.0),
        np.array([], dtype=bool),
    )


def test_multi_equal_to_single_element_match() -> None:
    assert objects_are_equal(
        multi_equal_to([np.array([2.0]), np.array([1.0])], value=2.0),
        np.array([True]),
    )


def test_multi_equal_to_single_element_no_match() -> None:
    assert objects_are_equal(
        multi_equal_to([np.array([1.0]), np.array([3.0])], value=2.0),
        np.array([False]),
    )


def test_multi_equal_to_all_match() -> None:
    assert objects_are_equal(
        multi_equal_to([np.array([2, 2, 2]), np.array([2, 2, 2])], value=2),
        np.array([True, True, True]),
    )


def test_multi_equal_to_three_arrays_combined() -> None:
    assert objects_are_equal(
        multi_equal_to(
            [
                np.array([1.0, float("nan"), 3.0]),
                np.array([4.0, 5.0, float("nan")]),
                np.array([float("nan"), 8.0, 9.0]),
            ],
            value=float("nan"),
        ),
        np.array([True, True, True]),
    )
