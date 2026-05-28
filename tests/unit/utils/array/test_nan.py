from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.utils.array import (
    NAN_POLICIES,
    NanPolicy,
    check_nan_policy,
    contains_nan,
    is_nan,
    multi_is_nan,
    remove_duplicate_nans,
    remove_nans,
    validate_nan_policy,
)

#########################################
#     Tests for validate_nan_policy     #
#########################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_validate_nan_policy_valid(nan_policy: str) -> None:
    validate_nan_policy(nan_policy)  # should not raise


@pytest.mark.parametrize(
    "nan_policy",
    [
        pytest.param("invalid", id="invalid-string"),
        pytest.param("", id="empty-string"),
        pytest.param("OMIT", id="uppercase"),
        pytest.param("Propagate", id="mixed-case"),
        pytest.param("drop", id="similar-but-wrong"),
    ],
)
def test_validate_nan_policy_invalid_raises(nan_policy: str) -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'nan_policy'"):
        validate_nan_policy(nan_policy)


def test_validate_nan_policy_error_message_contains_valid_values() -> None:
    with pytest.raises(ValueError, match=r"'omit', 'propagate', 'raise'"):
        validate_nan_policy("invalid")


##################################
#     Tests for contains_nan     #
##################################


# --- float arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), False, id="float-no-nan"),
        pytest.param(np.array([1.0, float("nan"), 3.0]), True, id="float-nan-middle"),
        pytest.param(np.array([float("nan"), 2.0, 3.0]), True, id="float-nan-start"),
        pytest.param(np.array([1.0, 2.0, float("nan")]), True, id="float-nan-end"),
        pytest.param(np.array([float("nan"), float("nan")]), True, id="float-all-nan"),
        pytest.param(np.array([float("nan")]), True, id="float-single-nan"),
        pytest.param(np.array([1.0]), False, id="float-single-no-nan"),
        pytest.param(np.array([], dtype=float), False, id="float-empty"),
    ],
)
def test_contains_nan_float_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- int arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 2, 3]), False, id="int-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int8), False, id="int8-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int16), False, id="int16-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int32), False, id="int32-no-nan"),
        pytest.param(np.array([1, 2, 3], dtype=np.int64), False, id="int64-no-nan"),
        pytest.param(np.array([], dtype=int), False, id="int-empty"),
    ],
)
def test_contains_nan_int_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- object arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 2, 3], dtype=object), False, id="object-no-nan"),
        pytest.param(np.array([1, float("nan"), 3], dtype=object), True, id="object-nan-middle"),
        pytest.param(np.array([float("nan"), 2, 3], dtype=object), True, id="object-nan-start"),
        pytest.param(np.array([1, 2, float("nan")], dtype=object), True, id="object-nan-end"),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object), True, id="object-all-nan"
        ),
        pytest.param(np.array([None, 2, 3], dtype=object), False, id="object-none-not-nan"),
        pytest.param(np.array([None, float("nan")], dtype=object), True, id="object-none-and-nan"),
        pytest.param(np.array([], dtype=object), False, id="object-empty"),
    ],
)
def test_contains_nan_object_array(arr: np.ndarray, expected: bool) -> None:
    assert contains_nan(arr) == expected


# --- non-numeric dtypes (TypeError fallback) ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array(["a", "b", "c"]), id="str"),
        pytest.param(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), id="datetime64"),
        pytest.param(np.array([True, False, True]), id="bool"),
    ],
)
def test_contains_nan_non_numeric_array(arr: np.ndarray) -> None:
    assert not contains_nan(arr)


# --- nan not confused with other special values ---


def test_contains_nan_inf_is_not_nan() -> None:
    assert not contains_nan(np.array([float("inf"), float("-inf")]))


def test_contains_nan_none_is_not_nan() -> None:
    assert not contains_nan(np.array([None, None], dtype=object))


######################################
#     Tests for check_nan_policy     #
######################################


# --- no NaN — all policies return False ---


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_check_nan_policy_no_nan(nan_policy: NanPolicy) -> None:
    assert not check_nan_policy(np.array([1, 2, 3]), nan_policy=nan_policy)


# --- missing_policy='propagate' ---


def test_check_nan_policy_propagate_returns_true() -> None:
    assert check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="propagate")


def test_check_nan_policy_propagate_does_not_raise() -> None:
    result = check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="propagate")
    assert result is True


# --- nan_policy='omit' ---


def test_check_nan_policy_omit_returns_true() -> None:
    assert check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="omit")


def test_check_nan_policy_omit_does_not_raise() -> None:
    result = check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="omit")
    assert result is True


# --- nan_policy='raise' ---


def test_check_nan_policy_raise_no_nan_does_not_raise() -> None:
    assert not check_nan_policy(np.array([1, 2, 3]), nan_policy="raise")


def test_check_nan_policy_raise_with_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"input contains at least one NaN value"):
        check_nan_policy(np.array([1.0, float("nan"), 3.0]), nan_policy="raise")


def test_check_nan_policy_raise_custom_name() -> None:
    with pytest.raises(ValueError, match=r"my_array contains at least one NaN value"):
        check_nan_policy(
            np.array([1.0, float("nan"), 3.0]),
            nan_policy="raise",
            name="my_array",
        )


# --- invalid nan_policy ---


def test_check_nan_policy_invalid_raises() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'nan_policy'"):
        check_nan_policy(np.array([1, 2, 3]), nan_policy="invalid")


# --- different dtypes ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, float("nan"), 3.0], dtype=np.float32), True, id="float32"),
        pytest.param(np.array([1.0, float("nan"), 3.0], dtype=np.float64), True, id="float64"),
        pytest.param(np.array([1, float("nan"), 3], dtype=object), True, id="object"),
        pytest.param(np.array([1, 2, 3], dtype=np.int32), False, id="int32"),
        pytest.param(np.array(["a", "b", "c"]), False, id="str"),
        pytest.param(
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), False, id="datetime64"
        ),
    ],
)
def test_check_nan_policy_dtypes(arr: np.ndarray, expected: bool) -> None:
    assert check_nan_policy(arr) == expected


# --- edge cases ---


def test_check_nan_policy_empty_array() -> None:
    assert not check_nan_policy(np.array([], dtype=float))


def test_check_nan_policy_all_nan() -> None:
    assert check_nan_policy(np.array([float("nan"), float("nan")]))


def test_check_nan_policy_single_nan() -> None:
    assert check_nan_policy(np.array([float("nan")]))


def test_check_nan_policy_single_no_nan() -> None:
    assert not check_nan_policy(np.array([1.0]))


##################################
#       Tests for is_nan         #
##################################


# --- float arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            np.array([False, False, False]),
            id="float-no-nan",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 3.0]),
            np.array([False, True, False]),
            id="float-nan-middle",
        ),
        pytest.param(
            np.array([float("nan"), 2.0, 3.0]),
            np.array([True, False, False]),
            id="float-nan-start",
        ),
        pytest.param(
            np.array([1.0, 2.0, float("nan")]),
            np.array([False, False, True]),
            id="float-nan-end",
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
            np.array([1.0]),
            np.array([False]),
            id="float-single-no-nan",
        ),
        pytest.param(
            np.array([], dtype=float),
            np.array([], dtype=bool),
            id="float-empty",
        ),
    ],
)
def test_is_nan_float_array(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(is_nan(arr), expected)


# --- int arrays ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1, 2, 3]), id="int64"),
        pytest.param(np.array([1, 2, 3], dtype=np.int8), id="int8"),
        pytest.param(np.array([1, 2, 3], dtype=np.int16), id="int16"),
        pytest.param(np.array([1, 2, 3], dtype=np.int32), id="int32"),
        pytest.param(np.array([1, 2, 3], dtype=np.uint8), id="uint8"),
    ],
)
def test_is_nan_int_array(arr: np.ndarray) -> None:
    assert objects_are_equal(is_nan(arr), np.array([False, False, False]))


# --- object arrays ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1, 2, 3], dtype=object),
            np.array([False, False, False]),
            id="object-no-nan",
        ),
        pytest.param(
            np.array([1, float("nan"), 3], dtype=object),
            np.array([False, True, False]),
            id="object-nan-middle",
        ),
        pytest.param(
            np.array([float("nan"), 2, 3], dtype=object),
            np.array([True, False, False]),
            id="object-nan-start",
        ),
        pytest.param(
            np.array([1, 2, float("nan")], dtype=object),
            np.array([False, False, True]),
            id="object-nan-end",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object),
            np.array([True, True]),
            id="object-all-nan",
        ),
        pytest.param(
            np.array([None, float("nan"), 1], dtype=object),
            np.array([False, True, False]),
            id="object-none-not-nan",
        ),
        pytest.param(
            np.array([None, None], dtype=object),
            np.array([False, False]),
            id="object-all-none-not-nan",
        ),
        pytest.param(
            np.array([], dtype=object),
            np.array([], dtype=bool),
            id="object-empty",
        ),
    ],
)
def test_is_nan_object_array(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(is_nan(arr), expected)


# --- non-numeric dtypes (TypeError fallback) ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array(["a", "b", "c"]), id="str"),
        pytest.param(np.array(["2021-01-01", "2021-01-02"], dtype="datetime64"), id="datetime64"),
        pytest.param(np.array([True, False, True]), id="bool"),
    ],
)
def test_is_nan_non_numeric_array(arr: np.ndarray) -> None:
    assert objects_are_equal(is_nan(arr), np.zeros(arr.shape, dtype=bool))


# --- special values not confused with nan ---


def test_is_nan_inf_is_not_nan() -> None:
    assert objects_are_equal(
        is_nan(np.array([float("inf"), float("-inf")])),
        np.array([False, False]),
    )


def test_is_nan_none_is_not_nan() -> None:
    assert objects_are_equal(
        is_nan(np.array([None, None], dtype=object)),
        np.array([False, False]),
    )


def test_is_nan_zero_is_not_nan() -> None:
    assert objects_are_equal(
        is_nan(np.array([0.0, 0.0])),
        np.array([False, False]),
    )


# --- output properties ---


def test_is_nan_returns_bool_dtype() -> None:
    assert is_nan(np.array([1.0, float("nan")])).dtype == bool


def test_is_nan_object_returns_bool_dtype() -> None:
    assert is_nan(np.array([1, float("nan")], dtype=object)).dtype == bool


def test_is_nan_preserves_shape_1d() -> None:
    arr = np.array([1.0, float("nan"), 3.0])
    assert is_nan(arr).shape == arr.shape


def test_is_nan_preserves_shape_2d() -> None:
    arr = np.array([[1.0, float("nan")], [3.0, 4.0]])
    assert objects_are_equal(
        is_nan(arr),
        np.array([[False, True], [False, False]]),
    )


def test_is_nan_preserves_shape_2d_object() -> None:
    arr = np.array([[1, float("nan")], [None, 4]], dtype=object)
    assert objects_are_equal(
        is_nan(arr),
        np.array([[False, True], [False, False]]),
    )


##################################
#     Tests for multi_is_nan     #
##################################


# --- single array ---


def test_multi_is_nan_single_array_no_nan() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([1.0, 2.0, 3.0])]),
        np.array([False, False, False]),
    )


def test_multi_is_nan_single_array_with_nan() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([1.0, float("nan"), 3.0])]),
        np.array([False, True, False]),
    )


def test_multi_is_nan_single_array_all_nan() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([float("nan"), float("nan")])]),
        np.array([True, True]),
    )


# --- multiple arrays ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            np.array([False, False, False]),
            id="two-arrays-no-nan",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])],
            np.array([False, True, False]),
            id="two-arrays-nan-in-first",
        ),
        pytest.param(
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, float("nan"), 6.0])],
            np.array([False, True, False]),
            id="two-arrays-nan-in-second",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([float("nan"), 5.0, 6.0])],
            np.array([True, True, False]),
            id="two-arrays-nan-in-both",
        ),
        pytest.param(
            [np.array([1.0, float("nan"), 3.0]), np.array([4.0, float("nan"), 6.0])],
            np.array([False, True, False]),
            id="two-arrays-nan-overlap",
        ),
        pytest.param(
            [
                np.array([1.0, float("nan"), 3.0]),
                np.array([4.0, 5.0, float("nan")]),
                np.array([float("nan"), 8.0, 9.0]),
            ],
            np.array([True, True, True]),
            id="three-arrays",
        ),
    ],
)
def test_multi_is_nan_multiple_arrays(arrays: list[np.ndarray], expected: np.ndarray) -> None:
    assert objects_are_equal(multi_is_nan(arrays), expected)


# --- object arrays ---


@pytest.mark.parametrize(
    ("arrays", "expected"),
    [
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
        pytest.param(
            [
                np.array([1, float("nan"), 3], dtype=object),
                np.array([None, 2, 3], dtype=object),
            ],
            np.array([False, True, False]),
            id="object-mixed-nan-and-none",
        ),
    ],
)
def test_multi_is_nan_object_arrays(arrays: list[np.ndarray], expected: np.ndarray) -> None:
    assert objects_are_equal(multi_is_nan(arrays), expected)


# --- output properties ---


def test_multi_is_nan_returns_bool_dtype() -> None:
    assert multi_is_nan([np.array([1.0, float("nan")])]).dtype == bool


def test_multi_is_nan_preserves_shape() -> None:
    arrays = [np.array([1.0, float("nan"), 3.0]), np.array([4.0, 5.0, 6.0])]
    assert multi_is_nan(arrays).shape == arrays[0].shape


def test_multi_is_nan_2d_arrays() -> None:
    assert objects_are_equal(
        multi_is_nan(
            [
                np.array([[1.0, float("nan")], [3.0, 4.0]]),
                np.array([[float("nan"), 2.0], [3.0, 4.0]]),
            ]
        ),
        np.array([[True, True], [False, False]]),
    )


# --- empty input raises ---


def test_multi_is_nan_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"'arrays' cannot be empty"):
        multi_is_nan([])


# --- edge cases ---


def test_multi_is_nan_empty_arrays() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([], dtype=float), np.array([], dtype=float)]),
        np.array([], dtype=bool),
    )


def test_multi_is_nan_single_element_nan() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([float("nan")]), np.array([1.0])]),
        np.array([True]),
    )


def test_multi_is_nan_single_element_no_nan() -> None:
    assert objects_are_equal(
        multi_is_nan([np.array([1.0]), np.array([2.0])]),
        np.array([False]),
    )


def test_multi_is_nan_all_nan() -> None:
    assert objects_are_equal(
        multi_is_nan(
            [np.array([float("nan"), float("nan")]), np.array([float("nan"), float("nan")])]
        ),
        np.array([True, True]),
    )


##################################################
#           Tests for remove_nans                #
##################################################


# --- no NaNs: array returned unchanged ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), id="float"),
        pytest.param(np.array([1, 2, 3]), id="int"),
        pytest.param(np.array(["cat", "dog", "bear"]), id="string"),
        pytest.param(np.array(["cat", "dog", "bear"], dtype=object), id="object-string"),
        pytest.param(np.array([1.0, 2.0, 3.0], dtype=object), id="object-float"),
        pytest.param(np.array([True, False, True]), id="bool"),
    ],
)
def test_remove_nans_no_nans(arr: np.ndarray) -> None:
    assert objects_are_equal(remove_nans(arr), arr)


# --- all NaNs: empty array returned ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([float("nan")]), id="float-single"),
        pytest.param(np.array([float("nan"), float("nan")]), id="float-two"),
        pytest.param(np.array([float("nan")] * 5), id="float-five"),
        pytest.param(np.array([float("nan")], dtype=object), id="object-single"),
        pytest.param(np.array([float("nan"), float("nan")], dtype=object), id="object-two"),
    ],
)
def test_remove_nans_all_nans(arr: np.ndarray) -> None:
    result = remove_nans(arr)
    assert result.size == 0
    assert result.dtype == arr.dtype


# --- NaNs removed, finite values preserved ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 2.0]),
            np.array([1.0, 2.0]),
            id="float-nan-middle",
        ),
        pytest.param(
            np.array([float("nan"), 1.0, 2.0]),
            np.array([1.0, 2.0]),
            id="float-nan-start",
        ),
        pytest.param(
            np.array([1.0, 2.0, float("nan")]),
            np.array([1.0, 2.0]),
            id="float-nan-end",
        ),
        pytest.param(
            np.array([float("nan"), 1.0, float("nan"), 2.0, float("nan")]),
            np.array([1.0, 2.0]),
            id="float-nan-multiple",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 2.0], dtype=object),
            np.array([1.0, 2.0], dtype=object),
            id="object-float-nan-middle",
        ),
        pytest.param(
            np.array([float("nan"), 1.0, float("nan"), 2.0], dtype=object),
            np.array([1.0, 2.0], dtype=object),
            id="object-float-nan-multiple",
        ),
    ],
)
def test_remove_nans_removes_nans(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_nans(arr), expected)


# --- mixed object arrays with strings and NaN ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array(["cat", float("nan"), "dog"], dtype=object),
            np.array(["cat", "dog"], dtype=object),
            id="string-nan-middle",
        ),
        pytest.param(
            np.array([float("nan"), "cat", "dog"], dtype=object),
            np.array(["cat", "dog"], dtype=object),
            id="string-nan-start",
        ),
        pytest.param(
            np.array(["cat", "dog", float("nan")], dtype=object),
            np.array(["cat", "dog"], dtype=object),
            id="string-nan-end",
        ),
        pytest.param(
            np.array([float("nan"), "cat", float("nan"), "dog", float("nan")], dtype=object),
            np.array(["cat", "dog"], dtype=object),
            id="string-nan-multiple",
        ),
    ],
)
def test_remove_nans_mixed_object(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_nans(arr), expected)


# --- dtype is preserved ---


@pytest.mark.parametrize(
    ("arr", "expected_dtype"),
    [
        pytest.param(np.array([1.0, float("nan")]), np.float64, id="float64"),
        pytest.param(np.array([1.0, float("nan")], dtype=np.float32), np.float32, id="float32"),
        pytest.param(np.array([1.0, float("nan")], dtype=object), object, id="object"),
    ],
)
def test_remove_nans_preserves_dtype(arr: np.ndarray, expected_dtype: np.dtype) -> None:
    result = remove_nans(arr)
    assert result.dtype == expected_dtype


# --- empty array ---


def test_remove_nans_empty() -> None:
    arr = np.array([])
    assert objects_are_equal(remove_nans(arr), arr)


# --- non-1D input raises ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), id="2d"),
        pytest.param(np.ones((2, 3, 4)), id="3d"),
    ],
)
def test_remove_nans_non_1d_raises(arr: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"input: expected .*D array, got shape"):
        remove_nans(arr)


def test_remove_nans_non_1d_error_includes_shape() -> None:
    arr = np.ones((3, 4))
    with pytest.raises(ValueError, match=r"input: expected .*D array, got shape"):
        remove_nans(arr)


##################################################
#        Tests for remove_duplicate_nans         #
##################################################


# --- no NaNs: array returned unchanged ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), id="float"),
        pytest.param(np.array([1, 2, 3]), id="int"),
        pytest.param(np.array(["cat", "dog", "bear"]), id="string"),
        pytest.param(np.array(["cat", "dog", "bear"], dtype=object), id="object-string"),
        pytest.param(np.array([1.0, 2.0, 3.0], dtype=object), id="object-float"),
        pytest.param(np.array([True, False, True]), id="bool"),
    ],
)
def test_remove_duplicate_nans_no_nans(arr: np.ndarray) -> None:
    assert objects_are_equal(remove_duplicate_nans(arr), arr)


# --- single NaN: array returned unchanged ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([1.0, float("nan"), 2.0]), id="float-nan-middle"),
        pytest.param(np.array([float("nan"), 1.0, 2.0]), id="float-nan-start"),
        pytest.param(np.array([1.0, 2.0, float("nan")]), id="float-nan-end"),
        pytest.param(np.array([float("nan")]), id="float-nan-only"),
        pytest.param(
            np.array([1.0, float("nan"), 2.0], dtype=object), id="object-float-nan-middle"
        ),
        pytest.param(np.array([float("nan")], dtype=object), id="object-float-nan-only"),
    ],
)
def test_remove_duplicate_nans_single_nan(arr: np.ndarray) -> None:
    assert objects_are_equal(remove_duplicate_nans(arr), arr, equal_nan=True)


# --- duplicate NaNs: collapsed to one NaN at the end ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1.0, float("nan"), 2.0, float("nan")]),
            np.array([1.0, 2.0, float("nan")]),
            id="float-nan-two",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")]),
            np.array([float("nan")]),
            id="float-nan-only-two",
        ),
        pytest.param(
            np.array([float("nan"), 1.0, float("nan"), 2.0, float("nan")]),
            np.array([1.0, 2.0, float("nan")]),
            id="float-nan-three",
        ),
        pytest.param(
            np.array([float("nan")] * 5),
            np.array([float("nan")]),
            id="float-nan-only-five",
        ),
        pytest.param(
            np.array([1.0, float("nan"), 2.0, float("nan")], dtype=object),
            np.array([1.0, 2.0, float("nan")], dtype=object),
            id="object-float-nan-two",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object),
            np.array([float("nan")], dtype=object),
            id="object-float-nan-only-two",
        ),
    ],
)
def test_remove_duplicate_nans_duplicates(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_duplicate_nans(arr).tolist(), expected.tolist(), equal_nan=True)


# --- mixed object arrays with strings and NaN ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array(["cat", float("nan"), "dog", float("nan")], dtype=object),
            np.array(["cat", "dog", float("nan")], dtype=object),
            id="object-mixed-string-nan-two",
        ),
        pytest.param(
            np.array([float("nan"), "cat", float("nan")], dtype=object),
            np.array(["cat", float("nan")], dtype=object),
            id="object-mixed-nan-string-nan",
        ),
        pytest.param(
            np.array(["cat", float("nan")], dtype=object),
            np.array(["cat", float("nan")], dtype=object),
            id="object-mixed-string-single-nan",
        ),
    ],
)
def test_remove_duplicate_nans_mixed_object(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_duplicate_nans(arr).tolist(), expected.tolist(), equal_nan=True)


# --- dtype is preserved ---


@pytest.mark.parametrize(
    ("arr", "expected_dtype"),
    [
        pytest.param(np.array([1.0, float("nan"), float("nan")]), np.float64, id="float64"),
        pytest.param(
            np.array([1.0, float("nan"), float("nan")], dtype=np.float32), np.float32, id="float32"
        ),
        pytest.param(
            np.array([1.0, float("nan"), float("nan")], dtype=object), object, id="object"
        ),
    ],
)
def test_remove_duplicate_nans_preserves_dtype(arr: np.ndarray, expected_dtype: np.dtype) -> None:
    assert remove_duplicate_nans(arr).dtype == expected_dtype


# --- empty array ---


def test_remove_duplicate_nans_empty() -> None:
    assert objects_are_equal(remove_duplicate_nans(np.array([])), np.array([]))


# --- non-1D input raises ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), id="2d"),
        pytest.param(np.ones((2, 3, 4)), id="3d"),
    ],
)
def test_remove_duplicate_nans_non_1d_raises(arr: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"input: expected .*D array, got shape"):
        remove_duplicate_nans(arr)


def test_remove_duplicate_nans_non_1d_error_includes_shape() -> None:
    arr = np.ones((3, 4))
    with pytest.raises(ValueError, match=r"input: expected 1D array, got shape"):
        remove_duplicate_nans(arr)
