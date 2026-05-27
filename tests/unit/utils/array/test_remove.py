from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from metriclab.utils.array import remove_values

##################################################
#           Tests for remove_values              #
##################################################


# --- value not present: array returned unchanged ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), 4.0, id="float-absent"),
        pytest.param(np.array([1, 2, 3]), 4, id="int-absent"),
        pytest.param(np.array(["cat", "dog"]), "bear", id="string-absent"),
        pytest.param(np.array(["cat", "dog"], dtype=object), "bear", id="object-string-absent"),
        pytest.param(np.array([1.0, 2.0], dtype=object), 3.0, id="object-float-absent"),
        pytest.param(np.array([True, False]), None, id="bool-none-absent"),
    ],
)
def test_remove_values_value_not_present(arr: np.ndarray, value: object) -> None:
    assert objects_are_equal(remove_values(arr, value), arr)


# --- all values match: empty array returned ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1.0, 1.0, 1.0]), 1.0, id="float-all-match"),
        pytest.param(np.array([1, 1, 1]), 1, id="int-all-match"),
        pytest.param(np.array(["cat", "cat"]), "cat", id="string-all-match"),
        pytest.param(np.array(["cat", "cat"], dtype=object), "cat", id="object-string-all-match"),
        pytest.param(np.array([float("nan"), float("nan")]), float("nan"), id="float-nan-all"),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object),
            float("nan"),
            id="object-nan-all",
        ),
    ],
)
def test_remove_values_all_match(arr: np.ndarray, value: object) -> None:
    result = remove_values(arr, value)
    assert result.size == 0
    assert result.dtype == arr.dtype


# --- partial match: matching values removed, others preserved ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1.0, 2.0, 1.0, 3.0]),
            1.0,
            np.array([2.0, 3.0]),
            id="float-start-and-end",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0, 2.0]),
            2.0,
            np.array([1.0, 3.0]),
            id="float-middle-and-end",
        ),
        pytest.param(
            np.array([1, 2, 1, 3, 1]),
            1,
            np.array([2, 3]),
            id="int-multiple",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat", "bear"]),
            "cat",
            np.array(["dog", "bear"]),
            id="string-multiple",
        ),
        pytest.param(
            np.array(["cat", "dog", "cat"], dtype=object),
            "cat",
            np.array(["dog"], dtype=object),
            id="object-string-multiple",
        ),
        pytest.param(
            np.array([1.0, 2.0, 1.0], dtype=object),
            1.0,
            np.array([2.0], dtype=object),
            id="object-float-multiple",
        ),
        pytest.param(
            np.array([True, False, True]),
            True,
            np.array([False]),
            id="bool-true",
        ),
        pytest.param(
            np.array([True, False, True]),
            False,
            np.array([True, True]),
            id="bool-false",
        ),
    ],
)
def test_remove_values_partial_match(arr: np.ndarray, value: object, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_values(arr, value), expected)


# --- NaN as value ---


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
            np.array([float("nan"), "cat", float("nan")], dtype=object),
            np.array(["cat"], dtype=object),
            id="object-mixed-nan-multiple",
        ),
    ],
)
def test_remove_values_nan(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_values(arr, float("nan")), expected)


# --- None as value ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1, None, 2, None], dtype=object),
            np.array([1, 2], dtype=object),
            id="object-int-none",
        ),
        pytest.param(
            np.array(["cat", None, "dog", None], dtype=object),
            np.array(["cat", "dog"], dtype=object),
            id="object-string-none",
        ),
        pytest.param(
            np.array([None, None], dtype=object),
            np.array([], dtype=object),
            id="object-none-all",
        ),
        pytest.param(
            np.array([1, 2, 3], dtype=object),
            np.array([1, 2, 3], dtype=object),
            id="object-none-absent",
        ),
    ],
)
def test_remove_values_none(arr: np.ndarray, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_values(arr, None), expected)


# --- mixed object arrays with heterogeneous types ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(
            np.array([1, "cat", 2, "cat"], dtype=object),
            "cat",
            np.array([1, 2], dtype=object),
            id="object-int-string-remove-string",
        ),
        pytest.param(
            np.array([1, "cat", 2, "cat"], dtype=object),
            1,
            np.array(["cat", 2, "cat"], dtype=object),
            id="object-int-string-remove-int",
        ),
        pytest.param(
            np.array([1, float("nan"), "cat", float("nan")], dtype=object),
            float("nan"),
            np.array([1, "cat"], dtype=object),
            id="object-int-string-nan-remove-nan",
        ),
        pytest.param(
            np.array([None, 1, "cat", None], dtype=object),
            None,
            np.array([1, "cat"], dtype=object),
            id="object-none-int-string-remove-none",
        ),
    ],
)
def test_remove_values_mixed_object(arr: np.ndarray, value: object, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_values(arr, value), expected)


# --- dtype is preserved ---


@pytest.mark.parametrize(
    ("arr", "value", "expected_dtype"),
    [
        pytest.param(np.array([1.0, 2.0, 1.0]), 1.0, np.float64, id="float64"),
        pytest.param(np.array([1.0, 2.0, 1.0], dtype=np.float32), 1.0, np.float32, id="float32"),
        pytest.param(np.array([1, 2, 1]), 1, np.intp, id="int"),
        pytest.param(np.array([1.0, 2.0, 1.0], dtype=object), 1.0, object, id="object"),
    ],
)
def test_remove_values_preserves_dtype(
    arr: np.ndarray, value: object, expected_dtype: np.dtype
) -> None:
    assert remove_values(arr, value).dtype == expected_dtype


# --- empty array ---


def test_remove_values_empty() -> None:
    arr = np.array([])
    assert objects_are_equal(remove_values(arr, 1.0), arr)


# --- single element ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1.0]), 1.0, np.array([]), id="float-match"),
        pytest.param(np.array([1.0]), 2.0, np.array([1.0]), id="float-no-match"),
        pytest.param(np.array([float("nan")]), float("nan"), np.array([]), id="float-nan"),
    ],
)
def test_remove_values_single_element(arr: np.ndarray, value: object, expected: np.ndarray) -> None:
    assert objects_are_equal(remove_values(arr, value), expected)


# --- non-1D input raises ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), id="2d"),
        pytest.param(np.ones((2, 3, 4)), id="3d"),
    ],
)
def test_remove_values_non_1d_raises(arr: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"input: expected .*D array, got shape"):
        remove_values(arr, 1.0)


def test_remove_values_non_1d_error_includes_shape() -> None:
    arr = np.ones((3, 4))
    with pytest.raises(ValueError, match=r"input: expected 1D array, got shape"):
        remove_values(arr, 1.0)
