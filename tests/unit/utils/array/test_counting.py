from __future__ import annotations

import numpy as np
import pytest

from metriclab.utils.array import count_values

##################################################
#           Tests for count_values               #
##################################################


# --- value not present: returns 0 ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1, 2, 3]), 4, id="int-absent"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 4.0, id="float-absent"),
        pytest.param(np.array(["cat", "dog"]), "bear", id="string-absent"),
        pytest.param(np.array(["cat", "dog"], dtype=object), "bear", id="object-string-absent"),
        pytest.param(np.array([1.0, 2.0], dtype=object), 3.0, id="object-float-absent"),
        pytest.param(np.array([1.0, 2.0, 3.0]), float("nan"), id="float-no-nan"),
        pytest.param(np.array([]), 1.0, id="empty"),
    ],
)
def test_count_values_not_present(arr: np.ndarray, value: object) -> None:
    assert count_values(arr, value) == 0


# --- single occurrence ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1, 2, 3]), 1, id="int-start"),
        pytest.param(np.array([1, 2, 3]), 2, id="int-middle"),
        pytest.param(np.array([1, 2, 3]), 3, id="int-end"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 2.0, id="float"),
        pytest.param(np.array(["cat", "dog", "bear"]), "dog", id="string"),
        pytest.param(np.array([1.0, float("nan"), 2.0]), float("nan"), id="float-nan"),
        pytest.param(
            np.array([1.0, float("nan"), 2.0], dtype=object), float("nan"), id="object-nan"
        ),
        pytest.param(np.array([1, None, 2], dtype=object), None, id="object-none"),
    ],
)
def test_count_values_single_occurrence(arr: np.ndarray, value: object) -> None:
    assert count_values(arr, value) == 1


# --- multiple occurrences ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1, 2, 1, 3, 1]), 1, 3, id="int-three"),
        pytest.param(np.array([1.0, 2.0, 1.0]), 1.0, 2, id="float-two"),
        pytest.param(np.array(["cat", "dog", "cat", "bear", "cat"]), "cat", 3, id="string-three"),
        pytest.param(
            np.array(["cat", "dog", "cat"], dtype=object), "cat", 2, id="object-string-two"
        ),
        pytest.param(
            np.array([float("nan"), 1.0, float("nan"), 2.0, float("nan")]),
            float("nan"),
            3,
            id="float-nan-three",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object),
            float("nan"),
            2,
            id="object-nan-two",
        ),
        pytest.param(
            np.array([None, 1, None, 2, None], dtype=object), None, 3, id="object-none-three"
        ),
    ],
)
def test_count_values_multiple_occurrences(arr: np.ndarray, value: object, expected: int) -> None:
    assert count_values(arr, value) == expected


# --- all elements match ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1, 1, 1]), 1, id="int"),
        pytest.param(np.array([2.0, 2.0, 2.0]), 2.0, id="float"),
        pytest.param(np.array(["cat", "cat"]), "cat", id="string"),
        pytest.param(np.array([float("nan"), float("nan")]), float("nan"), id="float-nan"),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object), float("nan"), id="object-nan"
        ),
        pytest.param(np.array([None, None], dtype=object), None, id="object-none"),
    ],
)
def test_count_values_all_match(arr: np.ndarray, value: object) -> None:
    assert count_values(arr, value) == arr.size


# --- multi-dimensional arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([[1, 2], [1, 3]]), 1, 2, id="2d-int"),
        pytest.param(np.array([[1.0, 2.0], [3.0, 1.0]]), 1.0, 2, id="2d-float"),
        pytest.param(
            np.array([[float("nan"), 1.0], [float("nan"), 2.0]]), float("nan"), 2, id="2d-nan"
        ),
        pytest.param(np.ones((2, 3, 4)), 1.0, 24, id="3d-float"),
        pytest.param(np.zeros((2, 3, 4)), 1.0, 0, id="3d-float-absent"),
    ],
)
def test_count_values_multidimensional(arr: np.ndarray, value: object, expected: int) -> None:
    assert count_values(arr, value) == expected


# --- mixed object arrays ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1, "cat", 1, "dog"], dtype=object), 1, 2, id="int-in-mixed"),
        pytest.param(np.array([1, "cat", 1, "dog"], dtype=object), "cat", 1, id="string-in-mixed"),
        pytest.param(
            np.array([None, 1, "cat", None, float("nan")], dtype=object),
            None,
            2,
            id="none-in-mixed",
        ),
        pytest.param(
            np.array([None, 1, "cat", None, float("nan")], dtype=object),
            float("nan"),
            1,
            id="nan-in-mixed",
        ),
    ],
)
def test_count_values_mixed_object(arr: np.ndarray, value: object, expected: int) -> None:
    assert count_values(arr, value) == expected


# --- return type is always int ---


@pytest.mark.parametrize(
    ("arr", "value"),
    [
        pytest.param(np.array([1, 2, 1]), 1, id="int"),
        pytest.param(np.array([1.0, float("nan")]), float("nan"), id="nan"),
        pytest.param(np.array([1, 2, 3]), 4, id="absent"),
        pytest.param(np.array([]), 1.0, id="empty"),
    ],
)
def test_count_values_returns_int(arr: np.ndarray, value: object) -> None:
    assert isinstance(count_values(arr, value), int)
