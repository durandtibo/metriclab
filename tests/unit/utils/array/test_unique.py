from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from metriclab.utils.array import count_unique_non_missing

############################################
#   Tests for count_unique_non_missing     #
############################################


# --- empty arrays ---


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([]), id="empty-float"),
        pytest.param(np.array([], dtype=int), id="empty-int"),
        pytest.param(np.array([], dtype=object), id="empty-object"),
    ],
)
def test_count_unique_non_missing_empty(arr: np.ndarray) -> None:
    assert count_unique_non_missing(arr) == 0


def test_count_unique_non_missing_empty_missing_values_set() -> None:
    assert count_unique_non_missing(np.array([], dtype=float), missing_values=float("nan")) == 0


# --- missing_values not set ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 2, 3]), 3, id="int-all-unique"),
        pytest.param(np.array([1, 1, 1]), 1, id="int-all-same"),
        pytest.param(np.array([1, 0, 0, 1, 1, 2]), 3, id="int-duplicates"),
        pytest.param(np.array([1.0, 2.0, 3.0]), 3, id="float-all-unique"),
        pytest.param(np.array([1.0, 0.0, 0.0, 1.0, 2.0]), 3, id="float-duplicates"),
        pytest.param(np.array(["cat", "dog", "bird"]), 3, id="str-all-unique"),
        pytest.param(np.array(["cat", "cat", "dog"]), 2, id="str-duplicates"),
        pytest.param(np.array([True, False]), 2, id="bool"),
        pytest.param(np.array([1]), 1, id="single-element"),
    ],
)
def test_count_unique_non_missing_no_missing_values(arr: np.ndarray, expected: int) -> None:
    assert count_unique_non_missing(arr) == expected


# --- object arrays without missing_values — np.unique TypeError path ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1, "cat", 3.0, "dog"], dtype=object),
            4,
            id="mixed-int-str-float",
        ),
        pytest.param(
            np.array([1, "cat", 1, "cat"], dtype=object),
            2,
            id="mixed-duplicates",
        ),
        pytest.param(
            np.array([1, "cat"], dtype=object),
            2,
            id="mixed-two-elements",
        ),
        pytest.param(
            np.array([None, 1, "cat"], dtype=object),
            3,
            id="mixed-with-none-no-missing-values",
        ),
    ],
)
def test_count_unique_non_missing_object_mixed_no_missing_values(
    arr: np.ndarray, expected: int
) -> None:
    # np.unique raises TypeError on mixed types — falls back to set-based
    assert count_unique_non_missing(arr) == expected


# --- NaN counted as unique when missing_values not set ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(
            np.array([1.0, 0.0, 2.0, float("nan")]),
            4,
            id="float-one-nan",
        ),
        pytest.param(
            np.array([1.0, float("nan"), float("nan")]),
            2,
            id="float-multiple-nan-counted-once",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")]),
            1,
            id="float-all-nan",
        ),
        pytest.param(
            np.array([1, float("nan"), 2], dtype=object),
            3,
            id="object-nan",
        ),
        pytest.param(
            np.array([float("nan"), float("nan")], dtype=object),
            1,
            id="object-multiple-nan-counted-once",
        ),
    ],
)
def test_count_unique_non_missing_nan_counted_without_missing_values(
    arr: np.ndarray, expected: int
) -> None:
    assert count_unique_non_missing(arr) == expected


# --- missing_values=float("nan") ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1.0, 0.0, 2.0, float("nan")]), 3, id="float-one-nan"),
        pytest.param(np.array([1.0, float("nan"), float("nan")]), 1, id="float-multiple-nan"),
        pytest.param(np.array([float("nan"), float("nan")]), 0, id="float-all-nan"),
        pytest.param(np.array([1.0, 0.0, 2.0]), 3, id="float-no-nan"),
        pytest.param(np.array([1, float("nan"), 2], dtype=object), 2, id="object-nan"),
        pytest.param(np.array([float("nan"), float("nan")], dtype=object), 0, id="object-all-nan"),
    ],
)
def test_count_unique_non_missing_nan(arr: np.ndarray, expected: int) -> None:
    assert count_unique_non_missing(arr, missing_values=float("nan")) == expected


# --- missing_values=None ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, None, 3], dtype=object), 2, id="object-one-none"),
        pytest.param(np.array([None, None, None], dtype=object), 0, id="object-all-none"),
        pytest.param(np.array([1, None, None, 2], dtype=object), 2, id="object-multiple-none"),
        pytest.param(np.array([1, 2, 3], dtype=object), 3, id="object-no-none"),
        pytest.param(np.array([1, 2, 3]), 3, id="non-object-no-none"),
    ],
)
def test_count_unique_non_missing_none(arr: np.ndarray, expected: int) -> None:
    assert count_unique_non_missing(arr, missing_values=None) == expected


# --- missing_values=float("inf") ---


@pytest.mark.parametrize(
    ("arr", "value", "expected"),
    [
        pytest.param(np.array([1.0, float("inf"), 2.0]), float("inf"), 2, id="pos-inf"),
        pytest.param(np.array([1.0, float("-inf"), 2.0]), float("-inf"), 2, id="neg-inf"),
        pytest.param(np.array([float("inf"), float("inf")]), float("inf"), 0, id="all-pos-inf"),
        pytest.param(
            np.array([float("inf"), float("-inf"), 1.0]),
            float("inf"),
            2,
            id="pos-inf-not-confused-with-neg-inf",
        ),
        pytest.param(np.array([1.0, 2.0, 3.0]), float("inf"), 3, id="inf-absent"),
    ],
)
def test_count_unique_non_missing_inf(arr: np.ndarray, value: float, expected: int) -> None:
    assert count_unique_non_missing(arr, missing_values=value) == expected


# --- missing_values=int sentinel ---


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        pytest.param(np.array([1, 99, 2, 3]), 3, id="int-sentinel-present"),
        pytest.param(np.array([1, 2, 3]), 3, id="int-sentinel-absent"),
        pytest.param(np.array([99, 99, 99]), 0, id="int-sentinel-all"),
    ],
)
def test_count_unique_non_missing_int_sentinel(arr: np.ndarray, expected: int) -> None:
    assert count_unique_non_missing(arr, missing_values=99) == expected


# --- object arrays with mixed types and missing_values ---


@pytest.mark.parametrize(
    ("arr", "missing_values", "expected"),
    [
        pytest.param(
            np.array([1, "cat", 3.0], dtype=object),
            None,
            3,
            id="mixed-no-none",
        ),
        pytest.param(
            np.array([1, "cat", None, 3.0], dtype=object),
            None,
            3,
            id="mixed-with-none",
        ),
        pytest.param(
            np.array([1, "cat", float("nan"), 3.0], dtype=object),
            float("nan"),
            3,
            id="mixed-with-nan",
        ),
        pytest.param(
            np.array([1, 1, "cat", "cat"], dtype=object),
            None,
            2,
            id="mixed-duplicates",
        ),
        pytest.param(
            np.array([None, "cat", None, "cat"], dtype=object),
            None,
            1,
            id="mixed-all-none-and-one-unique",
        ),
    ],
)
def test_count_unique_non_missing_object_mixed(
    arr: np.ndarray, missing_values: Any, expected: int
) -> None:
    assert count_unique_non_missing(arr, missing_values=missing_values) == expected


# --- NaN and None in the same array ---


@pytest.mark.parametrize(
    ("arr", "missing_values", "expected"),
    [
        pytest.param(
            np.array([1, float("nan"), None, 2], dtype=object),
            float("nan"),
            3,
            id="nan-missing-none-kept",
        ),
        pytest.param(
            np.array([1, float("nan"), None, 2], dtype=object),
            None,
            3,
            id="none-missing-nan-kept",
        ),
        pytest.param(
            np.array([float("nan"), None, float("nan"), None], dtype=object),
            float("nan"),
            1,
            id="nan-missing-only-none-remains",
        ),
        pytest.param(
            np.array([float("nan"), None, float("nan"), None], dtype=object),
            None,
            1,
            id="none-missing-only-nan-remains",
        ),
        pytest.param(
            np.array([float("nan"), None], dtype=object),
            float("nan"),
            1,
            id="nan-and-none-nan-excluded",
        ),
        pytest.param(
            np.array([float("nan"), None], dtype=object),
            None,
            1,
            id="nan-and-none-none-excluded",
        ),
    ],
)
def test_count_unique_non_missing_nan_and_none(
    arr: np.ndarray, missing_values: Any, expected: int
) -> None:
    assert count_unique_non_missing(arr, missing_values=missing_values) == expected


# --- single element ---


@pytest.mark.parametrize(
    ("arr", "missing_values", "expected"),
    [
        pytest.param(np.array([1]), None, 1, id="int-not-missing"),
        pytest.param(np.array([None], dtype=object), None, 0, id="none-is-missing"),
        pytest.param(np.array([float("nan")]), float("nan"), 0, id="nan-is-missing"),
        pytest.param(np.array([1.0]), float("nan"), 1, id="float-not-missing"),
        pytest.param(
            np.array([float("nan")], dtype=object), float("nan"), 0, id="object-nan-is-missing"
        ),
        pytest.param(np.array(["cat"], dtype=object), float("nan"), 1, id="object-str-not-missing"),
    ],
)
def test_count_unique_non_missing_single_element(
    arr: np.ndarray, missing_values: Any, expected: int
) -> None:
    assert count_unique_non_missing(arr, missing_values=missing_values) == expected


# --- return type ---


def test_count_unique_non_missing_returns_int() -> None:
    assert isinstance(count_unique_non_missing(np.array([1, 2, 3])), int)


def test_count_unique_non_missing_returns_int_with_missing_values() -> None:
    assert isinstance(
        count_unique_non_missing(np.array([1.0, float("nan")]), missing_values=float("nan")),
        int,
    )
