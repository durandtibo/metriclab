from __future__ import annotations

import numpy as np
import pytest

from metriclab.utils.array import to_numpy, to_numpy_1d

####################
#     to_numpy     #
####################


def test_to_numpy_numpy_array() -> None:
    arr = np.array([1, 2, 3])
    assert np.array_equal(to_numpy(arr), np.array([1, 2, 3]))


def test_to_numpy_numpy_array_returns_same_object() -> None:
    arr = np.array([1, 2, 3])
    assert to_numpy(arr) is arr


def test_to_numpy_list() -> None:
    assert np.array_equal(to_numpy([1, 2, 3]), np.array([1, 2, 3]))


def test_to_numpy_tuple() -> None:
    assert np.array_equal(to_numpy((1, 2, 3)), np.array([1, 2, 3]))


def test_to_numpy_empty_list() -> None:
    assert np.array_equal(to_numpy([]), np.array([]))


def test_to_numpy_empty_tuple() -> None:
    assert np.array_equal(to_numpy(()), np.array([]))


def test_to_numpy_2d_list() -> None:
    assert np.array_equal(to_numpy([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))


def test_to_numpy_float_values() -> None:
    assert np.array_equal(to_numpy([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_to_numpy_unsupported_type() -> None:
    with pytest.raises(TypeError, match="input: unsupported type"):
        to_numpy({"a": 1})


def test_to_numpy_unsupported_type_custom_name() -> None:
    with pytest.raises(TypeError, match="predictions: unsupported type"):
        to_numpy({"a": 1}, name="predictions")


#######################
#     to_numpy_1d     #
#######################


def test_to_numpy_1d_numpy_array() -> None:
    arr = np.array([1, 2, 3])
    assert np.array_equal(to_numpy_1d(arr), np.array([1, 2, 3]))


def test_to_numpy_1d_numpy_array_returns_same_object() -> None:
    arr = np.array([1, 2, 3])
    assert to_numpy_1d(arr) is arr


def test_to_numpy_1d_list() -> None:
    assert np.array_equal(to_numpy_1d([1, 2, 3]), np.array([1, 2, 3]))


def test_to_numpy_1d_tuple() -> None:
    assert np.array_equal(to_numpy_1d((1, 2, 3)), np.array([1, 2, 3]))


def test_to_numpy_1d_empty_list() -> None:
    assert np.array_equal(to_numpy_1d([]), np.array([]))


def test_to_numpy_1d_2d_array() -> None:
    with pytest.raises(ValueError, match="input: expected 1D array"):
        to_numpy_1d(np.array([[1, 2], [3, 4]]))


def test_to_numpy_1d_0d_array() -> None:
    with pytest.raises(ValueError, match="input: expected 1D array"):
        to_numpy_1d(np.array(1))


def test_to_numpy_1d_2d_array_custom_name() -> None:
    with pytest.raises(ValueError, match="predictions: expected 1D array"):
        to_numpy_1d(np.array([[1, 2], [3, 4]]), name="predictions")


def test_to_numpy_1d_unsupported_type() -> None:
    with pytest.raises(TypeError, match="input: unsupported type"):
        to_numpy_1d({"a": 1})


def test_to_numpy_1d_unsupported_type_custom_names() -> None:
    with pytest.raises(TypeError, match="predictions: unsupported type"):
        to_numpy_1d({"a": 1}, name="predictions")
