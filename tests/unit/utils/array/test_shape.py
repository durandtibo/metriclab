from __future__ import annotations

import numpy as np
import pytest

from metriclab.utils.array import validate_array_ndim, validate_same_shape

#########################################
#     Tests for validate_array_ndim     #
#########################################


@pytest.mark.parametrize("shape", [(2,), (1,), (3,)])
def test_validate_array_ndim_1(shape: tuple[int, ...]) -> None:
    validate_array_ndim(arr=np.ones(shape), ndim=1)


@pytest.mark.parametrize("shape", [(2, 3), (1, 1), (3, 2)])
def test_validate_array_ndim_2(shape: tuple[int, ...]) -> None:
    validate_array_ndim(arr=np.ones(shape), ndim=2)


def test_validate_array_ndim_incorrect() -> None:
    with pytest.raises(ValueError, match="input: expected 3D array"):
        validate_array_ndim(np.ones((2, 3)), ndim=3)


def test_validate_array_ndim_incorrect_custom_name() -> None:
    with pytest.raises(ValueError, match="predictions: expected 4D array"):
        validate_array_ndim(np.ones((2, 3)), ndim=4, name="predictions")


#########################################
#     Tests for validate_same_shape     #
#########################################


def test_validate_same_shape_1_array() -> None:
    validate_same_shape([np.array([1, 0, 0, 1, 1])])


def test_validate_same_shape_2_arrays_correct() -> None:
    validate_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5])])


def test_validate_same_shape_2_arrays_incorrect() -> None:
    with pytest.raises(ValueError, match="arrays have different shapes"):
        validate_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 0, 0, 1])])


def test_validate_same_shape_3_arrays_correct() -> None:
    validate_same_shape(
        [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1])]
    )


def test_validate_same_shape_3_arrays_incorrect() -> None:
    with pytest.raises(ValueError, match="arrays have different shapes"):
        validate_same_shape(
            [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4]), np.array([6, 5, 4, 3, 2, 1])]
        )
