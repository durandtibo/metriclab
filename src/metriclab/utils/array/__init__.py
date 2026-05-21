r"""Helpers to validate and process array-like inputs."""

from __future__ import annotations

__all__ = [
    "NAN_POLICIES",
    "NanPolicy",
    "check_nan_policy",
    "contains_nan",
    "contains_value",
    "equal_to",
    "is_nan",
    "multi_contains_value",
    "multi_equal_to",
    "multi_is_nan",
    "preprocess_1d",
    "to_numpy",
    "to_numpy_1d",
    "validate_array_ndim",
    "validate_nan_policy",
    "validate_same_shape",
]

from metriclab.utils.array.conversion import to_numpy, to_numpy_1d
from metriclab.utils.array.equal import equal_to, multi_equal_to
from metriclab.utils.array.nan import (
    NAN_POLICIES,
    NanPolicy,
    check_nan_policy,
    contains_nan,
    is_nan,
    multi_is_nan,
    validate_nan_policy,
)
from metriclab.utils.array.preprocessing import preprocess_1d
from metriclab.utils.array.search import contains_value, multi_contains_value
from metriclab.utils.array.shape import validate_array_ndim, validate_same_shape
