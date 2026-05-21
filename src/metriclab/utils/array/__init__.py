r"""Helpers to validate and process array-like inputs."""

from __future__ import annotations

__all__ = [
    "NAN_POLICIES",
    "NanPolicy",
    "check_nan_policy",
    "contains_nan",
    "contains_value",
    "to_numpy",
    "to_numpy_1d",
    "validate_array_ndim",
    "validate_nan_policy",
    "validate_same_shape",
]

from metriclab.utils.array.conversion import to_numpy, to_numpy_1d
from metriclab.utils.array.nan import (
    NAN_POLICIES,
    NanPolicy,
    check_nan_policy,
    contains_nan,
    validate_nan_policy,
)
from metriclab.utils.array.search import contains_value
from metriclab.utils.array.shape import validate_array_ndim, validate_same_shape
