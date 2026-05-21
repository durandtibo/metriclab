r"""Helpers to validate and process array-like inputs."""

from __future__ import annotations

__all__ = [
    "NAN_POLICIES",
    "NanPolicy",
    "check_nan_policy",
    "contains_nan",
    "contains_value",
    "validate_nan_policy",
]

from metriclab.utils.array.nan import (
    NAN_POLICIES,
    NanPolicy,
    check_nan_policy,
    contains_nan,
    validate_nan_policy,
)
from metriclab.utils.array.search import contains_value
