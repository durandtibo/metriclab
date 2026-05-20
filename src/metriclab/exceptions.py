r"""Define the base exceptions."""

from __future__ import annotations

__all__ = ["EmptyMetricError"]


class EmptyMetricError(Exception):
    r"""Raised when an empty metric is evaluated."""
