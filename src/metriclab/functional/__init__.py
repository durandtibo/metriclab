r"""Namespace for stateless metric functions.

This package is reserved for functional APIs that compute metrics
directly from inputs without storing intermediate state.
"""

from __future__ import annotations

__all__ = ["accuracy"]

from metriclab.functional.classification.accuracy import accuracy
