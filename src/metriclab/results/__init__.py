r"""Immutable containers that store computed metric values.

Result objects expose a consistent API to combine partial results and
export aggregates as dictionaries.
"""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult", "Result"]

from metriclab.results.base import BaseResult
from metriclab.results.classification.accuracy import AccuracyResult
from metriclab.results.vanilla import Result
