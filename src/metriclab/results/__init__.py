r"""Immutable containers that store computed metric values.

Result objects expose a consistent API to combine partial results and
export aggregates as dictionaries.
"""

from __future__ import annotations

__all__ = ["BaseResult"]

from metriclab.results.base import BaseResult
