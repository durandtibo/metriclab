r"""Immutable containers that store computed metric values.

Result objects expose a consistent API to compare metric values, export
aggregates as dictionaries, and format them for display.

Example:
    ```pycon
    >>> from metriclab.results import Result
    >>> result = Result({"accuracy": 0.9, "loss": 0.1})
    >>> result.to_dict(prefix="val_")
    {'val_accuracy': 0.9, 'val_loss': 0.1}

    ```
"""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult", "Result"]

from metriclab.results.base import BaseResult
from metriclab.results.classification.accuracy import AccuracyResult
from metriclab.results.vanilla import Result
