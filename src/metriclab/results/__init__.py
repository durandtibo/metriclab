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

__all__ = ["AccuracyResult", "BalancedAccuracyResult", "BaseResult", "Result", "ResultDict"]

from metriclab.results.base import BaseResult
from metriclab.results.classification.accuracy import AccuracyResult
from metriclab.results.classification.balanced_accuracy import BalancedAccuracyResult
from metriclab.results.mapping import ResultDict
from metriclab.results.vanilla import Result
