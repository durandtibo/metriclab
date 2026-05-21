r"""Utilities and result containers for evaluating model predictions.

The public API is currently centered around immutable result objects such as
``metriclab.results.Result`` and ``metriclab.results.AccuracyResult``.

Example:
    ```pycon
    >>> from metriclab.results import AccuracyResult
    >>> AccuracyResult(num_correct_predictions=8, num_predictions=10).accuracy
    0.8

    ```
"""

from __future__ import annotations

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"
