from __future__ import annotations

from metriclab.exceptions import EmptyMetricError

######################################
#     Tests for EmptyMetricError     #
######################################


def test_empty_metric_error_is_exception() -> None:
    assert issubclass(EmptyMetricError, Exception)
