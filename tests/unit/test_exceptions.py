from __future__ import annotations

import pytest

from metriclab.exceptions import EmptyMetricError

######################################
#     Tests for EmptyMetricError     #
######################################


def test_empty_metric_error_is_exception() -> None:
    assert issubclass(EmptyMetricError, Exception)


def test_empty_metric_error_can_be_raised_and_caught() -> None:
    msg = "no predictions"
    with pytest.raises(EmptyMetricError, match=msg):
        raise EmptyMetricError(msg)


def test_empty_metric_error_stores_message() -> None:
    msg = "Cannot compute metric: no valid predictions"
    err = EmptyMetricError(msg)
    assert str(err) == msg


def test_empty_metric_error_can_be_caught_as_exception() -> None:
    msg = "no predictions"
    with pytest.raises(Exception, match=msg):
        raise EmptyMetricError(msg)
