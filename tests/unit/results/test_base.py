from __future__ import annotations

from coola.equality.tester import get_default_registry

from metriclab.results import BaseResult


def test_has_equality_tester_for_base_result() -> None:
    assert get_default_registry().has_equality_tester(BaseResult)
