from __future__ import annotations

import pytest

from metriclab.utils.missing import MISSING_POLICIES, check_missing_policy

##########################################
#     Tests for check_missing_policy     #
##########################################


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_check_missing_policy_valid(missing_policy: str) -> None:
    check_missing_policy(missing_policy)


def test_check_missing_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'missing_policy': incorrect"):
        check_missing_policy("incorrect")
