from __future__ import annotations

import pytest

from metriclab.utils.missing import MISSING_POLICIES, validate_missing_policy

#############################################
#     Tests for validate_missing_policy     #
#############################################


@pytest.mark.parametrize("missing_policy", MISSING_POLICIES)
def test_validate_missing_policy_valid(missing_policy: str) -> None:
    validate_missing_policy(missing_policy)


def test_validate_missing_policy_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'missing_policy': incorrect"):
        validate_missing_policy("incorrect")
