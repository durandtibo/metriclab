r"""Validation helpers for missing-value handling policies."""

from __future__ import annotations

__all__ = ["MISSING_POLICIES", "MissingPolicy", "validate_missing_policy"]

from typing import Literal

MISSING_POLICIES = ["omit", "propagate", "raise"]

MissingPolicy = Literal["omit", "propagate", "raise"]


def validate_missing_policy(missing_policy: str) -> None:
    r"""Validate a missing-value policy value.

    Args:
        missing_policy: The policy name to validate.

    Raises:
        ValueError: If ``missing_policy`` is not one of
            :obj:`metriclab.utils.missing.MISSING_POLICIES`.

    Example:
        ```pycon
        >>> from metriclab.utils.missing import validate_missing_policy
        >>> validate_missing_policy("omit")

        ```
    """
    if missing_policy not in set(MISSING_POLICIES):
        msg = (
            f"Incorrect 'missing_policy': {missing_policy}. The valid values are: "
            f"'omit', 'propagate', 'raise'"
        )
        raise ValueError(msg)
