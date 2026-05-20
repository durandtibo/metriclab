r"""Define pytest mark decorators for conditional test skipping.

``pytest`` is required to use these decorators.
"""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "colorlog_not_available",
    "rich_available",
    "rich_not_available",
    "sklearn_available",
    "sklearn_not_available",
]

import pytest

from metriclab.utils.imports import (
    is_colorlog_available,
    is_rich_available,
    is_sklearn_available,
)

colorlog_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_colorlog_available(), reason="Requires colorlog"
)
colorlog_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_colorlog_available(), reason="Skip if colorlog is available"
)
rich_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_rich_available(), reason="Requires rich"
)
rich_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_rich_available(), reason="Skip if rich is available"
)
sklearn_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_sklearn_available(), reason="Requires scikit-learn"
)
sklearn_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_sklearn_available(), reason="Skip if scikit-learn is available"
)
