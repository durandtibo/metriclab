from __future__ import annotations

import pytest

ignore_single_label_warning = pytest.mark.filterwarnings(
    "ignore:A single label was found in 'y_true' and 'y_pred':UserWarning"
)
