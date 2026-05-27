from __future__ import annotations

import math

import numpy as np
import pytest

from metriclab.utils.nan import is_nan

##################################
#       Tests for is_nan         #
##################################


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(float("nan"), True, id="python_nan"),
        pytest.param(math.nan, True, id="math_nan"),
        pytest.param(np.nan, True, id="numpy_nan"),
        pytest.param(np.float64("nan"), True, id="numpy_float64_nan"),
        pytest.param(1.0, False, id="float"),
        pytest.param(0.0, False, id="zero"),
        pytest.param(float("inf"), False, id="positive_inf"),
        pytest.param(-float("inf"), False, id="negative_inf"),
        pytest.param(1, False, id="int"),
        pytest.param(True, False, id="bool_true"),
        pytest.param(False, False, id="bool_false"),
        pytest.param("nan", False, id="string"),
        pytest.param(None, False, id="none"),
        pytest.param(object(), False, id="object"),
    ],
)
def test_is_nan(value: object, expected: bool) -> None:
    """Test is_nan for valid and invalid inputs."""
    assert is_nan(value) is expected
