r"""Result objects tailored to classification metrics.

Example:
    ```pycon
    >>> from metriclab.results import AccuracyResult
    >>> AccuracyResult(num_correct_predictions=3, num_predictions=4).to_dict()
    {'accuracy': 0.75, 'num_correct_predictions': 3, 'num_predictions': 4}

    ```
"""
