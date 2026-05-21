# metriclab

`metriclab` provides lightweight result containers and utility helpers for
evaluating machine-learning predictions.

The library focuses on immutable result objects that expose a consistent API
for comparing metric values, exporting them as dictionaries, and formatting
them for display.

## Quickstart

```pycon
>>> from metriclab.results import AccuracyResult, Result
>>> accuracy = AccuracyResult(num_correct_predictions=7, num_predictions=10)
>>> accuracy.accuracy
0.7
>>> accuracy.to_dict(prefix="val_")
{'val_accuracy': 0.7, 'val_num_correct_predictions': 7, 'val_num_predictions': 10}
>>> Result({"loss": 0.2, "accuracy": 0.7}).to_display()
"{'loss': 0.2, 'accuracy': 0.7}"

```
