from __future__ import annotations

import numpy as np

from industrial_patch_inspector.metrics import compute_binary_classification_metrics


def test_compute_binary_classification_metrics() -> None:
    metrics = compute_binary_classification_metrics(
        sample_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        sample_probabilities=np.asarray([0.1, 0.8, 0.6, 0.4], dtype=np.float32),
        threshold=0.5,
    )

    assert metrics.sample_count == 4
    assert metrics.ok_count == 2
    assert metrics.ng_count == 2
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.specificity == 0.5
    assert metrics.f1 == 0.5
    assert metrics.predicted_ng_count == 2
    assert metrics.predicted_ok_count == 2
