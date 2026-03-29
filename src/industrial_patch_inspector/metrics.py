from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(slots=True)
class BinaryClassificationMetrics:
    threshold: float
    sample_count: int
    ok_count: int
    ng_count: int
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    predicted_ng_count: int
    predicted_ok_count: int
    mean_probability: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_binary_classification_metrics(
    sample_labels: np.ndarray,
    sample_probabilities: np.ndarray,
    threshold: float,
) -> BinaryClassificationMetrics:
    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(sample_probabilities, dtype=np.float32).reshape(-1)
    if labels.shape != probabilities.shape:
        msg = "sample_labels and sample_probabilities must have the same shape"
        raise ValueError(msg)

    predictions = (probabilities >= threshold).astype(np.int64)
    negatives = labels == 0
    positives = labels == 1

    true_negative = int(np.sum((predictions == 0) & negatives))
    false_positive = int(np.sum((predictions == 1) & negatives))
    false_negative = int(np.sum((predictions == 0) & positives))
    true_positive = int(np.sum((predictions == 1) & positives))

    accuracy = _safe_divide(true_negative + true_positive, int(labels.size))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return BinaryClassificationMetrics(
        threshold=float(threshold),
        sample_count=int(labels.size),
        ok_count=int(np.sum(negatives)),
        ng_count=int(np.sum(positives)),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        predicted_ng_count=int(np.sum(predictions == 1)),
        predicted_ok_count=int(np.sum(predictions == 0)),
        mean_probability=float(probabilities.mean()) if probabilities.size > 0 else 0.0,
    )
