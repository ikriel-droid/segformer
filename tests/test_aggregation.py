from __future__ import annotations

import numpy as np

from industrial_patch_inspector.aggregation import PatchScoreAggregator
from industrial_patch_inspector.config import AggregationConfig


def test_aggregate_scores_connected_components() -> None:
    aggregator = PatchScoreAggregator(
        AggregationConfig(
            top_k=2,
            patch_threshold=0.6,
            sample_threshold=0.5,
            topk_weight=0.55,
            component_weight=0.30,
            adjacency_weight=0.15,
            min_component_size=2,
        )
    )
    patch_probabilities = np.asarray([0.9, 0.8, 0.2, 0.7], dtype=np.float32)
    active_mask = np.ones(4, dtype=np.float32)
    grid_indices = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

    result = aggregator.aggregate(
        patch_probabilities,
        active_mask,
        grid_indices,
        grid_shape=(2, 2),
        alignment_score=1.0,
    )

    assert np.array_equal(result["binary_map"], np.asarray([[1, 1], [0, 1]], dtype=np.uint8))
    assert len(result["components"]) == 1
    assert result["components"][0]["size"] == 3
    assert result["components"][0]["coordinates"] == [(0, 0), (0, 1), (1, 1)]
    assert result["sample_probability"] == np.float32(0.7375)


def test_aggregate_returns_zero_for_inactive_grid() -> None:
    aggregator = PatchScoreAggregator(AggregationConfig())
    result = aggregator.aggregate(
        patch_probabilities=np.asarray([0.9, 0.1], dtype=np.float32),
        active_mask=np.asarray([0.0, 0.0], dtype=np.float32),
        grid_indices=np.asarray([[0, 0], [0, 1]], dtype=np.int64),
        grid_shape=(1, 2),
        alignment_score=1.0,
    )

    assert result["sample_probability"] == 0.0
    assert result["components"] == []
    assert np.array_equal(result["binary_map"], np.zeros((1, 2), dtype=np.uint8))
