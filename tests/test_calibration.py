from __future__ import annotations

import numpy as np

from industrial_patch_inspector.calibration import calibrate_sample_threshold, with_sample_threshold
from industrial_patch_inspector.config import AppConfig, CalibrationConfig, DatasetConfig


def test_calibration_prefers_highest_threshold_with_same_recall() -> None:
    result = calibrate_sample_threshold(
        sample_probabilities=np.asarray([0.1, 0.2, 0.3, 0.8], dtype=np.float32),
        sample_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        calibration=CalibrationConfig(
            ok_fpr_target=0.5,
            min_threshold=0.1,
            max_threshold=0.4,
            num_thresholds=4,
        ),
    )

    assert result.recommended_threshold == np.float32(0.3)
    assert result.ok_fpr == 0.0
    assert result.ng_recall == 1.0
    assert result.ok_count == 2
    assert result.ng_count == 2


def test_calibration_handles_missing_ok_examples() -> None:
    result = calibrate_sample_threshold(
        sample_probabilities=np.asarray([0.7, 0.9], dtype=np.float32),
        sample_labels=np.asarray([1, 1], dtype=np.int64),
        calibration=CalibrationConfig(),
    )

    assert result.recommended_threshold == 0.5
    assert result.ok_fpr == 0.0
    assert result.ng_recall == 1.0
    assert result.ok_count == 0
    assert result.ng_count == 2


def test_with_sample_threshold_overrides_runtime_threshold() -> None:
    config = AppConfig(dataset=DatasetConfig(root="dataset"))

    updated = with_sample_threshold(config, 0.73)

    assert updated is config
    assert updated.aggregation.sample_threshold == 0.73
    assert updated.aggregation.patch_threshold == 0.55
