from __future__ import annotations

import numpy as np

from industrial_patch_inspector.analysis import (
    build_thresholds,
    select_best_f1,
    select_best_recall_under_ok_fpr,
    summarize_patch_zones,
    sweep_binary_thresholds,
)


def test_threshold_sweep_selection_helpers() -> None:
    thresholds = build_thresholds(0.1, 0.3, 3)
    entries = sweep_binary_thresholds(
        probabilities=np.asarray([0.1, 0.2, 0.3, 0.8], dtype=np.float32),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        thresholds=thresholds,
    )

    best_f1 = select_best_f1(entries)
    best_recall = select_best_recall_under_ok_fpr(entries, ok_fpr_target=0.5)

    assert [round(float(item["threshold"]), 2) for item in entries] == [0.1, 0.2, 0.3]
    assert round(float(best_f1["threshold"]), 2) == 0.3
    assert round(float(best_recall["threshold"]), 2) == 0.3


def test_patch_zone_summary_surfaces_high_fp_zone() -> None:
    thresholds = build_thresholds(0.3, 0.8, 6)
    summary = summarize_patch_zones(
        patch_probabilities=np.asarray([0.7, 0.8, 0.4, 0.6, 0.2, 0.9], dtype=np.float32),
        patch_labels=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
        zone_rows=np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64),
        zone_cols=np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64),
        thresholds=thresholds,
        default_threshold=0.55,
        ok_fpr_target=0.0,
        top_k=2,
    )

    assert summary["zone_count"] == 2
    assert summary["high_fp_zones"][0]["zone"] == "r00_c00"
    assert round(float(summary["high_fp_zones"][0]["default_ok_fpr"]), 2) == 1.0
    assert round(float(summary["zones"][0]["recommended_patch_threshold"]), 2) == 0.8
