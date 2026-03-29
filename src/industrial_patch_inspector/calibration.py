from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .aggregation import PatchScoreAggregator
from .config import AggregationConfig, AppConfig, CalibrationConfig
from .dataset import CityscapesInspectionDataset, collate_samples
from .models import ContextAwarePatchClassifier


@dataclass(slots=True)
class CalibrationResult:
    recommended_threshold: float
    ok_fpr: float
    ng_recall: float
    ok_count: int
    ng_count: int


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def collect_sample_probabilities(
    config: AppConfig,
    model: ContextAwarePatchClassifier,
    device: torch.device,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = CityscapesInspectionDataset(config, split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_samples,
    )
    aggregator = PatchScoreAggregator(config.aggregation)
    sample_probs: list[float] = []
    sample_labels: list[int] = []
    model.eval()
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(
            batch["tiles"],
            batch["contexts"],
            batch["roi_tiles"],
            batch["context_roi_tiles"],
            batch["positions"],
            batch["active_mask"],
            batch["alignment_score"],
            tuple(batch["grid_shape"][0].tolist()),
        )
        patch_probs = torch.sigmoid(outputs["patch_logits"]).cpu().numpy()
        active_masks = batch["active_mask"].cpu().numpy()
        grid_indices = batch["grid_indices"].cpu().numpy()
        grid_shapes = batch["grid_shape"].cpu().numpy()
        alignment_scores = batch["alignment_score"].cpu().numpy()
        sample_targets = batch["sample_label"].cpu().numpy()
        for idx in range(patch_probs.shape[0]):
            aggregated = aggregator.aggregate(
                patch_probs[idx],
                active_masks[idx],
                grid_indices[idx],
                tuple(grid_shapes[idx].tolist()),
                float(alignment_scores[idx]),
            )
            sample_probs.append(float(aggregated["sample_probability"]))
            sample_labels.append(int(sample_targets[idx] >= 0.5))
    return np.asarray(sample_probs, dtype=np.float32), np.asarray(sample_labels, dtype=np.int64)


def calibrate_sample_threshold(
    sample_probabilities: np.ndarray,
    sample_labels: np.ndarray,
    calibration: CalibrationConfig,
) -> CalibrationResult:
    ok_probs = sample_probabilities[sample_labels == 0]
    ng_probs = sample_probabilities[sample_labels == 1]
    if ok_probs.size == 0:
        threshold = 0.5
        ok_fpr = 0.0
        ng_recall = float((ng_probs >= threshold).mean()) if ng_probs.size > 0 else 0.0
        return CalibrationResult(threshold, ok_fpr, ng_recall, 0, int(ng_probs.size))

    thresholds = np.linspace(
        calibration.min_threshold,
        calibration.max_threshold,
        num=calibration.num_thresholds,
        dtype=np.float32,
    )
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        ok_fpr = float((ok_probs >= threshold).mean())
        ng_recall = float((ng_probs >= threshold).mean()) if ng_probs.size > 0 else 0.0
        candidates.append((float(threshold), ok_fpr, ng_recall))

    feasible = [candidate for candidate in candidates if candidate[1] <= calibration.ok_fpr_target]
    if feasible:
        threshold, ok_fpr, ng_recall = max(feasible, key=lambda item: (item[2], item[0]))
    else:
        threshold, ok_fpr, ng_recall = min(candidates, key=lambda item: item[1])

    return CalibrationResult(
        recommended_threshold=float(threshold),
        ok_fpr=float(ok_fpr),
        ng_recall=float(ng_recall),
        ok_count=int(ok_probs.size),
        ng_count=int(ng_probs.size),
    )


def with_sample_threshold(config: AppConfig, sample_threshold: float) -> AppConfig:
    config.aggregation = AggregationConfig(
        top_k=config.aggregation.top_k,
        patch_threshold=config.aggregation.patch_threshold,
        sample_threshold=sample_threshold,
        topk_weight=config.aggregation.topk_weight,
        component_weight=config.aggregation.component_weight,
        adjacency_weight=config.aggregation.adjacency_weight,
        min_component_size=config.aggregation.min_component_size,
    )
    return config
