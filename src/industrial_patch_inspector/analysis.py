from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import AppConfig
from .dataset import CityscapesInspectionDataset, collate_samples
from .metrics import compute_binary_classification_metrics
from .models import ContextAwarePatchClassifier


@dataclass(slots=True)
class ActivePatchZoneOutputs:
    patch_probabilities: np.ndarray
    patch_labels: np.ndarray
    zone_rows: np.ndarray
    zone_cols: np.ndarray


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_thresholds(min_threshold: float, max_threshold: float, num_thresholds: int) -> np.ndarray:
    return np.linspace(
        float(min_threshold),
        float(max_threshold),
        num=max(int(num_thresholds), 2),
        dtype=np.float32,
    )


def sweep_binary_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = compute_binary_classification_metrics(labels, probabilities, float(threshold))
        payload = metrics.to_dict()
        payload["ok_fpr"] = 1.0 - float(metrics.specificity) if metrics.ok_count > 0 else 0.0
        payload["ng_recall"] = float(metrics.recall)
        results.append(payload)
    return results


def select_best_f1(entries: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not entries:
        return {}
    return dict(max(entries, key=lambda item: (float(item["f1"]), float(item["threshold"]))))


def select_best_recall_under_ok_fpr(
    entries: list[dict[str, float | int]],
    ok_fpr_target: float,
) -> dict[str, float | int]:
    if not entries:
        return {}
    feasible = [entry for entry in entries if float(entry["ok_fpr"]) <= float(ok_fpr_target)]
    if feasible:
        chosen = max(feasible, key=lambda item: (float(item["ng_recall"]), float(item["threshold"])))
    else:
        chosen = min(entries, key=lambda item: (float(item["ok_fpr"]), -float(item["ng_recall"])))
    return dict(chosen)


@torch.no_grad()
def collect_active_patch_zone_outputs(
    config: AppConfig,
    model: ContextAwarePatchClassifier,
    device: torch.device,
    split: str,
) -> ActivePatchZoneOutputs:
    dataset = CityscapesInspectionDataset(config, split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_samples,
    )

    patch_probabilities: list[float] = []
    patch_labels: list[int] = []
    zone_rows: list[int] = []
    zone_cols: list[int] = []

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
        batch_patch_probs = torch.sigmoid(outputs["patch_logits"]).cpu().numpy()
        batch_patch_labels = batch["patch_labels"].cpu().numpy()
        batch_active_masks = batch["active_mask"].cpu().numpy()
        batch_grid_indices = batch["grid_indices"].cpu().numpy()

        for sample_idx in range(batch_patch_probs.shape[0]):
            active = batch_active_masks[sample_idx] > 0
            if not active.any():
                continue
            patch_probabilities.extend(batch_patch_probs[sample_idx][active].tolist())
            patch_labels.extend((batch_patch_labels[sample_idx][active] >= 0.5).astype(np.int64).tolist())
            zone_rows.extend(batch_grid_indices[sample_idx][active, 0].tolist())
            zone_cols.extend(batch_grid_indices[sample_idx][active, 1].tolist())

    return ActivePatchZoneOutputs(
        patch_probabilities=np.asarray(patch_probabilities, dtype=np.float32),
        patch_labels=np.asarray(patch_labels, dtype=np.int64),
        zone_rows=np.asarray(zone_rows, dtype=np.int64),
        zone_cols=np.asarray(zone_cols, dtype=np.int64),
    )


def summarize_patch_zones(
    patch_probabilities: np.ndarray,
    patch_labels: np.ndarray,
    zone_rows: np.ndarray,
    zone_cols: np.ndarray,
    thresholds: np.ndarray,
    default_threshold: float,
    ok_fpr_target: float,
    top_k: int = 8,
) -> dict[str, object]:
    if patch_probabilities.size == 0:
        return {
            "zone_count": 0,
            "zones": [],
            "high_fp_zones": [],
            "hard_ok_zones": [],
        }

    zones: list[dict[str, object]] = []
    unique_zones = sorted({(int(row), int(col)) for row, col in zip(zone_rows.tolist(), zone_cols.tolist(), strict=True)})
    for row, col in unique_zones:
        mask = (zone_rows == row) & (zone_cols == col)
        zone_probs = patch_probabilities[mask]
        zone_labels = patch_labels[mask]
        ok_probs = zone_probs[zone_labels == 0]
        ng_probs = zone_probs[zone_labels == 1]

        sweep = sweep_binary_thresholds(zone_probs, zone_labels, thresholds)
        recommended = select_best_recall_under_ok_fpr(sweep, ok_fpr_target)
        default_metrics = compute_binary_classification_metrics(zone_labels, zone_probs, float(default_threshold))

        ok_mean_probability = float(ok_probs.mean()) if ok_probs.size > 0 else 0.0
        ng_mean_probability = float(ng_probs.mean()) if ng_probs.size > 0 else 0.0
        default_ok_fpr = 1.0 - float(default_metrics.specificity) if default_metrics.ok_count > 0 else 0.0

        zones.append(
            {
                "zone": f"r{row:02d}_c{col:02d}",
                "row": row,
                "col": col,
                "active_count": int(zone_probs.size),
                "normal_patch_count": int(ok_probs.size),
                "defect_patch_count": int(ng_probs.size),
                "defect_rate": float(zone_labels.mean()) if zone_labels.size > 0 else 0.0,
                "mean_probability": float(zone_probs.mean()),
                "ok_mean_probability": ok_mean_probability,
                "ng_mean_probability": ng_mean_probability,
                "max_probability": float(zone_probs.max()),
                "separation_gap": float(ng_mean_probability - ok_mean_probability),
                "default_patch_threshold": float(default_threshold),
                "default_ok_fpr": default_ok_fpr,
                "default_ng_recall": float(default_metrics.recall),
                "recommended_patch_threshold": float(recommended["threshold"]),
                "recommended_ok_fpr": float(recommended["ok_fpr"]),
                "recommended_ng_recall": float(recommended["ng_recall"]),
            }
        )

    zones_sorted = sorted(zones, key=lambda item: (int(item["row"]), int(item["col"])))
    high_fp_zones = sorted(
        zones,
        key=lambda item: (float(item["default_ok_fpr"]), float(item["ok_mean_probability"]), int(item["active_count"])),
        reverse=True,
    )[:top_k]
    hard_ok_zones = sorted(
        zones,
        key=lambda item: (float(item["ok_mean_probability"]), float(item["default_ok_fpr"]), int(item["active_count"])),
        reverse=True,
    )[:top_k]
    return {
        "zone_count": len(zones_sorted),
        "zones": zones_sorted,
        "high_fp_zones": high_fp_zones,
        "hard_ok_zones": hard_ok_zones,
    }
