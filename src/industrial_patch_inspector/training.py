from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import AppConfig
from .dataset import CityscapesInspectionDataset, collate_samples
from .losses import InspectionCriterion
from .models import ContextAwarePatchClassifier
from .refiner import SuspiciousTileRefiner


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_patch_epoch(
    model: ContextAwarePatchClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: InspectionCriterion,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_patch = 0.0
    total_sample = 0.0
    total_contrastive = 0.0
    steps = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
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
        losses = criterion(
            outputs,
            batch["patch_labels"],
            batch["sample_label"],
            batch["active_mask"],
        )
        losses["loss"].backward()
        optimizer.step()

        total_loss += float(losses["loss"].detach().cpu())
        total_patch += float(losses["patch_loss"].cpu())
        total_sample += float(losses["sample_loss"].cpu())
        total_contrastive += float(losses["contrastive_loss"].cpu())
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "patch_loss": total_patch / max(steps, 1),
        "sample_loss": total_sample / max(steps, 1),
        "contrastive_loss": total_contrastive / max(steps, 1),
    }


@torch.no_grad()
def evaluate_patch_epoch(
    model: ContextAwarePatchClassifier,
    loader,
    criterion: InspectionCriterion,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    steps = 0
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
        losses = criterion(
            outputs,
            batch["patch_labels"],
            batch["sample_label"],
            batch["active_mask"],
        )
        total_loss += float(losses["loss"].cpu())
        predictions = (torch.sigmoid(outputs["sample_logits"]) >= 0.5).float()
        correct += int((predictions == batch["sample_label"]).sum().cpu())
        count += int(batch["sample_label"].numel())
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "sample_accuracy": correct / max(count, 1),
    }


def train_refiner_epoch(
    refiner: SuspiciousTileRefiner,
    backbone: ContextAwarePatchClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    suspicious_threshold: float,
) -> float:
    refiner.train()
    backbone.eval()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.no_grad():
            outputs = backbone(
                batch["tiles"],
                batch["contexts"],
                batch["roi_tiles"],
                batch["context_roi_tiles"],
                batch["positions"],
                batch["active_mask"],
                batch["alignment_score"],
                tuple(batch["grid_shape"][0].tolist()),
            )
            patch_probs = torch.sigmoid(outputs["patch_logits"])

        candidate_mask = (patch_probs >= suspicious_threshold) & (batch["active_mask"] > 0)
        if candidate_mask.sum() == 0:
            continue

        flat_candidates = candidate_mask.reshape(-1)
        tiles = batch["tiles"].reshape(-1, *batch["tiles"].shape[2:])[flat_candidates]
        contexts = batch["contexts"].reshape(-1, *batch["contexts"].shape[2:])[flat_candidates]
        roi_tiles = batch["roi_tiles"].reshape(-1, *batch["roi_tiles"].shape[2:])[flat_candidates]
        context_roi_tiles = batch["context_roi_tiles"].reshape(-1, *batch["context_roi_tiles"].shape[2:])[flat_candidates]
        targets = batch["defect_tiles"].reshape(-1, *batch["defect_tiles"].shape[2:])[flat_candidates]

        optimizer.zero_grad(set_to_none=True)
        logits = refiner(tiles, contexts, roi_tiles, context_roi_tiles)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


def build_dataloaders(config: AppConfig):
    train_dataset = CityscapesInspectionDataset(config, config.dataset.train_split)
    val_dataset = CityscapesInspectionDataset(config, config.dataset.val_split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_samples,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_samples,
    )
    return train_loader, val_loader


def save_checkpoint(path: str | Path, payload: dict[str, object]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
