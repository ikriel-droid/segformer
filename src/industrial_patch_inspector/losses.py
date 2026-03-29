from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import LossConfig


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_factor * (1.0 - pt).pow(gamma) * ce


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.t() / temperature
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.t()).float()
    identity = torch.eye(embeddings.shape[0], device=embeddings.device)
    positive_mask = positive_mask - identity
    logits_mask = 1.0 - identity
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive_mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count
    return -mean_log_prob.mean()


class InspectionCriterion(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        patch_labels: torch.Tensor,
        sample_labels: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        patch_loss_map = binary_focal_loss(
            outputs["patch_logits"],
            patch_labels,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        patch_loss = (patch_loss_map * active_mask).sum() / active_mask.sum().clamp_min(1.0)
        sample_loss = F.binary_cross_entropy_with_logits(outputs["sample_logits"], sample_labels)

        flat_mask = active_mask.reshape(-1) > 0
        flat_embeddings = outputs["patch_embeddings"].reshape(-1, outputs["patch_embeddings"].shape[-1])[flat_mask]
        flat_labels = patch_labels.reshape(-1)[flat_mask]
        contrastive = supervised_contrastive_loss(
            flat_embeddings,
            flat_labels,
            temperature=self.config.contrastive_temperature,
        )

        total = (
            self.config.patch_weight * patch_loss
            + self.config.sample_weight * sample_loss
            + self.config.contrastive_weight * contrastive
        )
        return {
            "loss": total,
            "patch_loss": patch_loss.detach(),
            "sample_loss": sample_loss.detach(),
            "contrastive_loss": contrastive.detach(),
        }
