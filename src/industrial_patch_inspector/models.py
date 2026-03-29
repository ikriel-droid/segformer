from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TinyPatchEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        widths = [48, 96, embedding_dim]
        self.stem = ConvStem(4, widths[0])
        self.stage1 = nn.Sequential(
            DepthwiseResidual(widths[0]),
            nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(widths[1]),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            DepthwiseResidual(widths[1]),
            nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(widths[2]),
            nn.GELU(),
            DepthwiseResidual(widths[2]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, roi], dim=1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.pool(x).flatten(1)


class AttentionMILHead(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, tokens: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(tokens).squeeze(-1)
        scores = scores.masked_fill(active_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(-1), weights


class ContextAwarePatchClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = TinyPatchEncoder(config.embedding_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(4, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, config.embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.embedding_dim * 4),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.patch_head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim // 2, 1),
        )
        self.sample_head = AttentionMILHead(config.embedding_dim, config.dropout)

    def forward(
        self,
        tiles: torch.Tensor,
        contexts: torch.Tensor,
        roi_tiles: torch.Tensor,
        context_roi_tiles: torch.Tensor,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        alignment_scores: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch_size, patch_count = tiles.shape[:2]
        flat_tiles = tiles.reshape(batch_size * patch_count, *tiles.shape[2:])
        flat_contexts = contexts.reshape(batch_size * patch_count, *contexts.shape[2:])
        flat_rois = roi_tiles.reshape(batch_size * patch_count, *roi_tiles.shape[2:])
        flat_context_rois = context_roi_tiles.reshape(batch_size * patch_count, *context_roi_tiles.shape[2:])

        center_features = self.encoder(flat_tiles, flat_rois)
        context_features = self.encoder(flat_contexts, flat_context_rois)
        roi_fraction = flat_rois.float().mean(dim=(2, 3))
        position_features = torch.cat(
            [
                positions.reshape(batch_size * patch_count, 2),
                roi_fraction,
                alignment_scores[:, None].repeat(1, patch_count).reshape(-1, 1),
            ],
            dim=1,
        )
        position_embedding = self.position_mlp(position_features)

        center_features = center_features.reshape(batch_size, patch_count, -1)
        context_features = context_features.reshape(batch_size, patch_count, -1)
        position_embedding = position_embedding.reshape(batch_size, patch_count, -1)

        base_tokens = self.fusion(
            torch.cat(
                [
                    center_features,
                    context_features,
                    position_embedding,
                    center_features - context_features,
                ],
                dim=-1,
            )
        )
        neighbor_tokens = self._neighborhood_summary(base_tokens, active_mask, grid_shape)
        tokens = base_tokens + neighbor_tokens
        patch_logits = self.patch_head(tokens).squeeze(-1)
        sample_logits, attn_weights = self.sample_head(tokens, active_mask)
        return {
            "patch_logits": patch_logits,
            "sample_logits": sample_logits,
            "patch_embeddings": tokens,
            "attention": attn_weights,
        }

    @staticmethod
    def _neighborhood_summary(
        tokens: torch.Tensor,
        active_mask: torch.Tensor,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, patch_count, channels = tokens.shape
        rows, cols = grid_shape
        grid = tokens.reshape(batch_size, rows, cols, channels)
        mask = active_mask.reshape(batch_size, rows, cols, 1)
        summary = torch.zeros_like(grid)
        counts = torch.zeros_like(mask)

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in offsets:
            shifted = torch.zeros_like(grid)
            shifted_mask = torch.zeros_like(mask)
            row_src = slice(max(0, -dy), rows - max(0, dy))
            row_dst = slice(max(0, dy), rows - max(0, -dy))
            col_src = slice(max(0, -dx), cols - max(0, dx))
            col_dst = slice(max(0, dx), cols - max(0, -dx))
            shifted[:, row_dst, col_dst] = grid[:, row_src, col_src]
            shifted_mask[:, row_dst, col_dst] = mask[:, row_src, col_src]
            summary += shifted * shifted_mask
            counts += shifted_mask

        summary = summary / counts.clamp_min(1.0)
        return summary.reshape(batch_size, patch_count, channels)
