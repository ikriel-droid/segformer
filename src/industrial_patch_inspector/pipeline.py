from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .aggregation import PatchScoreAggregator
from .alignment import CanonicalAligner
from .config import AppConfig
from .models import ContextAwarePatchClassifier
from .refiner import SuspiciousTileRefiner
from .roi import ROIExtractor
from .structures import InspectionResult, PatchGrid
from .tiling import FixedGridTiler


class PatchInspectionSystem:
    def __init__(
        self,
        config: AppConfig,
        classifier: ContextAwarePatchClassifier,
        refiner: SuspiciousTileRefiner | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.classifier = classifier.to(self.device).eval()
        self.refiner = refiner.to(self.device).eval() if refiner is not None else None
        self.aligner = CanonicalAligner(config.alignment)
        self.roi_extractor = ROIExtractor(config.roi)
        self.tiler = FixedGridTiler(config.tiling, config.dataset.image_size)
        self.aggregator = PatchScoreAggregator(config.aggregation)

    def predict_path(self, image_path: str | Path) -> InspectionResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Failed to read image: {image_path}"
            raise FileNotFoundError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != self.config.dataset.image_size:
            image = cv2.resize(
                image,
                (self.config.dataset.image_size[1], self.config.dataset.image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return self.predict_array(image.astype(np.uint8))

    @torch.inference_mode()
    def predict_array(self, image: np.ndarray) -> InspectionResult:
        alignment = self.aligner.align(image)
        roi_mask = self.roi_extractor.build(alignment.image, alignment.roi_mask, None)
        patch_grid = self.tiler.extract(alignment.image, roi_mask, None)

        batch = {
            "tiles": torch.from_numpy(patch_grid.tiles).permute(0, 3, 1, 2).unsqueeze(0).to(self.device),
            "contexts": torch.from_numpy(patch_grid.contexts).permute(0, 3, 1, 2).unsqueeze(0).to(self.device),
            "roi_tiles": torch.from_numpy(patch_grid.roi_tiles).permute(0, 3, 1, 2).unsqueeze(0).to(self.device),
            "context_roi_tiles": torch.from_numpy(patch_grid.context_roi_tiles)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .to(self.device),
            "positions": torch.from_numpy(patch_grid.positions).unsqueeze(0).to(self.device),
            "active_mask": torch.from_numpy(patch_grid.active_mask).unsqueeze(0).to(self.device),
            "alignment_score": torch.tensor([alignment.score], dtype=torch.float32, device=self.device),
        }

        outputs = self.classifier(
            batch["tiles"],
            batch["contexts"],
            batch["roi_tiles"],
            batch["context_roi_tiles"],
            batch["positions"],
            batch["active_mask"],
            batch["alignment_score"],
            patch_grid.grid_shape,
        )
        patch_probs = torch.sigmoid(outputs["patch_logits"]).squeeze(0).cpu().numpy()
        aggregated = self.aggregator.aggregate(
            patch_probs,
            patch_grid.active_mask,
            patch_grid.grid_indices,
            patch_grid.grid_shape,
            alignment.score,
        )

        refined_mask = None
        if self.refiner is not None and aggregated["components"]:
            suspicious = np.where(patch_probs >= self.config.refiner.suspicious_threshold)[0]
            if suspicious.size > 0:
                refined_mask = self._refine_suspicious(batch, suspicious, patch_grid)

        probability = float(aggregated["sample_probability"])
        return InspectionResult(
            sample_probability=probability,
            sample_prediction=int(probability >= self.config.aggregation.sample_threshold),
            alignment_score=float(alignment.score),
            patch_probabilities=patch_probs,
            suspicious_components=list(aggregated["components"]),
            refined_mask=refined_mask,
        )

    @torch.inference_mode()
    def _refine_suspicious(
        self,
        batch: dict[str, torch.Tensor],
        suspicious: np.ndarray,
        patch_grid: PatchGrid,
    ) -> np.ndarray:
        indices = torch.from_numpy(suspicious).long().to(self.device)
        logits = self.refiner(
            batch["tiles"][0, indices],
            batch["contexts"][0, indices],
            batch["roi_tiles"][0, indices],
            batch["context_roi_tiles"][0, indices],
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        mask = np.zeros(patch_grid.image_shape, dtype=np.float32)
        tile_size = self.config.tiling.tile_size
        for idx, patch_idx in enumerate(suspicious.tolist()):
            row, col = patch_grid.grid_indices[patch_idx]
            y = min(row * self.config.tiling.stride, patch_grid.image_shape[0] - tile_size)
            x = min(col * self.config.tiling.stride, patch_grid.image_shape[1] - tile_size)
            patch_mask = probs[idx, 0]
            mask[y : y + tile_size, x : x + tile_size] = np.maximum(
                mask[y : y + tile_size, x : x + tile_size],
                patch_mask,
            )
        return mask
