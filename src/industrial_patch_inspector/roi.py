from __future__ import annotations

import cv2
import numpy as np

from .config import ROIConfig


class ROIExtractor:
    def __init__(self, config: ROIConfig) -> None:
        self.config = config

    def build(
        self,
        image: np.ndarray,
        aligned_roi: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if aligned_roi is not None and self.config.use_template_roi:
            mask = (aligned_roi > 0).astype(np.uint8) * 255
        elif label_mask is not None:
            mask = (label_mask >= 0).astype(np.uint8) * 255
        elif self.config.use_otsu:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(
                gray,
                self.config.threshold_value,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        if self.config.exclude_border > 0:
            border = self.config.exclude_border
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0

        kernel = np.ones((5, 5), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        keep = np.zeros_like(cleaned)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.config.min_component_area:
                keep[labels == label] = 255
        if keep.max() == 0:
            return cleaned
        return keep
