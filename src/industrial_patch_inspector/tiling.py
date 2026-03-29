from __future__ import annotations

from math import ceil

import cv2
import numpy as np

from .config import TilingConfig
from .structures import PatchGrid


class FixedGridTiler:
    def __init__(self, config: TilingConfig, image_size: tuple[int, int]) -> None:
        self.config = config
        self.image_height, self.image_width = image_size
        self.grid_positions = self._build_grid_positions()
        self.grid_shape = self._grid_shape()

    def _grid_shape(self) -> tuple[int, int]:
        rows = ceil((self.image_height - self.config.tile_size) / self.config.stride) + 1
        cols = ceil((self.image_width - self.config.tile_size) / self.config.stride) + 1
        return rows, cols

    def _build_grid_positions(self) -> list[tuple[int, int, int, int]]:
        rows, cols = self._grid_shape()
        positions: list[tuple[int, int, int, int]] = []
        for row in range(rows):
            for col in range(cols):
                y = min(row * self.config.stride, self.image_height - self.config.tile_size)
                x = min(col * self.config.stride, self.image_width - self.config.tile_size)
                positions.append((y, x, row, col))
        return positions

    def extract(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
        defect_mask: np.ndarray | None = None,
    ) -> PatchGrid:
        tile_size = self.config.tile_size
        context_size = int(round(tile_size * self.config.context_scale))
        half_margin = max((context_size - tile_size) // 2, 0)

        image_padded = np.pad(
            image,
            ((half_margin, half_margin), (half_margin, half_margin), (0, 0)),
            mode="edge",
        )
        roi_padded = np.pad(roi_mask, ((half_margin, half_margin), (half_margin, half_margin)), mode="constant")

        tiles: list[np.ndarray] = []
        contexts: list[np.ndarray] = []
        roi_tiles: list[np.ndarray] = []
        context_roi_tiles: list[np.ndarray] = []
        defect_tiles: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        active_mask: list[float] = []
        patch_labels: list[float] = []

        for y, x, row, col in self.grid_positions:
            tile = image[y : y + tile_size, x : x + tile_size]
            roi_tile = roi_mask[y : y + tile_size, x : x + tile_size]
            context = image_padded[y : y + context_size, x : x + context_size]
            context_roi = roi_padded[y : y + context_size, x : x + context_size]
            context = cv2.resize(context, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            context_roi = cv2.resize(context_roi, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

            roi_fraction = float((roi_tile > 0).mean())
            active = 1.0 if roi_fraction >= self.config.min_roi_fraction else 0.0

            patch_label = 0.0
            defect_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
            if defect_mask is not None:
                defect_tile = defect_mask[y : y + tile_size, x : x + tile_size]
                patch_label = 1.0 if float((defect_tile > 0).mean()) >= self.config.positive_fraction else 0.0

            tiles.append(tile.astype(np.float32) / 255.0)
            contexts.append(context.astype(np.float32) / 255.0)
            roi_tiles.append((roi_tile > 0).astype(np.float32)[..., None])
            context_roi_tiles.append((context_roi > 0).astype(np.float32)[..., None])
            defect_tiles.append((defect_tile > 0).astype(np.float32)[..., None])
            positions.append(
                np.asarray(
                    [
                        (x + tile_size * 0.5) / self.image_width,
                        (y + tile_size * 0.5) / self.image_height,
                    ],
                    dtype=np.float32,
                )
            )
            indices.append(np.asarray([row, col], dtype=np.int64))
            active_mask.append(active)
            patch_labels.append(patch_label)

        sample_label = 1.0 if any(label > 0 for label in patch_labels) else 0.0
        return PatchGrid(
            tiles=np.stack(tiles, axis=0),
            contexts=np.stack(contexts, axis=0),
            roi_tiles=np.stack(roi_tiles, axis=0),
            context_roi_tiles=np.stack(context_roi_tiles, axis=0),
            defect_tiles=np.stack(defect_tiles, axis=0),
            positions=np.stack(positions, axis=0),
            grid_indices=np.stack(indices, axis=0),
            active_mask=np.asarray(active_mask, dtype=np.float32),
            patch_labels=np.asarray(patch_labels, dtype=np.float32),
            sample_label=sample_label,
            image_shape=(self.image_height, self.image_width),
            grid_shape=self.grid_shape,
        )
