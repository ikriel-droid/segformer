from __future__ import annotations

from collections import deque

import numpy as np

from .config import AggregationConfig


class PatchScoreAggregator:
    def __init__(self, config: AggregationConfig) -> None:
        self.config = config

    def aggregate(
        self,
        patch_probabilities: np.ndarray,
        active_mask: np.ndarray,
        grid_indices: np.ndarray,
        grid_shape: tuple[int, int],
        alignment_score: float,
    ) -> dict[str, object]:
        active_probs = patch_probabilities[active_mask > 0]
        if active_probs.size == 0:
            return {
                "sample_probability": 0.0,
                "components": [],
                "binary_map": np.zeros(grid_shape, dtype=np.uint8),
            }

        top_k = min(self.config.top_k, active_probs.size)
        topk_mean = float(np.sort(active_probs)[-top_k:].mean())

        prob_grid = np.zeros(grid_shape, dtype=np.float32)
        active_grid = np.zeros(grid_shape, dtype=np.uint8)
        for index, (row, col) in enumerate(grid_indices):
            prob_grid[row, col] = patch_probabilities[index]
            active_grid[row, col] = int(active_mask[index] > 0)

        binary_map = ((prob_grid >= self.config.patch_threshold) & (active_grid > 0)).astype(np.uint8)
        components = self._connected_components(prob_grid, binary_map)
        component_score = max((component["score"] for component in components), default=0.0)
        adjacency_score = self._adjacency_score(binary_map)

        raw_score = (
            self.config.topk_weight * topk_mean
            + self.config.component_weight * component_score
            + self.config.adjacency_weight * adjacency_score
        )
        adjusted = raw_score * (0.5 + 0.5 * float(alignment_score))
        adjusted = float(max(0.0, min(1.0, adjusted)))
        return {
            "sample_probability": adjusted,
            "components": components,
            "binary_map": binary_map,
        }

    def _connected_components(
        self,
        prob_grid: np.ndarray,
        binary_map: np.ndarray,
    ) -> list[dict[str, object]]:
        rows, cols = binary_map.shape
        visited = np.zeros_like(binary_map, dtype=bool)
        components: list[dict[str, object]] = []
        for row in range(rows):
            for col in range(cols):
                if visited[row, col] or binary_map[row, col] == 0:
                    continue
                queue: deque[tuple[int, int]] = deque([(row, col)])
                visited[row, col] = True
                coords: list[tuple[int, int]] = []
                scores: list[float] = []
                while queue:
                    y, x = queue.popleft()
                    coords.append((y, x))
                    scores.append(float(prob_grid[y, x]))
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                            continue
                        if visited[ny, nx] or binary_map[ny, nx] == 0:
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                if len(coords) < self.config.min_component_size:
                    continue
                mean_score = float(np.mean(scores))
                weighted_score = float(mean_score * min(len(coords) / self.config.min_component_size, 3.0) / 3.0)
                components.append(
                    {
                        "size": len(coords),
                        "score": weighted_score,
                        "mean_probability": mean_score,
                        "coordinates": coords,
                    }
                )
        components.sort(key=lambda item: float(item["score"]), reverse=True)
        return components

    @staticmethod
    def _adjacency_score(binary_map: np.ndarray) -> float:
        horizontal = (binary_map[:, :-1] * binary_map[:, 1:]).sum()
        vertical = (binary_map[:-1, :] * binary_map[1:, :]).sum()
        active = binary_map.sum()
        if active <= 1:
            return 0.0
        return float((horizontal + vertical) / max(active - 1, 1))
