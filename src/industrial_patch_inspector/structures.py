from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AlignmentResult:
    image: Any
    roi_mask: Any
    transform: Any
    score: float


@dataclass(slots=True)
class PatchGrid:
    tiles: Any
    contexts: Any
    roi_tiles: Any
    context_roi_tiles: Any
    defect_tiles: Any
    positions: Any
    grid_indices: Any
    active_mask: Any
    patch_labels: Any
    sample_label: float
    image_shape: tuple[int, int]
    grid_shape: tuple[int, int]


@dataclass(slots=True)
class InspectionResult:
    sample_probability: float
    sample_prediction: int
    alignment_score: float
    patch_probabilities: Any
    suspicious_components: list[dict[str, Any]]
    refined_mask: Any | None = None
