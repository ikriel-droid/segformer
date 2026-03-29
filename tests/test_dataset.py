from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from industrial_patch_inspector.config import (
    AlignmentConfig,
    AppConfig,
    DatasetConfig,
    ROIConfig,
    TilingConfig,
    TrainingConfig,
)
from industrial_patch_inspector.dataset import CityscapesInspectionDataset


def _write_sample(root: Path, split: str, city: str, sample_id: str, label_mask: np.ndarray) -> None:
    image_dir = root / "leftImg8bit" / split / city
    label_dir = root / "gtFine" / split / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    image_path = image_dir / f"{sample_id}_leftImg8bit.png"
    label_path = label_dir / f"{sample_id}_gtFine_labelIds.png"

    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(label_path), label_mask)


def test_cityscapes_dataset_reads_semantic_masks_and_ignore_regions(tmp_path: Path) -> None:
    label_mask = np.zeros((32, 32), dtype=np.uint8)
    label_mask[:16, :16] = 255
    label_mask[16:, 16:] = 3
    _write_sample(tmp_path, "train", "bottle", "sample_001", label_mask)

    config = AppConfig(
        dataset=DatasetConfig(
            root=str(tmp_path),
            image_size=(32, 32),
            defect_label_ids=[],
            ignore_label_ids=[255],
        ),
        alignment=AlignmentConfig(method="identity"),
        roi=ROIConfig(use_template_roi=False, exclude_border=0, min_component_area=1),
        tiling=TilingConfig(
            tile_size=16,
            stride=16,
            context_scale=1.0,
            min_roi_fraction=0.01,
            positive_fraction=0.2,
        ),
        training=TrainingConfig(batch_size=1, num_workers=0, epochs=1, device="cpu"),
    )

    dataset = CityscapesInspectionDataset(config, "train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image_path"].endswith("sample_001_leftImg8bit.png")
    assert tuple(sample["tiles"].shape) == (4, 3, 16, 16)
    assert tuple(sample["defect_tiles"].shape) == (4, 1, 16, 16)
    assert sample["grid_shape"].tolist() == [2, 2]
    assert sample["active_mask"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert sample["patch_labels"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert float(sample["sample_label"]) == 1.0
