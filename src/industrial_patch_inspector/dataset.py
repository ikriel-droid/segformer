from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .alignment import CanonicalAligner
from .config import AppConfig
from .roi import ROIExtractor
from .tiling import FixedGridTiler


def _load_rgb(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to read image: {path}"
        raise FileNotFoundError(msg)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != image_size:
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8)


def _load_mask(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"Failed to read mask: {path}"
        raise FileNotFoundError(msg)
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != image_size:
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    return image.astype(np.int64)


class CityscapesInspectionDataset(Dataset):
    def __init__(self, config: AppConfig, split: str) -> None:
        self.config = config
        self.split = split
        self.image_root = Path(config.dataset.root) / config.dataset.image_dir / split
        self.label_root = Path(config.dataset.root) / config.dataset.label_dir / split
        self.samples = self._discover_samples()
        self.samples = self._apply_sample_limit(self.samples)
        if not self.samples:
            msg = f"No images found in {self.image_root}"
            raise FileNotFoundError(msg)
        self.aligner = CanonicalAligner(config.alignment)
        self.roi_extractor = ROIExtractor(config.roi)
        self.tiler = FixedGridTiler(config.tiling, config.dataset.image_size)

    def _discover_samples(self) -> list[tuple[Path, Path | None]]:
        image_paths = sorted(
            path
            for path in self.image_root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        samples: list[tuple[Path, Path | None]] = []
        for image_path in image_paths:
            relative = image_path.relative_to(self.image_root)
            label_relative = self._resolve_label_relative(relative)
            label_path = self.label_root / label_relative
            if not label_path.exists():
                label_path = None
            samples.append((image_path, label_path))
        return samples

    def _apply_sample_limit(self, samples: list[tuple[Path, Path | None]]) -> list[tuple[Path, Path | None]]:
        limit_map = {
            self.config.dataset.train_split: self.config.dataset.train_sample_limit,
            self.config.dataset.val_split: self.config.dataset.val_sample_limit,
            self.config.dataset.test_split: self.config.dataset.test_sample_limit,
        }
        limit = limit_map.get(self.split, 0)
        if limit <= 0 or len(samples) <= limit:
            return samples
        rng = random.Random(self.config.dataset.subset_seed + len(self.split))
        limited = list(samples)
        rng.shuffle(limited)
        return sorted(limited[:limit], key=lambda item: str(item[0]))

    def _resolve_label_relative(self, image_relative: Path) -> Path:
        path_str = str(image_relative)
        image_token = self.config.dataset.image_name_token
        label_token = self.config.dataset.label_name_token
        if image_token and image_token in path_str:
            return Path(path_str.replace(image_token, label_token))
        stem = image_relative.stem
        suffix = image_relative.suffix
        if stem.endswith("_leftImg8bit"):
            return image_relative.with_name(stem.replace("_leftImg8bit", "_gtFine_labelIds") + suffix)
        return image_relative

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, label_path = self.samples[index]
        image = _load_rgb(image_path, self.config.dataset.image_size)
        label_mask = _load_mask(label_path, self.config.dataset.image_size) if label_path is not None else None

        alignment = self.aligner.align(image)
        if label_mask is not None:
            label_mask = self._warp_mask(label_mask, alignment.transform, alignment.image.shape[:2])

        roi_mask = self.roi_extractor.build(alignment.image, alignment.roi_mask, label_mask)
        defect_mask = None
        if label_mask is not None:
            ignore_mask = np.isin(label_mask, self.config.dataset.ignore_label_ids)
            if ignore_mask.any():
                roi_mask = roi_mask.copy()
                roi_mask[ignore_mask] = 0
            if self.config.dataset.defect_label_ids:
                defect_mask = np.isin(label_mask, self.config.dataset.defect_label_ids).astype(np.uint8) * 255
            else:
                defect_mask = ((label_mask > 0) & ~ignore_mask).astype(np.uint8) * 255
            if ignore_mask.any():
                defect_mask[ignore_mask] = 0

        patch_grid = self.tiler.extract(alignment.image, roi_mask, defect_mask)

        return {
            "tiles": self._to_image_tensor(patch_grid.tiles),
            "contexts": self._to_image_tensor(patch_grid.contexts),
            "roi_tiles": self._to_mask_tensor(patch_grid.roi_tiles),
            "context_roi_tiles": self._to_mask_tensor(patch_grid.context_roi_tiles),
            "defect_tiles": self._to_mask_tensor(patch_grid.defect_tiles),
            "positions": torch.from_numpy(patch_grid.positions),
            "grid_indices": torch.from_numpy(patch_grid.grid_indices),
            "active_mask": torch.from_numpy(patch_grid.active_mask),
            "patch_labels": torch.from_numpy(patch_grid.patch_labels),
            "sample_label": torch.tensor(patch_grid.sample_label, dtype=torch.float32),
            "alignment_score": torch.tensor(alignment.score, dtype=torch.float32),
            "grid_shape": torch.tensor(patch_grid.grid_shape, dtype=torch.long),
            "image_path": str(image_path),
        }

    def _warp_mask(
        self,
        mask: np.ndarray,
        transform: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        if np.allclose(transform, np.eye(3), atol=1e-6):
            return mask
        is_affine = np.allclose(transform[2], np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)
        if is_affine:
            return cv2.warpAffine(
                mask,
                transform[:2],
                (output_shape[1], output_shape[0]),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return cv2.warpPerspective(
            mask,
            transform,
            (output_shape[1], output_shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _to_image_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _to_mask_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def collate_samples(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    result: dict[str, torch.Tensor | list[str]] = {}
    tensor_keys = {
        "tiles",
        "contexts",
        "roi_tiles",
        "context_roi_tiles",
        "defect_tiles",
        "positions",
        "grid_indices",
        "active_mask",
        "patch_labels",
        "sample_label",
        "alignment_score",
        "grid_shape",
    }
    for key in batch[0]:
        if key in tensor_keys:
            result[key] = torch.stack([item[key] for item in batch])  # type: ignore[index]
        else:
            result[key] = [item[key] for item in batch]  # type: ignore[index]
    return result
