from __future__ import annotations

import argparse
import html.parser
import json
import math
from pathlib import Path
import shutil
import tarfile
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPS_DIR = PROJECT_ROOT / ".deps"
import sys
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))

import cv2
import numpy as np


DOWNLOAD_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads"
LICENSE_NAME = "CC BY-NC-SA 4.0"


class DownloadLinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_href = ""
        self.links: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        self.current_href = attr_map.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.current_href = ""

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or "mydrive.ch" not in self.current_href:
            return
        normalized = text.lower().replace(" ", "_")
        if normalized in {
            "download_the_whole_dataset",
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        }:
            self.links[normalized] = self.current_href


def fetch_category_links() -> dict[str, str]:
    with urllib.request.urlopen(DOWNLOAD_PAGE) as response:
        content = response.read().decode("utf-8", errors="ignore")
    parser = DownloadLinkParser()
    parser.feed(content)
    if not parser.links:
        msg = "Failed to extract MVTec AD category links from the official download page."
        raise RuntimeError(msg)
    return parser.links


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def safe_extract(archive_path: Path, destination: Path) -> Path:
    existing_category_dirs = [
        path
        for path in destination.glob("*")
        if path.is_dir() and (path / "train").exists() and (path / "test").exists() and (path / "ground_truth").exists()
    ]
    if existing_category_dirs:
        return existing_category_dirs[0]
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:xz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                msg = f"Unsafe archive member: {member.name}"
                raise RuntimeError(msg)
        archive.extractall(destination)
    top_dirs = [path for path in destination.iterdir() if path.is_dir()]
    if len(top_dirs) == 1:
        return top_dirs[0]
    return destination


def _copy_pair(
    image_path: Path,
    mask_path: Path | None,
    output_root: Path,
    split: str,
    city_name: str,
    sample_id: str,
    class_id: int = 0,
) -> None:
    image_dir = output_root / "leftImg8bit" / split / city_name
    label_dir = output_root / "gtFine" / split / city_name
    image_target = image_dir / f"{sample_id}_leftImg8bit.png"
    label_target = label_dir / f"{sample_id}_gtFine_labelIds.png"
    label_train_target = label_dir / f"{sample_id}_gtFine_labelTrainIds.png"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to read image: {image_path}"
        raise FileNotFoundError(msg)
    cv2.imwrite(str(image_target), image)
    if mask_path is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.imwrite(str(label_target), mask)
        cv2.imwrite(str(label_train_target), mask)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            msg = f"Failed to read mask: {mask_path}"
            raise FileNotFoundError(msg)
        if mask.ndim == 3:
            mask = mask[..., 0]
        semantic_mask = np.where(mask > 0, class_id, 0).astype(np.uint8)
        cv2.imwrite(str(label_target), semantic_mask)
        cv2.imwrite(str(label_train_target), semantic_mask)


def _split_items(items: list[tuple[Path, Path | None]], ratios: tuple[float, float, float]) -> tuple[list, list, list]:
    total = len(items)
    if total == 0:
        return [], [], []
    train_count = max(1, math.floor(total * ratios[0]))
    val_count = max(1, math.floor(total * ratios[1])) if total > 2 else max(0, total - train_count - 1)
    test_count = max(total - train_count - val_count, 1)
    if train_count + val_count + test_count > total:
        overflow = train_count + val_count + test_count - total
        test_count = max(test_count - overflow, 1)
    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count : train_count + val_count + test_count]
    return train_items, val_items, test_items


def convert_mvtec_to_cityscapes(
    extracted_category_dir: Path,
    output_root: Path,
    category: str,
    anomaly_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    good_test_ratios: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    train_good = sorted((extracted_category_dir / "train" / "good").glob("*.png"))
    test_good = sorted((extracted_category_dir / "test" / "good").glob("*.png"))
    anomaly_dirs = [
        path
        for path in (extracted_category_dir / "test").iterdir()
        if path.is_dir() and path.name != "good"
    ]
    class_map = {"background": 0}
    for class_id, anomaly_dir in enumerate(sorted(anomaly_dirs), start=1):
        class_map[anomaly_dir.name] = class_id
    city_name = category

    def build_sample_id(split: str, defect_name: str, image_stem: str) -> str:
        safe_defect = defect_name.replace("-", "_")
        return f"{category}_{split}_{safe_defect}_{image_stem}"

    for image_path in train_good:
        sample_id = build_sample_id("train", "good", image_path.stem)
        _copy_pair(image_path, None, output_root, "train", city_name, sample_id)

    good_train, good_val, good_test = _split_items([(path, None) for path in test_good], good_test_ratios)
    for split, items in (("train", good_train), ("val", good_val), ("test", good_test)):
        for image_path, mask_path in items:
            sample_id = build_sample_id(split, "good", image_path.stem)
            _copy_pair(image_path, mask_path, output_root, split, city_name, sample_id)

    for anomaly_dir in sorted(anomaly_dirs):
        defect_name = anomaly_dir.name
        class_id = class_map[defect_name]
        images = sorted(anomaly_dir.glob("*.png"))
        pairs: list[tuple[Path, Path]] = []
        for image_path in images:
            mask_name = f"{image_path.stem}_mask.png"
            mask_path = extracted_category_dir / "ground_truth" / defect_name / mask_name
            if not mask_path.exists():
                msg = f"Missing mask for {image_path}"
                raise FileNotFoundError(msg)
            pairs.append((image_path, mask_path))
        an_train, an_val, an_test = _split_items(pairs, anomaly_ratios)
        for split, items in (("train", an_train), ("val", an_val), ("test", an_test)):
            for image_path, mask_path in items:
                sample_id = build_sample_id(split, defect_name, image_path.stem)
                _copy_pair(image_path, mask_path, output_root, split, city_name, sample_id, class_id=class_id)

    metadata = {
        "source_dataset": "MVTec AD",
        "source_url": DOWNLOAD_PAGE,
        "category": category,
        "format": "cityscapes-semantic",
        "license": LICENSE_NAME,
        "class_map": class_map,
        "note": "This is a derived supervised split built from the public MVTec AD category. It is not benchmark-preserving against the original unsupervised evaluation protocol.",
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one MVTec AD category and convert it to cityscapes-style splits")
    parser.add_argument("--category", default="bottle", help="Category name such as bottle, screw, capsule")
    parser.add_argument("--download-root", default="data/raw/mvtec_ad", help="Where the raw archives/extracted files should go")
    parser.add_argument("--output-root", default="data/mvtec_ad_cityscapes", help="Where the converted dataset should go")
    args = parser.parse_args()

    category = args.category.lower().replace(" ", "_")
    download_root = Path(args.download_root)
    output_root = Path(args.output_root)
    archive_path = download_root / f"{category}.tar.xz"
    extracted_root = download_root / category
    converted_root = output_root / category

    if not archive_path.exists():
        links = fetch_category_links()
        if category not in links:
            available = ", ".join(sorted(links))
            msg = f"Unknown category '{category}'. Available: {available}"
            raise ValueError(msg)
        download_file(links[category], archive_path)
    category_dir = safe_extract(archive_path, extracted_root)
    convert_mvtec_to_cityscapes(category_dir, converted_root, category)

    print(json.dumps({"category": category, "archive": str(archive_path), "converted_root": str(converted_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
