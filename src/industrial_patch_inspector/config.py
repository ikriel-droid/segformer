from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class DatasetConfig:
    root: str
    image_dir: str = "leftImg8bit"
    label_dir: str = "gtFine"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    image_name_token: str = "_leftImg8bit"
    label_name_token: str = "_gtFine_labelIds"
    defect_label_ids: list[int] = field(default_factory=lambda: [1])
    ignore_label_ids: list[int] = field(default_factory=lambda: [255])
    image_size: tuple[int, int] = (640, 640)
    train_sample_limit: int = 0
    val_sample_limit: int = 0
    test_sample_limit: int = 0
    subset_seed: int = 13


@dataclass(slots=True)
class AlignmentConfig:
    method: str = "identity"
    template_image: str = ""
    template_roi: str = ""
    ecc_iterations: int = 80
    ecc_eps: float = 1e-5
    max_keypoints: int = 500


@dataclass(slots=True)
class ROIConfig:
    use_template_roi: bool = True
    use_otsu: bool = False
    exclude_border: int = 0
    min_component_area: int = 128
    threshold_value: int = 0


@dataclass(slots=True)
class TilingConfig:
    tile_size: int = 128
    stride: int = 64
    context_scale: float = 2.0
    min_roi_fraction: float = 0.05
    positive_fraction: float = 0.01


@dataclass(slots=True)
class ModelConfig:
    embedding_dim: int = 192
    dropout: float = 0.1


@dataclass(slots=True)
class LossConfig:
    patch_weight: float = 1.0
    sample_weight: float = 1.0
    contrastive_weight: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    contrastive_temperature: float = 0.2


@dataclass(slots=True)
class AggregationConfig:
    top_k: int = 6
    patch_threshold: float = 0.55
    sample_threshold: float = 0.50
    topk_weight: float = 0.55
    component_weight: float = 0.30
    adjacency_weight: float = 0.15
    min_component_size: int = 2


@dataclass(slots=True)
class RefinerConfig:
    enabled: bool = True
    base_channels: int = 32
    suspicious_threshold: float = 0.55


@dataclass(slots=True)
class CalibrationConfig:
    ok_fpr_target: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    num_thresholds: int = 181


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _coerce_tuple(value: Any, size: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        if len(value) != size:
            msg = f"Expected {size} values but received {len(value)}"
            raise ValueError(msg)
        return tuple(int(item) for item in value)
    msg = f"Expected a list or tuple of length {size}, received {type(value)!r}"
    raise TypeError(msg)


def _build_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_name, field_def in cls.__dataclass_fields__.items():
        if field_name not in data:
            continue
        value = data[field_name]
        if field_name == "image_size":
            kwargs[field_name] = _coerce_tuple(value, 2)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    dataset = _build_dataclass(DatasetConfig, raw["dataset"])
    return AppConfig(
        dataset=dataset,
        alignment=_build_dataclass(AlignmentConfig, raw.get("alignment", {})),
        roi=_build_dataclass(ROIConfig, raw.get("roi", {})),
        tiling=_build_dataclass(TilingConfig, raw.get("tiling", {})),
        model=_build_dataclass(ModelConfig, raw.get("model", {})),
        loss=_build_dataclass(LossConfig, raw.get("loss", {})),
        aggregation=_build_dataclass(AggregationConfig, raw.get("aggregation", {})),
        refiner=_build_dataclass(RefinerConfig, raw.get("refiner", {})),
        calibration=_build_dataclass(CalibrationConfig, raw.get("calibration", {})),
        training=_build_dataclass(TrainingConfig, raw.get("training", {})),
    )
