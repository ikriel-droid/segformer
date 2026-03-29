from __future__ import annotations

from pathlib import Path

from industrial_patch_inspector.config import load_config


def test_load_config_coerces_image_size_and_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[dataset]
root = "data/demo"
image_size = [32, 48]

[training]
device = "cpu"
epochs = 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.root == "data/demo"
    assert config.dataset.image_size == (32, 48)
    assert config.training.device == "cpu"
    assert config.training.epochs == 3
    assert config.aggregation.sample_threshold == 0.5
