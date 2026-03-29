# Industrial Patch Inspector

`industrial-patch-inspector` is a reference implementation of the inspection architecture we discussed:

`alignment -> ROI cleanup -> fixed patch grid -> context-aware patch classifier -> top-k/component aggregation -> optional tile refiner`

The project assumes:

- `640x640` images
- the canonical Cityscapes directory and file naming convention
- patch classification is the primary decision path
- segmentation is a secondary explainer/refiner

## Public dataset setup

The project now includes an official MVTec AD downloader/converter:

```bash
python scripts/fetch_mvtec_ad.py --category bottle
```

This downloads one category from the official MVTec page and converts it into a derived supervised dataset that follows the usual Cityscapes semantic layout:

```text
data/cityscapes_mvtec_v2/<category>/
  leftImg8bit/
    train/<category>/*_leftImg8bit.png
    val/<category>/*_leftImg8bit.png
    test/<category>/*_leftImg8bit.png
  gtFine/
    train/<category>/*_gtFine_labelIds.png
    train/<category>/*_gtFine_labelTrainIds.png
    val/<category>/*_gtFine_labelIds.png
    test/<category>/*_gtFine_labelIds.png
```

The converted labels are semantic masks, not instance masks:

- `0` is background/good
- each defect type gets its own semantic class id
- the `metadata.json` file stores the generated `class_map`

Use the ready-made example config:

```text
examples/mvtec_bottle.toml
```

Important:

- MVTec AD is released under `CC BY-NC-SA 4.0`
- it is not for commercial use
- the converter creates a supervised split from the public files, so it is not benchmark-preserving against the original MVTec AD protocol

## Dataset layout

The default loader recursively scans standard Cityscapes-style folders:

```text
dataset/
  leftImg8bit/
    train/
      part_a/
        sample_001_leftImg8bit.png
    val/
      part_a/
        sample_101_leftImg8bit.png
  gtFine/
    train/
      part_a/
        sample_001_gtFine_labelIds.png
    val/
      part_a/
        sample_101_gtFine_labelIds.png
```

If your image and label names differ, set `image_name_token` and `label_name_token` in the config. The loader replaces the token in the relative path when looking up semantic masks.

Masks should use:

- `0` for normal/background
- one or more positive ids for defects
- optional ignore ids such as `255`

## What is implemented

- Canonical alignment with identity, ECC, or keypoint-homography modes
- ROI extraction from template ROI, label-derived ROI, or threshold cleanup
- Fixed grid tiling with center tile, context crop, ROI tiles, and normalized coordinates
- Context-aware patch classifier with center/context/position/neighborhood branches
- Combined patch/sample/contrastive losses
- Separate patch and final sample thresholds
- Top-k + adjacency + connected-component aggregation
- Optional suspicious-tile UNet-style refiner
- Validation-based sample-threshold calibration
- Single-image and batch-directory inference CLIs
- CLI entry points for training and prediction

## Quick start

1. Install dependencies:

```bash
python -m pip install --target .deps numpy opencv-python torch torchvision
```

2. Edit the example config:

```text
examples/config.toml
```

3. Train the patch model:

```bash
python run_cli.py train-patch --config examples/config.toml
```

4. Train the refiner:

```bash
python run_cli.py train-refiner --config examples/config.toml --checkpoint checkpoints/patch_last.pt
```

5. Run prediction:

```bash
python run_cli.py predict --config examples/config.toml --checkpoint checkpoints/patch_last.pt --image path/to/image.png
```

6. Calibrate the final sample threshold on the validation split:

```bash
python run_cli.py calibrate --config examples/config.toml --checkpoint checkpoints/patch_last.pt --output-json checkpoints/calibration.json
```

7. Run a whole directory:

```bash
python run_cli.py predict-dir --config examples/config.toml --checkpoint checkpoints/patch_last.pt --image-dir path/to/images --output-json outputs/predictions.json
```

## One-command pipeline

On this Windows workspace, the easiest way to run everything in order is:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_pipeline.ps1 -Category bottle -Mode smoke
```

You can also use the wrapper directly from the VS Code terminal:

```bat
.\scripts\run_full_pipeline.cmd -Category bottle -Mode smoke
```

What it automates:

1. verifies or installs `.deps`
2. fetches and converts the public MVTec category into Cityscapes semantic format
3. generates a run-specific config
4. trains the patch classifier
5. calibrates the sample threshold
6. trains the refiner
7. applies the calibrated sample threshold during test-directory inference and exports predictions

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_pipeline.ps1 -Category bottle -Mode full
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_pipeline.ps1 -Category bottle -Mode smoke -SkipRefiner
```

Artifacts are written under:

```text
outputs/pipeline/<category>/<mode>/
```

The run also writes a markdown summary with the remaining steps to reach fuller completion.

## Remaining Work

What still remains after the automated pipeline finishes:

1. Run `full` mode or increase epochs for a real training result.
2. Recalibrate on a harsher OK validation set that reflects line lighting/background drift.
3. Add alignment priors if the real part pose is not fixed.
4. Replace the public benchmark with your in-domain Cityscapes-format dataset.
5. Tune zone thresholds and hard-OK replay using real production false positives.

## Notes

- If your line is already tightly aligned, leave `alignment.method = "identity"`.
- If false positives come mostly from lighting/background drift, prioritize collecting more hard OK images.
- `aggregation.patch_threshold` is for patch adjacency/component logic. `aggregation.sample_threshold` is the final OK/NG cutoff.
- If you only want the main classifier path, set `refiner.enabled = false`.
