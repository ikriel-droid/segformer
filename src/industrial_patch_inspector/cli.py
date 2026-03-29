from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .analysis import (
    build_thresholds,
    collect_active_patch_zone_outputs,
    select_best_f1,
    select_best_recall_under_ok_fpr,
    summarize_patch_zones,
    sweep_binary_thresholds,
)
from .calibration import calibrate_sample_threshold, collect_sample_probabilities, with_sample_threshold
from .config import load_config
from .losses import InspectionCriterion
from .metrics import compute_binary_classification_metrics
from .models import ContextAwarePatchClassifier
from .pipeline import PatchInspectionSystem
from .refiner import SuspiciousTileRefiner
from .training import (
    build_dataloaders,
    evaluate_patch_epoch,
    save_checkpoint,
    train_patch_epoch,
    train_refiner_epoch,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_classifier(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[object, ContextAwarePatchClassifier]:
    config = load_config(config_path)
    classifier = ContextAwarePatchClassifier(config.model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])
    return config, classifier


def _load_optional_refiner(config, refiner_checkpoint: str, device: torch.device) -> SuspiciousTileRefiner | None:
    if not config.refiner.enabled or not refiner_checkpoint:
        return None
    refiner = SuspiciousTileRefiner(config.refiner)
    refiner_ckpt = torch.load(refiner_checkpoint, map_location=device)
    refiner.load_state_dict(refiner_ckpt["model_state"])
    return refiner


def _save_refined_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(mask * 255.0, 0, 255).astype(np.uint8))


def _resolve_sample_threshold(
    config,
    sample_threshold: float,
    calibration_json: str,
) -> float:
    threshold = float(config.aggregation.sample_threshold)
    if calibration_json:
        with Path(calibration_json).open("r", encoding="utf-8") as handle:
            calibration_payload = json.load(handle)
        if "recommended_sample_threshold" not in calibration_payload:
            msg = f"recommended_sample_threshold not found in {calibration_json}"
            raise KeyError(msg)
        threshold = float(calibration_payload["recommended_sample_threshold"])
    if sample_threshold >= 0.0:
        threshold = float(sample_threshold)
    return threshold


def cmd_train_patch(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = _resolve_device(config.training.device)
    model = ContextAwarePatchClassifier(config.model).to(device)
    criterion = InspectionCriterion(config.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader, val_loader = build_dataloaders(config)
    checkpoint_dir = Path(config.training.checkpoint_dir)
    for epoch in range(1, config.training.epochs + 1):
        train_metrics = train_patch_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_patch_epoch(model, val_loader, criterion, device)
        print(
            f"[patch][epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['sample_accuracy']:.4f}"
        )
        save_checkpoint(
            checkpoint_dir / "patch_last.pt",
            {
                "model_state": model.state_dict(),
                "config_path": str(args.config),
                "epoch": epoch,
            },
        )


def cmd_train_refiner(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = _resolve_device(config.training.device)
    backbone = ContextAwarePatchClassifier(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(checkpoint["model_state"])
    refiner = SuspiciousTileRefiner(config.refiner).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader, _ = build_dataloaders(config)
    checkpoint_dir = Path(config.training.checkpoint_dir)
    for epoch in range(1, config.training.epochs + 1):
        loss = train_refiner_epoch(
            refiner,
            backbone,
            train_loader,
            optimizer,
            device,
            config.refiner.suspicious_threshold,
        )
        print(f"[refiner][epoch {epoch:03d}] loss={loss:.4f}")
        save_checkpoint(
            checkpoint_dir / "refiner_last.pt",
            {
                "model_state": refiner.state_dict(),
                "config_path": str(args.config),
                "epoch": epoch,
            },
        )


def cmd_predict(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    effective_threshold = _resolve_sample_threshold(config, args.sample_threshold, args.calibration_json)
    config = with_sample_threshold(config, effective_threshold)
    device = _resolve_device(config.training.device)
    _, classifier = _load_classifier(args.config, args.checkpoint, device)
    refiner = _load_optional_refiner(config, args.refiner_checkpoint, device)
    system = PatchInspectionSystem(config, classifier, refiner, str(device))
    result = system.predict_path(args.image)
    print(f"sample_probability={result.sample_probability:.4f}")
    print(f"sample_prediction={result.sample_prediction}")
    print(f"alignment_score={result.alignment_score:.4f}")
    print(f"suspicious_components={len(result.suspicious_components)}")
    if args.save_mask and result.refined_mask is not None:
        _save_refined_mask(result.refined_mask, Path(args.save_mask))


def cmd_predict_dir(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    effective_threshold = _resolve_sample_threshold(config, args.sample_threshold, args.calibration_json)
    config = with_sample_threshold(config, effective_threshold)
    device = _resolve_device(config.training.device)
    _, classifier = _load_classifier(args.config, args.checkpoint, device)
    refiner = _load_optional_refiner(config, args.refiner_checkpoint, device)
    system = PatchInspectionSystem(config, classifier, refiner, str(device))

    image_root = Path(args.image_dir)
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    records: list[dict[str, object]] = []
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    for image_path in image_paths:
        result = system.predict_path(image_path)
        record = {
            "image": str(image_path),
            "sample_probability": result.sample_probability,
            "sample_prediction": result.sample_prediction,
            "alignment_score": result.alignment_score,
            "component_count": len(result.suspicious_components),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if mask_dir is not None and result.refined_mask is not None:
            output_mask = mask_dir / image_path.relative_to(image_root)
            output_mask = output_mask.with_suffix(".png")
            _save_refined_mask(result.refined_mask, output_mask)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)


def cmd_evaluate_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    effective_threshold = _resolve_sample_threshold(config, args.sample_threshold, args.calibration_json)
    config = with_sample_threshold(config, effective_threshold)
    device = _resolve_device(config.training.device)
    _, classifier = _load_classifier(args.config, args.checkpoint, device)
    sample_probs, sample_labels = collect_sample_probabilities(
        config,
        classifier.to(device),
        device,
        args.split,
    )
    metrics = compute_binary_classification_metrics(sample_labels, sample_probs, effective_threshold)
    payload: dict[str, Any] = metrics.to_dict()
    payload["split"] = args.split
    payload["checkpoint"] = args.checkpoint
    payload["config"] = args.config
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def cmd_diagnose_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    effective_threshold = _resolve_sample_threshold(config, args.sample_threshold, args.calibration_json)
    config = with_sample_threshold(config, effective_threshold)
    device = _resolve_device(config.training.device)
    _, classifier = _load_classifier(args.config, args.checkpoint, device)

    thresholds = build_thresholds(
        args.min_threshold if args.min_threshold >= 0.0 else config.calibration.min_threshold,
        args.max_threshold if args.max_threshold >= 0.0 else config.calibration.max_threshold,
        args.num_thresholds if args.num_thresholds > 0 else config.calibration.num_thresholds,
    )

    sample_probs, sample_labels = collect_sample_probabilities(
        config,
        classifier.to(device),
        device,
        args.split,
    )
    sample_sweep = sweep_binary_thresholds(sample_probs, sample_labels, thresholds)
    current_metrics = compute_binary_classification_metrics(sample_labels, sample_probs, effective_threshold).to_dict()
    current_metrics["ok_fpr"] = 1.0 - float(current_metrics["specificity"]) if int(current_metrics["ok_count"]) > 0 else 0.0
    current_metrics["ng_recall"] = float(current_metrics["recall"])

    patch_outputs = collect_active_patch_zone_outputs(
        config,
        classifier.to(device),
        device,
        args.split,
    )
    zone_summary = summarize_patch_zones(
        patch_outputs.patch_probabilities,
        patch_outputs.patch_labels,
        patch_outputs.zone_rows,
        patch_outputs.zone_cols,
        thresholds=thresholds,
        default_threshold=float(config.aggregation.patch_threshold),
        ok_fpr_target=float(config.calibration.ok_fpr_target),
        top_k=args.top_k,
    )

    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "sample_threshold_in_use": float(effective_threshold),
        "patch_threshold_in_use": float(config.aggregation.patch_threshold),
        "ok_fpr_target": float(config.calibration.ok_fpr_target),
        "sample_threshold_sweep": {
            "best_f1": select_best_f1(sample_sweep),
            "best_recall_under_ok_fpr": select_best_recall_under_ok_fpr(sample_sweep, config.calibration.ok_fpr_target),
            "current_threshold": current_metrics,
            "entries": sample_sweep,
        },
        "patch_zone_analysis": zone_summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def cmd_calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = _resolve_device(config.training.device)
    _, classifier = _load_classifier(args.config, args.checkpoint, device)
    sample_probs, sample_labels = collect_sample_probabilities(
        config,
        classifier.to(device),
        device,
        config.dataset.val_split,
    )
    result = calibrate_sample_threshold(sample_probs, sample_labels, config.calibration)
    payload = {
        "recommended_sample_threshold": result.recommended_threshold,
        "ok_fpr": result.ok_fpr,
        "ng_recall": result.ng_recall,
        "ok_count": result.ok_count,
        "ng_count": result.ng_count,
        "patch_threshold": config.aggregation.patch_threshold,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial patch inspection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_patch = subparsers.add_parser("train-patch", help="Train the patch classifier")
    train_patch.add_argument("--config", required=True)
    train_patch.set_defaults(func=cmd_train_patch)

    train_refiner = subparsers.add_parser("train-refiner", help="Train the suspicious tile refiner")
    train_refiner.add_argument("--config", required=True)
    train_refiner.add_argument("--checkpoint", required=True)
    train_refiner.set_defaults(func=cmd_train_refiner)

    predict = subparsers.add_parser("predict", help="Run inference on one image")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--refiner-checkpoint", default="")
    predict.add_argument("--save-mask", default="")
    predict.add_argument("--calibration-json", default="")
    predict.add_argument("--sample-threshold", type=float, default=-1.0)
    predict.set_defaults(func=cmd_predict)

    predict_dir = subparsers.add_parser("predict-dir", help="Run inference for all images in a directory")
    predict_dir.add_argument("--config", required=True)
    predict_dir.add_argument("--checkpoint", required=True)
    predict_dir.add_argument("--image-dir", required=True)
    predict_dir.add_argument("--refiner-checkpoint", default="")
    predict_dir.add_argument("--output-json", default="")
    predict_dir.add_argument("--mask-dir", default="")
    predict_dir.add_argument("--calibration-json", default="")
    predict_dir.add_argument("--sample-threshold", type=float, default=-1.0)
    predict_dir.set_defaults(func=cmd_predict_dir)

    calibrate = subparsers.add_parser("calibrate", help="Calibrate the sample threshold on the validation split")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--output-json", default="")
    calibrate.set_defaults(func=cmd_calibrate)

    evaluate = subparsers.add_parser("evaluate-split", help="Evaluate sample-level classification on a split")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--calibration-json", default="")
    evaluate.add_argument("--sample-threshold", type=float, default=-1.0)
    evaluate.add_argument("--output-json", default="")
    evaluate.set_defaults(func=cmd_evaluate_split)

    diagnose = subparsers.add_parser("diagnose-split", help="Run threshold sweep and zone hotspot analysis on a split")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--checkpoint", required=True)
    diagnose.add_argument("--split", default="val")
    diagnose.add_argument("--calibration-json", default="")
    diagnose.add_argument("--sample-threshold", type=float, default=-1.0)
    diagnose.add_argument("--min-threshold", type=float, default=-1.0)
    diagnose.add_argument("--max-threshold", type=float, default=-1.0)
    diagnose.add_argument("--num-thresholds", type=int, default=0)
    diagnose.add_argument("--top-k", type=int, default=8)
    diagnose.add_argument("--output-json", default="")
    diagnose.set_defaults(func=cmd_diagnose_split)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
