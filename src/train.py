"""Train the redistribution-aware detector from a leakage-safe CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import (
    EVALUATION_CONDITIONS,
    EvaluationTransform,
    PairedRobustTransform,
    deterministic_corruption,
)
from src.data import ImageDataset, PairImageDataset, read_manifest
from src.model import FORENSIC_BRANCH_VERSION, FORENSIC_FEATURES, build_model
from src.utils import device_from_arg, set_seed


DEFAULT_SELECTION_CONDITIONS = ("clean", "jpeg30", "blur2.0", "resize0.25", "crop")


def loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    persistent: bool = False,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=persistent and workers > 0)


@torch.inference_mode()
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    scores, labels = [], []
    for images, targets, _ in dataloader:
        scores.extend(torch.sigmoid(model(images.to(device))).cpu().tolist())
        labels.extend(targets.tolist())
    auc = roc_auc_score(labels, scores)
    accuracy = sum((score >= 0.5) == bool(label) for score, label in zip(scores, labels)) / len(labels)
    return float(auc), float(accuracy)


def parse_forensic_features(value: str) -> tuple[str, ...]:
    features = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    unknown = set(features) - set(FORENSIC_FEATURES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown forensic features {sorted(unknown)}; choose from {list(FORENSIC_FEATURES)}"
        )
    if not features:
        raise argparse.ArgumentTypeError("select at least one forensic feature")
    return features


def parse_selection_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    unknown = set(conditions) - set(EVALUATION_CONDITIONS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown validation conditions: {sorted(unknown)}")
    if "clean" not in conditions or len(conditions) < 2:
        raise argparse.ArgumentTypeError("selection conditions must include clean and at least one corruption")
    return conditions


def combined_selection_metrics(
    condition_metrics: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return clean AUC/accuracy, mean robust AUC and the 50/50 score."""
    if "clean" not in condition_metrics or len(condition_metrics) < 2:
        raise ValueError("Checkpoint selection requires clean plus at least one corruption.")
    clean_auc, clean_accuracy = condition_metrics["clean"]
    robust_auc = sum(
        auc for condition, (auc, _) in condition_metrics.items() if condition != "clean"
    ) / (len(condition_metrics) - 1)
    selection_score = 0.5 * clean_auc + 0.5 * robust_auc
    return clean_auc, clean_accuracy, robust_auc, selection_score


def stratified_selection_frame(frame, max_samples: int, seed: int):
    """Take a deterministic label-stratified subset for checkpoint selection."""
    if len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    groups = [group for _, group in frame.groupby("label")]
    if max_samples < len(groups):
        raise ValueError("--selection-max-samples is too small to retain both labels")
    per_group = max_samples // len(groups)
    parts = [group.sample(n=min(len(group), per_group), random_state=seed) for group in groups]
    selected = pd.concat(parts)
    remaining_count = max_samples - len(selected)
    if remaining_count:
        remaining = frame.drop(index=selected.index)
        selected = pd.concat(
            [selected, remaining.sample(n=min(remaining_count, len(remaining)), random_state=seed)]
        )
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load and validate a checkpoint produced by this training script."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"{path} is not a compatible detector checkpoint.")
    architecture = checkpoint["model_config"].get("architecture", "convnext_tiny")
    if architecture not in {"convnext_tiny", "convnext_small"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    if checkpoint["model_config"].get("forensic_branch") != FORENSIC_BRANCH_VERSION:
        raise ValueError(
            "This checkpoint predates the multi-domain forensic branch and cannot be resumed. "
            "Start a new run with the updated architecture."
        )
    return checkpoint


def checkpoint_payload(model: nn.Module, optimizer, scheduler, scaler, epoch: int,
                       best_score: float, args: argparse.Namespace) -> dict:
    """Capture everything needed to resume after a completed epoch."""
    return {
        "model_state": model.state_dict(),
        "model_config": {
            "architecture": args.architecture,
            "use_frequency": not args.spatial_only,
            "forensic_branch": FORENSIC_BRANCH_VERSION,
            "forensic_features": list(args.forensic_features),
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "planned_epochs": args.epochs,
        "image_size": args.image_size,
        "best_selection_score": best_score,
        "training_args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust AIGC image detector.")
    parser.add_argument("--manifest", required=True, help="CSV generated by scripts/build_manifest.py")
    parser.add_argument("--output-dir", default="artifacts/robust_detector")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Use 8 if ConvNeXt-Small exhausts GPU memory.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--feature-consistency-weight", type=float, default=0.1)
    parser.add_argument("--reliability-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture", choices=["convnext_tiny", "convnext_small"], default="convnext_small",
        help="ConvNeXt-Small is the recommended spatial backbone; use Tiny for lower-memory ablations.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--spatial-only", action="store_true", help="Baseline ablation without the forensic branch.")
    parser.add_argument(
        "--forensic-features",
        type=parse_forensic_features,
        default=FORENSIC_FEATURES,
        metavar="FEATURES",
        help="Comma-separated forensic inputs: wavelet,dct,fft (default: all).",
    )
    parser.add_argument(
        "--selection-conditions",
        type=parse_selection_conditions,
        default=DEFAULT_SELECTION_CONDITIONS,
        metavar="CONDITIONS",
        help="Deterministic conditions used to choose best.pt.",
    )
    parser.add_argument(
        "--selection-max-samples",
        type=int,
        default=2000,
        help="Maximum validation samples used per checkpoint-selection condition.",
    )
    parser.add_argument(
        "--selection-workers",
        type=int,
        default=0,
        help="Workers per robust-selection loader; zero avoids multiple persistent worker pools.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", metavar="LAST_PT",
        help="Resume an interrupted run from last.pt. Keep --epochs equal to the original total.",
    )
    checkpoint_group.add_argument(
        "--init-checkpoint", metavar="CHECKPOINT_PT",
        help="Fine-tune model weights from a prior checkpoint with a fresh optimizer and schedule.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.selection_max_samples < 2:
        parser.error("epochs/batch size must be positive and selection samples must be at least 2")
    if args.workers < 0 or args.selection_workers < 0:
        parser.error("worker counts must be non-negative")
    if min(args.consistency_weight, args.feature_consistency_weight, args.reliability_weight) < 0:
        parser.error("loss weights must be non-negative")

    set_seed(args.seed)
    device = device_from_arg(args.device)
    loaded_checkpoint = load_checkpoint(args.resume or args.init_checkpoint, device) if (args.resume or args.init_checkpoint) else None
    if loaded_checkpoint is not None:
        checkpoint_uses_frequency = loaded_checkpoint["model_config"]["use_frequency"]
        if args.spatial_only and checkpoint_uses_frequency:
            parser.error("--spatial-only does not match the selected checkpoint. Omit it to retain the checkpoint architecture.")
        # Always reconstruct the architecture stored in the checkpoint.
        args.spatial_only = not checkpoint_uses_frequency
        args.architecture = loaded_checkpoint["model_config"].get("architecture", "convnext_tiny")
        args.forensic_features = tuple(
            loaded_checkpoint["model_config"].get("forensic_features", FORENSIC_FEATURES)
        )
    train_frame, val_frame = read_manifest(args.manifest, "train"), read_manifest(args.manifest, "val")
    train_set = PairImageDataset(train_frame, PairedRobustTransform(args.image_size))
    train_loader = loader(
        train_set, args.batch_size, args.workers, shuffle=True, persistent=True
    )
    selection_frame = stratified_selection_frame(
        val_frame, args.selection_max_samples, args.seed
    )
    selection_loaders = {
        condition: loader(
            ImageDataset(
                selection_frame,
                EvaluationTransform(args.image_size),
                corruption=deterministic_corruption(condition, args.seed),
            ),
            args.batch_size,
            args.selection_workers,
            shuffle=False,
        )
        for condition in args.selection_conditions
    }
    model = build_model(
        pretrained=not (args.no_pretrained or loaded_checkpoint is not None),
        use_frequency=not args.spatial_only,
        architecture=args.architecture,
        forensic_features=args.forensic_features,
    ).to(device)
    if loaded_checkpoint is not None:
        model.load_state_dict(loaded_checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                     epochs=args.epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_score = -1.0
    start_epoch = 1

    if args.resume:
        required = {"optimizer_state", "scheduler_state", "epoch", "planned_epochs"}
        missing = required - set(loaded_checkpoint)
        if missing:
            parser.error(
                f"{args.resume} cannot be resumed because it lacks {sorted(missing)}. "
                "Use --init-checkpoint to fine-tune it instead."
            )
        if loaded_checkpoint["planned_epochs"] != args.epochs:
            parser.error(
                "--resume requires the same --epochs value as the original interrupted run. "
                "Use --init-checkpoint for additional fine-tuning epochs."
            )
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
        scheduler.load_state_dict(loaded_checkpoint["scheduler_state"])
        if "scaler_state" in loaded_checkpoint:
            scaler.load_state_dict(loaded_checkpoint["scaler_state"])
        best_score = float(loaded_checkpoint.get("best_selection_score", -1.0))
        start_epoch = int(loaded_checkpoint["epoch"]) + 1
        if start_epoch > args.epochs:
            parser.error("This checkpoint has already completed all requested epochs. Use --init-checkpoint to fine-tune further.")
        print(f"Resuming {args.resume} at epoch {start_epoch}/{args.epochs}.")
    elif args.init_checkpoint:
        print(f"Fine-tuning weights from {args.init_checkpoint} with a fresh optimizer and schedule.")

    append_history = bool(args.resume and history_path.exists())
    with open(history_path, "a" if append_history else "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch", "loss", "val_auc", "val_accuracy", "robust_auc", "selection_score"
            ],
        )
        if not append_history:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            losses = []
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for clean, damaged, targets in progress:
                clean, damaged, targets = clean.to(device), damaged.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    clean_logits, clean_details = model(clean, return_details=True)
                    damaged_logits, damaged_details = model(damaged, return_details=True)
                    classification = 0.5 * (F.binary_cross_entropy_with_logits(clean_logits, targets) +
                                            F.binary_cross_entropy_with_logits(damaged_logits, targets))
                    consistency = F.mse_loss(torch.sigmoid(clean_logits), torch.sigmoid(damaged_logits))
                    feature_consistency = 1.0 - F.cosine_similarity(
                        clean_details["fused"], damaged_details["fused"], dim=1
                    ).mean()
                    reliability_loss = torch.zeros((), device=device)
                    if damaged_details["reliability"] is not None:
                        forensic_stability = F.cosine_similarity(
                            clean_details["forensic"].detach(),
                            damaged_details["forensic"].detach(),
                            dim=1,
                        ).clamp(0.0, 1.0)
                        gate_strength = damaged_details["reliability"].mean(dim=1)
                        reliability_loss = F.mse_loss(gate_strength, forensic_stability)
                    loss = (
                        classification
                        + args.consistency_weight * consistency
                        + args.feature_consistency_weight * feature_consistency
                        + args.reliability_weight * reliability_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}")

            condition_metrics = {
                condition: validate(model, dataloader, device)
                for condition, dataloader in selection_loaders.items()
            }
            val_auc, val_accuracy, robust_auc, selection_score = combined_selection_metrics(
                condition_metrics
            )
            row = {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "val_auc": val_auc,
                "val_accuracy": val_accuracy,
                "robust_auc": robust_auc,
                "selection_score": selection_score,
            }
            writer.writerow(row)
            history_file.flush()
            print(row)
            if selection_score > best_score:
                best_score = selection_score
                payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_score, args)
                torch.save(payload, output_dir / "best.pt")
            payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_score, args)
            torch.save(payload, output_dir / "last.pt")
    print(f"Best clean/robust selection score: {best_score:.4f}; best checkpoint: {output_dir / 'best.pt'}")
    print(f"Latest resumable checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
