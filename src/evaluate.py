"""Report clean and transformation-specific ROC-AUC for a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import EVALUATION_CONDITIONS, EvaluationTransform, deterministic_corruption
from src.data import ImageDataset, read_manifest
from src.utils import device_from_arg, load_detector, load_temperature, set_seed


CONDITIONS = list(EVALUATION_CONDITIONS)


def corruption_for(name: str, seed: int = 42):
    return deterministic_corruption(name, seed)


@torch.inference_mode()
def score(model, dataloader, device: torch.device, temperature: float) -> tuple[float, float, int]:
    labels, predictions = [], []
    for images, batch_labels, _ in tqdm(dataloader, leave=False):
        logits = model(images.to(device)) / temperature
        predictions.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_labels.tolist())
    return roc_auc_score(labels, predictions), accuracy_score(labels, [p >= 0.5 for p in predictions]), len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean and corrupted AIGC detection performance.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="artifacts/evaluation.csv")
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", help="temperature.json from src.calibrate")
    args = parser.parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    model, checkpoint = load_detector(args.checkpoint, device)
    image_size = checkpoint.get("image_size", 224)
    manifest = read_manifest(args.manifest, args.split)
    temperature = load_temperature(args.calibration)
    requested = CONDITIONS if args.condition == "all" else [args.condition]
    results = []
    auc_by_condition = {}
    for condition in requested:
        dataset = ImageDataset(
            manifest,
            EvaluationTransform(image_size),
            corruption=corruption_for(condition, args.seed),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        auc, accuracy, count = score(model, loader, device, temperature)
        auc_by_condition[condition] = auc
        results.append({"condition": condition, "roc_auc": round(auc, 6), "accuracy@0.5": round(accuracy, 6), "n": count})
        print(results[-1])
    if args.condition == "all":
        robust_rows = [auc for condition, auc in auc_by_condition.items() if condition != "clean"]
        mean_robust = sum(robust_rows) / len(robust_rows)
        clean_auc = auc_by_condition["clean"]
        combined_score = 0.5 * clean_auc + 0.5 * mean_robust
        results.append({"condition": "mean_robust", "roc_auc": round(mean_robust, 6), "accuracy@0.5": "", "n": len(manifest)})
        results.append({"condition": "combined_score", "roc_auc": round(combined_score, 6), "accuracy@0.5": "", "n": len(manifest)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "roc_auc", "accuracy@0.5", "n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved robustness summary to {args.output}")


if __name__ == "__main__":
    main()
