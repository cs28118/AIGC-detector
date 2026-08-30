"""Report clean and transformation-specific ROC-AUC for a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import EvaluationTransform, apply_condition
from src.data import ImageDataset, read_manifest
from src.utils import device_from_arg, load_detector, load_temperature


CONDITIONS = ["clean", "jpeg30", "jpeg50", "jpeg70", "jpeg90", "blur0.5", "blur1.0", "blur2.0",
              "resize0.5", "resize0.25", "noise0.02", "noise0.05", "noise0.10", "color", "crop"]


def corruption_for(name: str):
    if name == "clean":
        return None
    if name.startswith("jpeg"):
        return lambda image: apply_condition(image, "jpeg", name)
    if name.startswith("blur"):
        return lambda image: apply_condition(image, "blur", name)
    if name.startswith("resize"):
        return lambda image: apply_condition(image, "resize", name)
    if name.startswith("noise"):
        return lambda image: apply_condition(image, "noise", name)
    if name in {"color", "crop"}:
        return lambda image: apply_condition(image, name)
    raise ValueError(f"Unsupported condition {name}")


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
    parser.add_argument("--calibration", help="temperature.json from src.calibrate")
    args = parser.parse_args()
    device = device_from_arg(args.device)
    model, checkpoint = load_detector(args.checkpoint, device)
    image_size = checkpoint.get("image_size", 224)
    manifest = read_manifest(args.manifest, args.split)
    temperature = load_temperature(args.calibration)
    requested = CONDITIONS if args.condition == "all" else [args.condition]
    results = []
    for condition in requested:
        dataset = ImageDataset(manifest, EvaluationTransform(image_size), corruption=corruption_for(condition))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
        auc, accuracy, count = score(model, loader, device, temperature)
        results.append({"condition": condition, "roc_auc": round(auc, 6), "accuracy@0.5": round(accuracy, 6), "n": count})
        print(results[-1])
    robust_rows = [r["roc_auc"] for r in results if r["condition"] != "clean"]
    if args.condition == "all":
        results.append({"condition": "mean_robust", "roc_auc": round(sum(robust_rows) / len(robust_rows), 6), "accuracy@0.5": "", "n": len(manifest)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "roc_auc", "accuracy@0.5", "n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved robustness summary to {args.output}")


if __name__ == "__main__":
    main()

