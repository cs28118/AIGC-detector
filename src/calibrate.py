"""Fit a temperature on the development split, never on the challenge benchmark."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import EvaluationTransform
from src.data import ImageDataset, read_manifest
from src.utils import device_from_arg, load_detector, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit post-hoc probability temperature scaling.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--split",
        default="calibration",
        help="Use a split separate from checkpoint selection (default: calibration).",
    )
    parser.add_argument("--output", default="artifacts/temperature.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = device_from_arg(args.device)
    model, checkpoint = load_detector(args.checkpoint, device)
    dataset = ImageDataset(read_manifest(args.manifest, args.split), EvaluationTransform(checkpoint.get("image_size", 224)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets, _ in tqdm(loader, desc="collecting validation logits"):
            logits.append(model(images.to(device)).cpu())
            labels.append(targets)
    logits, labels = torch.cat(logits).to(device), torch.cat(labels).to(device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    before = float(F.binary_cross_entropy_with_logits(logits, labels))
    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach().cpu())
    after = float(F.binary_cross_entropy_with_logits(logits / temperature, labels))
    save_json(args.output, {"temperature": temperature, "validation_nll_before": before, "validation_nll_after": after})
    print(f"Saved temperature={temperature:.5f} to {args.output}")


if __name__ == "__main__":
    main()
