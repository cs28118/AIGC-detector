"""Challenge-required directory inference: image directory to JSON predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.augmentations import EvaluationTransform
from src.utils import device_from_arg, load_detector, load_temperature, save_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        self.paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        self.transform = EvaluationTransform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image), str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AIGC likelihood for every image in a directory.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Required JSON output path.")
    parser.add_argument("--calibration", help="Optional temperature.json from src.calibrate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory")
    device = device_from_arg(args.device)
    model, checkpoint = load_detector(args.checkpoint, device)
    dataset = FolderDataset(args.input_dir, checkpoint.get("image_size", 224))
    if not dataset.paths:
        parser.error("No supported image files found in --input-dir")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    temperature = load_temperature(args.calibration)
    records = []
    with torch.inference_mode():
        for images, paths in tqdm(loader, desc="inferring"):
            probabilities = torch.sigmoid(model(images.to(device)) / temperature).cpu().tolist()
            records.extend({"image_path": path, "pred": round(float(prediction), 6)} for path, prediction in zip(paths, probabilities))
    save_json(args.output, records)
    print(f"Wrote {len(records)} predictions to {args.output}")


if __name__ == "__main__":
    main()

