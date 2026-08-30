"""Manifest-backed datasets. Labels: 0 = authentic, 1 = AI-generated."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_MANIFEST_COLUMNS = {"image_path", "label"}


def read_manifest(path: str | Path, split: str | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    manifest["label"] = manifest["label"].astype(int)
    if not manifest["label"].isin([0, 1]).all():
        raise ValueError("Manifest labels must be 0 (real) or 1 (AI).")
    if split is not None:
        if "split" not in manifest.columns:
            raise ValueError("A split was requested but the manifest has no 'split' column.")
        manifest = manifest[manifest["split"] == split].copy()
    if manifest.empty:
        raise ValueError(f"No records available in {path} for split={split!r}.")
    missing_paths = [p for p in manifest.image_path if not Path(p).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} manifest image paths do not exist. First: {missing_paths[0]}")
    return manifest.reset_index(drop=True)


class PairImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: Callable) -> None:
        self.manifest, self.transform = manifest, transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        with Image.open(row.image_path) as image:
            clean, damaged = self.transform(image)
        return clean, damaged, torch.tensor(row.label, dtype=torch.float32)


class ImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: Callable, corruption: Callable | None = None) -> None:
        self.manifest, self.transform, self.corruption = manifest, transform, corruption

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.manifest.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            if self.corruption is not None:
                image = self.corruption(image)
            tensor = self.transform(image)
        return tensor, torch.tensor(row.label, dtype=torch.float32), str(row.image_path)

