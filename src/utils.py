"""Shared reproducibility and checkpoint helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_arg(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: str | Path, content: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)


def load_temperature(path: str | None) -> float:
    if path is None:
        return 1.0
    with open(path, encoding="utf-8") as handle:
        temperature = float(json.load(handle)["temperature"])
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive.")
    return temperature


def load_detector(checkpoint_path: str, device: torch.device):
    from src.model import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint["model_config"].get("architecture", "convnext_tiny")
    if architecture != "convnext_tiny":
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model = build_model(pretrained=False, use_frequency=checkpoint["model_config"]["use_frequency"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
