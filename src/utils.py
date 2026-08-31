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
    from src.model import FORENSIC_BRANCH_VERSION, FORENSIC_FEATURES, build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint["model_config"].get("architecture", "convnext_tiny")
    if checkpoint["model_config"].get("forensic_branch") != FORENSIC_BRANCH_VERSION:
        raise ValueError(
            "This checkpoint predates the multi-domain forensic branch and is incompatible with this code."
        )
    model = build_model(
        pretrained=False,
        use_frequency=checkpoint["model_config"]["use_frequency"],
        architecture=architecture,
        forensic_features=checkpoint["model_config"].get("forensic_features", FORENSIC_FEATURES),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
