"""Image transforms used for robust training and controlled evaluation."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


Condition = Literal["clean", "jpeg", "blur", "resize", "noise", "color", "crop"]


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG to simulate common platform re-encoding."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def _resize_upscale(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(small, Image.Resampling.BICUBIC).resize(
        (width, height), Image.Resampling.BICUBIC
    )


def _color_jitter(image: Image.Image, amount: float = 0.20) -> Image.Image:
    factor = lambda: random.uniform(1 - amount, 1 + amount)
    image = ImageEnhance.Brightness(image).enhance(factor())
    image = ImageEnhance.Contrast(image).enhance(factor())
    return ImageEnhance.Color(image).enhance(factor())


def _add_noise(image: Image.Image, sigma: float) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels += np.random.normal(0.0, sigma, size=pixels.shape).astype(np.float32)
    return Image.fromarray(np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0), mode="RGB")


def apply_condition(image: Image.Image, condition: Condition, strength: str = "random") -> Image.Image:
    """Apply one documented redistribution transform to an RGB PIL image.

    ``strength=random`` samples the challenge's documented values. Evaluation can
    instead pass an exact value such as ``jpeg30`` or ``blur2``.
    """
    image = image.convert("RGB")
    if condition == "clean":
        return image
    if condition == "jpeg":
        quality = int(strength.removeprefix("jpeg")) if strength.startswith("jpeg") else random.choice([90, 70, 50, 30])
        return _jpeg(image, quality)
    if condition == "blur":
        sigma = float(strength.removeprefix("blur")) if strength.startswith("blur") else random.choice([0.5, 1.0, 2.0])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if condition == "resize":
        scale = float(strength.removeprefix("resize")) if strength.startswith("resize") else random.choice([0.5, 0.25])
        return _resize_upscale(image, scale)
    if condition == "noise":
        sigma = float(strength.removeprefix("noise")) if strength.startswith("noise") else random.choice([0.02, 0.05, 0.10])
        return _add_noise(image, sigma)
    if condition == "color":
        return _color_jitter(image, 0.20)
    if condition == "crop":
        width, height = image.size
        crop_w, crop_h = max(1, int(width * 0.8)), max(1, int(height * 0.8))
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    raise ValueError(f"Unknown condition: {condition}")


@dataclass
class PairedRobustTransform:
    """Return aligned clean and redistribution-damaged views of one image."""

    image_size: int = 224

    def __post_init__(self) -> None:
        self.geometry = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.70, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.geometry(image.convert("RGB"))
        # One or two independently sampled real-world transforms per robust view.
        damaged = base.copy()
        choices: list[Condition] = ["jpeg", "blur", "resize", "noise", "color", "crop"]
        for condition in random.sample(choices, k=random.choice([1, 1, 2])):
            damaged = apply_condition(damaged, condition)
        return self.to_tensor(base), self.to_tensor(damaged)


class EvaluationTransform:
    """Deterministic model input transform, after any requested corruption."""

    def __init__(self, image_size: int = 224) -> None:
        resize_size = round(image_size / 0.875)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

