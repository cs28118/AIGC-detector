"""An original spatial + frequency fusion AIGC detector."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class FrequencyBranch(nn.Module):
    """Learn from log Fourier magnitude of the luminance channel."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, output_dim, 3, stride=2, padding=1), nn.BatchNorm2d(output_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    @staticmethod
    def spectrum(images: torch.Tensor) -> torch.Tensor:
        # Input remains in [0, 1]; FFT features should not use ImageNet normalisation.
        luminance = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        spectrum = torch.fft.fft2(luminance, norm="ortho")
        magnitude = torch.log1p(torch.abs(torch.fft.fftshift(spectrum, dim=(-2, -1))))
        return magnitude

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.spectrum(images))


class RedistributionAwareDetector(nn.Module):
    """Fuse generic image features with learned frequency-forensic evidence."""

    def __init__(self, pretrained: bool = True, use_frequency: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.spatial = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.use_frequency = use_frequency
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if use_frequency:
            self.frequency = FrequencyBranch(output_dim=256)
            fusion_dim = 1280 + 256
        else:
            self.frequency = None
            fusion_dim = 1280
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.GELU(), nn.Dropout(0.25), nn.Linear(512, 1)
        )

    def forward(self, images: torch.Tensor, return_features: bool = False):
        normalized = (images - self.mean) / self.std
        spatial = self.spatial_pool(self.spatial(normalized)).flatten(1)
        features = torch.cat([spatial, self.frequency(images)], dim=1) if self.frequency else spatial
        logits = self.head(features).squeeze(1)
        return (logits, features) if return_features else logits


def build_model(pretrained: bool = True, use_frequency: bool = True) -> RedistributionAwareDetector:
    return RedistributionAwareDetector(pretrained=pretrained, use_frequency=use_frequency)

