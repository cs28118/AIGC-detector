"""Redistribution Reliability Fusion CNN for robust AIGC detection.

The model has two top-level branches:

1. an ImageNet-pretrained ConvNeXt spatial branch; and
2. a custom forensic branch combining aligned high-pass, Haar-wavelet and
   block-DCT maps with a separately encoded global FFT spectrum.

The two feature vectors are projected to the same width and combined through
a learned reliability gate. All forensic transforms are implemented here and
remain differentiable, while their analysis kernels are fixed to reduce
dataset-source shortcut learning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_small,
    convnext_tiny,
)


BACKBONES = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
}
FORENSIC_FEATURES = ("wavelet", "dct", "fft")
FORENSIC_BRANCH_VERSION = "multidomain_v2"


class MultiDomainForensicBranch(nn.Module):
    """Encode local residual/frequency maps and a global FFT spectrum."""

    block_size = 8

    def __init__(
        self,
        output_dim: int = 256,
        features: Sequence[str] = FORENSIC_FEATURES,
    ) -> None:
        super().__init__()
        selected = tuple(dict.fromkeys(features))
        unknown = set(selected) - set(FORENSIC_FEATURES)
        if unknown:
            raise ValueError(f"Unsupported forensic features: {sorted(unknown)}")
        if not selected:
            raise ValueError("At least one forensic feature must be selected.")
        self.features = selected

        # Fixed Haar LL/LH/HL/HH filters.
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [1.0, -1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer("haar", haar)

        # A fixed 5x5 binomial blur produces the high-pass residual.
        gaussian_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        blur_kernel = torch.outer(gaussian_1d, gaussian_1d).view(1, 1, 5, 5)
        self.register_buffer("blur_kernel", blur_kernel)

        dct_basis, dct_masks = self._build_dct_components(self.block_size)
        self.register_buffer("dct_basis", dct_basis)
        self.register_buffer("dct_masks", dct_masks)

        local_channels = 0
        if "wavelet" in selected:
            local_channels += 7  # residual + three detail maps at two scales
        if "dct" in selected:
            local_channels += 3  # low-, mid- and high-frequency block energy

        self.local_encoder = (
            nn.Sequential(
                self._block(local_channels, 32),
                self._block(32, 64),
                self._block(64, 128),
                self._block(128, 192),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            if local_channels
            else None
        )
        self.fft_encoder = (
            nn.Sequential(
                self._block(1, 16),
                self._block(16, 32),
                self._block(32, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            if "fft" in selected
            else None
        )
        encoded_dim = (192 if self.local_encoder is not None else 0) + (64 if self.fft_encoder is not None else 0)
        self.projection = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, output_dim),
            nn.GELU(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _build_dct_components(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(size, dtype=torch.float32)
        frequencies = torch.arange(size, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(math.pi * (positions + 0.5) * frequencies / size)
        basis[0] *= math.sqrt(1.0 / size)
        basis[1:] *= math.sqrt(2.0 / size)

        vertical, horizontal = torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing="ij"
        )
        frequency_sum = vertical + horizontal
        masks = torch.stack(
            [
                (frequency_sum >= 1) & (frequency_sum <= 3),
                (frequency_sum >= 4) & (frequency_sum <= 7),
                frequency_sum >= 8,
            ]
        ).to(torch.float32)
        return basis, masks

    @staticmethod
    def _luminance(images: torch.Tensor) -> torch.Tensor:
        return 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

    @staticmethod
    def _grid_size(images: torch.Tensor) -> tuple[int, int]:
        height, width = images.shape[-2:]
        return math.ceil(height / 8), math.ceil(width / 8)

    @staticmethod
    def _standardize_maps(maps: torch.Tensor) -> torch.Tensor:
        """Normalize each forensic channel independently within each image."""
        mean = maps.mean(dim=(-2, -1), keepdim=True)
        std = maps.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (maps - mean) / std

    def _haar_decompose(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        if height % 2 or width % 2:
            image = F.pad(image, (0, width % 2, 0, height % 2), mode="replicate")
        return F.conv2d(image, self.haar, stride=2)

    def _wavelet_maps(
        self,
        luminance: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        padded = F.pad(luminance, (2, 2, 2, 2), mode="replicate")
        residual = luminance - F.conv2d(padded, self.blur_kernel)
        level_one = self._haar_decompose(luminance)
        level_two = self._haar_decompose(level_one[:, :1])
        maps = torch.cat(
            [
                F.adaptive_avg_pool2d(residual.abs(), grid_size),
                F.adaptive_avg_pool2d(level_one[:, 1:].abs(), grid_size),
                F.adaptive_avg_pool2d(level_two[:, 1:].abs(), grid_size),
            ],
            dim=1,
        )
        return self._standardize_maps(maps)

    def _dct_maps_at_offset(
        self,
        luminance: torch.Tensor,
        offset_y: int,
        offset_x: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        luminance = luminance[..., offset_y:, offset_x:]
        height, width = luminance.shape[-2:]
        pad_height = (-height) % self.block_size
        pad_width = (-width) % self.block_size
        padded = F.pad(luminance, (0, pad_width, 0, pad_height), mode="replicate")
        grid_height = padded.shape[-2] // self.block_size
        grid_width = padded.shape[-1] // self.block_size
        patches = F.unfold(padded, kernel_size=self.block_size, stride=self.block_size)
        blocks = patches.transpose(1, 2).reshape(
            luminance.shape[0], -1, self.block_size, self.block_size
        )
        coefficients = torch.matmul(self.dct_basis, blocks)
        coefficients = torch.matmul(coefficients, self.dct_basis.transpose(0, 1))
        log_energy = torch.log1p(coefficients.square())
        band_energy = torch.einsum("blij,cij->blc", log_energy, self.dct_masks)
        band_energy = band_energy / self.dct_masks.sum(dim=(1, 2)).view(1, 1, -1)
        maps = band_energy.transpose(1, 2).reshape(
            luminance.shape[0], 3, grid_height, grid_width
        )
        if maps.shape[-2:] != grid_size:
            maps = F.interpolate(maps, size=grid_size, mode="bilinear", align_corners=False)
        return maps

    def _dct_maps(self, luminance: torch.Tensor) -> torch.Tensor:
        grid_size = self._grid_size(luminance)
        height, width = luminance.shape[-2:]
        if self.training:
            offset_y = int(torch.randint(0, min(self.block_size, height), ()).item())
            offset_x = int(torch.randint(0, min(self.block_size, width), ()).item())
            maps = self._dct_maps_at_offset(luminance, offset_y, offset_x, grid_size)
        else:
            offsets = [
                (offset_y, offset_x)
                for offset_y, offset_x in ((0, 0), (0, 4), (4, 0), (4, 4))
                if offset_y < height and offset_x < width
            ]
            maps = torch.stack(
                [
                    self._dct_maps_at_offset(luminance, offset_y, offset_x, grid_size)
                    for offset_y, offset_x in offsets
                ]
            ).mean(dim=0)
        return self._standardize_maps(maps)

    @staticmethod
    def _fft_map(
        luminance: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        height, width = luminance.shape[-2:]
        centered = luminance - luminance.mean(dim=(-2, -1), keepdim=True)
        window_y = torch.hann_window(
            height, periodic=False, device=luminance.device, dtype=luminance.dtype
        )
        window_x = torch.hann_window(
            width, periodic=False, device=luminance.device, dtype=luminance.dtype
        )
        window = window_y.view(1, 1, height, 1) * window_x.view(1, 1, 1, width)
        spectrum = torch.fft.fft2(centered * window, norm="ortho")
        magnitude = torch.log1p(torch.abs(torch.fft.fftshift(spectrum, dim=(-2, -1))))
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (magnitude - mean) / std
        return F.adaptive_avg_pool2d(normalized, grid_size)

    def forensic_maps(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return aligned local maps and a frequency-plane FFT map."""
        luminance = self._luminance(images).float()
        grid_size = self._grid_size(images)
        local_parts = []
        if "wavelet" in self.features:
            local_parts.append(self._wavelet_maps(luminance, grid_size))
        if "dct" in self.features:
            local_parts.append(self._dct_maps(luminance))
        local_maps = torch.cat(local_parts, dim=1) if local_parts else None
        fft_map = self._fft_map(luminance, grid_size) if "fft" in self.features else None
        return local_maps, fft_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        local_maps, fft_map = self.forensic_maps(images)
        encoded = []
        if self.local_encoder is not None:
            encoded.append(self.local_encoder(local_maps))
        if self.fft_encoder is not None:
            encoded.append(self.fft_encoder(fft_map))
        return self.projection(torch.cat(encoded, dim=1))


class RedistributionAwareDetector(nn.Module):
    """Two-branch CNN with reliability-gated spatial/forensic fusion."""

    def __init__(
        self,
        pretrained: bool = True,
        use_frequency: bool = True,
        architecture: str = "convnext_small",
        forensic_features: Sequence[str] = FORENSIC_FEATURES,
        forensic_dim: int = 256,
        fusion_dim: int = 512,
    ) -> None:
        super().__init__()
        if architecture not in BACKBONES:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose from {sorted(BACKBONES)}")
        backbone_builder, default_weights = BACKBONES[architecture]
        backbone = backbone_builder(weights=default_weights if pretrained else None)
        self.spatial = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.architecture = architecture
        self.use_frequency = use_frequency  # Retained for CLI/checkpoint compatibility.
        self.forensic_features = tuple(forensic_features)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        spatial_dim = backbone.classifier[-1].in_features
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(spatial_dim), nn.Linear(spatial_dim, fusion_dim), nn.GELU()
        )
        if use_frequency:
            self.frequency = MultiDomainForensicBranch(
                output_dim=forensic_dim,
                features=self.forensic_features,
            )
            self.forensic_projection = nn.Sequential(
                nn.LayerNorm(forensic_dim), nn.Linear(forensic_dim, fusion_dim), nn.GELU()
            )
            self.gate = nn.Sequential(
                nn.Linear(spatial_dim + forensic_dim, fusion_dim // 2),
                nn.GELU(),
                nn.Linear(fusion_dim // 2, fusion_dim),
                nn.Sigmoid(),
            )
        else:
            self.frequency = None
            self.forensic_projection = None
            self.gate = None
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(0.25), nn.Linear(fusion_dim, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        return_details: bool = False,
    ):
        if return_features and return_details:
            raise ValueError("Choose either return_features or return_details, not both.")
        normalized = (images - self.mean) / self.std
        spatial_raw = self.spatial_pool(self.spatial(normalized)).flatten(1)
        fused = self.spatial_projection(spatial_raw)
        forensic_raw = None
        reliability = None
        if self.frequency is not None:
            forensic_raw = self.frequency(images)
            forensic = self.forensic_projection(forensic_raw)
            spatial_gate_input = F.layer_norm(spatial_raw, spatial_raw.shape[1:])
            forensic_gate_input = F.layer_norm(forensic_raw, forensic_raw.shape[1:])
            reliability = self.gate(
                torch.cat([spatial_gate_input, forensic_gate_input], dim=1)
            )
            fused = fused + reliability * forensic
        logits = self.head(fused).squeeze(1)
        if return_details:
            return logits, {
                "fused": fused,
                "spatial": spatial_raw,
                "forensic": forensic_raw,
                "reliability": reliability,
            }
        return (logits, fused) if return_features else logits


def build_model(
    pretrained: bool = True,
    use_frequency: bool = True,
    architecture: str = "convnext_small",
    forensic_features: Sequence[str] = FORENSIC_FEATURES,
) -> RedistributionAwareDetector:
    return RedistributionAwareDetector(
        pretrained=pretrained,
        use_frequency=use_frequency,
        architecture=architecture,
        forensic_features=forensic_features,
    )
