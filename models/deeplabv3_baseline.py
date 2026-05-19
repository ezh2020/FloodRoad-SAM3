from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3FloodRoad(nn.Module):
    """Torchvision DeepLabV3-ResNet50 baseline with a single binary output channel."""

    def __init__(self, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = None
        weights_backbone = None
        if pretrained_backbone:
            weights_backbone = ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def build_deeplab(cfg: dict) -> DeepLabV3FloodRoad:
    return DeepLabV3FloodRoad(pretrained_backbone=bool(cfg.get("pretrained_backbone", True)))
