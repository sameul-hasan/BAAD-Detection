"""Broadcasted Residual Network (BC-ResNet) for ultra-lightweight KWS.

Based on Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting"
(Interspeech 2021). Uses standard BatchNorm2d, SiLU/ReLU-friendly ops, and Conv2d
so the model exports cleanly to ONNX and quantizes to < 1 MB INT8.
"""
from __future__ import annotations

import torch
from torch import nn


class SubSpectralNorm(nn.Module):
    """Standard BatchNorm2d wrapper (kept simple & ONNX/INT8-friendly)."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BCResBlock(nn.Module):
    """Broadcasted residual block: 2D freq-depthwise -> freq-pool -> 1D temporal-depthwise + broadcast."""

    def __init__(self, in_c: int, out_c: int, stride: tuple[int, int] = (1, 1), dilation: int = 1):
        super().__init__()
        self.transition = (in_c != out_c) or (stride != (1, 1))
        layers = []
        if in_c != out_c:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ])
        # Frequency depthwise 3x1
        layers.extend([
            nn.Conv2d(out_c, out_c, kernel_size=(3, 1), stride=stride, padding=(1, 0), groups=out_c, bias=False),
            SubSpectralNorm(out_c),
        ])
        self.f2 = nn.Sequential(*layers)

        # Temporal 1D depthwise 1x3 (represented as Conv2d on [B, C, 1, T])
        pad_t = dilation
        self.f1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(1, 3), padding=(0, pad_t), dilation=(1, dilation), groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y2 = self.f2(x)
        # Global frequency average pooling -> [B, C, 1, T]
        y1 = y2.mean(dim=2, keepdim=True)
        y1 = self.f1(y1)
        out = y2 + y1  # broadcast across frequency axis
        if not self.transition:
            out = out + identity
        return self.act(out)


class BCResNetBackbone(nn.Module):
    """BC-ResNet backbone taking [B, 1, n_mels, T] log-Mel and outputting [B, num_classes] logits."""

    def __init__(self, num_classes: int, scale: int = 8, dropout: float = 0.15):
        super().__init__()
        c1, c2, c3, c4 = scale * 8, scale * 12, scale * 16, scale * 20
        head_c = c4

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            BCResBlock(c1, c1, stride=(1, 1), dilation=1),
            BCResBlock(c1, c1, stride=(1, 1), dilation=1),
        )
        self.stage2 = nn.Sequential(
            BCResBlock(c1, c2, stride=(2, 1), dilation=1),
            BCResBlock(c2, c2, stride=(1, 1), dilation=2),
        )
        self.stage3 = nn.Sequential(
            BCResBlock(c2, c3, stride=(2, 2), dilation=1),
            BCResBlock(c3, c3, stride=(1, 1), dilation=2),
            BCResBlock(c3, c3, stride=(1, 1), dilation=4),
        )
        self.stage4 = nn.Sequential(
            BCResBlock(c3, c4, stride=(1, 1), dilation=1),
            BCResBlock(c4, c4, stride=(1, 1), dilation=2),
            BCResBlock(c4, c4, stride=(1, 1), dilation=4),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c4, head_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_c),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        x = x.mean(dim=(2, 3))
        x = self.drop(x)
        return self.classifier(x)


class EndToEndKWS(nn.Module):
    """Audio-to-logits wrapper embedding LogMel + optional train-only SpecAugment + BCResNetBackbone."""

    def __init__(self, logmel: nn.Module, backbone: nn.Module, spec_augment: nn.Module | None = None):
        super().__init__()
        self.logmel = logmel
        self.spec_augment = spec_augment
        self.backbone = backbone

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.logmel(wav)
        if self.training and self.spec_augment is not None:
            x = self.spec_augment(x)
        return self.backbone(x)
