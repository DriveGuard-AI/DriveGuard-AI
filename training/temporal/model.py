"""Causal TCN for fixed windows of normalized features and validity masks."""

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.projection = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        residual = self.projection(x)
        x = self.dropout1(F.relu(self.conv1(F.pad(x, (self.left_padding, 0)))))
        x = self.dropout2(F.relu(self.conv2(F.pad(x, (self.left_padding, 0)))))
        return F.relu(x + residual)


class TemporalTCN(nn.Module):
    """Map ``[B, T, 2F]`` windows to four independent, unnormalized logits.

    The first F channels are normalized feature values; the next F are binary
    validity masks. Output feature t only depends on input times <= t. Call
    ``eval()`` when comparing deterministic features because training uses dropout.
    """

    def __init__(self, input_channels: int = 24,
                 channels: Sequence[int] = (32, 32, 32, 32, 32),
                 dilations: Sequence[int] = (1, 2, 4, 8, 16),
                 kernel_size: int = 3, dropout: float = 0.1,
                 window_steps: int = 100) -> None:
        super().__init__()
        _positive_int(input_channels, "input_channels")
        if input_channels % 2:
            raise ValueError("input_channels must be even (F values plus F masks)")
        _positive_int(kernel_size, "kernel_size")
        _positive_int(window_steps, "window_steps")
        if (not isinstance(channels, Sequence) or not isinstance(dilations, Sequence)
                or not channels or len(channels) != len(dilations)):
            raise ValueError("channels and dilations must be nonempty sequences of equal length")
        for width in channels:
            _positive_int(width, "channels entries")
        for dilation in dilations:
            _positive_int(dilation, "dilations entries")
        if (isinstance(dropout, bool) or not isinstance(dropout, (float, int))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        self.input_channels = input_channels
        self.window_steps = window_steps
        blocks = []
        previous_channels = input_channels
        for width, dilation in zip(channels, dilations):
            blocks.append(_ResidualBlock(previous_channels, width, kernel_size, dilation, float(dropout)))
            previous_channels = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(previous_channels, 4)

    def forward_features(self, x: Tensor) -> Tensor:
        """Return causal features with shape ``[B, channels[-1], T]``."""
        if not isinstance(x, Tensor) or x.dtype != torch.float32:
            raise TypeError("x must be a float32 tensor")
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1:] != (self.window_steps, self.input_channels):
            raise ValueError(f"x must have shape [B>0, {self.window_steps}, {self.input_channels}]")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        masks = x[:, :, self.input_channels // 2:]
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("feature validity masks must be binary")
        return self.blocks(x.transpose(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x)[:, :, -1])
