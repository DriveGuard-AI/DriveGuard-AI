"""Lightweight shared causal encoding and task-specific relation-group gates."""

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn

from training.temporal.model import _positive_int, _ResidualBlock


DEFAULT_GROUP_INDICES = (
    (0, 1, 2, 3, 4, 5, 6, 7, 12),
    (8, 9),
    (13, 14, 15, 16, 17, 18),
    (10, 11),
)


class RelationGatedTCN(nn.Module):
    """Map normalized values/masks and quality summaries to four task logits.

    ``x`` is FP32 ``[B, T, 2F]``: F values followed by F binary masks.
    Invalid feature values are zeroed before projection. Four groups have distinct
    projections and share one causal temporal encoder. Each task independently
    gates the final group embeddings using FP32 quality ``[B, 4, 2]`` and boolean
    eligibility ``[B, 4]``. Ineligible embeddings cannot influence any task gate
    or aggregation. Fully unavailable rows return zero logits and weights.
    """

    def __init__(self, input_channels: int = 38,
                 group_indices: Sequence[Sequence[int]] = DEFAULT_GROUP_INDICES,
                 projection_dim: int = 16, gate_hidden_dim: int = 16,
                 channels: Sequence[int] = (16, 16, 16, 16, 16),
                 dilations: Sequence[int] = (1, 2, 4, 8, 16),
                 kernel_size: int = 3, dropout: float = 0.1,
                 window_steps: int = 100) -> None:
        super().__init__()
        for name, value in (("input_channels", input_channels),
                            ("projection_dim", projection_dim),
                            ("gate_hidden_dim", gate_hidden_dim),
                            ("kernel_size", kernel_size),
                            ("window_steps", window_steps)):
            _positive_int(value, name)
        if input_channels % 2:
            raise ValueError("input_channels must be even (F values plus F masks)")
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
        if not isinstance(group_indices, Sequence) or len(group_indices) != 4:
            raise ValueError("group_indices must contain exactly four nonempty groups")
        indices = []
        for group in group_indices:
            if not isinstance(group, Sequence) or not group:
                raise ValueError("group_indices must contain exactly four nonempty groups")
            for index in group:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("group indices must be integers")
                indices.append(index)
        if sorted(indices) != list(range(input_channels // 2)):
            raise ValueError("group indices must partition all features exactly once")

        self.input_channels = input_channels
        self.window_steps = window_steps
        self.group_indices = tuple(tuple(group) for group in group_indices)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * len(group), projection_dim), nn.ReLU())
            for group in self.group_indices
        ])
        blocks = []
        previous_channels = projection_dim
        for width, dilation in zip(channels, dilations):
            blocks.append(_ResidualBlock(previous_channels, width, kernel_size, dilation, float(dropout)))
            previous_channels = width
        self.blocks = nn.Sequential(*blocks)
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(4 * previous_channels + 8, gate_hidden_dim),
                          nn.ReLU(), nn.Linear(gate_hidden_dim, 4))
            for _ in range(4)
        ])
        self.heads = nn.ModuleList([nn.Linear(previous_channels, 1) for _ in range(4)])

    def encode(self, x: Tensor) -> Tensor:
        """Return causal group features ``[B, 4, channels[-1], T]``.

        Feature time t only depends on inputs at times <= t. Use ``eval()`` for
        deterministic comparisons because the shared encoder contains dropout.
        Groups are folded into the batch dimension for a single encoder call.
        """
        if not isinstance(x, Tensor) or x.dtype != torch.float32:
            raise TypeError("x must be a float32 tensor")
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1:] != (self.window_steps, self.input_channels):
            raise ValueError(f"x must have shape [B>0, {self.window_steps}, {self.input_channels}]")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        feature_count = self.input_channels // 2
        masks = x[:, :, feature_count:]
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("feature validity masks must be binary")
        values = x[:, :, :feature_count].masked_fill(masks == 0, 0)
        projected = []
        for group, projection in zip(self.group_indices, self.projections):
            group_input = torch.cat((values[:, :, list(group)], masks[:, :, list(group)]), dim=-1)
            projected.append(projection(group_input).transpose(1, 2))
        grouped = torch.stack(projected, dim=1)
        encoded = self.blocks(grouped.flatten(0, 1))
        return encoded.reshape(x.shape[0], 4, encoded.shape[1], self.window_steps)

    def forward(self, x: Tensor, quality: Tensor, eligible: Tensor,
                return_aux: bool = False) -> Tensor | dict[str, Tensor]:
        embeddings = self.encode(x)[:, :, :, -1]
        if not isinstance(quality, Tensor) or quality.dtype != torch.float32:
            raise TypeError("quality must be a float32 tensor")
        if quality.shape != (x.shape[0], 4, 2):
            raise ValueError("quality must have shape [B, 4, 2]")
        if quality.device != x.device:
            raise ValueError("quality and x must be on the same device")
        if not torch.isfinite(quality).all() or not ((quality >= 0) & (quality <= 1)).all():
            raise ValueError("quality must contain finite values in [0, 1]")
        if not isinstance(eligible, Tensor) or eligible.dtype != torch.bool:
            raise TypeError("eligible must be a boolean tensor")
        if eligible.shape != (x.shape[0], 4):
            raise ValueError("eligible must have shape [B, 4]")
        if eligible.device != x.device:
            raise ValueError("eligible and x must be on the same device")

        embeddings = embeddings.masked_fill(~eligible[:, :, None], 0)
        gate_input = torch.cat((embeddings.flatten(1), quality.flatten(1)), dim=-1)
        scores = torch.stack([gate(gate_input) for gate in self.gates], dim=1)
        # A finite sentinel keeps the all-missing softmax and its gradients defined.
        weights = torch.softmax(scores.masked_fill(~eligible[:, None, :], torch.finfo(scores.dtype).min), dim=-1)
        weights = weights.masked_fill(~eligible[:, None, :], 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        task_embeddings = torch.matmul(weights, embeddings)
        logits = torch.cat([head(task_embeddings[:, task]) for task, head in enumerate(self.heads)], dim=-1)
        available = eligible.any(dim=-1)
        logits = logits.masked_fill(~available[:, None], 0)
        if return_aux:
            return {"logits": logits, "weights": weights, "available": available}
        return logits
