"""Class-balanced supervision for partially labeled multilabel windows."""

import torch
from torch import Tensor, nn


def _binary(tensor: Tensor, name: str) -> None:
    if tensor.is_complex() or not torch.isfinite(tensor).all() or not ((tensor == 0) | (tensor == 1)).all():
        raise ValueError(f"{name} must contain only finite binary values")


def masked_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    label_mask: Tensor,
    active_mask: Tensor | None = None,
) -> tuple[Tensor | None, Tensor, Tensor]:
    """Average known BCE within each supported class, then across classes.

    Unknown targets must be zero placeholders. Inactive classes and unknown
    labels contribute neither gradient nor statistics. Returned per-class sums
    and integer counts are detached for epoch aggregation; an unsupervised batch
    returns ``None`` so the caller can skip both backward and the optimizer step.
    All tensors must be on the same device.
    """
    for value, name in ((logits, "logits"), (targets, "targets"), (label_mask, "label_mask")):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
    if logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] == 0:
        raise ValueError("logits must have shape [B>0, 4]")
    if targets.shape != logits.shape or label_mask.shape != logits.shape:
        raise ValueError("targets and label_mask must match logits shape [B, 4]")
    if not logits.is_floating_point() or targets.dtype != torch.float32:
        raise TypeError("logits must be floating point and targets must be float32")
    if targets.device != logits.device or label_mask.device != logits.device:
        raise ValueError("logits, targets, and label_mask must be on the same device")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    _binary(targets, "targets")
    _binary(label_mask, "label_mask")
    if active_mask is None:
        active_mask = torch.ones(4, dtype=torch.bool, device=logits.device)
    if not isinstance(active_mask, Tensor):
        raise TypeError("active_mask must be a tensor")
    if active_mask.shape != (4,) or active_mask.device != logits.device:
        raise ValueError("active_mask must have shape [4] and be on the logits device")
    _binary(active_mask, "active_mask")
    known = label_mask.bool()
    if (targets[~known] != 0).any():
        raise ValueError("unknown targets must use zero placeholders")
    effective = known & active_mask.bool().unsqueeze(0)
    counts = effective.sum(dim=0)
    supported = counts > 0
    if not supported.any():
        return None, logits.new_zeros(4), counts.detach()
    element_losses = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    sums = element_losses.masked_fill(~effective, 0).sum(dim=0)
    loss = (sums[supported] / counts[supported]).mean()
    return loss, sums.detach(), counts.detach()
