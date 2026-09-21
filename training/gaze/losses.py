"""Equal per-task means with configurable positive task weights."""
import torch
from torch.nn import functional as F
from training.temporal.losses import masked_bce_with_logits, _binary


def joint_loss(behavior_logits,gaze_logits,targets,label_masks,gaze_targets,gaze_label_masks,task_weights=None):
    """Unknown tasks do not enter the autograd graph or the weighted divisor."""
    _,_,behavior_counts=masked_bce_with_logits(behavior_logits,targets,label_masks)
    batch=len(behavior_logits)
    if gaze_logits.shape != (batch,9) or not gaze_logits.is_floating_point() or not torch.isfinite(gaze_logits).all():
        raise ValueError('gaze_logits must be finite [B,9]')
    if gaze_targets.dtype != torch.int64 or gaze_targets.shape != (batch,) or gaze_label_masks.shape != (batch,):
        raise ValueError('gaze targets/masks must have shape [B], int64 targets')
    if any(v.device != behavior_logits.device for v in (gaze_logits,gaze_targets,gaze_label_masks)):
        raise ValueError('All loss inputs must be on the same device')
    _binary(gaze_label_masks,'gaze_label_masks')
    if ((gaze_targets<0)|(gaze_targets>=9)).any() or (gaze_targets[~gaze_label_masks.bool()] != 0).any():
        raise ValueError('Invalid gaze class or nonzero unknown placeholder')
    weights=torch.as_tensor(task_weights if task_weights is not None else [1.]*5,device=behavior_logits.device,dtype=torch.float32)
    if weights.shape != (5,) or not torch.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('task_weights must contain five finite positive weights')
    counts=torch.cat((behavior_counts,gaze_label_masks.bool().sum().reshape(1)))
    sums=behavior_logits.new_zeros(5)
    losses=[]; present=[]
    for index in range(4):
        if counts[index] > 0:
            known=label_masks[:,index].bool()
            task_sum=F.binary_cross_entropy_with_logits(behavior_logits[known,index],targets[known,index],reduction='sum')
            sums[index]=task_sum.detach()
            losses.append(task_sum/counts[index]*weights[index]); present.append(index)
    if counts[4] > 0:
        known=gaze_label_masks.bool()
        task_sum=F.cross_entropy(gaze_logits[known],gaze_targets[known],reduction='sum')
        sums[4]=task_sum.detach()
        losses.append(task_sum/counts[4]*weights[4]); present.append(4)
    loss=torch.stack(losses).sum()/weights[present].sum() if losses else None
    return loss,sums,counts.detach()
