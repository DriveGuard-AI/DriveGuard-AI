"""Warm-started fatigue backbone, frame actions and annotation-free event logic."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ActionModel(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        if backbone.frames!=32: raise ValueError('Actions require dense 32-frame input')
        self.backbone=backbone
        self.actions=nn.Linear(512,2)

    def encode(self,images):
        base=self.backbone
        outputs=[]
        for chunk in images.split(base.frame_chunk):
            pixels=(chunk.float()/255.-base.rgb_mean)/base.rgb_std
            features=base.encoder(pixels)
            outputs.append((((features-base.head_mean_hi)-base.head_mean_lo)/base.head_std).clamp(-5,5))
        return torch.cat(outputs)

    def frame_actions(self,images):
        """Per-frame logits without labels, temporal future, or batch statistics."""
        return self.actions(self.encode(images))

    def forward(self,images,visual_mask):
        base=self.backbone
        if (images.dtype!=torch.uint8 or images.ndim!=5 or tuple(images.shape[1:])!=(32,3,224,224)
                or visual_mask.dtype!=torch.bool or tuple(visual_mask.shape)!=tuple(images.shape[:2])):
            raise ValueError('Expected dense uint8 pixels and boolean masks')
        available=visual_mask.all(1)
        logits=base.gate.new_zeros(len(images))
        actions=base.gate.new_zeros((len(images),32,2))
        if available.any():
            features=self.encode(images[available].flatten(0,1)).reshape(-1,32,512)
            visual=torch.cat((torch.zeros_like(features),features),-1)
            hidden=F.relu(base.probe.visual_projection(visual)).transpose(1,2)
            for layer in base.probe.temporal:
                hidden=hidden+base.probe.dropout(F.relu(layer(F.pad(hidden,(2*layer.dilation[0],0)))))
            hidden=hidden+base.gate*torch.tanh(base.extra(F.pad(hidden,(18,0))))
            logits[available]=base.probe.head(base.probe.dropout(hidden[:,:,-1])).squeeze(-1)
            actions[available]=self.actions(features)
        return dict(logits=logits,actions=actions,available=available)


def loss_terms(output,target,known,actions,masks):
    result={}
    for name,pred,y,mask in (
            ('fatigue',output['logits'],target,known&output['available']),
            ('eye',output['actions'][:,:,0],actions[:,:,0],masks[:,:,0]&output['available'][:,None]),
            ('yawn',output['actions'][:,:,1],actions[:,:,1],masks[:,:,1]&output['available'][:,None])):
        pred,y=pred[mask],y[mask]
        if not torch.isfinite(pred).all() or not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():
            raise ValueError('Invalid supervised action/fatigue values')
        result[name]=(F.binary_cross_entropy_with_logits(pred,y,reduction='sum'),y.numel())
    return result


class EventDecoder:
    """Carry predicted closure within a recording using actual observed PTS.

    Fixed logit>=0 for actions; no validation threshold search. A reset is based
    solely on original source gaps. Missing pixels reset closure, never close eyes.
    """
    def __init__(self,closed_eye_min_ms):
        if type(closed_eye_min_ms)!=int or closed_eye_min_ms<=0: raise ValueError('Invalid closure duration')
        self.duration=closed_eye_min_ms
        self.last=None; self.since=None; self.margin=None

    def step(self,time,eye,yawn,valid,reset=False):
        if self.last is not None and time<self.last: raise ValueError('Unordered event timestamps')
        if not valid:
            self.last=time; self.since=self.margin=None
            return None
        if not math.isfinite(eye) or not math.isfinite(yawn): raise ValueError('Nonfinite action prediction')
        if reset or eye<0: self.since=self.margin=None
        if eye>=0:
            if self.since is None: self.since=time; self.margin=eye
            else: self.margin=min(self.margin,eye)
        closure=self.margin if self.since is not None and time-self.since>=self.duration else -1e6
        self.last=time
        return float(max(yawn,closure))
