"""A nine-class gaze head on the same F embedding used by behavior gates."""
import torch
from torch import nn
from training.relation.model import RelationGatedTCN
from training.gaze.profile import GROUP_INDICES, GAZE_HEAD_INDICES, GAZE_IRIS_PAIRS, GAZE_MIN_VALID_RATIO


class RelationGazeTCN(RelationGatedTCN):
    def __init__(self,input_channels=46,group_indices=GROUP_INDICES,**kwargs):
        if input_channels != 46 or tuple(tuple(g) for g in group_indices) != GROUP_INDICES:
            raise ValueError('Joint model requires canonical 23 features and four groups')
        super().__init__(input_channels=input_channels,group_indices=group_indices,**kwargs)
        self.gaze_head=nn.Linear(self.heads[0].in_features,9)

    @staticmethod
    def gaze_availability(x,eligible):
        masks=x[:,:,23:].bool()
        good=masks[:,:,list(GAZE_HEAD_INDICES)].all(-1) & (masks[:,:,list(GAZE_IRIS_PAIRS[0])].all(-1) | masks[:,:,list(GAZE_IRIS_PAIRS[1])].all(-1))
        return eligible[:,0] & good[:,-1] & (good.float().mean(-1) >= GAZE_MIN_VALID_RATIO)

    @staticmethod
    def behavior_availability(x,eligible):
        return eligible.any(-1)[:,None].expand(-1,4)

    def checked_embeddings(self,x,quality,eligible):
        embeddings=self.encode(x)[:,:,:,-1]
        if not isinstance(quality,torch.Tensor) or quality.dtype != torch.float32:
            raise TypeError('quality must be float32')
        if quality.shape != (len(x),4,2) or quality.device != x.device:
            raise ValueError('quality shape/device mismatch')
        if not torch.isfinite(quality).all() or not ((quality>=0)&(quality<=1)).all():
            raise ValueError('quality must be finite in [0,1]')
        if not isinstance(eligible,torch.Tensor) or eligible.dtype != torch.bool:
            raise TypeError('eligible must be boolean')
        if eligible.shape != (len(x),4) or eligible.device != x.device:
            raise ValueError('eligible shape/device mismatch')
        return embeddings.masked_fill(~eligible[:,:,None],0)

    def forward(self,x,quality,eligible,return_aux=False):
        # One encoder invocation produces the embeddings for all five tasks.
        embeddings=self.checked_embeddings(x,quality,eligible)
        gate_input=torch.cat((embeddings.flatten(1),quality.flatten(1)),dim=-1)
        scores=torch.stack([gate(gate_input) for gate in self.gates],dim=1)
        weights=torch.softmax(scores.masked_fill(~eligible[:,None,:],torch.finfo(scores.dtype).min),dim=-1)
        weights=weights.masked_fill(~eligible[:,None,:],0)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        task_embeddings=torch.matmul(weights,embeddings)
        behavior=torch.cat([head(task_embeddings[:,task]) for task,head in enumerate(self.heads)],dim=-1)
        available=eligible.any(-1)
        gaze_available=self.gaze_availability(x,eligible)
        return {'behavior_logits':behavior.masked_fill(~available[:,None],0),
                'gaze_logits':self.gaze_head(embeddings[:,0]).masked_fill(~gaze_available[:,None],0),
                'available':available,'gaze_available':gaze_available,'weights':weights}
