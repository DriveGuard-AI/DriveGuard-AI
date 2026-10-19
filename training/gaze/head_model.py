"""An independently eligible head-pose stream, used only by distraction."""
import torch
from torch import nn

from training.gaze.model import RelationGazeTCN
from training.gaze.head_policy import resolve_head_policy, minimum_head_count
from training.temporal.model import _ResidualBlock


def _weights(scores, eligible):
    weights = torch.softmax(scores.masked_fill(~eligible,torch.finfo(scores.dtype).min),dim=-1)
    weights = weights.masked_fill(~eligible,0)
    return weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class RelationGazeHeadTCN(RelationGazeTCN):
    def __init__(self, head_policy=None, **kwargs):
        super().__init__(**kwargs)
        self.head_policy = resolve_head_policy(head_policy, kwargs.get('window_steps', 100))
        projection = kwargs.get('projection_dim',16)
        hidden = kwargs.get('gate_hidden_dim',16)
        self.head_projection = nn.Sequential(nn.Linear(6,projection),nn.ReLU())
        previous = projection
        blocks = []
        for width,dilation in zip(kwargs.get('channels',(16,)*5),kwargs.get('dilations',(1,2,4,8,16))):
            blocks.append(_ResidualBlock(previous,width,kwargs.get('kernel_size',3),dilation,kwargs.get('dropout',.1)))
            previous = width
        self.head_blocks = nn.Sequential(*blocks)
        self.gates[3] = nn.Sequential(nn.Linear(5*previous+10,hidden),nn.ReLU(),nn.Linear(hidden,5))

    def head_statistics(self, x):
        good=x[:,:,28:31].bool().all(-1)
        ratio=good.float().mean(-1)
        steps = good.shape[-1]
        last = torch.where(good, torch.arange(steps, device=x.device), -1).amax(-1)
        gap = steps - 1 - last
        available = (gap <= self.head_policy['max_gap_steps']) & (good.sum(-1) >= minimum_head_count(self.head_policy, steps))
        return available, ratio, gap

    def head_quality(self, x):
        return self.head_statistics(x)[:2]

    def behavior_availability(self,x,eligible):
        result=eligible.any(-1)[:,None].expand(-1,4).clone()
        result[:,3] |= self.head_quality(x)[0]
        return result

    def encode_head(self,x):
        masks=x[:,:,28:31]
        values=x[:,:,5:8].masked_fill(~masks.bool(),0)
        projected=self.head_projection(torch.cat((values,masks),dim=-1)).transpose(1,2)
        return self.head_blocks(projected)

    def forward(self,x,quality,eligible,return_aux=False):
        embeddings=self.checked_embeddings(x,quality,eligible)
        h_eligible,h_ratio,h_gap=self.head_statistics(x)
        h=self.encode_head(x)[:,:,-1].masked_fill(~h_eligible[:,None],0)
        gate_input=torch.cat((embeddings.flatten(1),quality.flatten(1)),dim=-1)
        ordinary_scores=torch.stack([gate(gate_input) for gate in self.gates[:3]],dim=1)
        ordinary_weights=_weights(ordinary_scores,eligible[:,None,:])
        ordinary_embeddings=torch.matmul(ordinary_weights,embeddings)
        ordinary_logits=torch.cat([head(ordinary_embeddings[:,i]) for i,head in enumerate(self.heads[:3])],dim=-1)
        # Current H preserves legacy freshness. A permitted gap uses an explicit
        # slot-recency score; no values/masks are filled or presented as current.
        recency = (1. - h_gap.float() / (self.head_policy['max_gap_steps'] + 1)).clamp_min(0.)
        freshness = torch.where(h_gap == 0, quality[:,0,1], recency)
        h_quality=torch.stack((h_ratio,freshness),dim=-1).masked_fill(~h_eligible[:,None],0)
        dist_embeddings=torch.cat((embeddings,h[:,None,:]),dim=1)
        dist_quality=torch.cat((quality,h_quality[:,None,:]),dim=1)
        dist_eligible=torch.cat((eligible,h_eligible[:,None]),dim=1)
        dist_scores=self.gates[3](torch.cat((dist_embeddings.flatten(1),dist_quality.flatten(1)),dim=-1))
        dist_weights=_weights(dist_scores,dist_eligible)
        dist_logit=self.heads[3]((dist_weights[:,:,None]*dist_embeddings).sum(1))
        task_available=self.behavior_availability(x,eligible)
        gaze_available=self.gaze_availability(x,eligible)
        return dict(behavior_logits=torch.cat((ordinary_logits,dist_logit),dim=-1).masked_fill(~task_available,0),
            gaze_logits=self.gaze_head(embeddings[:,0]).masked_fill(~gaze_available[:,None],0),
            available=task_available.any(-1),task_available=task_available,gaze_available=gaze_available,
            weights=torch.cat((ordinary_weights,dist_weights[:,None,:4]),dim=1),
            head_available=h_eligible,head_dist_weight=dist_weights[:,4],head_valid_ratio=h_ratio,
            head_gap_steps=h_gap,head_current_valid=h_gap==0)
