"""Small causal temporal heads over precomputed, frozen RGB features."""
import torch
from torch import nn
from torch.nn import functional as F


class VisualProbe(nn.Module):
    """Eight causal embeddings, with optional frozen baseline context.

    Encoders deliberately live outside this model. Three left-padded temporal
    layers give the last position a receptive field spanning all eight frames.
    """

    def __init__(self, visual_dim, context_dim, hidden_dim=64, dropout=.1, mode='visual'):
        super().__init__()
        if mode not in ('visual', 'fusion'):
            raise ValueError('Probe mode must be visual or fusion')
        if any(type(value) is not int or value <= 0 for value in (visual_dim, hidden_dim)):
            raise ValueError('visual_dim and hidden_dim must be positive integers')
        if type(context_dim) is not int or context_dim < 0 or (mode == 'fusion' and not context_dim):
            raise ValueError('Fusion requires a positive context_dim')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.model_params = dict(visual_dim=visual_dim, context_dim=context_dim,
                                 hidden_dim=hidden_dim, dropout=dropout, mode=mode)
        self.mode = mode
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.temporal = nn.ModuleList(nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=d)
                                      for d in (1, 2, 4))
        self.dropout = nn.Dropout(dropout)
        if mode == 'fusion':
            self.context_projection = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim * (2 if mode == 'fusion' else 1), 1)

    def forward(self, visual, context, visual_mask):
        if visual.ndim != 3 or tuple(visual.shape[1:]) != (8, self.model_params['visual_dim']):
            raise ValueError('Expected visual [batch,8,visual_dim]')
        if tuple(visual_mask.shape) != tuple(visual.shape[:2]) or visual_mask.dtype != torch.bool:
            raise ValueError('Expected boolean visual_mask [batch,8]')
        available = visual_mask.all(dim=1)
        # Exclude invalid rows before any arithmetic, including placeholder NaNs.
        clean = visual.masked_fill(~available[:, None, None], 0)
        hidden = F.relu(self.visual_projection(clean)).transpose(1, 2)
        for layer in self.temporal:
            transformed = layer(F.pad(hidden, (2 * layer.dilation[0], 0)))
            hidden = hidden + self.dropout(F.relu(transformed))
        embedding = hidden[:, :, -1]
        if self.mode == 'fusion':
            if tuple(context.shape) != (len(visual), self.model_params['context_dim']):
                raise ValueError('Expected context [batch,context_dim]')
            frozen_context = context.masked_fill(~available[:, None], 0)
            embedding = torch.cat((embedding, self.context_projection(frozen_context)), dim=1)
        logits = self.head(self.dropout(embedding)).squeeze(-1)
        return {'logits': logits.masked_fill(~available, 0), 'available': available}
