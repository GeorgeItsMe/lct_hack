"""Optional raw-onset embeddings and GRU, sharing the current-snapshot baseline."""

import torch
from torch import nn

from moscollector.experiments.neural_count_model import CountNetwork


class EventCountNetwork(nn.Module):
    def __init__(self, numeric_dim, category_sizes, event_category_sizes, use_history, initial_log_mean):
        super().__init__()
        self.use_history = use_history
        self.backbone = CountNetwork(numeric_dim, category_sizes, use_history, initial_log_mean)
        if use_history:
            self.event_embeddings = nn.ModuleList(
                [
                    nn.Embedding(size, width, padding_idx=0)
                    for size, width in zip(event_category_sizes, (12, 4, 4), strict=True)
                ]
            )
            self.backbone.history_projection = nn.Sequential(nn.Linear(23, 32), nn.Tanh())

    def forward(self, numeric, categorical, event_categorical=None, event_numeric=None, lengths=None):
        if not self.use_history:
            return self.backbone(numeric, categorical)
        if event_categorical is None or event_numeric is None or lengths is None:
            raise ValueError("Missing raw-onset sequence inputs")
        embeddings = [layer(event_categorical[:, :, i]) for i, layer in enumerate(self.event_embeddings)]
        tokens = torch.cat([*embeddings, event_numeric], dim=2)
        return self.backbone(numeric, categorical, tokens, lengths)
