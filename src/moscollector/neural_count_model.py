"""Optional PyTorch count networks; production serving does not import this."""

import torch
from torch import nn


class CountNetwork(nn.Module):
    def __init__(self, numeric_dim, category_sizes, use_history, initial_log_mean):
        super().__init__()
        self.use_history = use_history
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, width, padding_idx=0)
                for size, width in zip(category_sizes, (8, 4, 2), strict=True)
            ]
        )
        self.current = nn.Sequential(nn.Linear(numeric_dim + 14, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        if use_history:
            self.history_projection = nn.Sequential(nn.Linear(numeric_dim + 1, 32), nn.Tanh())
            self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 if use_history else 32, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, initial_log_mean)

    def forward(self, numeric, categorical, history=None, lengths=None):
        embeddings = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        current = self.current(torch.cat([numeric, *embeddings], dim=1))
        if self.use_history:
            if history is None or lengths is None:
                raise ValueError("Missing sequence inputs")
            output, _ = self.gru(self.history_projection(history))
            # A unidirectional GRU's last valid output cannot depend on later
            # padding. Empty histories contribute the exact zero vector.
            state = output[torch.arange(len(output), device=output.device), (lengths - 1).clamp(min=0)]
            state = state * (lengths > 0).unsqueeze(1)
            current = torch.cat([current, state], dim=1)
        return self.head(current).squeeze(1).clamp(-12, 8)
