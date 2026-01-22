import torch
import torch.nn as nn
from typing import Optional

class SessionGRUModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_size: int = 128, num_layers: int = 1, padding_idx: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru_cells = nn.ModuleList([nn.GRUCell(emb_dim if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.head = nn.Linear(hidden_size, vocab_size)
    def forward_step(self, inputs: torch.LongTensor, hidden: torch.Tensor):
        emb = self.emb(inputs)
        h = emb
        new_hidden_layers = []
        for layer_idx in range(self.num_layers):
            cell = self.gru_cells[layer_idx]
            h_prev = hidden[layer_idx]
            h_new = cell(h, h_prev)
            new_hidden_layers.append(h_new.unsqueeze(0))
            h = h_new
        new_hidden = torch.cat(new_hidden_layers, dim=0)
        logits = self.head(h_new)
        return logits, new_hidden
