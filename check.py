# Run this in your Python environment
import torch
from gru4rec_pytorch import GRU4Rec

model = GRU4Rec(layers=[100], device='cuda:0')

# Check these methods exist:
print(hasattr(model, 'forward_step'))      # Should be True
print(hasattr(model, 'loss_function'))     # Should be True
print(hasattr(model, 'optimizer'))         # Should be True
print(hasattr(model, 'savemodel'))         # Should be True
print(hasattr(model, 'layers'))            # Should be True