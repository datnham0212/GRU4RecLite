import torch
import torch.nn as nn
import torch.optim as optim
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should be False
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
