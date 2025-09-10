import os, sys
import torch
import pandas as pd

# ensure both the gru4rec_torch folder and its parent are on sys.path
GRU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))           # .../gru4rec_torch
PARENT_DIR = os.path.abspath(os.path.join(GRU_DIR, '..'))                         # .../model (or project parent)
for p in (GRU_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try to import the model module from the local gru4rec_torch package
try:
    # prefer direct import from the local folder
    from gru4rec_pytorch import GRU4Rec
except Exception as e:
    # fallback to package-style import if available on sys.path
    try:
        from model.gru4rec_torch.gru4rec_pytorch import GRU4Rec
    except Exception as e2:
        print("Failed to import GRU4Rec module (tried local and package imports).")
        print("Local import error:", e)
        print("Package import error:", e2)
        raise

CKPT_PATH = os.path.join(GRU_DIR, "output_data", "save_model_test.pt")
if not os.path.isfile(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

# use map_location to avoid GPU issues
gru = GRU4Rec.loadmodel(CKPT_PATH, device="cpu")

print("=== Option 0: Basic model summary ===")
print("GRU4Rec object loaded.")
print("Hidden layer sizes:", getattr(gru.model, "layers", None))
print("Constrained embedding:", getattr(gru.model, "constrained_embedding", None))
print("Embedding mode:", getattr(gru.model, "embedding", None))

print("\nParameter summary:")
total = 0
for name, p in gru.model.named_parameters():
    n = p.numel()
    total += n
    print(f"{name:30s} shape={tuple(p.shape)} params={n}")
print("Total params:", total)

# Option A: ItemId -> internal index mapping
print("\n=== Option A: ItemId -> internal index (first 5) ===")
mapping = getattr(gru.data_iterator, "itemidmap", None)
if mapping is None:
    print("No itemidmap found on data_iterator")
else:
    print(mapping.head())

# Also show reverse mapping first few
if mapping is not None:
    rev = {v: k for k, v in mapping.items()}
    print("\nFirst 5 internal indices -> ItemId:")
    for i in range(min(5, len(rev))):
        print(i, "->", rev[i])

# Option B: Raw torch.load of entire checkpoint object
print("\n=== Option B: Raw torch.load object inspection ===")
raw_obj = torch.load(CKPT_PATH, map_location="cpu")
print("Raw type:", type(raw_obj))
print("Some attributes:", [k for k in dir(raw_obj) if not k.startswith('_')][:40])

# Option C: List state_dict parameter tensors
print("\n=== Option C: state_dict parameter shapes ===")
sd = gru.model.state_dict()
for k, v in sd.items():
    print(f"{k:30s} {tuple(v.shape)}")

# Option D: External item mapping CSV (if needed separately)
print("\n=== Option D: Item mapping CSV ===")
csv_path = os.path.join(GRU_DIR, "output_data", "af_item_mapping.csv")
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
else:
    print("Mapping CSV not found:", csv_path)
