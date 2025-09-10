import os, sys, numpy as np, pandas as pd
from pathlib import Path

# ensure local package import
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.gru4rec_torch.gru4rec_pytorch import GRU4Rec

def extract_and_cluster(ckpt_path, out_csv, n_clusters=50, random_state=42):
    gru = GRU4Rec.loadmodel(str(ckpt_path), device="cpu")
    gru.model.eval()

    # Use output weight matrix as item vectors (rows indexed by internal item idx)
    Wy = gru.model.Wy.weight.detach().cpu().numpy()  # shape: (num_items, hidden)
    # Build idx -> item_id mapping
    itemidmap = gru.data_iterator.itemidmap  # pandas Series index=item_id, value=internal_idx
    idx_to_item = {int(v): str(k) for k, v in itemidmap.items()}

    # Ensure rows correspond to indices 0..N-1
    N = Wy.shape[0]
    item_ids = [idx_to_item.get(i, "") for i in range(N)]

    # Run k-means
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn required: pip install scikit-learn") from e

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Wy)

    # Save CSV: item_id, idx, cluster
    df = pd.DataFrame({"item_id": item_ids, "idx": list(range(N)), "cluster": labels})
    df.to_csv(out_csv, index=False)
    print(f"Saved clusters to: {out_csv} (n_items={N}, n_clusters={n_clusters})")
    return df

if __name__ == "__main__":
    CKPT = BASE_DIR / "output_data" / "save_model_test.pt"
    OUT = BASE_DIR / "output_data" / "yoochoose_item_clusters.csv"
    extract_and_cluster(CKPT, OUT, n_clusters=50)
