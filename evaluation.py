from gru4rec_pytorch import SessionDataIterator
import torch
import numpy as np

# Đo lường số lượng mục duy nhất xuất hiện trong tất cả các đề xuất Top-K.
def item_coverage(topk_preds, item_catalog):
    """
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_catalog: set or list of tất cả các item_id trong tập dữ liệu

    Trả về: float ∈ [0, 1]
    """
    recommended_items = set(topk_preds.flatten().tolist())
    return len(recommended_items) / len(item_catalog)

# Đo lường số lượng phiên có ít nhất một mục thuộc danh mục đầy đủ.
def catalog_coverage(topk_preds, item_catalog):
    """
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_catalog: set of tất cả các item_id trong tập dữ liệu

    Trả về: float ∈ [0, 1]
    """
    count = 0
    for session_items in topk_preds:
        if any(item in item_catalog for item in session_items):
            count += 1
    return count / len(topk_preds)

# Tính độ đa dạng trong danh sách đề xuất bằng cách sử dụng độ tương phản cosine giữa các embedding.
def intra_list_diversity(topk_preds, item_embeddings):
    """
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_embeddings: np.ndarray of shape (num_items, D), đã chuẩn hóa

    Trả về: float ∈ [0, 1]
    """
    ild_scores = []
    for session in topk_preds:
        emb = item_embeddings[session]  # (K, D)
        sim_matrix = emb @ emb.T        # ma trận độ tương đồng cosine
        K = len(session)
        diversity = 1 - (sim_matrix.sum() - K) / (K * (K - 1))  # loại bỏ đường chéo
        ild_scores.append(diversity)
    return np.mean(ild_scores)


# Đo lường độ đa dạng tổng hợp: số lượng mục duy nhất được đề xuất trên tất cả người dùng.
def aggregate_diversity(topk_preds, item_catalog):
    """
    Computes aggregate diversity (coverage of the catalog by recommended items)
    in a fail-safe manner.
    
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_catalog: set of all item_ids in the dataset

    Returns:
        float ∈ [0, 1] : fraction of catalog covered by top-K predictions
    """
    if topk_preds.size == 0:
        return 0.0  # No predictions at all
    if not item_catalog:
        return 0.0  # Empty catalog, avoid division by zero

    # Flatten predictions and filter only items in the catalog
    unique_items = set(topk_preds.flatten().tolist()) & item_catalog

    if not unique_items:
        return 0.0  # No items matched catalog

    return len(unique_items) / len(item_catalog)



# Đo lường độ đa dạng giữa người dùng: mức độ khác biệt giữa các danh sách đề xuất.
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import numpy as np

def inter_user_diversity_lsh(topk_preds, num_perm=128, lsh_threshold=0.0, sample_size=10000):
    """
    Approximate inter-user diversity using MinHash + LSH, with optional sampling for speed.
    
    topk_preds: np.ndarray of shape (num_sessions, K)
    num_perm: number of permutations for MinHash signature (higher -> more accurate)
    lsh_threshold: similarity threshold for LSH (0-1), controls candidate pairs
    sample_size: maximum number of sessions to use (randomly sampled if n_sessions > sample_size)

    Returns:
        float ∈ [0, 1] : approximate average inter-user diversity
    """
    n_sessions = topk_preds.shape[0]
    if n_sessions < 2:
        return 0.0

    # --- Step 0: Optional sampling ---
    if n_sessions > sample_size:
        idx = np.random.choice(n_sessions, size=sample_size, replace=False)
        topk_preds = topk_preds[idx]
        n_sessions = sample_size

    # --- Step 1: Create MinHash signatures for all sessions ---
    minhashes = []
    for items in tqdm(topk_preds, desc="Building MinHash signatures"):
        m = MinHash(num_perm=num_perm)
        for item in items:
            m.update(str(item).encode('utf8'))
        minhashes.append(m)

    # --- Step 2: Build LSH index ---
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(f"user_{i}", m)

    # --- Step 3: Collect candidate similar pairs and compute approximate Jaccard distances ---
    distances = []
    seen_pairs = set()
    for i, m in enumerate(tqdm(minhashes, desc="Querying LSH for candidate pairs")):
        sims = lsh.query(m)
        for sim_user in sims:
            j = int(sim_user.split("_")[1])
            if i < j and (i, j) not in seen_pairs:
                seen_pairs.add((i, j))
                approx_jaccard = m.jaccard(minhashes[j])
                distances.append(1 - approx_jaccard)  # Jaccard distance

    return float(np.mean(distances)) if distances else 0.0




def _get_item_embeddings(gru):
    """
    Trả về ma trận embedding cho mục (shape: [n_items, dim]).
    Ưu tiên:
    - Nếu constrained_embedding=True: dùng Wy.weight (ma trận đầu ra) như embedding mục.
    - Nếu có lớp input embedding riêng (gru.model.embedding hoặc gru.model.E): dùng .weight.
    - Nếu không, fallback sang các lớp đầu ra phổ biến (Wy/decoder/linear/...).
    - Nếu không tìm thấy, trả về None để bỏ qua ILD an toàn.
    """
    model = getattr(gru, 'model', None)
    if model is None:
        return None

    # 1) Trường hợp embedding bị ràng buộc: dùng trực tiếp ma trận đầu ra Wy làm embedding mục
    if getattr(model, 'constrained_embedding', False):
        Wy = getattr(model, 'Wy', None)
        if hasattr(Wy, 'weight'):
            return Wy.weight.detach().cpu().numpy()

    # 2) Embedding đầu vào riêng (tùy cấu hình): thường là `E` hoặc `embedding`
    for name in ['embedding', 'E']:
        emb = getattr(model, name, None)
        if hasattr(emb, 'weight'):
            return emb.weight.detach().cpu().numpy()

    # 3) Fallback các lớp đầu ra phổ biến (bao gồm Wy)
    for name in ['Wy', 'output', 'out', 'linear', 'fc', 'decoder']:
        layer = getattr(model, name, None)
        if hasattr(layer, 'weight'):
            return layer.weight.detach().cpu().numpy()

    return None

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='item_id', session_key='session_id', time_key='time', eval_metrics=('recall_mrr', 'coverage', 'ild')):
    if gru.error_during_train: 
        raise Exception('Đang cố gắng đánh giá một mô hình không được huấn luyện đúng cách (error_during_train=True)')

    # Filter test data to only include items in training vocabulary
    valid_items = set(gru.data_iterator.itemidmap.index)
    test_data_filtered = test_data[test_data[item_key].isin(valid_items)].copy()
    if test_data_filtered.empty:
        raise ValueError("No valid items in test data after filtering by training vocabulary")
    
    print(f"Original test data: {len(test_data)} events")
    print(f"Filtered test data: {len(test_data_filtered)} events (removed {len(test_data) - len(test_data_filtered)} unknown items)")
    train_vocab = set(gru.data_iterator.itemidmap.index)
    print(f"Training vocabulary size: {len(train_vocab)}")
    print(f"Test data unique items: {test_data[item_key].nunique()}")
    print(f"Items in both: {len(set(test_data[item_key]) & set(train_vocab))}")

    item_embeddings = _get_item_embeddings(gru)
    if item_embeddings is not None:
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_embeddings = item_embeddings / norms

    recall = {c: 0 for c in cutoff}
    mrr = {c: 0 for c in cutoff}
    H = [torch.zeros((batch_size, gru.layers[i]), device=gru.device) for i in range(len(gru.layers))]
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    data_iterator = SessionDataIterator(
        test_data_filtered, batch_size, 0, 0, 0, item_key, session_key, time_key,
        device=gru.device, itemidmap=gru.data_iterator.itemidmap
    )
    all_topk_preds = []
    item_catalog = set(gru.data_iterator.itemidmap.values)
    n = 0
    max_cutoff = max(cutoff)

    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H:
            h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])
        topk = torch.topk(oscores, k=max_cutoff, dim=0).indices.cpu().numpy().T
        all_topk_preds.append(topk)

        if 'recall_mrr' in eval_metrics:
            if mode == 'standard':
                ranks = (oscores > tscores).sum(dim=0) + 1
            elif mode == 'conservative':
                ranks = (oscores >= tscores).sum(dim=0)
            elif mode == 'median':
                ranks = (oscores > tscores).sum(dim=0) + 0.5 * ((oscores == tscores).sum(dim=0) - 1) + 1
            else:
                raise NotImplementedError
            for c in cutoff:
                recall[c] += (ranks <= c).sum().cpu().numpy()
                mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
        n += O.shape[0]

    results = {}
    if 'recall_mrr' in eval_metrics:
        for c in cutoff:
            recall[c] /= n
            mrr[c] /= n
        results['recall'] = recall
        results['mrr'] = mrr

    topk_preds = np.concatenate(all_topk_preds, axis=0)
    if 'coverage' in eval_metrics:
        results['item_coverage'] = item_coverage(topk_preds, item_catalog)
        results['catalog_coverage'] = catalog_coverage(topk_preds, item_catalog)
    if 'ild' in eval_metrics:
        results['ild'] = intra_list_diversity(topk_preds, item_embeddings) if item_embeddings is not None else float('nan')
    if 'diversity' in eval_metrics:
        results['aggregate_diversity'] = aggregate_diversity(topk_preds, item_catalog)
        results['inter_user_diversity'] = inter_user_diversity_lsh(topk_preds)

    return results
