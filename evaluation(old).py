from gru4rec_pytorch import SessionDataIterator
import torch
import numpy as np

# Đo lường số lượng mục duy nhất xuất hiện trong tất cả các đề xuất Top-K.
def item_coverage(topk_preds, item_catalog):
    """
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_catalog: set or list of tất cả các ItemId trong tập dữ liệu

    Trả về: float ∈ [0, 1]
    """
    recommended_items = set(topk_preds.flatten().tolist())
    return len(recommended_items) / len(item_catalog)

# Đo lường số lượng phiên có ít nhất một mục thuộc danh mục đầy đủ.
def catalog_coverage(topk_preds, item_catalog):
    """
    topk_preds: np.ndarray of shape (num_sessions, K)
    item_catalog: set of tất cả các ItemId trong tập dữ liệu

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


# Ghi chú liên quan lỗi trước đó:
# Trong quá trình đánh giá, mã cũ giả định gru.model.embedding là nn.Embedding và truy cập .weight.
# Với một số biến thể mô hình, gru.model.embedding có thể không tồn tại hoặc chỉ là một số (int),
# dẫn đến AttributeError: 'int' object has no attribute 'weight'.
# Helper dưới đây tìm cách lấy vector biểu diễn mục (item embeddings) một cách an toàn:
# 1) Ưu tiên dùng embedding đầu vào nếu có (gru.model.embedding.weight);
# 2) Nếu không có, fallback sang các lớp đầu ra phổ biến (decoder/linear) có trọng số tương ứng với vector mục;
# 3) Nếu vẫn không tìm thấy, trả về None để bỏ qua việc tính ILD thay vì gây lỗi.
def _get_item_embeddings(gru):
    # Ưu tiên embedding đầu vào nếu có
    emb = getattr(gru.model, 'embedding', None)
    if hasattr(emb, 'weight'):
        return emb.weight.detach().cpu().numpy()
    # Thử các lớp đầu ra phổ biến để suy ra embedding từ trọng số
    for name in ['output', 'out', 'linear', 'fc', 'decoder']:
        layer = getattr(gru.model, name, None)
        if hasattr(layer, 'weight'):
            # nn.Linear(out_features=n_items, in_features=hidden) -> weight: (n_items, hidden)
            return layer.weight.detach().cpu().numpy()
    return None

@torch.no_grad()  # Vô hiệu hóa tính toán gradient cho quá trình đánh giá
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    if gru.error_during_train: 
        raise Exception('Đang cố gắng đánh giá một mô hình không được huấn luyện đúng cách (error_during_train=True)')

    # Lấy embedding của các mục từ mô hình GRU (nếu có) và chuẩn hóa
    item_embeddings = _get_item_embeddings(gru)
    if item_embeddings is not None:
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_embeddings = item_embeddings / norms

    # Khởi tạo các dictionary để lưu trữ các chỉ số recall và MRR cho từng giá trị cutoff
    recall = {c: 0 for c in cutoff}
    mrr = {c: 0 for c in cutoff}

    # Khởi tạo trạng thái ẩn cho từng lớp GRU
    H = [torch.zeros((batch_size, gru.layers[i]), device=gru.device) for i in range(len(gru.layers))]

    # Hook để reset trạng thái ẩn khi phiên kết thúc
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

    # Tạo bộ lặp dữ liệu kiểm tra
    data_iterator = SessionDataIterator(
        test_data, batch_size, 0, 0, 0, item_key, session_key, time_key,
        device=gru.device, itemidmap=gru.data_iterator.itemidmap
    )

    all_topk_preds = []
    item_catalog = set(gru.data_iterator.itemidmap.values)
    n = 0  # Bộ đếm số lượng mẫu đã xử lý

    # Lặp qua từng batch dữ liệu
    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H:
            h.detach_()

        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])

        # Lấy top-K dự đoán cho mỗi phiên
        for c in cutoff:
            topk = torch.topk(oscores, k=c, dim=0).indices.cpu().numpy().T
            all_topk_preds.append(topk)

        # Tính thứ hạng dựa trên chế độ đánh giá
        if mode == 'standard':
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative':
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':
            ranks = (oscores > tscores).sum(dim=0) + 0.5 * ((oscores == tscores).sum(dim=0) - 1) + 1
        else:
            raise NotImplementedError

        # Cập nhật recall và MRR
        for c in cutoff:
            recall[c] += (ranks <= c).sum().cpu().numpy()
            mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu.numpy()

        n += O.shape[0]

    # Chuẩn hóa các chỉ số
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n

    # Tính coverage và ILD cho giá trị cutoff lớn nhất
    max_cutoff = max(cutoff)
    topk_preds = np.concatenate([batch for batch in all_topk_preds if batch.shape[1] == max_cutoff], axis=0)
    item_cov = item_coverage(topk_preds, item_catalog)
    catalog_cov = catalog_coverage(topk_preds, item_catalog)
    ild = intra_list_diversity(topk_preds, item_embeddings) if item_embeddings is not None else float('nan')

    return recall, mrr, item_cov, catalog_cov, ild
