from batching_utils import load_sessions_from_dat
from batching_datasets import SessionDataset, SessionParallelDataset, collate_fn
from batching_models import SessionGRUModel
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Đọc 2 file sẽ rất lâu, nên chia ra đọc từng file một
    # Đọc phiên từ file train của Yoochoose
    yoochoose_path = 'd:/output_data/yoochoose_train_tr.dat'
    try:
        yoochoose_sessions = load_sessions_from_dat(yoochoose_path, item_idx_col='item_idx', min_session_length=2)
        print(f'Số phiên Yoochoose train: {len(yoochoose_sessions)}')
    except Exception as e:
        print('Không đọc được file Yoochoose:', e)
        yoochoose_sessions = [[1,5,8,3],[2,4,6],[3,8,9,10,2],[2,7,2,9]]

    # Chọn dữ liệu để chạy demo (ở đây dùng Yoochoose, có thể đổi sang retailrocket_sessions)
    sessions = yoochoose_sessions

    # --- Prefix-based approach ---
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0, inplace_shuffle=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_idx=0, align='right'), num_workers=0)
    for batch in loader:
        print('Prefix batch shapes:', batch['input_ids'].shape, batch['attention_mask'].shape, batch['targets'].shape)
        break

    # --- Session-parallel approach ---
    sp_ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    sp_loader = DataLoader(sp_ds, batch_size=None, num_workers=0)
    vocab_size = max(max(s) for s in sessions) + 1
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=64, hidden_size=64, num_layers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    hidden = torch.zeros(model.num_layers, 2, model.hidden_size)
    for i, batch in enumerate(sp_loader):
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        new_mask = batch['new_session_mask'].to(device).bool()
        hidden[:, new_mask, :] = 0.0
        logits, hidden = model.forward_step(inputs.to(device), hidden.to(device))
        print('Session-parallel batch', i, 'inputs', inputs.shape, 'targets', targets.shape, 'logits', logits.shape)
        if i >= 5:
            break
    print('Done example run')
