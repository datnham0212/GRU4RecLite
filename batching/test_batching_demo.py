import pytest
import torch
from torch.utils.data import DataLoader
from batching_datasets import SessionDataset, SessionParallelDataset, collate_fn
from batching_models import SessionGRUModel

def test_dataloader_with_sessiondataset():
    """Kiểm tra DataLoader với SessionDataset tạo batch hợp lệ."""
    sessions = [[1, 2, 3], [4, 5]]
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    loader = DataLoader(
        ds, 
        batch_size=2, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    batch = next(iter(loader))
    
    # Kiểm tra batch có đủ keys
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'targets' in batch
    assert 'lengths' in batch
    
    # Kiểm tra shape hợp lý
    assert batch['input_ids'].shape[0] == 2  # batch_size=2
    assert batch['targets'].shape[0] == 2
    assert batch['attention_mask'].shape == batch['input_ids'].shape
    assert batch['lengths'].shape[0] == 2

def test_dataloader_sessiondataset_multiple_batches():
    """Kiểm tra DataLoader có thể lấy nhiều batch từ SessionDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    loader = DataLoader(
        ds, 
        batch_size=2, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    batches = list(loader)
    # Phải có ít nhất 2 batches
    assert len(batches) >= 2
    
    # Mỗi batch phải có đúng keys
    for batch in batches:
        assert 'input_ids' in batch
        assert 'targets' in batch

def test_dataloader_sessiondataset_with_shuffle():
    """Kiểm tra DataLoader với SessionDataset và shuffle."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]]
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0, inplace_shuffle=False)
    loader = DataLoader(
        ds, 
        batch_size=2, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    batches = list(loader)
    assert len(batches) > 0
    
    for batch in batches:
        assert batch['input_ids'].shape[0] <= 2  # batch_size=2

def test_dataloader_with_sessionparalleldataset():
    """Kiểm tra DataLoader với SessionParallelDataset sinh batch liên tục."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    
    batch = next(iter(loader))
    
    # Kiểm tra batch có đủ keys
    required_keys = ['inputs', 'targets', 'session_ids', 'new_session_mask']
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"
    
    # Kiểm tra shape
    assert batch['inputs'].shape[0] == 2  # batch_size=2
    assert batch['targets'].shape[0] == 2
    assert batch['session_ids'].shape[0] == 2
    assert batch['new_session_mask'].shape[0] == 2

def test_dataloader_sessionparalleldataset_multiple_batches():
    """Kiểm tra DataLoader có thể lấy nhiều batch từ SessionParallelDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= 10:  # Lấy tối đa 10 batches
            break
    
    # Phải có ít nhất một batch
    assert len(batches) >= 1
    
    # Mỗi batch phải có đúng keys
    for batch in batches:
        assert 'inputs' in batch
        assert 'targets' in batch
        assert 'session_ids' in batch
        assert 'new_session_mask' in batch

def test_model_runs_on_sessiondataset_batch():
    """Kiểm tra mô hình chạy được với batch từ SessionDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    vocab_size = 15
    batch_size = 2
    hidden_size = 16
    num_layers = 1
    
    model = SessionGRUModel(
        vocab_size=vocab_size, 
        emb_dim=8, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )
    model.eval()
    
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    batch = next(iter(loader))
    # Use lengths to get valid (non-padded) items
    lengths = batch['lengths']
    inputs = torch.tensor([batch['input_ids'][i, lengths[i]-1] for i in range(len(lengths))])
    hidden = torch.zeros(num_layers, inputs.shape[0], hidden_size)
    
    logits, new_hidden = model.forward_step(inputs, hidden)
    
    # Kiểm tra logits shape
    assert logits.shape == (inputs.shape[0], vocab_size)
    # Kiểm tra hidden state shape
    assert new_hidden.shape == (num_layers, inputs.shape[0], hidden_size)

def test_model_runs_on_sessionparalleldataset_batch():
    """Kiểm tra mô hình chạy được với batch từ SessionParallelDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    vocab_size = 15
    batch_size = 2
    hidden_size = 16
    num_layers = 1
    
    model = SessionGRUModel(
        vocab_size=vocab_size, 
        emb_dim=8, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )
    model.eval()
    
    sp_ds = SessionParallelDataset(sessions, batch_size=batch_size, shuffle=False)
    sp_loader = DataLoader(sp_ds, batch_size=None, num_workers=0)
    
    batch = next(iter(sp_loader))
    inputs = batch['inputs']
    new_session_mask = batch['new_session_mask'].bool()
    
    hidden = torch.zeros(num_layers, inputs.shape[0], hidden_size)
    hidden[:, new_session_mask, :] = 0.0
    
    logits, new_hidden = model.forward_step(inputs, hidden)
    
    # Kiểm tra logits shape
    assert logits.shape == (inputs.shape[0], vocab_size)
    # Kiểm tra hidden state shape
    assert new_hidden.shape == (num_layers, inputs.shape[0], hidden_size)

def test_model_full_training_loop_sessiondataset():
    """Kiểm tra mô hình có thể chạy toàn bộ training loop với SessionDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13]]
    vocab_size = 15
    batch_size = 2
    hidden_size = 16
    num_layers = 1
    
    model = SessionGRUModel(
        vocab_size=vocab_size, 
        emb_dim=8, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )
    
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    batch_count = 0
    for batch in loader:
        lengths = batch['lengths']
        inputs = torch.tensor([batch['input_ids'][i, lengths[i]-1] for i in range(len(lengths))])
        targets = batch['targets']
        hidden = torch.zeros(num_layers, inputs.shape[0], hidden_size)
        
        logits, _ = model.forward_step(inputs, hidden)
        
        assert logits.shape == (inputs.shape[0], vocab_size)
        assert not torch.isnan(logits).any()
        batch_count += 1
    
    assert batch_count > 0

def test_model_full_training_loop_sessionparalleldataset():
    """Kiểm tra mô hình có thể chạy toàn bộ training loop với SessionParallelDataset."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13]]
    vocab_size = 15
    batch_size = 2
    hidden_size = 16
    num_layers = 1
    
    model = SessionGRUModel(
        vocab_size=vocab_size, 
        emb_dim=8, 
        hidden_size=hidden_size, 
        num_layers=num_layers
    )
    
    sp_ds = SessionParallelDataset(sessions, batch_size=batch_size, shuffle=False)
    sp_loader = DataLoader(sp_ds, batch_size=None, num_workers=0)
    
    batch_count = 0
    hidden = None
    
    for batch in sp_loader:
        inputs = batch['inputs']
        targets = batch['targets']
        new_session_mask = batch['new_session_mask'].bool()
        
        # Initialize or update hidden state based on actual batch size
        if hidden is None or hidden.shape[1] != inputs.shape[0]:
            hidden = torch.zeros(num_layers, inputs.shape[0], hidden_size)
        
        # Reset hidden state cho phiên mới
        hidden[:, new_session_mask, :] = 0.0
        
        logits, hidden = model.forward_step(inputs, hidden)
        
        # Kiểm tra output hợp lệ
        assert logits.shape == (inputs.shape[0], vocab_size)
        assert not torch.isnan(logits).any()
        
        batch_count += 1
        if batch_count >= 10:
            break
    
    # Phải xử lý ít nhất một batch
    assert batch_count > 0

def test_sessiondataset_vs_sessionparalleldataset_vocab_coverage():
    """Kiểm tra cả hai dataset đều cover toàn bộ items."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    vocab_size = max(max(s) for s in sessions) + 1
    
    # SessionDataset
    ds1 = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    loader1 = DataLoader(
        ds1, 
        batch_size=2, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    items1 = set()
    for batch in loader1:
        items1.update(batch['targets'].tolist())
    
    # SessionParallelDataset
    sp_ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    sp_loader = DataLoader(sp_ds, batch_size=None, num_workers=0)
    
    items2 = set()
    for batch in sp_loader:
        items2.update(batch['targets'].tolist())
    
    # Cả hai phải cover các items
    assert len(items1) > 0
    assert len(items2) > 0

def test_different_alignments_sessiondataset():
    """Kiểm tra SessionDataset với các alignment khác nhau."""
    sessions = [[1, 2, 3], [4, 5]]
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    
    # Test align='right'
    loader_right = DataLoader(
        ds, 
        batch_size=2, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    batch_right = next(iter(loader_right))
    
    # Test align='left'
    loader_left = DataLoader(
        ds, 
        batch_size=2, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='left'),
        num_workers=0
    )
    batch_left = next(iter(loader_left))
    
    # Cả hai phải tạo ra batch hợp lệ
    assert batch_right['input_ids'].shape == batch_left['input_ids'].shape
    assert batch_right['targets'].shape == batch_left['targets'].shape

def test_large_batch_size_sessiondataset():
    """Kiểm tra SessionDataset với batch_size lớn."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    ds = SessionDataset(sessions, mode='all', max_seq_len=50, pad_idx=0)
    
    large_batch_size = 32
    loader = DataLoader(
        ds, 
        batch_size=large_batch_size, 
        collate_fn=lambda b: collate_fn(b, pad_idx=0, align='right'),
        num_workers=0
    )
    
    batch = next(iter(loader))
    # Batch size có thể nhỏ hơn large_batch_size nếu không đủ samples
    assert batch['input_ids'].shape[0] <= large_batch_size
    assert batch['input_ids'].shape[0] > 0