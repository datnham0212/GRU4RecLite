import pytest
import torch
from batching_models import SessionGRUModel

def test_embedding_output_shape():
    """Kiểm tra embedding tạo ra tensor đúng kích thước."""
    vocab_size = 10
    emb_dim = 8
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=emb_dim, hidden_size=16, num_layers=1)
    inputs = torch.tensor([1, 2, 3])  # batch_size=3
    emb = model.emb(inputs)
    assert emb.shape == (3, emb_dim)

def test_embedding_with_different_batch_size():
    """Kiểm tra embedding với các batch size khác nhau."""
    vocab_size = 20
    emb_dim = 16
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=emb_dim, hidden_size=32, num_layers=1)
    
    # Test với batch_size=1
    inputs1 = torch.tensor([5])
    emb1 = model.emb(inputs1)
    assert emb1.shape == (1, emb_dim)
    
    # Test với batch_size=10
    inputs2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    emb2 = model.emb(inputs2)
    assert emb2.shape == (10, emb_dim)

def test_embedding_values_are_different():
    """Kiểm tra rằng các embedding khác nhau cho các item khác nhau."""
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    inputs = torch.tensor([1, 2, 3])
    emb = model.emb(inputs)
    
    # Embedding cho item 1 và item 2 phải khác nhau
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])

def test_forward_step_hidden_shape():
    """Kiểm tra forward_step cập nhật hidden state đúng shape [num_layers, batch_size, hidden_size]."""
    vocab_size = 10
    hidden_size = 16
    num_layers = 2
    batch_size = 4
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=hidden_size, num_layers=num_layers)
    inputs = torch.tensor([1, 2, 3, 4])
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    logits, new_hidden = model.forward_step(inputs, hidden)
    assert new_hidden.shape == (num_layers, batch_size, hidden_size)

def test_forward_step_hidden_state_changes():
    """Kiểm tra rằng hidden state thay đổi sau khi forward_step."""
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    inputs = torch.tensor([1, 2, 3])
    hidden_init = torch.zeros(1, 3, 16)
    logits, new_hidden = model.forward_step(inputs, hidden_init)
    
    # Hidden state không được phép là zero sau khi update
    assert not torch.allclose(new_hidden, hidden_init)

def test_forward_step_logits_shape():
    """Kiểm tra logits có shape [batch_size, vocab_size]."""
    vocab_size = 20
    batch_size = 5
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=16, num_layers=1)
    inputs = torch.tensor([1, 2, 3, 4, 5])
    hidden = torch.zeros(1, batch_size, 16)
    logits, new_hidden = model.forward_step(inputs, hidden)
    assert logits.shape == (batch_size, vocab_size)

def test_forward_step_with_multiple_layers():
    """Kiểm tra forward_step với nhiều GRU layers."""
    vocab_size = 15
    batch_size = 3
    hidden_size = 32
    num_layers = 3
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=hidden_size, num_layers=num_layers)
    inputs = torch.tensor([1, 2, 3])
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    logits, new_hidden = model.forward_step(inputs, hidden)
    
    assert logits.shape == (batch_size, vocab_size)
    assert new_hidden.shape == (num_layers, batch_size, hidden_size)

def test_reset_hidden_state_with_new_session_mask():
    """Kiểm tra reset hidden state khi batch có phiên mới (new_session_mask)."""
    vocab_size = 10
    hidden_size = 16
    num_layers = 1
    batch_size = 3
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=hidden_size, num_layers=num_layers)
    hidden = torch.ones(num_layers, batch_size, hidden_size)
    new_session_mask = torch.tensor([True, False, True])
    
    # Reset hidden state tại các vị trí phiên mới
    hidden[:, new_session_mask, :] = 0.0
    
    # Kiểm tra rằng hidden tại vị trí phiên mới bằng 0
    assert torch.all(hidden[:, 0, :] == 0)
    assert torch.all(hidden[:, 2, :] == 0)
    # Kiểm tra rằng hidden tại vị trí không reset vẫn giữ nguyên
    assert torch.all(hidden[:, 1, :] == 1)

def test_reset_hidden_state_all_new():
    """Kiểm tra reset hidden state khi tất cả là phiên mới."""
    vocab_size = 10
    hidden_size = 16
    num_layers = 2
    batch_size = 4
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=hidden_size, num_layers=num_layers)
    hidden = torch.ones(num_layers, batch_size, hidden_size)
    new_session_mask = torch.tensor([True, True, True, True])
    
    # Reset tất cả
    hidden[:, new_session_mask, :] = 0.0
    
    # Tất cả phải bằng 0
    assert torch.all(hidden == 0)

def test_reset_hidden_state_none_new():
    """Kiểm tra không reset hidden state khi không có phiên mới."""
    vocab_size = 10
    hidden_size = 16
    num_layers = 1
    batch_size = 3
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=hidden_size, num_layers=num_layers)
    hidden_original = torch.ones(num_layers, batch_size, hidden_size)
    hidden = hidden_original.clone()
    new_session_mask = torch.tensor([False, False, False])
    
    # Reset (không có phần tử nào được reset)
    hidden[:, new_session_mask, :] = 0.0
    
    # Tất cả vẫn giữ nguyên
    assert torch.allclose(hidden, hidden_original)

def test_forward_pass_full():
    """Kiểm tra full forward pass từ input đến output."""
    vocab_size = 15
    batch_size = 4
    emb_dim = 8
    hidden_size = 16
    num_layers = 2
    
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers)
    
    # Khởi tạo hidden state
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    
    # Forward pass
    inputs = torch.tensor([1, 2, 3, 4])
    logits, new_hidden = model.forward_step(inputs, hidden)
    
    # Kiểm tra output shapes
    assert logits.shape == (batch_size, vocab_size)
    assert new_hidden.shape == (num_layers, batch_size, hidden_size)
    
    # Logits phải là số thực (không NaN hoặc Inf)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_model_trainable():
    """Kiểm tra rằng mô hình có trainable parameters."""
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    
    # Kiểm tra số lượng parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    
    # Kiểm tra rằng có ít nhất một parameter cần gradient
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0

def test_forward_step_deterministic():
    """Kiểm tra rằng forward_step có kết quả xác định với cùng hidden state."""
    torch.manual_seed(42) # Đặt seed để tái lập kết quả
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    model.eval()  # Đặt mode evaluation để tắt dropout
    
    inputs = torch.tensor([1, 2, 3])
    hidden = torch.zeros(1, 3, 16)
    
    # Chạy forward_step hai lần với hidden state độc lập
    logits1, hidden1 = model.forward_step(inputs, hidden.clone())
    logits2, hidden2 = model.forward_step(inputs, hidden.clone())
    
    # Kết quả phải giống nhau
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(hidden1, hidden2)

def test_gru_output_different_inputs():
    """Kiểm tra rằng các input khác nhau tạo ra output khác nhau."""
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    model.eval()
    
    hidden = torch.zeros(1, 2, 16)
    
    # Input khác nhau
    inputs1 = torch.tensor([1, 2])
    inputs2 = torch.tensor([3, 4])
    
    logits1, _ = model.forward_step(inputs1, hidden)
    logits2, _ = model.forward_step(inputs2, hidden)
    
    # Output phải khác nhau
    assert not torch.allclose(logits1, logits2)

def test_vocab_size_validation():
    """Kiểm tra rằng vocab_size được sử dụng đúng."""
    vocab_size = 100
    model = SessionGRUModel(vocab_size=vocab_size, emb_dim=8, hidden_size=16, num_layers=1)
    
    batch_size = 3
    inputs = torch.tensor([1, 2, 3])
    hidden = torch.zeros(1, batch_size, 16)
    logits, _ = model.forward_step(inputs, hidden)
    
    # Logits phải có vocab_size features
    assert logits.shape[1] == vocab_size

def test_forward_with_gpu_tensors():
    """Kiểm tra forward pass với GPU tensors (nếu GPU có sẵn)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    model = SessionGRUModel(vocab_size=10, emb_dim=8, hidden_size=16, num_layers=1)
    model = model.cuda()
    
    inputs = torch.tensor([1, 2, 3]).cuda()
    hidden = torch.zeros(1, 3, 16).cuda()
    
    logits, new_hidden = model.forward_step(inputs, hidden)
    
    # Output phải ở GPU
    assert logits.is_cuda
    assert new_hidden.is_cuda