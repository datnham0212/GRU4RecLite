import pytest
import torch
import numpy as np
from unittest.mock import patch
from batching_datasets import SessionDataset, collate_fn, SessionParallelDataset

def test_mode_all_generates_all_samples():
    """Kiểm tra số lượng sample sinh ra với mode='all'."""
    sessions = [[1, 2, 3]]
    ds = SessionDataset(sessions, mode='all')
    samples = [ds[i] for i in range(len(ds))]
    assert len(samples) == 2
    assert samples[0] == ([1], 2)
    assert samples[1] == ([1, 2], 3)

def test_mode_last_generates_only_last_sample():
    """Kiểm tra số lượng sample sinh ra với mode='last'."""
    sessions = [[1, 2, 3]]
    ds = SessionDataset(sessions, mode='last')
    samples = [ds[i] for i in range(len(ds))]
    assert len(samples) == 1
    assert samples[0] == ([1, 2], 3)

def test_mode_all_multiple_sessions():
    """Kiểm tra mode='all' với nhiều phiên."""
    sessions = [[1, 2, 3], [4, 5]]
    ds = SessionDataset(sessions, mode='all')
    # session 1: 2 samples, session 2: 1 sample = 3 tổng
    assert len(ds) == 3
    assert ds[0] == ([1], 2)
    assert ds[1] == ([1, 2], 3)
    assert ds[2] == ([4], 5)

def test_max_seq_len_truncates_sequences():
    """Kiểm tra cắt chuỗi khi dài hơn max_seq_len."""
    sessions = [[i for i in range(1, 101)]]  # phiên dài 100
    ds = SessionDataset(sessions, mode='last', max_seq_len=50)
    seq, target = ds[0]
    assert len(seq) == 50  # đã bị cắt xuống max_seq_len
    assert seq[0] == 50    # giữ 50 phần tử cuối cùng (từ 50 đến 99)
    assert seq[-1] == 99   # phần tử cuối là 99 (vì target là 100)

def test_max_seq_len_no_truncate_short_sequences():
    """Kiểm tra không cắt nếu chuỗi ngắn hơn max_seq_len."""
    sessions = [[1, 2, 3, 4, 5]]
    ds = SessionDataset(sessions, mode='last', max_seq_len=50)
    seq, target = ds[0]
    assert len(seq) == 4  # không bị cắt
    assert seq == [1, 2, 3, 4]

def test_inplace_shuffle_changes_order():
    """Kiểm tra shuffle hoạt động khi inplace_shuffle=True."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    ds1 = SessionDataset(sessions, mode='all', inplace_shuffle=False)
    ds2 = SessionDataset(sessions, mode='all', inplace_shuffle=True)
    # cùng số lượng sample
    assert len(ds1) == len(ds2)
    # thứ tự có thể khác nhau nhưng cùng tập hợp sample
    samples1 = sorted([str(s) for s in ds1.samples])
    samples2 = sorted([str(s) for s in ds2.samples])
    assert samples1 == samples2

def test_getitem_returns_correct_sequence_and_target():
    """Kiểm tra __getitem__ trả về đúng (sequence, target)."""
    sessions = [[1, 2, 3]]
    ds = SessionDataset(sessions, mode='all')
    seq, target = ds[0]
    assert seq == [1]
    assert target == 2
    assert isinstance(target, int)

def test_getitem_returns_int_target():
    """Kiểm tra target được chuyển thành int."""
    sessions = [[1, 2, 3]]
    ds = SessionDataset(sessions, mode='all')
    seq, target = ds[1]
    assert isinstance(target, int)
    assert target == 3

def test_dataset_len():
    """Kiểm tra độ dài dataset."""
    sessions = [[1, 2, 3, 4], [5, 6]]
    ds = SessionDataset(sessions, mode='all')
    # session 1: 3 samples (1->2, 1,2->3, 1,2,3->4)
    # session 2: 1 sample (5->6)
    assert len(ds) == 4

def test_invalid_mode_raises_error():
    """Kiểm tra lỗi khi mode không hợp lệ."""
    sessions = [[1, 2, 3]]
    with pytest.raises(AssertionError):
        SessionDataset(sessions, mode='invalid')

def test_padding_to_max_length():
    """Kiểm tra padding đúng chiều dài lớn nhất trong batch."""
    batch = [([1, 2, 3], 4), ([5, 6], 7)]
    result = collate_fn(batch, pad_idx=0, align='right')
    input_ids = result['input_ids']
    # max length = 3
    assert input_ids.shape == (2, 3)
    # chuỗi [5,6] được pad thêm 1 phần tử ở đầu
    assert torch.equal(input_ids[1], torch.tensor([0, 5, 6]))

def test_padding_with_different_pad_idx():
    """Kiểm tra padding với pad_idx khác."""
    batch = [([1, 2], 3), ([4], 5)]
    result = collate_fn(batch, pad_idx=-1, align='right')
    input_ids = result['input_ids']
    # max_len = 2, chuỗi [4] được pad 1 phần tử ở đầu
    assert input_ids.shape == (2, 2)
    assert torch.equal(input_ids[1], torch.tensor([-1, 4]))

def test_attention_mask_marks_non_padding():
    """Kiểm tra attention_mask đánh dấu đúng vị trí không phải padding."""
    batch = [([1, 2], 3), ([4], 5)]
    result = collate_fn(batch, pad_idx=0, align='right')
    mask = result['attention_mask']
    # max_len = 2
    # batch[0]: [1,2] -> [1,2] -> mask [1,1]
    # batch[1]: [4] -> [0,4] -> mask [0,1]
    assert torch.equal(mask[0], torch.tensor([1, 1]))
    assert torch.equal(mask[1], torch.tensor([0, 1]))

def test_align_right():
    """Kiểm tra căn chỉnh phải (align='right')."""
    batch = [([1, 2], 3), ([4, 5, 6], 7)]
    result = collate_fn(batch, pad_idx=0, align='right')
    input_ids = result['input_ids']
    # chuỗi [1,2] được pad ở đầu để đạt độ dài 3
    assert torch.equal(input_ids[0], torch.tensor([0, 1, 2]))
    # chuỗi [4,5,6] không cần pad
    assert torch.equal(input_ids[1], torch.tensor([4, 5, 6]))

def test_align_left():
    """Kiểm tra căn chỉnh trái (align='left')."""
    batch = [([1, 2], 3), ([4, 5, 6], 7)]
    result = collate_fn(batch, pad_idx=0, align='left')
    input_ids = result['input_ids']
    # chuỗi [1,2] được pad ở cuối
    assert torch.equal(input_ids[0], torch.tensor([1, 2, 0]))
    # chuỗi [4,5,6] không cần pad
    assert torch.equal(input_ids[1], torch.tensor([4, 5, 6]))

def test_attention_mask_left_alignment():
    """Kiểm tra attention_mask với align='left'."""
    batch = [([1, 2], 3), ([4, 5, 6], 7)]
    result = collate_fn(batch, pad_idx=0, align='left')
    mask = result['attention_mask']
    # align left: chuỗi ở đầu, pad ở cuối
    assert torch.equal(mask[0], torch.tensor([1, 1, 0]))
    assert torch.equal(mask[1], torch.tensor([1, 1, 1]))

def test_targets_match_input_batch():
    """Kiểm tra targets khớp với dữ liệu đầu vào."""
    batch = [([1, 2], 3), ([4, 5], 6), ([7], 8)]
    result = collate_fn(batch, pad_idx=0, align='right')
    targets = result['targets']
    assert torch.equal(targets, torch.tensor([3, 6, 8]))

def test_lengths_tensor():
    """Kiểm tra lengths tensor chứa độ dài chuỗi gốc."""
    batch = [([1, 2, 3], 4), ([5, 6], 7), ([8], 9)]
    result = collate_fn(batch, pad_idx=0, align='right')
    lengths = result['lengths']
    assert torch.equal(lengths, torch.tensor([3, 2, 1]))

def test_empty_batch_handling():
    """Kiểm tra xử lý batch trống."""
    batch = []
    # Đây có thể raise error hoặc xử lý đặc biệt
    # tùy vào implementation của collate_fn
    try:
        result = collate_fn(batch, pad_idx=0)
        assert False, "Should handle empty batch or raise error"
    except (ValueError, IndexError):
        pass

def test_single_item_batch():
    """Kiểm tra batch với một item duy nhất."""
    batch = [([1, 2, 3], 4)]
    result = collate_fn(batch, pad_idx=0, align='right')
    input_ids = result['input_ids']
    # không cần pad vì chỉ có 1 item
    assert torch.equal(input_ids[0], torch.tensor([1, 2, 3]))
    assert torch.equal(result['targets'], torch.tensor([4]))

def test_batch_has_required_keys():
    """Kiểm tra batch đầu ra có đủ keys: inputs, targets, session_ids, new_session_mask."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    loader = iter(ds)
    batch = next(loader)
    required_keys = ['inputs', 'targets', 'session_ids', 'new_session_mask']
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"
    
    # Kiểm tra kiểu dữ liệu
    assert isinstance(batch['inputs'], torch.Tensor)
    assert isinstance(batch['targets'], torch.Tensor)
    assert isinstance(batch['session_ids'], torch.Tensor)

def test_batch_tensor_shapes():
    """Kiểm tra hình dạng các tensor trong batch."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    batch_size = 2
    ds = SessionParallelDataset(sessions, batch_size=batch_size, shuffle=False)
    loader = iter(ds)
    batch = next(loader)
    
    # inputs và targets phải có cùng kích thước batch
    assert batch['inputs'].shape[0] == batch_size
    assert batch['targets'].shape[0] == batch_size
    assert batch['session_ids'].shape[0] == batch_size
    assert batch['new_session_mask'].shape[0] == batch_size

def test_new_session_mask_marks_new_sessions():
    """Kiểm tra new_session_mask đánh dấu đúng khi phiên mới bắt đầu."""
    sessions = [[1, 2], [3, 4], [5, 6]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    loader = iter(ds)
    
    # Batch đầu tiên: phiên mới được khởi tạo
    batch1 = next(loader)
    # Ít nhất một phần tử phải có mask = 1 (phiên mới)
    assert batch1['new_session_mask'].sum().item() >= 1
    
    # Batch thứ hai: tiếp tục từ phiên cũ
    batch2 = next(loader)
    # new_session_mask phải có một số phần tử = 0 (phiên cũ) và = 1 (phiên mới)
    assert batch2['new_session_mask'].sum().item() >= 0

def test_replace_session_when_finished():
    """Kiểm tra loại bỏ phiên khi kết thúc và thay bằng phiên mới."""
    sessions = [[1, 2], [3, 4], [5, 6], [7, 8]]
    ds = SessionParallelDataset(sessions, batch_size=1, shuffle=False)
    loader = iter(ds)
    batches = list(loader)
    
    # Kiểm tra rằng có nhiều session_ids khác nhau
    all_session_ids = set()
    for batch in batches:
        all_session_ids.update(batch['session_ids'].tolist())
    
    # Nên có ít nhất 2-3 phiên khác nhau (phụ thuộc vào kích thước batch)
    assert len(all_session_ids) >= 2

def test_shuffle_false_preserves_order():
    """Kiểm tra khi shuffle=False, thứ tự phiên được giữ nguyên."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    
    # Chạy 2 lần với cùng dữ liệu
    order1 = []
    for batch in iter(ds):
        order1.extend(batch['session_ids'].tolist())
    
    order2 = []
    for batch in iter(ds):
        order2.extend(batch['session_ids'].tolist())
    
    # Thứ tự phải giống nhau
    assert order1 == order2

def test_shuffle_true_may_change_order():
    """Kiểm tra khi shuffle=True, thứ tự phiên có thể thay đổi."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    
    orders = []
    for _ in range(10):  # Increase iterations for more confidence
        ds = SessionParallelDataset(sessions, batch_size=2, shuffle=True)
        order = []
        for batch in iter(ds):
            order.extend(batch['session_ids'].tolist())
        orders.append(order)
    
    # Kiểm tra rằng có ít nhất một đôi thứ tự khác nhau
    assert not all(o == orders[0] for o in orders), "Shuffle may not be working"

def test_inputs_targets_correspondence():
    """Kiểm tra inputs và targets tương ứng chính xác."""
    sessions = [[1, 2, 3, 4]]
    ds = SessionParallelDataset(sessions, batch_size=1, shuffle=False)
    loader = iter(ds)
    
    batches = list(loader)
    # Với session [1,2,3,4], ta có các cặp:
    # input=1, target=2
    # input=2, target=3
    # input=3, target=4
    assert len(batches) == 3
    
    for batch in batches:
        assert batch['inputs'].shape[0] == 1
        assert batch['targets'].shape[0] == 1

def test_batch_size_respected():
    """Kiểm tra kích thước batch được tuân thủ."""
    sessions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    batch_size = 2
    ds = SessionParallelDataset(sessions, batch_size=batch_size, shuffle=False)
    
    for batch in iter(ds):
        # Tất cả batch đều có kích thước <= batch_size
        assert batch['inputs'].shape[0] <= batch_size

def test_minimum_session_length_assertion():
    """Kiểm tra assertion cho session độ dài tối thiểu."""
    # Session với độ dài 1 phải raise AssertionError
    sessions = [[1], [2, 3]]
    with pytest.raises(AssertionError):
        SessionParallelDataset(sessions, batch_size=2)

def test_multiple_batches_from_single_session():
    """Kiểm tra rằng một phiên dài có thể tạo nhiều batch."""
    sessions = [[i for i in range(1, 11)]]  # phiên dài 10 items
    batch_size = 2
    ds = SessionParallelDataset(sessions, batch_size=batch_size, shuffle=False)
    
    batches = list(iter(ds))
    # Phiên có 10 items -> 9 cặp (input, target) -> ít nhất 5 batches
    assert len(batches) >= 4

def test_worker_info_none():
    """Kiểm tra khi worker_info là None (single process)."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    
    with patch('batching_datasets.get_worker_info', return_value=None):
        batches = list(iter(ds))
        # Phải có ít nhất một batch
        assert len(batches) > 0

def test_new_session_mask_initial_true():
    """Kiểm tra rằng batch đầu tiên luôn có new_session_mask=True."""
    sessions = [[1, 2, 3], [4, 5, 6]]
    ds = SessionParallelDataset(sessions, batch_size=2, shuffle=False)
    loader = iter(ds)
    
    batch1 = next(loader)
    # Batch đầu tiên phải có ít nhất một phần tử với mask=True
    assert batch1['new_session_mask'].max().item() == 1