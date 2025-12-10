import pytest
import pandas as pd
from pathlib import Path
from batching_utils import load_sessions_from_dat

def test_read_valid_file(tmp_path):
    """Kiểm tra đọc file hợp lệ với đủ cột."""
    data = """session_id,timestamp,item_idx
s1,2021-01-01 10:00:00,1
s1,2021-01-01 10:05:00,2
s2,2021-01-02 09:00:00,3"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    sessions = load_sessions_from_dat(str(file))
    assert isinstance(sessions, list)
    assert all(isinstance(s, list) for s in sessions)
    assert len(sessions) == 1 # chỉ có session s1 đủ dài

def test_missing_item_idx(tmp_path):
    """Kiểm tra lỗi khi thiếu cột item_idx."""
    data = """session_id,timestamp,item_id
s1,2021-01-01 10:00:00,1"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    with pytest.raises(KeyError):
        load_sessions_from_dat(str(file))

def test_sort_by_timestamp(tmp_path):
    """Kiểm tra sắp xếp theo timestamp đúng thứ tự."""
    data = """session_id,timestamp,item_idx
s1,2021-01-01 10:05:00,2
s1,2021-01-01 10:00:00,1"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    sessions = load_sessions_from_dat(str(file))
    assert sessions[0] == [1, 2]  # đã được sắp xếp

def test_filter_short_sessions(tmp_path):
    """Kiểm tra loại bỏ session ngắn hơn min_session_length."""
    data = """session_id,timestamp,item_idx
s1,2021-01-01 10:00:00,1
s2,2021-01-01 10:00:00,2
s2,2021-01-01 10:05:00,3"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    sessions = load_sessions_from_dat(str(file), min_session_length=2)
    assert len(sessions) == 1  # chỉ giữ lại session s2
    assert sessions[0] == [2, 3]

def test_timestamp_conversion(tmp_path):
    """Kiểm tra chuyển đổi timestamp sang datetime thành công."""
    data = """session_id,timestamp,item_idx
s1,2021-01-01 10:00:00,1
s1,2021-01-01 10:05:00,2"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    sessions = load_sessions_from_dat(str(file))
    assert len(sessions) == 1
    assert sessions[0] == [1, 2]

def test_no_timestamp_column(tmp_path):
    """Kiểm tra file không có timestamp column."""
    data = """session_id,item_idx
s1,1
s1,2
s2,3"""
    file = tmp_path / "test.csv"
    file.write_text(data)
    sessions = load_sessions_from_dat(str(file), min_session_length=2)
    assert len(sessions) == 1
    assert sessions[0] == [1, 2]