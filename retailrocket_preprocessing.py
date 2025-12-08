import pandas as pd
import numpy as np
from datetime import timedelta
import os

def read_retailrocket_events(path, use_events=['view']):
    """
    Đọc và chuẩn hóa dữ liệu Retail Rocket events.csv
    """
    df = pd.read_csv(path)
    print(f"Số dòng ban đầu: {df.shape[0]}")
    # Đổi tên cột cho thống nhất
    df = df.rename(columns={
        'visitorid': 'session_id',
        'itemid': 'item_id',
        'event': 'event',
        'timestamp': 'timestamp'
    })
    # Chỉ giữ các cột cần thiết
    df = df[['timestamp', 'session_id', 'event', 'item_id']]
    # Lọc theo event
    if use_events:
        df = df[df['event'].isin(use_events)]
    print(f"Sau khi lọc event: {df.shape[0]}")
    # Loại bỏ dòng thiếu thông tin
    df = df.dropna(subset=['session_id', 'item_id', 'timestamp'])
    print(f"Sau khi loại bỏ dòng thiếu thông tin: {df.shape[0]}")
    # Đảm bảo kiểu dữ liệu
    df['session_id'] = df['session_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    # Xử lý timestamp (Unix time)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    print(f"Sau khi chuyển timestamp: {df.shape[0]}")
    return df

def filter_sessions_items(df, min_session_length=2, min_item_support=5):
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_support].index
    df = df[df['item_id'].isin(valid_items)]
    print(f"Sau khi lọc item: {df.shape[0]}")
    sess_counts = df['session_id'].value_counts()
    valid_sess = sess_counts[sess_counts >= min_session_length].index
    df = df[df['session_id'].isin(valid_sess)]
    print(f"Sau khi lọc session: {df.shape[0]}")
    return df

def sort_and_remove_consecutive_duplicates(df):
    df = df.sort_values(['session_id', 'timestamp'])
    df['prev_item'] = df.groupby('session_id')['item_id'].shift(1)
    df = df[df['item_id'] != df['prev_item']]
    print(f"Sau khi loại bỏ click trùng lặp liên tiếp: {df.shape[0]}")
    return df.drop(columns=['prev_item'])

def map_item_indices(df, start_index=1):
    unique_items = df['item_id'].unique()
    idx_map = {item: idx for idx, item in enumerate(unique_items, start=start_index)}
    df['item_idx'] = df['item_id'].map(idx_map)
    return df, idx_map

def split_by_time(df, test_days=7, valid_days=7):
    max_t = df['timestamp'].max()
    test_boundary = max_t - timedelta(days=test_days)
    train_full = df[df['timestamp'] < test_boundary]
    test = df[df['timestamp'] >= test_boundary]
    max_train = train_full['timestamp'].max()
    valid_boundary = max_train - timedelta(days=valid_days)
    train_tr = train_full[train_full['timestamp'] < valid_boundary]
    train_valid = train_full[train_full['timestamp'] >= valid_boundary]
    return {
        'train_full': train_full,
        'test': test,
        'train_tr': train_tr,
        'train_valid': train_valid
    }

def preprocess_retailrocket(
    path,
    use_events=['view'],
    min_session_length=2,
    min_item_support=5,
    test_days=7,
    valid_days=7,
    index_start=1,
    output_dir=None):
    # Determine a sensible default output directory if none provided
    if output_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(repo_root, 'input_data', 'retailrocket-data')

    df = read_retailrocket_events(path, use_events)
    df = filter_sessions_items(df, min_session_length, min_item_support)
    df = sort_and_remove_consecutive_duplicates(df)
    df, idx_map = map_item_indices(df, index_start)
    splits = split_by_time(df, test_days, valid_days)
    os.makedirs(output_dir, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(os.path.join(output_dir, f'retailrocket_{name}.dat'), index=False, sep='\t')
    pd.to_pickle(idx_map, os.path.join(output_dir, 'retailrocket_map.pkl'))
    print('Done!')

def __main_block__():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    possible_rr = [
        os.path.join(repo_root, 'Retailrocket-data', 'events.csv'),
        os.path.join(repo_root, '..', 'Retailrocket-data', 'events.csv'),
        os.path.join(repo_root, 'retailrocket-data', 'events.csv'),
        os.path.join(repo_root, '..', 'retailrocket-data', 'events.csv'),
    ]
    rr_path = next((p for p in possible_rr if os.path.isfile(p)), None)
    if rr_path is None:
        print("Warning: Retailrocket events.csv not found in expected locations. Set path manually in __main__ if needed.")
    else:
        print("Using retailrocket file:", rr_path)
        preprocess_retailrocket(
            path=rr_path,
            use_events=['view'],
            min_session_length=2,
            min_item_support=5,
            test_days=7,
            valid_days=7,
            index_start=1,
            output_dir=None
        )

if __name__ == '__main__':
    __main_block__()
