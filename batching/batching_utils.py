import pandas as pd
from typing import List

def load_sessions_from_dat(path: str, item_idx_col: str = 'item_idx', min_session_length: int = 2) -> List[List[int]]:
    """Read a .dat/.csv file containing session_id,timestamp,item_idx and return list of sessions (list of item_idx).
    Expect file to have columns: 'session_id', 'timestamp', and either 'item_idx' (preferred) or 'item_id' if you map externally.
    """
    df = pd.read_csv(path, sep=None, engine='python', dtype={'session_id': str})
    if item_idx_col not in df.columns:
        raise KeyError(f"Column {item_idx_col} not found in {path}. Expected 'item_idx'.")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(['session_id', 'timestamp'])
    else:
        df = df.sort_values(['session_id'])
    grouped = df.groupby('session_id')[item_idx_col].apply(list)
    sessions = [list(map(int, s)) for s in grouped.tolist() if len(s) >= min_session_length]
    return sessions

if __name__ == '__main__':
    # đọc phiên dữ liệu từ file train Yoochoose
    yoochoose_path = r'D:/output_data/yoochoose_train_tr.dat'
    yoochoose_sessions = load_sessions_from_dat(yoochoose_path, item_idx_col='item_idx', min_session_length=2)
    print(f'Số phiên Yoochoose train: {len(yoochoose_sessions)}')
    print('Ví dụ một phiên Yoochoose:', yoochoose_sessions[0] if yoochoose_sessions else 'Không có phiên nào')

    # đọc phiên dữ liệu từ file train Retail Rocket
    # retailrocket_path = r'd:/output_data/retailrocket_train_tr.dat'
    # retailrocket_sessions = load_sessions_from_dat(retailrocket_path, item_idx_col='item_idx', min_session_length=2)
    # print(f'Số phiên Retail Rocket train: {len(retailrocket_sessions)}')
    # print('Ví dụ một phiên Retail Rocket:', retailrocket_sessions[0] if retailrocket_sessions else 'Không có phiên nào')
