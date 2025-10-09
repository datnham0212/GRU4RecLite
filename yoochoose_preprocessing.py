import pandas as pd
import numpy as np
from datetime import timedelta
import glob
import os


def _ensure_datetime_series(s):
    """Return a pd.Series of datetimes from s (which may be int/float epoch seconds or strings)."""
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # Treat as Unix epoch seconds
        return pd.to_datetime(s, unit='s', errors='coerce')
    # try to parse strings / mixed types
    return pd.to_datetime(s, errors='coerce')


def read_and_normalize(paths):
    def _read_file(path):
        # Yoochoose-clicks.dat: session_id, timestamp, item_id, Category
        # Auto-detect separator
        seps = ['\t', ',', ';', '|']
        df = None
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    header=None,
                    usecols=[0, 1, 2],
                    names=['session_id', 'timestamp', 'item_id'],
                    dtype={'session_id': object, 'item_id': object},
                    low_memory=False
                )
                if df.shape[1] == 3:
                    break
            except Exception:
                df = None
                continue
        if df is None:
            raise ValueError(f"Cannot determine separator / read file {path}")
        # Normalize timestamp: could be numeric epoch seconds or already date-like
        df['timestamp'] = _ensure_datetime_series(df['timestamp'])
        # Drop rows with invalid timestamps or missing ids
        df = df.dropna(subset=['session_id', 'item_id', 'timestamp'])
        return df

    if isinstance(paths, str) and any(c in paths for c in ['*', '?']):
        file_list = glob.glob(paths)
    elif isinstance(paths, (list, tuple)):
        file_list = list(paths)
    else:
        file_list = [paths]

    if not file_list:
        raise ValueError("No input files found for read_and_normalize")

    df_list = [_read_file(p) for p in file_list]
    df = pd.concat(df_list, ignore_index=True)
    # Ensure proper dtypes
    df['session_id'] = df['session_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    return df


def filter_data(df, min_session_length=2, min_item_support=5):
    # Lọc dữ liệu dựa trên độ dài session và số lần xuất hiện của item
    df = df.copy()
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_support].index
    df = df[df['item_id'].isin(valid_items)]

    sess_counts = df['session_id'].value_counts()
    valid_sess = sess_counts[sess_counts >= min_session_length].index
    df = df[df['session_id'].isin(valid_sess)]
    return df


def sort_and_dedup(df):
    # Sắp xếp và loại bỏ các mục trùng lặp trong cùng một session
    df = df.copy()
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = _ensure_datetime_series(df['timestamp'])
    df = df.sort_values(['session_id', 'timestamp'])
    df['prev_item'] = df.groupby('session_id')['item_id'].shift(1)
    df = df[df['item_id'] != df['prev_item']]
    return df.drop(columns=['prev_item'])


def map_indices(df, start_index=1):
    # Ánh xạ các item sang chỉ số duy nhất
    unique_items = pd.Index(df['item_id'].unique())
    idx_map = {item: idx for idx, item in enumerate(unique_items, start=start_index)}
    df = df.copy()
    df['item_idx'] = df['item_id'].map(idx_map)
    return df, idx_map


def split_time_based(df, test_days=7, valid_days=7):
    # Chia dữ liệu dựa trên thời gian
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = _ensure_datetime_series(df['timestamp'])
    max_t = df['timestamp'].max()
    if pd.isna(max_t):
        raise ValueError("timestamp column contains no valid datetime values")
    test_boundary = max_t - timedelta(days=test_days)
    train_full = df[df['timestamp'] < test_boundary].copy()
    test = df[df['timestamp'] >= test_boundary].copy()

    max_train = train_full['timestamp'].max()
    valid_boundary = max_train - timedelta(days=valid_days)
    train_tr = train_full[train_full['timestamp'] < valid_boundary].copy()
    train_valid = train_full[train_full['timestamp'] >= valid_boundary].copy()

    return {
        'train_full': train_full,
        'test': test,
        'train_tr': train_tr,
        'train_valid': train_valid
    }


def preprocess_pipeline(paths,
                        min_session_length=2,
                        min_item_support=5,
                        test_days=7,
                        valid_days=7,
                        index_start=1):
    # Pipeline xử lý dữ liệu Yoochoose
    df = read_and_normalize(paths)
    if df.empty:
        raise ValueError("No data returned from read_and_normalize")
    df = filter_data(df, min_session_length, min_item_support)
    if df.empty:
        raise ValueError("No data left after filter_data (adjust min_item_support / min_session_length)")
    df = sort_and_dedup(df)
    df, idx_map = map_indices(df, index_start)
    splits = split_time_based(df, test_days, valid_days)
    return splits, idx_map

if __name__ == '__main__':
    # Try sensible default locations but avoid hard failures if files absent.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Update possible_yoo paths to reflect new Yoochoose-data folder structure
    possible_yoo = [
        os.path.join(repo_root, 'Yoochoose-data', 'yoochoose-clicks.dat'),
        os.path.join(repo_root, 'Yoochoose-data', 'yoochoose-buys.dat'),
        os.path.join(repo_root, 'Yoochoose-data', 'yoochoose-test.dat'),
        os.path.join(repo_root, '..', 'Yoochoose-data', 'yoochoose-clicks.dat'),
        os.path.join(repo_root, '..', 'Yoochoose-data', 'yoochoose-buys.dat'),
        os.path.join(repo_root, '..', 'Yoochoose-data', 'yoochoose-test.dat')
    ]
    yoo_paths = next((p for p in possible_yoo if os.path.isfile(p)), None)
    if yoo_paths is None:
        print("Warning: Yoochoose file not found in expected locations. Set yoo_paths manually in __main__ if needed.")
    else:
        print("Using yoochoose file:", yoo_paths)
        yoo_splits, yoo_map = preprocess_pipeline(
            paths=yoo_paths,
            min_session_length=2,
            min_item_support=5,
            test_days=7,
            valid_days=7,
            index_start=1
        )
        output_dir = os.path.join(repo_root, 'input_data', 'yoochoose-data')
        os.makedirs(output_dir, exist_ok=True)
        for name, df in yoo_splits.items():
            # write timestamp in ISO format for consistency
            out_df = df.copy()
            out_df.loc[:, 'timestamp'] = out_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            out_df.to_csv(os.path.join(output_dir, f'yoochoose_{name}.dat'), index=False, sep='\t')
        pd.to_pickle(yoo_map, os.path.join(output_dir, 'yoochoose_map.pkl'))
        print("Yoochoose preprocessing complete. Files written to", output_dir)
