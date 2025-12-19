"""
Unit Tests for Data Preprocessing

Tests the data preprocessing pipelines for Retailrocket and Yoochoose datasets:
- Data loading and normalization
- Filtering by session length and item support
- Duplicate removal and sorting
- Item index mapping
- Time-based data splitting
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(__file__))

from retailrocket_preprocessing import (
    read_retailrocket_events,
    filter_sessions_items,
    sort_and_remove_consecutive_duplicates,
    map_item_indices,
    split_by_time
)

from yoochoose_preprocessing import (
    read_and_normalize,
    filter_data,
    sort_and_dedup,
    map_indices,
    split_time_based,
    preprocess_pipeline
)


# ============================================================================
# FIXTURES - SAMPLE DATA
# ============================================================================

@pytest.fixture
def sample_retailrocket_csv(tmp_path):
    """Create sample Retailrocket events.csv file."""
    csv_file = tmp_path / "events.csv"
    data = """timestamp,visitorid,itemid,event,transactionid
1000,1,100,view,
1005,1,200,view,
2000,2,150,view,
2010,2,250,view,
3000,3,100,view,
3005,3,100,view
"""
    csv_file.write_text(data)
    return str(csv_file)


@pytest.fixture
def sample_yoochoose_csv(tmp_path):
    """Create sample Yoochoose clicks.dat file."""
    dat_file = tmp_path / "yoochoose-clicks.dat"
    # session_id, timestamp, item_id
    data = """1\t1000\t100
1\t1005\t200
2\t2000\t150
2\t2010\t250
3\t3000\t100
3\t3005\t100
"""
    dat_file.write_text(data)
    return str(dat_file)


@pytest.fixture
def sample_retailrocket_df():
    """Create sample Retailrocket DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 10:05:00',
                                     '2025-01-02 11:00:00', '2025-01-02 11:10:00',
                                     '2025-01-03 12:00:00', '2025-01-03 12:05:00']),
        'session_id': ['s1', 's1', 's2', 's2', 's3', 's3'],
        'event': ['view', 'view', 'view', 'view', 'view', 'view'],
        'item_id': ['100', '200', '150', '250', '100', '100']
    })


@pytest.fixture
def sample_yoochoose_df():
    """Create sample Yoochoose DataFrame."""
    return pd.DataFrame({
        'session_id': ['1', '1', '2', '2', '3', '3'],
        'timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 10:05:00',
                                     '2025-01-02 11:00:00', '2025-01-02 11:10:00',
                                     '2025-01-03 12:00:00', '2025-01-03 12:05:00']),
        'item_id': ['100', '200', '150', '250', '100', '100']
    })


# ============================================================================
# RETAILROCKET PREPROCESSING TESTS
# ============================================================================

class TestRetailrocketDataLoading:
    """Tests for Retailrocket data loading and normalization."""
    
    def test_read_retailrocket_events_with_valid_file(self, sample_retailrocket_csv):
        """Verify reading valid Retailrocket events file."""
        df = read_retailrocket_events(sample_retailrocket_csv, use_events=['view'])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'session_id' in df.columns
        assert 'item_id' in df.columns
        assert 'timestamp' in df.columns
    
    def test_read_retailrocket_events_filters_by_event_type(self, sample_retailrocket_csv):
        """Verify event filtering works correctly."""
        df = read_retailrocket_events(sample_retailrocket_csv, use_events=['view'])
        
        # All rows should have event type 'view'
        assert all(df['event'] == 'view')
    
    def test_read_retailrocket_events_removes_nulls(self):
        """Verify null values are removed."""
        df = pd.DataFrame({
            'timestamp': [1000, 1005, 1010],
            'visitorid': ['1', '2', '3'],  # Required by retailrocket format
            'itemid': ['100', '200', None],
            'event': ['view', 'view', 'view'],  # Add event column
            'transactionid': [None, None, None]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            f.flush()
            temp_path = f.name
        
        try:
            result = read_retailrocket_events(temp_path, use_events=['view'])
            
            assert len(result) == 2
            assert result['item_id'].notna().all()
        finally:
            os.unlink(temp_path)  
            
    def test_read_retailrocket_events_converts_timestamp(self, sample_retailrocket_csv):
        """Verify timestamp conversion to datetime."""
        df = read_retailrocket_events(sample_retailrocket_csv, use_events=['view'])
        
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])


class TestRetailrocketFiltering:
    """Tests for Retailrocket data filtering."""
    
    def test_filter_sessions_items_removes_short_sessions(self, sample_retailrocket_df):
        """Verify sessions shorter than min_session_length are removed."""
        # Add a short session
        short_session = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-04 13:00:00')],
            'session_id': ['s4'],
            'event': ['view'],
            'item_id': ['300']
        })
        df = pd.concat([sample_retailrocket_df, short_session], ignore_index=True)
        
        result = filter_sessions_items(df, min_session_length=2, min_item_support=1)
        
        # s4 should be removed (only 1 item)
        assert 's4' not in result['session_id'].values
        assert len(result) == 6
    
    def test_filter_sessions_items_removes_low_support_items(self, sample_retailrocket_df):
        """Verify items with support < min_item_support are removed."""
        result = filter_sessions_items(sample_retailrocket_df, min_session_length=1, min_item_support=2)
        
        # Item '250' appears only once, should be removed
        assert '250' not in result['item_id'].values
    
    def test_filter_sessions_items_preserves_valid_data(self, sample_retailrocket_df):
        """Verify valid data is preserved."""
        result = filter_sessions_items(sample_retailrocket_df, min_session_length=1, min_item_support=1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestRetailrocketDeduplication:
    """Tests for Retailrocket duplicate removal."""
    
    def test_sort_and_remove_consecutive_duplicates(self, sample_retailrocket_df):
        """Verify consecutive duplicates are removed."""
        # sample has consecutive duplicates: item 100 appears twice in session s3
        result = sort_and_remove_consecutive_duplicates(sample_retailrocket_df)
        
        # Check that session s3 has only 1 item (duplicate removed)
        s3_items = result[result['session_id'] == 's3']['item_id'].values
        assert len(s3_items) == 1
    
    def test_sort_and_remove_does_not_remove_non_consecutive_duplicates(self):
        """Verify non-consecutive duplicates are NOT removed."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 10:05:00', '2025-01-01 10:10:00']),
            'session_id': ['s1', 's1', 's1'],
            'event': ['view', 'view', 'view'],
            'item_id': ['100', '200', '100']  # 100 appears twice but not consecutive
        })
        
        result = sort_and_remove_consecutive_duplicates(df)
        
        # Both 100s should remain (not consecutive)
        assert len(result) == 3
    
    def test_sort_and_remove_sorts_by_timestamp(self):
        """Verify data is sorted by session and timestamp."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:10:00', '2025-01-01 10:00:00', '2025-01-01 10:05:00']),
            'session_id': ['s1', 's1', 's1'],
            'event': ['view', 'view', 'view'],
            'item_id': ['200', '100', '150']
        })
        
        result = sort_and_remove_consecutive_duplicates(df)
        
        # Should be sorted: 100, 150, 200
        assert list(result['item_id'].values) == ['100', '150', '200']


class TestRetailrocketIndexMapping:
    """Tests for Retailrocket item index mapping."""
    
    def test_map_item_indices_creates_mapping(self, sample_retailrocket_df):
        """Verify item mapping is created correctly."""
        df, idx_map = map_item_indices(sample_retailrocket_df, start_index=1)
        
        assert isinstance(idx_map, dict)
        assert len(idx_map) > 0
        assert 'item_idx' in df.columns
    
    def test_map_item_indices_start_index(self, sample_retailrocket_df):
        """Verify start_index parameter works."""
        df, idx_map = map_item_indices(sample_retailrocket_df, start_index=100)
        
        # All indices should be >= 100
        assert df['item_idx'].min() >= 100
    
    def test_map_item_indices_unique_mapping(self, sample_retailrocket_df):
        """Verify each unique item gets unique index."""
        df, idx_map = map_item_indices(sample_retailrocket_df, start_index=1)
        
        # Number of unique items should equal mapping size
        assert len(idx_map) == df['item_id'].nunique()
    
    def test_map_item_indices_bidirectional(self, sample_retailrocket_df):
        """Verify mapping is consistent in both directions."""
        df, idx_map = map_item_indices(sample_retailrocket_df, start_index=1)
        
        # For each item_id, look up its index in the mapping
        for item_id in df['item_id'].unique():
            expected_idx = idx_map[item_id]
            actual_idx = df[df['item_id'] == item_id]['item_idx'].iloc[0]
            assert expected_idx == actual_idx


class TestRetailrocketTimeSplit:
    """Tests for Retailrocket time-based splitting."""
    
    def test_split_by_time_creates_all_splits(self, sample_retailrocket_df):
        """Verify all split keys are created."""
        splits = split_by_time(sample_retailrocket_df, test_days=1, valid_days=1)
        
        expected_keys = {'train_full', 'test', 'train_tr', 'train_valid'}
        assert set(splits.keys()) == expected_keys
    
    def test_split_by_time_all_data_accounted_for(self, sample_retailrocket_df):
        """Verify all data is accounted for in splits."""
        splits = split_by_time(sample_retailrocket_df, test_days=1, valid_days=1)
        
        # train_full should equal train_tr + train_valid
        train_full_ids = set(splits['train_full'].index)
        train_split_ids = set(splits['train_tr'].index) | set(splits['train_valid'].index)
        assert train_full_ids == train_split_ids
    
    def test_split_by_time_test_set_recent_data(self, sample_retailrocket_df):
        """Verify test set contains most recent data."""
        splits = split_by_time(sample_retailrocket_df, test_days=1, valid_days=1)
        
        test_time_min = splits['test']['timestamp'].min()
        train_full_time_max = splits['train_full']['timestamp'].max()
        
        # Test set should be more recent
        assert test_time_min >= train_full_time_max


# ============================================================================
# YOOCHOOSE PREPROCESSING TESTS
# ============================================================================

class TestYoochooseDataLoading:
    """Tests for Yoochoose data loading and normalization."""
    
    def test_read_and_normalize_with_valid_file(self, sample_yoochoose_csv):
        """Verify reading valid Yoochoose file."""
        df = read_and_normalize(sample_yoochoose_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'session_id' in df.columns
        assert 'item_id' in df.columns
        assert 'timestamp' in df.columns
    
    def test_read_and_normalize_with_glob_pattern(self, tmp_path):
        """Verify glob pattern support."""
        # Create multiple files
        for i in range(2):
            f = tmp_path / f"clicks_{i}.dat"
            f.write_text("1\t1000\t100\n2\t2000\t200\n")
        
        pattern = str(tmp_path / "clicks_*.dat")
        df = read_and_normalize(pattern)
        
        assert len(df) == 4  # 2 files x 2 rows
    
    def test_read_and_normalize_with_file_list(self, tmp_path):
        """Verify handling list of files."""
        files = []
        for i in range(2):
            f = tmp_path / f"clicks_{i}.dat"
            f.write_text("1\t1000\t100\n2\t2000\t200\n")
            files.append(str(f))
        
        df = read_and_normalize(files)
        
        assert len(df) == 4  # 2 files x 2 rows
    
    def test_read_and_normalize_removes_nulls(self, tmp_path):
        """Verify null values are removed."""
        f = tmp_path / "test.dat"
        f.write_text("1\t1000\t100\n\t2000\t200\n3\t3000\t")
        
        df = read_and_normalize(str(f))
        
        # Only first row should remain
        assert len(df) == 1


class TestYoochooseFiltering:
    """Tests for Yoochoose data filtering."""
    
    def test_filter_data_removes_short_sessions(self, sample_yoochoose_df):
        """Verify sessions shorter than min_session_length are removed."""
        short_session = pd.DataFrame({
            'session_id': ['4'],
            'timestamp': pd.to_datetime(['2025-01-04 13:00:00']),
            'item_id': ['300']
        })
        df = pd.concat([sample_yoochoose_df, short_session], ignore_index=True)
        
        result = filter_data(df, min_session_length=2, min_item_support=1)
        
        assert '4' not in result['session_id'].values
    
    def test_filter_data_removes_low_support_items(self, sample_yoochoose_df):
        """Verify items with low support are removed."""
        result = filter_data(sample_yoochoose_df, min_session_length=1, min_item_support=2)
        
        # Items 150 and 250 appear only once, should be removed
        assert '250' not in result['item_id'].values
    
    def test_filter_data_preserves_valid_data(self, sample_yoochoose_df):
        """Verify valid data is preserved."""
        result = filter_data(sample_yoochoose_df, min_session_length=1, min_item_support=1)
        
        assert len(result) > 0


class TestYoochooseDeduplication:
    """Tests for Yoochoose duplicate removal."""
    
    def test_sort_and_dedup_removes_consecutive_duplicates(self, sample_yoochoose_df):
        """Verify consecutive duplicates are removed."""
        result = sort_and_dedup(sample_yoochoose_df)
        
        # Session 3 had consecutive 100s, should be reduced to 1
        s3_items = result[result['session_id'] == '3']['item_id'].values
        assert len(s3_items) == 1
    
    def test_sort_and_dedup_preserves_non_consecutive_duplicates(self):
        """Verify non-consecutive duplicates are preserved."""
        df = pd.DataFrame({
            'session_id': ['1', '1', '1'],
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 10:05:00', '2025-01-01 10:10:00']),
            'item_id': ['100', '200', '100']
        })
        
        result = sort_and_dedup(df)
        
        assert len(result) == 3


class TestYoochooseIndexMapping:
    """Tests for Yoochoose item index mapping."""
    
    def test_map_indices_creates_mapping(self, sample_yoochoose_df):
        """Verify index mapping is created."""
        df, idx_map = map_indices(sample_yoochoose_df, start_index=1)
        
        assert isinstance(idx_map, dict)
        assert 'item_idx' in df.columns
    
    def test_map_indices_unique_values(self, sample_yoochoose_df):
        """Verify unique items get unique indices."""
        df, idx_map = map_indices(sample_yoochoose_df, start_index=1)
        
        assert len(idx_map) == df['item_id'].nunique()
        assert len(idx_map) == df['item_idx'].nunique()


class TestYoochooseTimeSplit:
    """Tests for Yoochoose time-based splitting."""
    
    def test_split_time_based_creates_splits(self, sample_yoochoose_df):
        """Verify all split keys are created."""
        splits = split_time_based(sample_yoochoose_df, test_days=1, valid_days=1)
        
        expected_keys = {'train_full', 'test', 'train_tr', 'train_valid'}
        assert set(splits.keys()) == expected_keys
    
    def test_split_time_based_all_data_accounted_for(self, sample_yoochoose_df):
        """Verify all data is in one of the splits."""
        splits = split_time_based(sample_yoochoose_df, test_days=1, valid_days=1)
        
        train_full_ids = set(splits['train_full'].index)
        train_split_ids = set(splits['train_tr'].index) | set(splits['train_valid'].index)
        assert train_full_ids == train_split_ids
    
    def test_split_time_based_no_overlap_test_train(self, sample_yoochoose_df):
        """Verify test and train sets don't overlap in time."""
        splits = split_time_based(sample_yoochoose_df, test_days=1, valid_days=1)
        
        test_time_min = splits['test']['timestamp'].min()
        train_time_max = splits['train_full']['timestamp'].max()
        
        assert test_time_min >= train_time_max


class TestYoochoosePipeline:
    """Tests for complete Yoochoose preprocessing pipeline."""
    
    def test_preprocess_pipeline_returns_splits_and_map(self, sample_yoochoose_csv):
        """Verify pipeline returns correct structure."""
        splits, idx_map = preprocess_pipeline(
            sample_yoochoose_csv,
            min_session_length=1,
            min_item_support=1,
            test_days=1,
            valid_days=1
        )
        
        assert isinstance(splits, dict)
        assert isinstance(idx_map, dict)
        assert 'train_full' in splits
        assert 'test' in splits
    
    def test_preprocess_pipeline_with_empty_data(self, tmp_path):
        """Verify error handling with empty data."""
        f = tmp_path / "empty.dat"
        f.write_text("")
        
        with pytest.raises(ValueError):
            preprocess_pipeline(str(f))
    
    def test_preprocess_pipeline_with_high_filter_thresholds(self, tmp_path):
        """Verify error when filters remove all data."""
        f = tmp_path / "test.dat"
        # Create data that won't meet threshold
        f.write_text("1\t1000\t100\n")
        
        with pytest.raises(ValueError):
            preprocess_pipeline(str(f), min_session_length=10, min_item_support=10)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDataPreprocessingIntegration:
    """Integration tests for complete preprocessing pipelines."""
    
    def test_retailrocket_preprocessing_produces_valid_splits(self, sample_retailrocket_df):
        """Verify Retailrocket pipeline produces valid output."""
        from retailrocket_preprocessing import filter_sessions_items, sort_and_remove_consecutive_duplicates, map_item_indices, split_by_time
        
        df = filter_sessions_items(sample_retailrocket_df, min_session_length=1, min_item_support=1)
        df = sort_and_remove_consecutive_duplicates(df)
        df, _ = map_item_indices(df)
        splits = split_by_time(df, test_days=1, valid_days=1)
        
        # Verify all splits are valid DataFrames with expected columns
        for split_name, split_df in splits.items():
            assert isinstance(split_df, pd.DataFrame)
            assert 'item_idx' in split_df.columns
    
    def test_yoochoose_pipeline_end_to_end(self, sample_yoochoose_csv):
        """Verify Yoochoose pipeline works end-to-end."""
        splits, idx_map = preprocess_pipeline(
            sample_yoochoose_csv,
            min_session_length=1,
            min_item_support=1
        )
        
        # Verify structure
        assert 'train_full' in splits
        assert 'test' in splits
        assert len(idx_map) > 0
        
        # Verify all splits have item_idx
        for split_df in splits.values():
            assert 'item_idx' in split_df.columns
