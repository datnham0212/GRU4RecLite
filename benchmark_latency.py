#!/usr/bin/env python3
"""
Latency Benchmarking Script for GRU4Rec
Measures per-session inference latency with percentile metrics
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru4rec_pytorch import GRU4Rec


def load_test_data(test_file):
    """Load test data from TAB-separated file using pandas"""
    print(f"Reading file: {test_file}")
    
    # Read without dtype constraints first, then convert
    df = pd.read_csv(test_file, sep='\t')
    
    print(f"File columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    # Convert session_id and item_id to int
    df['session_id'] = df['session_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    
    # Convert timestamp - handle both datetime string and numeric formats
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_numeric'] = df['timestamp'].astype(np.int64) // 10**9
    else:
        df['timestamp_numeric'] = df['timestamp']
    
    # Group by session
    sessions = {}
    for session_id, group in df.groupby('session_id'):
        items = group.sort_values('timestamp_numeric')['item_id'].tolist()
        sessions[session_id] = items
    
    print(f"Loaded {len(sessions)} unique sessions")
    return sessions, df


def benchmark_inference(gru, test_sessions, df, n_samples=1000):
    """
    Benchmark inference latency using the GRU4Rec model
    
    Args:
        gru: Trained GRU4Rec instance
        test_sessions: Dictionary of session_id -> [item_id, ...]
        df: Original dataframe with item mappings
        n_samples: Number of sessions to benchmark
    
    Returns:
        Dictionary with latency statistics
    """
    
    device = gru.device
    gru.model.eval()
    
    # Get item ID mapping from model
    itemidmap = gru.data_iterator.itemidmap
    
    # Debug: Print itemidmap info
    print(f"\nItemidmap info:")
    print(f"  Type: {type(itemidmap)}")
    print(f"  Length: {len(itemidmap)}")
    print(f"  Index dtype: {itemidmap.index.dtype}")
    
    # FIX: Convert itemidmap index to int for comparison
    # The itemidmap index is stored as string, but test data item_id is int
    itemidmap_int_index = itemidmap.copy()
    itemidmap_int_index.index = itemidmap_int_index.index.astype(int)
    
    print(f"  Converted index dtype: {itemidmap_int_index.index.dtype}")
    print(f"  First 5 items after conversion: {list(itemidmap_int_index.head().items())}")
    
    # Now valid_items contains int item IDs
    valid_items = set(itemidmap_int_index.index)
    
    # Debug: Check overlap with test data
    test_items = set(df['item_id'].unique())
    overlap = valid_items & test_items
    print(f"  Test data unique items: {len(test_items)}")
    print(f"  Valid items in model: {len(valid_items)}")
    print(f"  Overlap: {len(overlap)}")
    
    # Sample some test items to check
    sample_test_items = list(test_items)[:5]
    print(f"  Sample test item_ids: {sample_test_items}")
    print(f"  Are they in itemidmap? {[item in valid_items for item in sample_test_items]}")
    
    if len(overlap) == 0:
        print("ERROR: No overlap between test items and model vocabulary!")
        return None
    
    latencies = []
    session_ids = list(test_sessions.keys())[:n_samples]
    
    print(f"\nBenchmarking {len(session_ids)} sessions for inference latency...")
    print(f"Device: {device}")
    
    # Get layer sizes
    if isinstance(gru.layers, list):
        layer_sizes = gru.layers
    else:
        layer_sizes = [gru.layers]
    
    skipped = 0
    processed = 0
    skip_reasons = {'no_valid_items': 0, 'single_item': 0}
    
    with torch.no_grad():
        for i, session_id in enumerate(session_ids):
            items = test_sessions[session_id]
            
            # Filter to valid items only
            valid_session_items = [item for item in items if item in valid_items]
            
            if len(valid_session_items) == 0:
                skip_reasons['no_valid_items'] += 1
                skipped += 1
                continue
            
            if len(valid_session_items) < 2:
                skip_reasons['single_item'] += 1
                skipped += 1
                continue
            
            # Initialize hidden state for single session (batch_size=1)
            H = [torch.zeros((1, layer_sizes[j]), device=device) for j in range(len(layer_sizes))]
            
            # Measure time for processing this session
            start_time = time.perf_counter()
            
            # Process each item in sequence (simulate real-time recommendation)
            for item_idx in range(len(valid_session_items) - 1):
                current_item = valid_session_items[item_idx]
                
                # Map to internal ID using itemidmap (use int index version)
                internal_id = itemidmap_int_index[current_item]
                
                # Create input tensor
                X = torch.tensor([internal_id], dtype=torch.long, device=device)
                
                # Forward pass
                E, O, B = gru.model.embed(X, H, None)
                if gru.model.dropout_p_embed > 0 and gru.model.training:
                    E = gru.model.DE(E)
                if not (gru.model.constrained_embedding or gru.model.embedding):
                    H[0] = E
                Xh = gru.model.hidden_step(E, H, training=False)
                scores = gru.model.score_items(Xh, O, B)
                
                # Get top-20 recommendations (simulate production use)
                _, top_indices = torch.topk(scores, min(20, scores.shape[1]), dim=1)
                
                # Update hidden state for next step
                H = [h.detach() for h in Xh] if isinstance(Xh, list) else [Xh.detach()]
            
            end_time = time.perf_counter()
            
            # Calculate latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            processed += 1
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(session_ids)} sessions...")
    
    print(f"\nProcessed: {processed} sessions")
    print(f"Skipped: {skipped} sessions")
    print(f"  - No valid items: {skip_reasons['no_valid_items']}")
    print(f"  - Single item only: {skip_reasons['single_item']}")
    
    # Calculate statistics
    latencies = np.array(latencies)
    
    if len(latencies) == 0:
        print("ERROR: No valid sessions to benchmark!")
        return None
    
    stats = {
        'n_sessions': len(latencies),
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'p999_ms': np.percentile(latencies, 99.9),
    }
    
    return stats


def print_stats(stats, model_name, device):
    """Print formatted latency statistics"""
    print("\n" + "=" * 60)
    print(f"LATENCY BENCHMARK RESULTS: {model_name}")
    print(f"Device: {device}")
    print("=" * 60)
    print(f"Sessions tested:    {stats['n_sessions']}")
    print(f"Mean latency:       {stats['mean_ms']:.3f} ms")
    print(f"Std deviation:      {stats['std_ms']:.3f} ms")
    print(f"Min latency:        {stats['min_ms']:.3f} ms")
    print(f"Max latency:        {stats['max_ms']:.3f} ms")
    print("-" * 60)
    print("PERCENTILES:")
    print(f"  p50 (median):     {stats['p50_ms']:.3f} ms")
    print(f"  p90:              {stats['p90_ms']:.3f} ms")
    print(f"  p95:              {stats['p95_ms']:.3f} ms")
    print(f"  p99:              {stats['p99_ms']:.3f} ms")
    print(f"  p99.9:            {stats['p999_ms']:.3f} ms")
    print("-" * 60)
    
    # Production readiness check
    if stats['p99_ms'] < 100:
        print("PRODUCTION STATUS:  ✅ PASS (p99 < 100ms)")
    else:
        print("PRODUCTION STATUS:  ❌ FAIL (p99 >= 100ms)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark GRU4Rec inference latency')
    parser.add_argument('model', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('-t', '--test_data', type=str, required=True,
                        help='Path to test data file')
    parser.add_argument('-n', '--n_samples', type=int, default=1000,
                        help='Number of sessions to benchmark (default: 1000)')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda:0)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    
    # Load model
    gru = torch.load(args.model, map_location=args.device, weights_only=False)
    gru.device = args.device
    
    # Move model to device
    if hasattr(gru, 'model'):
        gru.model.to(args.device)
        gru.model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model layers: {gru.layers}")
    
    print(f"Loading test data from: {args.test_data}")
    test_sessions, df = load_test_data(args.test_data)
    
    if len(test_sessions) == 0:
        print("ERROR: No sessions loaded from test data!")
        sys.exit(1)
    
    print(f"Benchmarking {args.n_samples} sessions...")
    
    # Run benchmark
    stats = benchmark_inference(gru, test_sessions, df, args.n_samples)
    
    if stats is None:
        print("Benchmark failed - no valid sessions")
        sys.exit(1)
    
    # Extract model name from path
    model_name = Path(args.model).stem
    
    # Print results
    print_stats(stats, model_name, args.device)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Sessions: {stats['n_sessions']}\n")
            f.write(f"Mean: {stats['mean_ms']:.3f} ms\n")
            f.write(f"p50: {stats['p50_ms']:.3f} ms\n")
            f.write(f"p90: {stats['p90_ms']:.3f} ms\n")
            f.write(f"p95: {stats['p95_ms']:.3f} ms\n")
            f.write(f"p99: {stats['p99_ms']:.3f} ms\n")
            f.write(f"p99.9: {stats['p999_ms']:.3f} ms\n")
            f.write(f"Production Ready: {'YES' if stats['p99_ms'] < 100 else 'NO'}\n")
        print(f"\nResults appended to: {args.output}")
    
    return stats


if __name__ == '__main__':
    main()
