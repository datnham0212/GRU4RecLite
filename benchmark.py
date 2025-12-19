"""
Benchmark script: Compare GRU4Rec, MostPopular, and LastItem baselines.
Usage:
    python benchmark.py <train_data> <test_data> [--model-path MODEL] [--cutoff CUTOFF] [--batch-size BATCH_SIZE]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from datetime import datetime

# Add paths
WEB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GRU_PKG_DIR = os.path.dirname(__file__)

if WEB_ROOT not in sys.path:
    sys.path.insert(0, WEB_ROOT)
if GRU_PKG_DIR not in sys.path:
    sys.path.insert(0, GRU_PKG_DIR)

# Import models
try:
    from gru4rec_pytorch import GRU4Rec, SessionDataIterator
except ImportError:
    try:
        from model.gru4rec_torch.gru4rec_pytorch import GRU4Rec, SessionDataIterator
    except ImportError:
        print("ERROR: Could not import GRU4Rec")
        sys.exit(1)

from baselines import MostPopularBaseline, LastItemBaseline
import evaluation


def load_data(fname, item_key='item_id', session_key='session_id', time_key='timestamp'):
    """Load training or test data."""
    if fname.endswith('.pickle'):
        print(f'Loading pickle: {fname}')
        import joblib
        data = joblib.load(fname)
    else:
        print(f'Loading data: {fname}')
        # Try to detect and handle different formats
        try:
            data = pd.read_csv(fname, sep='\t', usecols=[session_key, item_key, time_key],
                              dtype={session_key: 'int32', item_key: 'str'})
        except Exception as e:
            print(f"Warning: Could not load with default format: {e}")
            print("Attempting flexible load...")
            data = pd.read_csv(fname, sep='\t', dtype={'session_id': 'int32'})
    
    if data.empty:
        raise ValueError(f"Data is empty: {fname}")
    
    # Ensure proper dtypes
    if session_key in data.columns:
        try:
            data[session_key] = data[session_key].astype('int32')
        except:
            pass
    if item_key in data.columns:
        data[item_key] = data[item_key].astype('str')
    
    # Ensure timestamp column exists and is sortable
    if time_key not in data.columns:
        print(f"Warning: Column '{time_key}' not found. Trying alternatives...")
        alt_names = ['timestamp', 'time', 'ts']
        for alt in alt_names:
            if alt in data.columns:
                data[time_key] = data[alt]
                break
    
    return data


def get_item_embeddings_from_gru(gru):
    """Extract item embeddings from GRU4Rec model for LastItem baseline."""
    try:
        model = getattr(gru, 'model', None)
        if model is None:
            return None

        # Try constrained embedding (output matrix Wy)
        if getattr(model, 'constrained_embedding', False):
            Wy = getattr(model, 'Wy', None)
            if hasattr(Wy, 'weight'):
                embs = Wy.weight.detach().cpu().numpy()
                # Normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs = embs / norms
                return embs

        # Try input embedding
        for name in ['embedding', 'E']:
            emb_layer = getattr(model, name, None)
            if hasattr(emb_layer, 'weight'):
                embs = emb_layer.weight.detach().cpu().numpy()
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs = embs / norms
                return embs

        # Fallback to output layer
        for name in ['Wy', 'output', 'out', 'linear', 'fc', 'decoder']:
            layer = getattr(model, name, None)
            if hasattr(layer, 'weight'):
                embs = layer.weight.detach().cpu().numpy()
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embs = embs / norms
                return embs
    except Exception as e:
        print(f"Warning: Could not extract embeddings: {e}")
    
    return None


def run_gru4rec_eval(gru, test_data, cutoff, item_key, session_key, time_key, 
                     batch_size=512, eval_metrics=('recall_mrr', 'coverage', 'ild')):
    """Evaluate GRU4Rec using batch_eval. (Removed 'diversity' by default for memory efficiency)"""
    return evaluation.batch_eval(
        gru, test_data,
        cutoff=cutoff,
        batch_size=batch_size,
        mode='conservative',
        item_key=item_key,
        session_key=session_key,
        time_key=time_key,
        eval_metrics=eval_metrics
    )


def run_baseline_eval(baseline, test_data, cutoff, item_key='item_id', session_key='session_id'):
    """Evaluate a baseline using standard recall/MRR computation."""
    results = {c: {'recall': 0.0, 'mrr': 0.0} for c in cutoff}
    
    # Group by session
    sessions = test_data.groupby(session_key)
    n_sessions = len(sessions)
    
    for sid, group in sessions:
        session_items = group[item_key].tolist()
        if len(session_items) < 2:
            continue
        
        # Use all but last as input, last as target
        input_items = session_items[:-1]
        target_item = session_items[-1]
        
        # Get recommendations
        recs = baseline.recommend(input_items, topk=max(cutoff), exclude_seen=True)
        
        # Compute metrics
        for c in cutoff:
            topk_recs = recs[:c]
            if target_item in topk_recs:
                rank = topk_recs.index(target_item) + 1
                results[c]['recall'] += 1.0
                results[c]['mrr'] += 1.0 / rank
    
    # Average
    for c in cutoff:
        results[c]['recall'] /= n_sessions
        results[c]['mrr'] /= n_sessions
    
    return results


def format_results_table(gru_results, popular_results, lastitem_results, cutoff):
    """Format comparison table."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("BASELINE COMPARISON REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Recall/MRR comparison
    lines.append("RECALL and MRR Metrics:")
    lines.append("-" * 80)
    header = "Model".ljust(20)
    for c in cutoff:
        header += f"Recall@{c}".ljust(12) + f"MRR@{c}".ljust(12)
    lines.append(header)
    lines.append("-" * 80)
    
    # GRU4Rec
    line = "GRU4Rec".ljust(20)
    for c in cutoff:
        recall = gru_results['recall'].get(c, 0.0)
        mrr = gru_results['mrr'].get(c, 0.0)
        line += f"{recall:.4f}".ljust(12) + f"{mrr:.4f}".ljust(12)
    lines.append(line)
    
    # MostPopular
    line = "MostPopular".ljust(20)
    for c in cutoff:
        recall = popular_results[c]['recall']
        mrr = popular_results[c]['mrr']
        line += f"{recall:.4f}".ljust(12) + f"{mrr:.4f}".ljust(12)
    lines.append(line)
    
    # LastItem
    line = "LastItem".ljust(20)
    for c in cutoff:
        recall = lastitem_results[c]['recall']
        mrr = lastitem_results[c]['mrr']
        line += f"{recall:.4f}".ljust(12) + f"{mrr:.4f}".ljust(12)
    lines.append(line)
    
    lines.append("-" * 80)
    
    # Coverage and Diversity (GRU4Rec only, as baselines don't compute these)
    lines.append("")
    lines.append("DIVERSITY Metrics (GRU4Rec):")
    lines.append("-" * 80)
    if 'coverage' in gru_results:
        lines.append(f"Item Coverage:        {gru_results['coverage'].get('item_coverage', np.nan):.4f}")
        lines.append(f"Catalog Coverage:     {gru_results['coverage'].get('catalog_coverage', np.nan):.4f}")
    if 'ild' in gru_results:
        lines.append(f"Intra-List Diversity: {gru_results['ild']:.4f}")
    if 'diversity' in gru_results:
        lines.append(f"Aggregate Diversity:  {gru_results['diversity'].get('aggregate_diversity', np.nan):.4f}")
        lines.append(f"Inter-User Diversity: {gru_results['diversity'].get('inter_user_diversity', np.nan):.4f}")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)


def find_preprocessed_data():
    """Auto-detect preprocessed .dat files."""
    input_dir = os.path.join(os.path.dirname(__file__), 'input_data')
    
    datasets = {
        'retailrocket': os.path.join(input_dir, 'retailrocket-data'),
        'yoochoose': os.path.join(input_dir, 'yoochoose-data'),
    }
    
    available = {}
    for name, data_dir in datasets.items():
        train_file = os.path.join(data_dir, f'{name}_train_full.dat')
        test_file = os.path.join(data_dir, f'{name}_test.dat')
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            available[name] = (train_file, test_file)
    
    return available


def main():
    # Auto-detect datasets
    available_datasets = find_preprocessed_data()
    
    parser = argparse.ArgumentParser(
        description='Benchmark GRU4Rec vs baselines',
        epilog=f"Available preprocessed datasets: {', '.join(available_datasets.keys())}"
    )
    parser.add_argument('train_data', type=str, nargs='?', default=None, 
                       help='Path to training data (or dataset name: retailrocket/yoochoose)')
    parser.add_argument('test_data', type=str, nargs='?', default=None,
                       help='Path to test data')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained GRU4Rec model')
    parser.add_argument('--cutoff', type=int, nargs='+', default=[20], help='Cutoff values for metrics')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for evaluation')
    parser.add_argument('--test-samples', type=int, default=100000, help='Limit test set to N random sessions (default: 100000, None for all)')
    parser.add_argument('--full-eval', action='store_true', help='Enable full evaluation including inter-user diversity (memory intensive)')
    parser.add_argument('--item-key', type=str, default='item_id', help='Item ID column name')
    parser.add_argument('--session-key', type=str, default='session_id', help='Session ID column name')
    parser.add_argument('--time-key', type=str, default='timestamp', help='Timestamp column name')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda:0)')
    parser.add_argument('--output', type=str, default=None, help='Output file for report (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Handle dataset name shortcuts
    if args.train_data in available_datasets:
        train_file, test_file = available_datasets[args.train_data]
        args.train_data = train_file
        args.test_data = test_file
        print(f"Using preprocessed {args.train_data.split(os.sep)[-2]} dataset")
    elif args.train_data is None and len(available_datasets) == 1:
        # If only one dataset available, use it
        dataset_name = list(available_datasets.keys())[0]
        train_file, test_file = available_datasets[dataset_name]
        args.train_data = train_file
        args.test_data = test_file
        print(f"Using default preprocessed {dataset_name} dataset")
    elif args.train_data is None or args.test_data is None:
        print(f"Available preprocessed datasets: {', '.join(available_datasets.keys())}")
        print(f"\nUsage:")
        print(f"  python benchmark.py retailrocket")
        print(f"  python benchmark.py yoochoose")
        print(f"  python benchmark.py <path/to/train.dat> <path/to/test.dat>")
        parser.print_help()
        sys.exit(1)
    
    # Load data
    print("[1/4] Loading data...")
    train_data = load_data(args.train_data, args.item_key, args.session_key, args.time_key)
    test_data = load_data(args.test_data, args.item_key, args.session_key, args.time_key)
    print(f"  Train: {len(train_data)} events, {train_data[args.session_key].nunique()} sessions")
    print(f"  Test:  {len(test_data)} events, {test_data[args.session_key].nunique()} sessions")
    
    # Optionally sample test data for memory efficiency
    if args.test_samples is not None:
        unique_sessions = test_data[args.session_key].unique()
        if len(unique_sessions) > args.test_samples:
            sampled_sessions = np.random.choice(unique_sessions, args.test_samples, replace=False)
            test_data = test_data[test_data[args.session_key].isin(sampled_sessions)]
            print(f"  Sampled test to {args.test_samples} sessions ({len(test_data)} events)")
    
    # Load/find GRU4Rec model
    print("[2/4] Loading GRU4Rec model...")
    if args.model_path is None:
        # Try to find default model
        default_models = [
            os.path.join(os.path.dirname(__file__), 'output_data', 'save_model_new.pt'),
            os.path.join(os.path.dirname(__file__), 'output_data', 'save_model_test.pt'),
        ]
        for model in default_models:
            if os.path.isfile(model):
                args.model_path = model
                break
        if args.model_path is None:
            print("ERROR: No model path specified and no default model found")
            sys.exit(1)
    
    gru = GRU4Rec.loadmodel(args.model_path, device=args.device)
    print(f"  Loaded: {args.model_path}")
    print(f"  Model vocabulary size: {len(gru.data_iterator.itemidmap)}")
    print(f"  Model layers: {gru.layers}")
    
    # Extract embeddings for LastItem
    item_embeddings = get_item_embeddings_from_gru(gru)
    if item_embeddings is not None:
        print(f"  Extracted embeddings shape: {item_embeddings.shape}")
    else:
        print("  Warning: Could not extract embeddings for LastItem")
    
    # Train baselines
    print("[3/4] Training baselines...")
    most_popular = MostPopularBaseline(args.item_key, args.session_key)
    most_popular.fit(train_data)
    print(f"  MostPopular: {len(most_popular.popular_items)} items")
    
    last_item = LastItemBaseline(item_key=args.item_key, session_key=args.session_key)
    if item_embeddings is not None:
        # Create embedding dict from GRU itemidmap
        itemidmap = gru.data_iterator.itemidmap
        idx_to_item = pd.Series(itemidmap.index.values, index=itemidmap.values)
        emb_dict = {str(idx_to_item[i]): item_embeddings[i] for i in range(len(item_embeddings))}
        last_item.fit(train_data, item_embeddings=emb_dict)
    else:
        last_item.fit(train_data)
    print(f"  LastItem: ready (fallback to MostPopular if needed)")
    
    # Evaluate
    print("[4/4] Evaluating models...")
    
    # Determine eval metrics based on --full-eval flag
    eval_metrics = ('recall_mrr', 'coverage', 'ild')
    if args.full_eval:
        eval_metrics = ('recall_mrr', 'coverage', 'ild', 'diversity')
        print("  (Full evaluation with inter-user diversity)")
    else:
        print("  (Memory-efficient mode. Use --full-eval for complete diversity metrics)")
    
    print("  GRU4Rec...", end='', flush=True)
    gru_results = run_gru4rec_eval(gru, test_data, args.cutoff, args.item_key, 
                                   args.session_key, args.time_key, args.batch_size, eval_metrics)
    print(" done")
    
    print("  MostPopular...", end='', flush=True)
    popular_results = run_baseline_eval(most_popular, test_data, args.cutoff, args.item_key, args.session_key)
    print(" done")
    
    print("  LastItem...", end='', flush=True)
    lastitem_results = run_baseline_eval(last_item, test_data, args.cutoff, args.item_key, args.session_key)
    print(" done")
    
    # Format and output report
    report = format_results_table(gru_results, popular_results, lastitem_results, args.cutoff)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
