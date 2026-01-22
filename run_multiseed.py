#!/usr/bin/env python
"""
Script to run GRU4Rec with multiple random seeds for stability analysis
Usage: python run_multiseed.py [dataset] [num_runs]

Uses best parameter values from paramfiles/ directory by default.
"""

import subprocess
import sys
import os
import importlib.util

def load_params_from_file(param_file):
    """Load GRU4Rec parameters from a Python parameter file"""
    if not os.path.exists(param_file):
        return None
    
    spec = importlib.util.spec_from_file_location("params", param_file)
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)
    
    # Convert OrderedDict to comma-separated string
    params = params_module.gru4rec_params
    param_list = []
    for k, v in params.items():
        # Handle list values (e.g., layers=[224] → layers=224)
        if isinstance(v, list):
            v = '/'.join(str(x) for x in v)
        param_list.append(f"{k}={v}")
    return ','.join(param_list)

def run_with_seed(seed, dataset, config_params, output_prefix, output_dir, device='cpu', attention=False):
    """Run training with a specific random seed"""
    print(f"\n{'='*80}")
    print(f"RUN {seed}: Training with seed={seed} on device={device}")
    print(f"{'='*80}\n")
    
    # Extract base name from dataset (retailrocket-data → retailrocket, yoochoose-data → yoochoose)
    dataset_base = dataset.replace('-data', '')
    
    cmd = [
        'python', 'run.py',
        f'input_data/{dataset}/{dataset_base}_train_full.dat',
        '-ps', config_params,
        '-t', f'input_data/{dataset}/{dataset_base}_test.dat',
        '-m', '1', '5', '10', '20',
        '-d', device,
        '-s', f'{output_dir}/{output_prefix}_seed{seed}.pt',
        '--seed', str(seed)
    ]
    if attention:
        cmd.append('--attention')
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0

def main():
    """Run experiments with multiple seeds using best parameters"""
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'retailrocket-data'
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    attention = '--attention' in sys.argv
    
    # Detect if running on Kaggle and set output directory
    if os.path.exists('/kaggle/working/'):
        output_dir = '/kaggle/working/output_data'
    else:
        output_dir = 'output_data'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Map datasets to their best parameter files
    param_file_map = {
        'retailrocket-data': 'paramfiles/retailrocket_bprmax_shared_best.py',
        # 'yoochoose-data': 'paramfiles/yoochoose_xe_tuned_fast.py',
        'yoochoose-data': 'paramfiles/yoochoose_xe_shared_best.py',
    }
    
    if dataset not in param_file_map:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(param_file_map.keys())}")
        sys.exit(1)
    
    # Load best parameters from file
    param_file = param_file_map[dataset]
    config_params = load_params_from_file(param_file)
    
    if config_params is None:
        print(f"ERROR: Parameter file not found: {param_file}")
        print(f"Expected location: {os.path.join(os.path.dirname(__file__), param_file)}")
        sys.exit(1)
    
    seeds = [42, 123, 456, 789, 999][:num_runs]
    dataset_short = dataset.split('-')[0]
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  MULTI-SEED EVALUATION FOR REPRODUCIBILITY                   ║
╚════════════════════════════════════════════════════════════════╝

Dataset:        {dataset}
Number of runs: {num_runs}
Seeds:          {seeds}
Parameter file: {param_file}

Configuration Parameters:
{config_params}
""")
    
    results = {}
    successful_runs = 0
    
    for seed in seeds:
        output_prefix = f'best_{dataset_short}' + ('_attn' if attention else '')
        success = run_with_seed(seed, dataset, config_params, output_prefix, output_dir, device=device, attention=attention)
        results[seed] = success
        if success:
            successful_runs += 1
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successful runs: {successful_runs}/{num_runs}")
    for seed, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Seed {seed}: {status}")
    
    print("\nModels saved to:")
    for seed in seeds:
        print(f"  {output_dir}/best_{dataset_short}_seed{seed}.pt")
    
    print("\nNext step: Extract metrics from output above")
    print("          python -c \"import numpy as np; recalls=[...]; print(f'Recall@20: {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f}')\"")

if __name__ == '__main__':
    main()
