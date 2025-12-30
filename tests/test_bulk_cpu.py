#!/usr/bin/env python3
"""
Test Script: CPU Bulk RNA-seq Inference Validation

Validates SecActPy bulk RNA-seq inference against RidgeR output.

Dataset: Ly86-Fc_vs_Vehicle_logFC.txt
Reference: R output from RidgeR::SecAct.activity.inference

Usage:
    python tests/test_bulk_cpu.py
    python tests/test_bulk_cpu.py --save
    python tests/test_bulk_cpu.py --resample 1000  # Test with 1000 resampled samples
    python tests/test_bulk_cpu.py data.csv --gene-col 0
    python tests/test_bulk_cpu.py data.tsv --no-validate

Expected output:
    All arrays should match R output exactly (or within numerical tolerance).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time

# Add package to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy import secact_activity_inference, load_expression_data


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "Ly86-Fc_vs_Vehicle_logFC.txt"
OUTPUT_DIR = DATA_DIR / "output" / "signature" / "bulk"

# Parameters matching R defaults
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9


# =============================================================================
# Resampling Function
# =============================================================================

def resample_expression(Y: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Resample expression data to create more samples for benchmarking.
    
    Creates n_samples by resampling columns with replacement and adding
    small random noise to create variation.
    
    Parameters
    ----------
    Y : pd.DataFrame
        Expression data (genes × samples)
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Resampled expression data (genes × n_samples)
    """
    np.random.seed(seed)
    
    n_genes = Y.shape[0]
    n_original = Y.shape[1]
    
    # Resample columns with replacement
    sample_indices = np.random.choice(n_original, size=n_samples, replace=True)
    
    # Create resampled data
    resampled = Y.iloc[:, sample_indices].copy()
    
    # Add small random noise to create variation (scale by data range)
    data_std = np.std(Y.values)
    noise = np.random.normal(0, data_std * 0.01, size=(n_genes, n_samples))
    resampled_values = resampled.values + noise
    
    # Create new column names
    new_columns = [f"Sample_{i+1}" for i in range(n_samples)]
    
    return pd.DataFrame(resampled_values, index=Y.index, columns=new_columns)


# =============================================================================
# Comparison Functions
# =============================================================================

def load_r_output(output_dir: Path) -> dict:
    """Load R output files."""
    result = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        filepath = output_dir / f"{name}.txt"
        if filepath.exists():
            df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
            result[name] = df
            print(f"  Loaded {name}: {df.shape}")
        else:
            print(f"  Warning: {name}.txt not found")
    
    return result


def compare_results(py_result: dict, r_result: dict, tolerance: float = 1e-10) -> dict:
    """Compare Python and R results."""
    comparison = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name not in py_result or name not in r_result:
            comparison[name] = {'status': 'MISSING', 'message': 'Array not found'}
            continue
        
        py_arr = py_result[name]
        r_arr = r_result[name]
        
        # Check shape
        if py_arr.shape != r_arr.shape:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Shape mismatch: Python {py_arr.shape} vs R {r_arr.shape}'
            }
            continue
        
        # Align by row names
        common_rows = py_arr.index.intersection(r_arr.index)
        if len(common_rows) != len(py_arr):
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row name mismatch. Common: {len(common_rows)}, Python: {len(py_arr)}'
            }
            continue
        
        py_aligned = py_arr.loc[common_rows]
        r_aligned = r_arr.loc[common_rows]
        
        # Calculate difference
        diff = np.abs(py_aligned.values - r_aligned.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        
        if max_diff <= tolerance:
            comparison[name] = {
                'status': 'PASS',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'message': f'Max diff: {max_diff:.2e}'
            }
        else:
            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'message': f'Max diff: {max_diff:.2e} (tolerance: {tolerance:.2e})'
            }
    
    return comparison


# =============================================================================
# Main Test
# =============================================================================

def main(input_file=None, gene_col=None, validate=True, save_output=False, resample=None):
    """
    Run bulk RNA-seq inference.
    
    Parameters
    ----------
    input_file : str, optional
        Path to input expression file. If None, uses default test file.
    gene_col : int or str, optional
        Column containing gene symbols (if not row index).
    validate : bool, default=True
        If True, compare against R reference output.
    save_output : bool, default=False
        If True, save results to HDF5 file.
    resample : int, optional
        If provided, resample to this many samples for benchmarking.
    """
    print("=" * 70)
    print("SecActPy Bulk RNA-seq Validation Test (CPU)")
    print("=" * 70)
    
    # Determine input file
    if input_file is not None:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            return False
        validate = False  # Custom file, no R reference
    else:
        input_path = INPUT_FILE
        if not input_path.exists():
            print(f"ERROR: Default input file not found: {input_path}")
            return False
    
    # Check files
    print("\n1. Checking files...")
    print(f"   Input: {input_path}")
    
    # Disable validation if resampling (different sample count)
    if resample is not None:
        validate = False
        print(f"   Resampling: {resample} samples (validation disabled)")
    
    if validate:
        if not OUTPUT_DIR.exists():
            print(f"   Warning: Reference output not found: {OUTPUT_DIR}")
            print("   Skipping validation.")
            validate = False
        else:
            print(f"   Reference: {OUTPUT_DIR}")
    
    # Load data
    print("\n2. Loading input data...")
    Y = load_expression_data(input_path, gene_col=gene_col)
    print(f"   Original expression data: {Y.shape} (genes × samples)")
    print(f"   Original samples: {list(Y.columns)}")
    
    # Resample if requested
    if resample is not None:
        print(f"\n   Resampling to {resample} samples...")
        Y = resample_expression(Y, n_samples=resample, seed=42)
        print(f"   Resampled expression data: {Y.shape} (genes × samples)")
        print(f"   Memory: {Y.values.nbytes / 1e6:.1f} MB")
    
    # Run inference
    print("\n3. Running SecActPy inference (CPU)...")
    start_time = time.time()
    
    try:
        py_result = secact_activity_inference(
            input_profile=Y,
            is_differential=True,
            sig_matrix="secact",
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            backend="numpy",
            use_cache=True,  # Cache permutation tables for faster repeated runs
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f} seconds")
        print(f"   Result shape: {py_result['beta'].shape}")
        
        # Performance stats
        n_samples = Y.shape[1]
        print(f"   Throughput: {n_samples / elapsed:.0f} samples/sec")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    all_passed = True
    
    if validate:
        # Load R reference
        print("\n4. Loading R reference output...")
        r_result = load_r_output(OUTPUT_DIR)
        
        if not r_result:
            print("   Warning: No R output files found!")
            validate = False
        else:
            # Compare
            print("\n5. Comparing results...")
            comparison = compare_results(py_result, r_result)
            
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            
            for name, result in comparison.items():
                status = result['status']
                message = result['message']
                
                if status == 'PASS':
                    print(f"  {name:8s}: ✓ PASS - {message}")
                else:
                    print(f"  {name:8s}: ✗ {status} - {message}")
                    all_passed = False
            
            print("\n" + "=" * 70)
            if all_passed:
                print("ALL TESTS PASSED! ✓")
                print("SecActPy bulk produces identical results to RidgeR.")
            else:
                print("SOME TESTS FAILED! ✗")
                print("Check the detailed output above for discrepancies.")
            print("=" * 70)
    
    if not validate:
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE (validation skipped)")
        print("=" * 70)
    
    # Show sample output
    step_num = 6 if validate else 4
    print(f"\n{step_num}. Sample output (first 10 rows of zscore):")
    print(py_result['zscore'].head(10))
    
    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_results
            
            output_h5 = PACKAGE_ROOT / "dataset" / "output" / "bulk_cpu_activity.h5"
            results_to_save = {
                'beta': py_result['beta'].values,
                'se': py_result['se'].values,
                'zscore': py_result['zscore'].values,
                'pvalue': py_result['pvalue'].values,
                'feature_names': list(py_result['beta'].index),
                'sample_names': list(py_result['beta'].columns),
            }
            save_results(results_to_save, output_h5)
            print(f"   Saved to: {output_h5}")
        except Exception as e:
            print(f"   Could not save: {e}")
    
    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy Bulk RNA-seq Inference (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test file and validate against R
  python tests/test_bulk_cpu.py
  
  # Run with resampled data (1000 samples) for benchmarking
  python tests/test_bulk_cpu.py --resample 1000
  
  # Run with custom CSV file
  python tests/test_bulk_cpu.py data.csv --gene-col 0
  
  # Run without validation and save results
  python tests/test_bulk_cpu.py --no-validate --save
        """
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default=None,
        help='Path to expression data file (default: test dataset)'
    )
    parser.add_argument(
        '--gene-col',
        type=str,
        default=None,
        help='Column containing gene symbols (name or index)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation against R reference output'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to HDF5 file'
    )
    parser.add_argument(
        '--resample',
        type=int,
        default=None,
        metavar='N',
        help='Resample to N samples for benchmarking (e.g., --resample 1000)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    gene_col = args.gene_col
    if gene_col is not None:
        try:
            gene_col = int(gene_col)
        except ValueError:
            pass
    
    success = main(
        input_file=args.input_file,
        gene_col=gene_col,
        validate=not args.no_validate,
        save_output=args.save,
        resample=args.resample
    )
    sys.exit(0 if success else 1)
