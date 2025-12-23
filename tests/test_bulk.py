#!/usr/bin/env python3
"""
Test Script: Bulk RNA-seq Validation

Validates SecActPy against RidgeR output for bulk RNA-seq data.

Dataset: Ly86-Fc_vs_Vehicle_logFC.txt (differential expression)
Reference: R output from RidgeR::SecAct.activity.inference with method="legacy"

Usage:
    python tests/test_bulk.py

Expected output:
    All arrays should match R output exactly (or within numerical tolerance).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add package to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy import secact_activity_inference


# =============================================================================
# Configuration
# =============================================================================

# Paths (adjust as needed)
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
# Comparison Functions
# =============================================================================

def load_r_output(output_dir: Path) -> dict:
    """Load R output files."""
    result = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        # Try different file names (pval.txt vs pvalue.txt)
        for filename in [f"{name}.txt", "pval.txt" if name == "pvalue" else None]:
            if filename is None:
                continue
            filepath = output_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep=r'\s+', index_col=0)
                result[name] = df
                print(f"  Loaded {name}: {df.shape}")
                break
        
        if name not in result:
            print(f"  Warning: {name}.txt not found")
    
    return result


def compare_results(py_result: dict, r_result: dict, tolerance: float = 1e-10) -> dict:
    """
    Compare Python and R results.
    
    Returns
    -------
    dict
        Comparison results for each array
    """
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
        
        # Check row names
        py_rows = set(py_arr.index)
        r_rows = set(r_arr.index)
        if py_rows != r_rows:
            missing_in_py = r_rows - py_rows
            missing_in_r = py_rows - r_rows
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row name mismatch. Missing in Python: {len(missing_in_py)}, Missing in R: {len(missing_in_r)}'
            }
            if len(missing_in_py) <= 5:
                print(f"    Missing in Python: {missing_in_py}")
            if len(missing_in_r) <= 5:
                print(f"    Missing in R: {missing_in_r}")
            continue
        
        # Align by row names
        py_aligned = py_arr.loc[r_arr.index]
        
        # Calculate difference
        diff = np.abs(py_aligned.values - r_arr.values)
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
            # Find location of max difference
            max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
            row_name = py_aligned.index[max_idx[0]]
            col_name = py_aligned.columns[max_idx[1]]
            py_val = py_aligned.iloc[max_idx[0], max_idx[1]]
            r_val = r_arr.iloc[max_idx[0], max_idx[1]]
            
            comparison[name] = {
                'status': 'FAIL',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'location': f'{row_name}, {col_name}',
                'py_value': py_val,
                'r_value': r_val,
                'message': f'Max diff: {max_diff:.2e} at ({row_name}, {col_name})'
            }
    
    return comparison


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("SecActPy Bulk RNA-seq Validation Test")
    print("=" * 70)
    
    # Check files exist
    print("\n1. Checking files...")
    if not INPUT_FILE.exists():
        print(f"   ERROR: Input file not found: {INPUT_FILE}")
        print("   Please ensure the test data is in place.")
        return False
    print(f"   Input: {INPUT_FILE}")
    
    if not OUTPUT_DIR.exists():
        print(f"   ERROR: Output directory not found: {OUTPUT_DIR}")
        return False
    print(f"   Reference outputs: {OUTPUT_DIR}")
    
    # Load input data
    print("\n2. Loading input data...")
    # The file is space-separated (R's write.table default)
    Y = pd.read_csv(INPUT_FILE, sep=r'\s+', index_col=0)
    print(f"   Expression data: {Y.shape} (genes × samples)")
    print(f"   Sample names: {Y.columns.tolist()}")
    print(f"   First 5 genes: {Y.index[:5].tolist()}")
    
    # Run Python inference
    print("\n3. Running SecActPy inference...")
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
            verbose=True
        )
        print(f"   Result shape: {py_result['beta'].shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load R reference output
    print("\n4. Loading R reference output...")
    r_result = load_r_output(OUTPUT_DIR)
    
    if not r_result:
        print("   ERROR: No R output files found!")
        return False
    
    # Compare results
    print("\n5. Comparing results...")
    comparison = compare_results(py_result, r_result)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    all_passed = True
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
        print("SecActPy produces identical results to RidgeR.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Check the detailed output above for discrepancies.")
    print("=" * 70)
    
    # Show sample output
    print("\n6. Sample output (first 10 rows of zscore):")
    print(py_result['zscore'].head(10))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
