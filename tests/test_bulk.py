#!/usr/bin/env python3
"""
Test Script: Bulk RNA-seq Validation

Validates SecActPy against RidgeR output for bulk RNA-seq data.

Dataset: Ly86-Fc_vs_Vehicle_logFC.txt (differential expression)
Reference: R output from RidgeR::SecAct.activity.inference with method="legacy"

Supports multiple input formats:
- Gene symbols as row names, single column with log-FC
- First column as gene symbols, remaining columns as samples
- TSV, CSV, or space-separated files

Usage:
    python tests/test_bulk.py
    python tests/test_bulk.py path/to/expression.csv
    python tests/test_bulk.py path/to/expression.tsv --gene-col 0

Expected output:
    All arrays should match R output exactly (or within numerical tolerance).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Add package to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from secactpy import secact_activity_inference, load_expression_data


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

def main(input_file=None, gene_col=None, validate=True, save_output=False):
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
    """
    print("=" * 70)
    print("SecActPy Bulk RNA-seq Validation Test")
    print("=" * 70)
    
    # Determine input file
    if input_file is None:
        input_path = INPUT_FILE
    else:
        input_path = Path(input_file)
    
    # Check files exist
    print("\n1. Checking files...")
    if not input_path.exists():
        print(f"   ERROR: Input file not found: {input_path}")
        print("   Please ensure the test data is in place.")
        return False
    print(f"   Input: {input_path}")
    
    if validate and OUTPUT_DIR.exists():
        print(f"   Reference outputs: {OUTPUT_DIR}")
    elif validate:
        print(f"   Warning: Reference output directory not found: {OUTPUT_DIR}")
        print("   Will run inference but skip validation.")
        validate = False
    
    # Load input data using flexible loader
    print("\n2. Loading input data...")
    try:
        Y = load_expression_data(input_path, gene_col=gene_col)
        print(f"   Expression data: {Y.shape} (genes × samples)")
        print(f"   Sample names: {Y.columns.tolist()[:5]}{'...' if len(Y.columns) > 5 else ''}")
        print(f"   First 5 genes: {Y.index[:5].tolist()}")
        
        # Show detected format
        suffix = input_path.suffix.lower()
        print(f"   File format: {suffix}")
    except Exception as e:
        print(f"   ERROR loading file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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
    
    all_passed = True
    
    if validate:
        # Load R reference output
        print("\n4. Loading R reference output...")
        r_result = load_r_output(OUTPUT_DIR)
        
        if not r_result:
            print("   Warning: No R output files found! Skipping validation.")
            validate = False
        else:
            # Compare results
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
                print("SecActPy produces identical results to RidgeR.")
            else:
                print("SOME TESTS FAILED! ✗")
                print("Check the detailed output above for discrepancies.")
            print("=" * 70)
    
    if not validate:
        print("\n" + "=" * 70)
        print("INFERENCE COMPLETE")
        print("=" * 70)
    
    # Show sample output
    step_num = 6 if validate else 4
    print(f"\n{step_num}. Sample output (first 10 rows of zscore):")
    print(py_result['zscore'].head(10))
    
    # Optional: Save results to h5
    if save_output:
        step_num = 7 if validate else 5
        print(f"\n{step_num}. Saving results to HDF5...")
        try:
            from secactpy.io import save_results
            
            output_h5 = PACKAGE_ROOT / "dataset" / "output" / "bulk_with_activity.h5"
            
            # Convert DataFrames to arrays for save_results
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
        description="SecActPy Bulk RNA-seq Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test file and validate
  python tests/test_bulk.py
  
  # Run with custom CSV file (genes in first column)
  python tests/test_bulk.py data.csv --gene-col 0
  
  # Run with custom TSV file (genes as row names)
  python tests/test_bulk.py data.tsv
  
  # Run without validation
  python tests/test_bulk.py data.csv --no-validate

Supported file formats:
  - CSV (comma-separated)
  - TSV (tab-separated)
  - TXT (space or tab-separated)

Input file structure:
  1. Gene symbols as row names (index):
     Gene    Sample1  Sample2
     GENE1   1.5      2.3
     GENE2   0.8      1.2
  
  2. Gene symbols in first column:
     GeneSymbol,Sample1,Sample2
     GENE1,1.5,2.3
     GENE2,0.8,1.2
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
        help='Column containing gene symbols (name or index). '
             'If not specified, assumes genes are row names.'
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Convert gene_col to int if it's a number string
    gene_col = args.gene_col
    if gene_col is not None:
        try:
            gene_col = int(gene_col)
        except ValueError:
            pass  # Keep as string (column name)
    
    success = main(
        input_file=args.input_file,
        gene_col=gene_col,
        validate=not args.no_validate,
        save_output=args.save
    )
    sys.exit(0 if success else 1)
