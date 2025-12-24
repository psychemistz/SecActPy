#!/usr/bin/env python3
"""
Test Script: Cell-Type Resolution scRNAseq Validation (Pseudo-bulk)

Validates SecActPy scRNAseq inference at cell-type resolution (pseudo-bulk).

Dataset: OV_scRNAseq_CD4.h5ad (converted from Seurat RDS)
Reference: R output from RidgeR::SecAct.activity.inference.scRNAseq with is_single_cell_level=FALSE

Usage:
    python tests/test_scrnaseq_ct_res.py

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

from secactpy import secact_activity_inference_scrnaseq


# =============================================================================
# Configuration
# =============================================================================

# Paths (adjust as needed)
PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "OV_scRNAseq_CD4.h5ad"
OUTPUT_DIR = DATA_DIR / "output" / "signature" / "scRNAseq_ct_res"

# Parameters matching R defaults
LAMBDA = 5e5
NRAND = 1000
SEED = 0
GROUP_COR = 0.9
CELL_TYPE_COL = "Annotation"  # Match R's cellType_meta


# =============================================================================
# Comparison Functions
# =============================================================================

def load_r_output(output_dir: Path) -> dict:
    """Load R output files."""
    result = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        # Try different file names
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
        
        # Check column names
        py_cols = set(py_arr.columns)
        r_cols = set(r_arr.columns)
        if py_cols != r_cols:
            missing_in_py = r_cols - py_cols
            missing_in_r = py_cols - r_cols
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Column name mismatch. Missing in Python: {len(missing_in_py)}, Missing in R: {len(missing_in_r)}'
            }
            continue
        
        # Align by row and column names
        py_aligned = py_arr.loc[r_arr.index, r_arr.columns]
        
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

def main(save_output=False):
    """
    Run cell-type resolution scRNAseq inference.
    
    Parameters
    ----------
    save_output : bool, default=False
        If True, save results to HDF5 file.
    """
    print("=" * 70)
    print("SecActPy scRNAseq Cell-Type Resolution (Pseudo-bulk) Validation Test")
    print("=" * 70)
    
    # Check if anndata is available
    try:
        import anndata as ad
    except ImportError:
        print("ERROR: anndata is required for this test.")
        print("Install with: pip install anndata")
        return False
    
    # Check files exist
    print("\n1. Checking files...")
    if not INPUT_FILE.exists():
        print(f"   ERROR: Input file not found: {INPUT_FILE}")
        print("   Please ensure the test data is in place.")
        return False
    print(f"   Input: {INPUT_FILE}")
    
    if not OUTPUT_DIR.exists():
        print(f"   ERROR: Output directory not found: {OUTPUT_DIR}")
        print("   Please run the R script to generate reference output first.")
        return False
    print(f"   Reference outputs: {OUTPUT_DIR}")
    
    # Load AnnData
    print("\n2. Loading AnnData...")
    adata = ad.read_h5ad(INPUT_FILE)
    print(f"   Shape: {adata.shape} (cells × genes)")
    print(f"   Obs columns: {list(adata.obs.columns)}")
    
    if CELL_TYPE_COL not in adata.obs.columns:
        print(f"   ERROR: Cell type column '{CELL_TYPE_COL}' not found!")
        print(f"   Available columns: {list(adata.obs.columns)}")
        return False
    
    cell_types = adata.obs[CELL_TYPE_COL].unique()
    print(f"   Cell types ({len(cell_types)}): {list(cell_types)[:5]}...")
    
    # Check raw counts
    if adata.raw is not None:
        print(f"   Raw counts available: {adata.raw.X.shape}")
    else:
        print("   Using adata.X (no raw layer)")
    
    # Run Python inference
    print("\n3. Running SecActPy scRNAseq inference...")
    try:
        py_result = secact_activity_inference_scrnaseq(
            adata=adata,
            cell_type_col=CELL_TYPE_COL,
            sig_matrix="secact",
            is_single_cell_level=False,  # Pseudo-bulk by cell type
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            verbose=True
        )
        print(f"   Result shape: {py_result['beta'].shape}")
        print(f"   Cell types in result: {list(py_result['beta'].columns)}")
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
    
    # Show R column names for comparison
    if 'zscore' in r_result:
        print(f"   R cell types: {list(r_result['zscore'].columns)}")
    
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
        print("SecActPy scRNAseq (cell-type resolution) produces identical results to RidgeR.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Check the detailed output above for discrepancies.")
    print("=" * 70)
    
    # Show sample output
    print("\n6. Sample output (first 5 rows, first 3 columns of zscore):")
    cols = py_result['zscore'].columns[:3]
    print(py_result['zscore'].iloc[:5][cols])
    
    # Optional: Save results
    if save_output:
        print("\n7. Saving results...")
        try:
            from secactpy.io import save_results
            
            output_h5 = PACKAGE_ROOT / "dataset" / "output" / "scRNAseq_ct_res_activity.h5"
            
            # For pseudo-bulk analysis, activity results are per cell-type, not per cell
            # So we save just the activity results (not combined with cell-level counts)
            results_to_save = {
                'beta': py_result['beta'].values,
                'se': py_result['se'].values,
                'zscore': py_result['zscore'].values,
                'pvalue': py_result['pvalue'].values,
                'feature_names': list(py_result['beta'].index),  # proteins
                'sample_names': list(py_result['beta'].columns),  # cell types
            }
            
            save_results(results_to_save, output_h5)
            print(f"   Saved activity results to: {output_h5}")
            print(f"   Shape: {py_result['beta'].shape} (proteins × cell_types)")
            
            # Also save as CSV for easy viewing
            csv_path = PACKAGE_ROOT / "dataset" / "output" / "scRNAseq_ct_res_zscore.csv"
            py_result['zscore'].to_csv(csv_path)
            print(f"   Saved z-scores to: {csv_path}")
            
        except Exception as e:
            print(f"   Could not save results: {e}")
    
    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy scRNAseq Cell-Type Resolution (Pseudo-bulk) Validation"
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to HDF5 and CSV files'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(save_output=args.save)
    sys.exit(0 if success else 1)
