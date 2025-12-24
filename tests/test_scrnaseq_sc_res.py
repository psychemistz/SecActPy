#!/usr/bin/env python3
"""
Test Script: Single-Cell Resolution scRNAseq Validation

Validates SecActPy scRNAseq inference at single-cell resolution (per cell).

Dataset: OV_scRNAseq_CD4.h5ad
Reference: R output from RidgeR::SecAct.activity.inference.scRNAseq with is_single_cell_level=TRUE

Usage:
    python tests/test_scrnaseq_sc_res.py

Expected output:
    All arrays should match R output exactly (or within numerical tolerance).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
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
OUTPUT_DIR = DATA_DIR / "output" / "signature" / "scRNAseq_sc_res"

# Parameters matching R defaults
CELL_TYPE_COL = "Annotation"  # Match R's cellType_meta
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
        
        # Check row names (proteins)
        py_rows = set(py_arr.index)
        r_rows = set(r_arr.index)
        if py_rows != r_rows:
            missing_in_py = r_rows - py_rows
            missing_in_r = py_rows - r_rows
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row name mismatch. Missing in Python: {len(missing_in_py)}, Missing in R: {len(missing_in_r)}'
            }
            continue
        
        # Check column names (cells)
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
    Run single-cell resolution scRNAseq inference.
    
    Parameters
    ----------
    save_output : bool, default=False
        If True, save results to h5ad and HDF5 files.
    """
    print("=" * 70)
    print("SecActPy scRNAseq Single-Cell Resolution Validation Test")
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
        return False
    print(f"   Input: {INPUT_FILE}")
    
    if not OUTPUT_DIR.exists():
        print(f"   Warning: Reference output directory not found: {OUTPUT_DIR}")
        print("   Will run inference but skip validation.")
        print("   To generate R reference, run:")
        print("   ```R")
        print("   library(RidgeR)")
        print("   Seurat_obj <- readRDS('OV_scRNAseq_CD4.rds')")
        print("   Seurat_obj <- SecAct.activity.inference.scRNAseq(Seurat_obj, cellType_meta='Annotation', is_single_cell_level=TRUE)")
        print("   dir.create('dataset/output/signature/scRNAseq_sc_res', showWarnings=FALSE)")
        print("   write.table(Seurat_obj@misc$SecAct_output$SecretedProteinActivity$beta, 'dataset/output/signature/scRNAseq_sc_res/beta.txt', quote=F)")
        print("   # ... same for se, zscore, pvalue")
        print("   ```")
        validate = False
    else:
        print(f"   Reference outputs: {OUTPUT_DIR}")
        validate = True
    
    # Load AnnData
    print("\n2. Loading AnnData...")
    adata = ad.read_h5ad(INPUT_FILE)
    print(f"   Shape: {adata.shape} (cells × genes)")
    print(f"   Cell types: {adata.obs[CELL_TYPE_COL].nunique()}")
    print(f"   Cells: {adata.n_obs}")
    
    # Run single-cell level inference
    print("\n3. Running SecActPy Single-Cell Resolution inference...")
    start_time = time.time()
    
    try:
        py_result = secact_activity_inference_scrnaseq(
            adata,
            cell_type_col=CELL_TYPE_COL,
            is_single_cell_level=True,  # Single-cell resolution
            sig_matrix="secact",
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.1f} seconds")
        print(f"   Result shape: {py_result['beta'].shape} (proteins × cells)")
        print(f"   Sample cell IDs: {list(py_result['beta'].columns)[:5]}")
        
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
            # Show R column names for comparison
            if 'zscore' in r_result:
                print(f"   R sample cells: {list(r_result['zscore'].columns)[:5]}")
            
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
                print("SecActPy scRNAseq (single-cell resolution) produces identical results to RidgeR.")
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
    print(f"\n{step_num}. Sample output (first 5 proteins, first 5 cells):")
    print(py_result['zscore'].iloc[:5, :5])
    
    # Activity statistics by cell type
    step_num += 1
    print(f"\n{step_num}. Activity statistics by cell type:")
    cell_types = adata.obs[CELL_TYPE_COL].values
    cell_names = list(adata.obs_names)
    
    for ct in sorted(set(cell_types)):
        mask = cell_types == ct
        ct_cells = [c for c, m in zip(cell_names, mask) if m]
        ct_data = py_result['zscore'][ct_cells]
        
        mean_activity = ct_data.mean(axis=1).sort_values(ascending=False)
        
        print(f"\n   {ct} ({len(ct_cells)} cells):")
        print(f"     Top 3 active: {', '.join(mean_activity.head(3).index)}")
        print(f"     Z-score range: [{mean_activity.min():.2f}, {mean_activity.max():.2f}]")
    
    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import add_activity_to_anndata, save_results
            
            # Add activity to AnnData
            adata = add_activity_to_anndata(adata, py_result)
            
            # Save h5ad
            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / "scRNAseq_sc_res_activity.h5ad"
            adata.write_h5ad(output_h5ad)
            print(f"   Saved h5ad to: {output_h5ad}")
            
            # Also save as HDF5 for easy loading
            output_h5 = PACKAGE_ROOT / "dataset" / "output" / "scRNAseq_sc_res_activity.h5"
            results_to_save = {
                'beta': py_result['beta'].values,
                'se': py_result['se'].values,
                'zscore': py_result['zscore'].values,
                'pvalue': py_result['pvalue'].values,
                'feature_names': list(py_result['beta'].index),
                'sample_names': list(py_result['beta'].columns),
            }
            save_results(results_to_save, output_h5)
            print(f"   Saved HDF5 to: {output_h5}")
            
        except Exception as e:
            print(f"   Could not save results: {e}")
    
    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy scRNAseq Single-Cell Resolution Validation"
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to h5ad and HDF5 files'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(save_output=args.save)
    sys.exit(0 if success else 1)
