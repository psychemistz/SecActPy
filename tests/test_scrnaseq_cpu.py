#!/usr/bin/env python3
"""
Test Script: CPU scRNA-seq Inference Validation

Validates SecActPy scRNA-seq inference against RidgeR output.
Tests both cell-type resolution (pseudo-bulk) and single-cell resolution.

Dataset: OV_scRNAseq_CD4.h5ad
Reference: R output from RidgeR::SecAct.activity.inference.scRNAseq

Usage:
    python tests/test_scrnaseq_cpu.py                  # Cell-type resolution
    python tests/test_scrnaseq_cpu.py --single-cell    # Single-cell resolution
    python tests/test_scrnaseq_cpu.py --save

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

from secactpy import secact_activity_inference_scrnaseq


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"
INPUT_FILE = DATA_DIR / "input" / "OV_scRNAseq_CD4.h5ad"

# Reference output directories
OUTPUT_DIR_CT = DATA_DIR / "output" / "signature" / "scRNAseq_ct_res"
OUTPUT_DIR_SC = DATA_DIR / "output" / "signature" / "scRNAseq_sc_res"

# Parameters matching R defaults
CELL_TYPE_COL = "Annotation"
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
        
        # Check row names (proteins)
        py_rows = set(py_arr.index)
        r_rows = set(r_arr.index)
        if py_rows != r_rows:
            missing_in_py = len(r_rows - py_rows)
            missing_in_r = len(py_rows - r_rows)
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Row mismatch. Missing in Py: {missing_in_py}, Missing in R: {missing_in_r}'
            }
            continue
        
        # Check column names
        py_cols = set(py_arr.columns)
        r_cols = set(r_arr.columns)
        if py_cols != r_cols:
            missing_in_py = len(r_cols - py_cols)
            missing_in_r = len(py_cols - r_cols)
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Column mismatch. Missing in Py: {missing_in_py}, Missing in R: {missing_in_r}'
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

def main(single_cell=False, save_output=False):
    """
    Run scRNA-seq inference validation.
    
    Parameters
    ----------
    single_cell : bool, default=False
        If True, run single-cell resolution. If False, run cell-type resolution.
    save_output : bool, default=False
        If True, save results to files.
    """
    resolution = "Single-Cell" if single_cell else "Cell-Type"
    output_dir = OUTPUT_DIR_SC if single_cell else OUTPUT_DIR_CT
    
    print("=" * 70)
    print(f"SecActPy scRNA-seq Validation Test (CPU) - {resolution} Resolution")
    print("=" * 70)
    
    # Check anndata
    try:
        import anndata as ad
    except ImportError:
        print("ERROR: anndata is required for this test.")
        print("Install with: pip install anndata")
        return False
    
    # Check files
    print("\n1. Checking files...")
    if not INPUT_FILE.exists():
        print(f"   ERROR: Input file not found: {INPUT_FILE}")
        return False
    print(f"   Input: {INPUT_FILE}")
    
    validate = True
    if not output_dir.exists():
        print(f"   Warning: Reference output not found: {output_dir}")
        print("   Will run inference but skip validation.")
        print("   To generate R reference, run:")
        print("   ```R")
        print("   library(RidgeR)")
        print("   Seurat_obj <- readRDS('OV_scRNAseq_CD4.rds')")
        if single_cell:
            print("   Seurat_obj <- SecAct.activity.inference.scRNAseq(Seurat_obj, cellType_meta='Annotation', is_single_cell_level=TRUE)")
            print("   dir.create('dataset/output/signature/scRNAseq_sc_res', showWarnings=FALSE)")
            print("   # Save outputs to scRNAseq_sc_res/")
        else:
            print("   Seurat_obj <- SecAct.activity.inference.scRNAseq(Seurat_obj, cellType_meta='Annotation')")
            print("   dir.create('dataset/output/signature/scRNAseq_ct_res', showWarnings=FALSE)")
            print("   # Save outputs to scRNAseq_ct_res/")
        print("   ```")
        validate = False
    else:
        print(f"   Reference: {output_dir}")
    
    # Load data
    print("\n2. Loading AnnData...")
    adata = ad.read_h5ad(INPUT_FILE)
    print(f"   Shape: {adata.shape} (cells × genes)")
    print(f"   Cell types: {adata.obs[CELL_TYPE_COL].nunique()}")
    print(f"   Cells: {adata.n_obs}")
    
    # Run inference
    print(f"\n3. Running SecActPy inference (CPU, {resolution})...")
    start_time = time.time()
    
    try:
        py_result = secact_activity_inference_scrnaseq(
            adata,
            cell_type_col=CELL_TYPE_COL,
            is_single_cell_level=single_cell,
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
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    all_passed = True
    
    if validate:
        # Load R reference
        print("\n4. Loading R reference output...")
        r_result = load_r_output(output_dir)
        
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
                print(f"SecActPy scRNAseq ({resolution}) produces identical results to RidgeR.")
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
    print(f"\n{step_num}. Sample output (first 5 proteins, first 5 samples):")
    print(py_result['zscore'].iloc[:5, :5])
    
    # Activity statistics by cell type
    step_num += 1
    print(f"\n{step_num}. Activity statistics by cell type:")
    
    if single_cell:
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
    else:
        for col in py_result['zscore'].columns:
            top_proteins = py_result['zscore'][col].sort_values(ascending=False).head(3)
            print(f"\n   {col}:")
            print(f"     Top 3 active: {', '.join(top_proteins.index)}")
    
    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            suffix = "sc_res" if single_cell else "ct_res"
            
            if single_cell:
                from secactpy.io import add_activity_to_anndata, save_results
                
                # Save h5ad with activity
                adata = add_activity_to_anndata(adata, py_result)
                output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_cpu_activity.h5ad"
                adata.write_h5ad(output_h5ad)
                print(f"   Saved h5ad to: {output_h5ad}")
                
                # Save HDF5
                output_h5 = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_cpu_activity.h5"
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
            else:
                from secactpy.io import save_results
                
                output_h5 = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_cpu_activity.h5"
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
                
                # Also save CSV
                csv_path = PACKAGE_ROOT / "dataset" / "output" / f"scRNAseq_{suffix}_cpu_zscore.csv"
                py_result['zscore'].to_csv(csv_path)
                print(f"   Saved CSV to: {csv_path}")
                
        except Exception as e:
            print(f"   Could not save: {e}")
    
    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy scRNA-seq Inference (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cell-type resolution (pseudo-bulk)
  python tests/test_scrnaseq_cpu.py
  
  # Single-cell resolution
  python tests/test_scrnaseq_cpu.py --single-cell
  
  # Save results
  python tests/test_scrnaseq_cpu.py --save
  python tests/test_scrnaseq_cpu.py --single-cell --save
        """
    )
    parser.add_argument(
        '--single-cell',
        action='store_true',
        help='Run single-cell resolution (default: cell-type resolution)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(single_cell=args.single_cell, save_output=args.save)
    sys.exit(0 if success else 1)
