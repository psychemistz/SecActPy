#!/usr/bin/env python3
"""
Test Script: CPU Spatial Transcriptomics Inference Validation

Validates SecActPy ST inference against RidgeR output.
Tests Visium data by default, can also test CosMx with --cosmx flag.

Dataset: 
    - Visium_HCC (10X Visium hepatocellular carcinoma)
    - LIHC_CosMx_data.h5ad (single-cell resolution ST)
    
Reference: R output from RidgeR::SecAct.activity.inference.ST

Usage:
    python tests/test_st_cpu.py              # Visium dataset
    python tests/test_st_cpu.py --cosmx      # CosMx dataset
    python tests/test_st_cpu.py --save

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

from secactpy import secact_activity_inference_st, load_visium_10x, load_signature


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"

# Visium configuration
VISIUM_INPUT = DATA_DIR / "input" / "Visium_HCC"
VISIUM_OUTPUT = DATA_DIR / "output" / "signature" / "ST"
VISIUM_MIN_GENES = 1000
VISIUM_SCALE_FACTOR = 1e5

# CosMx configuration
COSMX_INPUT = DATA_DIR / "input" / "LIHC_CosMx_data.h5ad"
COSMX_OUTPUT = DATA_DIR / "output" / "signature" / "CosMx"
COSMX_MIN_GENES = 50
COSMX_SCALE_FACTOR = 1000

# Common parameters
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


def load_cosmx_data(input_file, min_genes=50, verbose=True):
    """Load CosMx data from h5ad file."""
    import anndata as ad
    from scipy import sparse
    
    adata = ad.read_h5ad(input_file)
    
    # Always use adata.X and adata.var_names (not .raw)
    # The .raw layer may have different gene IDs
    counts_matrix = adata.X
    gene_names = list(adata.var_names)
    cell_names = list(adata.obs_names)
    
    if verbose:
        print(f"   Using adata.X (shape: {counts_matrix.shape})")
        print(f"   Total genes: {len(gene_names)}")
        print(f"   Total cells: {len(cell_names)}")
        print(f"   Sample gene names: {gene_names[:10]}")
    
    # Apply QC filter
    if sparse.issparse(counts_matrix):
        genes_per_cell = np.asarray((counts_matrix > 0).sum(axis=1)).ravel()
    else:
        genes_per_cell = (counts_matrix > 0).sum(axis=1)
    
    keep_cells = genes_per_cell >= min_genes
    n_before = len(cell_names)
    n_after = keep_cells.sum()
    
    counts_matrix = counts_matrix[keep_cells, :]
    cell_names = [c for c, k in zip(cell_names, keep_cells) if k]
    
    if verbose:
        print(f"   Cells after QC (>={min_genes} genes): {n_after} / {n_before}")
    
    # Transpose to (genes × cells)
    if sparse.issparse(counts_matrix):
        counts_transposed = counts_matrix.T.tocsr()
        counts_df = pd.DataFrame.sparse.from_spmatrix(
            counts_transposed, index=gene_names, columns=cell_names
        ).sparse.to_dense()
    else:
        counts_df = pd.DataFrame(counts_matrix.T, index=gene_names, columns=cell_names)
    
    return counts_df


# =============================================================================
# Main Test
# =============================================================================

def main(cosmx=False, save_output=False):
    """
    Run spatial transcriptomics inference validation.
    
    Parameters
    ----------
    cosmx : bool, default=False
        If True, use CosMx dataset. If False, use Visium dataset.
    save_output : bool, default=False
        If True, save results to files.
    """
    platform = "CosMx" if cosmx else "Visium"
    
    print("=" * 70)
    print(f"SecActPy Spatial Transcriptomics Validation Test (CPU) - {platform}")
    print("=" * 70)
    
    # Set configuration based on platform
    if cosmx:
        input_path = COSMX_INPUT
        output_dir = COSMX_OUTPUT
        min_genes = COSMX_MIN_GENES
        scale_factor = COSMX_SCALE_FACTOR
        sig_filter = True
    else:
        input_path = VISIUM_INPUT
        output_dir = VISIUM_OUTPUT
        min_genes = VISIUM_MIN_GENES
        scale_factor = VISIUM_SCALE_FACTOR
        sig_filter = False
    
    # Check files
    print("\n1. Checking files...")
    if not input_path.exists():
        print(f"   ERROR: Input not found: {input_path}")
        return False
    print(f"   Input: {input_path}")
    
    validate = True
    if not output_dir.exists():
        print(f"   Warning: Reference output not found: {output_dir}")
        print("   Will run inference but skip validation.")
        validate = False
    else:
        print(f"   Reference: {output_dir}")
    
    # Load data
    print("\n2. Loading data...")
    
    if cosmx:
        input_data = load_cosmx_data(COSMX_INPUT, min_genes=min_genes, verbose=True)
        print(f"   Shape: {input_data.shape} (genes × cells)")
    else:
        input_data = load_visium_10x(VISIUM_INPUT, min_genes=min_genes, verbose=True)
        print(f"   Spots: {len(input_data['spot_names'])}")
        print(f"   Genes: {len(input_data['gene_names'])}")
    
    # Check gene overlap with signature
    print("\n   Checking gene overlap with signature...")
    sig_df = load_signature("secact")
    sig_genes = set(sig_df.index)
    
    if isinstance(input_data, pd.DataFrame):
        data_genes = set(input_data.index)
    else:
        data_genes = set(input_data.get('gene_names', []))
    
    overlap = sig_genes & data_genes
    print(f"   Signature genes: {len(sig_genes)}")
    print(f"   Data genes: {len(data_genes)}")
    print(f"   Overlap: {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n   WARNING: No direct gene overlap!")
        print(f"   Sample signature genes: {list(sig_genes)[:5]}")
        print(f"   Sample data genes: {list(data_genes)[:5]}")
        
        # Try case-insensitive matching
        sig_genes_upper = {g.upper(): g for g in sig_genes}
        data_genes_upper = {g.upper(): g for g in data_genes}
        case_overlap = set(sig_genes_upper.keys()) & set(data_genes_upper.keys())
        
        if len(case_overlap) > 0:
            print(f"\n   Case-insensitive overlap: {len(case_overlap)}")
            print("   The updated inference.py should handle this automatically.")
    
    # Run inference
    print(f"\n3. Running SecActPy inference (CPU, {platform})...")
    start_time = time.time()
    
    try:
        py_result = secact_activity_inference_st(
            input_data=input_data,
            scale_factor=scale_factor,
            sig_matrix="secact",
            is_group_sig=True,
            is_group_cor=GROUP_COR,
            lambda_=LAMBDA,
            n_rand=NRAND,
            seed=SEED,
            sig_filter=sig_filter,
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
                print(f"SecActPy ST ({platform}) produces identical results to RidgeR.")
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
    print(f"\n{step_num}. Sample output (first 5 proteins, first 3 samples):")
    cols = py_result['zscore'].columns[:3]
    print(py_result['zscore'].iloc[:5][cols])
    
    # Save results
    if save_output:
        step_num += 1
        print(f"\n{step_num}. Saving results...")
        try:
            from secactpy.io import save_st_results_to_h5ad
            
            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"{platform}_cpu_activity.h5ad"
            
            if cosmx:
                save_st_results_to_h5ad(
                    counts=input_data.values,
                    activity_results=py_result,
                    output_path=output_h5ad,
                    gene_names=list(input_data.index),
                    cell_names=list(input_data.columns),
                    platform=platform
                )
            else:
                save_st_results_to_h5ad(
                    counts=input_data['counts'],
                    activity_results=py_result,
                    output_path=output_h5ad,
                    gene_names=input_data['gene_names'],
                    cell_names=input_data['spot_names'],
                    spatial_coords=input_data['spot_coordinates'],
                    platform=platform
                )
            
            print(f"   Saved to: {output_h5ad}")
            
        except Exception as e:
            print(f"   Could not save: {e}")
            import traceback
            traceback.print_exc()
    
    return all_passed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SecActPy Spatial Transcriptomics Inference (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visium dataset
  python tests/test_st_cpu.py
  
  # CosMx dataset
  python tests/test_st_cpu.py --cosmx
  
  # Save results
  python tests/test_st_cpu.py --save
  python tests/test_st_cpu.py --cosmx --save
        """
    )
    parser.add_argument(
        '--cosmx',
        action='store_true',
        help='Use CosMx dataset (default: Visium)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to h5ad file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(cosmx=args.cosmx, save_output=args.save)
    sys.exit(0 if success else 1)
