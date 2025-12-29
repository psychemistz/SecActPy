#!/usr/bin/env python3
"""
Test Script: GPU-Accelerated Spatial Transcriptomics Inference

Compares CPU vs GPU computation for spatial transcriptomics activity inference.
Tests Visium data by default, can also test CosMx with --cosmx flag.

Requirements:
    - CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)
    - NVIDIA GPU with CUDA support

Usage:
    python tests/test_st_gpu.py              # Visium dataset
    python tests/test_st_gpu.py --cosmx      # CosMx dataset
    python tests/test_st_gpu.py --save

Expected output:
    GPU and CPU results should match within numerical tolerance.
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
from secactpy.ridge import CUPY_AVAILABLE


# =============================================================================
# Configuration
# =============================================================================

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "dataset"

# Visium configuration
VISIUM_INPUT = DATA_DIR / "input" / "Visium_HCC"
VISIUM_MIN_GENES = 1000
VISIUM_SCALE_FACTOR = 1e5

# CosMx configuration
COSMX_INPUT = DATA_DIR / "input" / "LIHC_CosMx_data.h5ad"
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

def compare_results(cpu_result: dict, gpu_result: dict, tolerance: float = 1e-8) -> dict:
    """Compare CPU and GPU results."""
    comparison = {}
    
    for name in ['beta', 'se', 'zscore', 'pvalue']:
        if name not in cpu_result or name not in gpu_result:
            comparison[name] = {'status': 'MISSING', 'message': 'Array not found'}
            continue
        
        cpu_arr = cpu_result[name]
        gpu_arr = gpu_result[name]
        
        # Check shape
        if cpu_arr.shape != gpu_arr.shape:
            comparison[name] = {
                'status': 'FAIL',
                'message': f'Shape mismatch: CPU {cpu_arr.shape} vs GPU {gpu_arr.shape}'
            }
            continue
        
        # Align by index/columns
        gpu_aligned = gpu_arr.loc[cpu_arr.index, cpu_arr.columns]
        
        # Calculate difference
        diff = np.abs(cpu_arr.values - gpu_aligned.values)
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
    n_kept = keep_cells.sum()
    counts_matrix = counts_matrix[keep_cells, :]
    cell_names = [c for c, k in zip(cell_names, keep_cells) if k]
    
    if verbose:
        print(f"   Cells after QC (>={min_genes} genes): {n_kept}")
    
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
    Run GPU vs CPU comparison for spatial transcriptomics.
    
    Parameters
    ----------
    cosmx : bool, default=False
        If True, use CosMx dataset. If False, use Visium dataset.
    save_output : bool, default=False
        If True, save results to files.
    """
    platform = "CosMx" if cosmx else "Visium"
    
    print("=" * 70)
    print(f"SecActPy GPU vs CPU Comparison: Spatial Transcriptomics ({platform})")
    print("=" * 70)
    
    # Check GPU availability
    print("\n1. Checking GPU availability...")
    if not CUPY_AVAILABLE:
        from secactpy.ridge import CUPY_INIT_ERROR
        print("   ERROR: GPU is not available!")
        if CUPY_INIT_ERROR:
            print(f"   Reason: {CUPY_INIT_ERROR}")
        else:
            print("   CuPy is not installed.")
            print("   Install with: pip install cupy-cuda11x  # or cupy-cuda12x")
        print("   Skipping GPU test.")
        return False
    
    try:
        import cupy as cp
        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
        print(f"   ✓ CuPy available")
        print(f"   GPU: {gpu_info['name'].decode()}")
        print(f"   Memory: {gpu_info['totalGlobalMem'] / 1e9:.1f} GB")
    except Exception as e:
        print(f"   ERROR: Could not initialize GPU: {e}")
        return False
    
    # Load data based on platform
    print("\n2. Loading data...")
    
    if cosmx:
        if not COSMX_INPUT.exists():
            print(f"   ERROR: Input file not found: {COSMX_INPUT}")
            return False
        print(f"   Input: {COSMX_INPUT}")
        
        input_data = load_cosmx_data(COSMX_INPUT, min_genes=COSMX_MIN_GENES, verbose=True)
        scale_factor = COSMX_SCALE_FACTOR
        sig_filter = True
        print(f"   Final shape: {input_data.shape} (genes × cells)")
    else:
        if not VISIUM_INPUT.exists():
            print(f"   ERROR: Input directory not found: {VISIUM_INPUT}")
            return False
        print(f"   Input: {VISIUM_INPUT}")
        
        input_data = load_visium_10x(VISIUM_INPUT, min_genes=VISIUM_MIN_GENES, verbose=True)
        scale_factor = VISIUM_SCALE_FACTOR
        sig_filter = False
        print(f"   Spots: {len(input_data['spot_names'])}")
    
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
        print("\n   ERROR: No overlapping genes!")
        print(f"   Sample signature genes: {list(sig_genes)[:5]}")
        print(f"   Sample data genes: {list(data_genes)[:5]}")
        
        # Try case-insensitive matching
        sig_genes_upper = {g.upper(): g for g in sig_genes}
        data_genes_upper = {g.upper(): g for g in data_genes}
        case_overlap = set(sig_genes_upper.keys()) & set(data_genes_upper.keys())
        
        if len(case_overlap) > 0:
            print(f"\n   Case-insensitive overlap: {len(case_overlap)}")
            print("   Gene names may have case mismatch. Converting data gene names to match signature...")
            
            # Create mapping from data genes to signature genes
            gene_mapping = {data_genes_upper[g]: sig_genes_upper[g] for g in case_overlap}
            
            if isinstance(input_data, pd.DataFrame):
                input_data.index = [gene_mapping.get(g, g) for g in input_data.index]
                data_genes = set(input_data.index)
                overlap = sig_genes & data_genes
                print(f"   New overlap: {len(overlap)}")
            
        if len(overlap) == 0:
            return False
    
    # Run CPU inference
    print("\n3. Running CPU inference...")
    cpu_start = time.time()
    
    cpu_result = secact_activity_inference_st(
        input_data=input_data,
        scale_factor=scale_factor,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        sig_filter=sig_filter,
        backend="numpy",  # CPU
        use_cache=True,
        verbose=False
    )
    
    cpu_time = time.time() - cpu_start
    print(f"   CPU time: {cpu_time:.2f} seconds")
    print(f"   Result shape: {cpu_result['beta'].shape}")
    
    # Run GPU inference
    print("\n4. Running GPU inference...")
    gpu_start = time.time()
    
    gpu_result = secact_activity_inference_st(
        input_data=input_data,
        scale_factor=scale_factor,
        sig_matrix="secact",
        is_group_sig=True,
        is_group_cor=GROUP_COR,
        lambda_=LAMBDA,
        n_rand=NRAND,
        seed=SEED,
        sig_filter=sig_filter,
        backend="cupy",  # GPU
        use_cache=True,
        verbose=False
    )
    
    gpu_time = time.time() - gpu_start
    print(f"   GPU time: {gpu_time:.2f} seconds")
    print(f"   Result shape: {gpu_result['beta'].shape}")
    
    # Compare results
    print("\n5. Comparing CPU vs GPU results...")
    comparison = compare_results(cpu_result, gpu_result)
    
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
    
    print("\n" + "-" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    print(f"  Platform: {platform}")
    print(f"  Samples:  {cpu_result['beta'].shape[1]}")
    print(f"  CPU time: {cpu_time:.2f} seconds")
    print(f"  GPU time: {gpu_time:.2f} seconds")
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"  Speedup:  {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("GPU produces identical results to CPU.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("GPU results differ from CPU.")
    print("=" * 70)
    
    # Save results
    if save_output:
        print("\n6. Saving results...")
        try:
            from secactpy.io import save_st_results_to_h5ad, save_results
            
            suffix = "CosMx" if cosmx else "Visium"
            output_h5ad = PACKAGE_ROOT / "dataset" / "output" / f"{suffix}_gpu_activity.h5ad"
            
            if cosmx:
                # For CosMx, save just the results
                save_st_results_to_h5ad(
                    counts=input_data.values,
                    activity_results=gpu_result,
                    output_path=output_h5ad,
                    gene_names=list(input_data.index),
                    cell_names=list(input_data.columns),
                    platform=platform
                )
            else:
                # For Visium, include spatial coordinates
                save_st_results_to_h5ad(
                    counts=input_data['counts'],
                    activity_results=gpu_result,
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
        description="SecActPy GPU vs CPU Comparison: Spatial Transcriptomics"
    )
    parser.add_argument(
        '--cosmx',
        action='store_true',
        help='Use CosMx dataset (default: Visium)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save GPU results to h5ad file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = main(cosmx=args.cosmx, save_output=args.save)
    sys.exit(0 if success else 1)
