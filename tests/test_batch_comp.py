#!/usr/bin/env python3
"""
Batch vs Non-Batch Processing Comparison Test.

Compares standard ridge inference with batch processing (including sparse-preserving)
for all data types: bulk RNA-seq, scRNA-seq, and spatial transcriptomics.

Usage:
    python tests/test_batch_comparison.py
    python tests/test_batch_comparison.py --bulk
    python tests/test_batch_comparison.py --scrnaseq
    python tests/test_batch_comparison.py --st
    python tests/test_batch_comparison.py --all
"""

import numpy as np
import pandas as pd
from scipy import sparse as sps
import time
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secactpy import (
    ridge,
    ridge_batch,
    precompute_population_stats,
    precompute_projection_components,
    ridge_batch_sparse_preserving,
    load_signature,
    clear_perm_cache,
)

# Test parameters
TOLERANCE = 1e-10
N_RAND = 100  # Reduced for faster testing
SEED = 0
LAMBDA = 5e5


def scale_columns(Y: np.ndarray) -> np.ndarray:
    """
    Scale columns of Y like R's scale() (mean=0, std=1, ddof=1).
    """
    mu = Y.mean(axis=0)
    sigma = Y.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y - mu) / sigma


def replicate_samples(Y: np.ndarray, n_target: int) -> np.ndarray:
    """Replicate samples to reach target count."""
    n_current = Y.shape[1]
    if n_current >= n_target:
        return Y[:, :n_target]
    
    n_reps = (n_target + n_current - 1) // n_current
    Y_rep = np.tile(Y, (1, n_reps))
    return Y_rep[:, :n_target]


def compare_results(result1: dict, result2: dict, name1: str, name2: str) -> dict:
    """Compare two result dictionaries."""
    metrics = {}
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        if key in result1 and key in result2:
            diff = np.abs(result1[key] - result2[key]).max()
            metrics[key] = diff
    
    all_pass = all(v < TOLERANCE for v in metrics.values())
    
    print(f"\n  Comparison: {name1} vs {name2}")
    for key, diff in metrics.items():
        status = "✓" if diff < TOLERANCE else "✗"
        print(f"    {status} {key}: max diff = {diff:.2e}")
    
    return {'metrics': metrics, 'pass': all_pass}


def test_bulk_comparison(n_samples: int = 100, use_gpu: bool = False):
    """
    Test batch vs non-batch for bulk RNA-seq.
    """
    print("=" * 70)
    print(f"BULK RNA-SEQ: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create test data
    n_genes = 1000
    n_features = 50
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
    # Generate signature matrix
    X = np.random.randn(n_genes, n_features).astype(np.float64)
    
    # Generate expression data
    Y_raw = np.random.randn(n_genes, n_samples).astype(np.float64)
    
    # Scale Y (like R's scale())
    Y_scaled = scale_columns(Y_raw)
    
    # =========================================================================
    # Method 1: Standard ridge (non-batch)
    # =========================================================================
    print(f"\n1. Standard ridge (non-batch, {backend})...")
    t_start = time.time()
    result_std = ridge(
        X, Y_scaled, 
        lambda_=LAMBDA, 
        n_rand=N_RAND, 
        seed=SEED, 
        backend=backend,
        verbose=False
    )
    t_std = time.time() - t_start
    print(f"   Time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense, {backend})...")
    batch_size = max(10, n_samples // 5)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=False
    )
    t_batch = time.time() - t_start
    print(f"   Time: {t_batch:.3f}s (batch_size={batch_size})")
    print(f"   Shape: {result_batch['beta'].shape}")
    
    # =========================================================================
    # Method 3: Sparse-preserving batch processing (CPU only for now)
    # =========================================================================
    print("\n3. Sparse-preserving batch processing (CPU)...")
    
    # Precompute components
    stats = precompute_population_stats(Y_raw)
    proj = precompute_projection_components(X, lambda_=LAMBDA)
    
    t_start = time.time()
    result_sparse = ridge_batch_sparse_preserving(
        proj, Y_raw, stats,
        n_rand=N_RAND,
        seed=SEED,
        use_cache=True,
        verbose=False
    )
    t_sparse = time.time() - t_start
    print(f"   Time: {t_sparse:.3f}s")
    print(f"   Shape: {result_sparse['beta'].shape}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n" + "-" * 50)
    print("COMPARISONS:")
    
    cmp1 = compare_results(result_std, result_batch, "Standard", "Batch")
    cmp2 = compare_results(result_std, result_sparse, "Standard", "Sparse-preserving")
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY:")
    print(f"  Standard ({backend}):   {t_std:.3f}s")
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_scrnaseq_comparison(n_cells: int = 500, use_gpu: bool = False):
    """
    Test batch vs non-batch for scRNA-seq.
    """
    print("\n" + "=" * 70)
    print(f"scRNA-SEQ: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create test data
    n_genes = 1000
    n_features = 50
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_cells: {n_cells}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
    # Generate signature matrix
    X = np.random.randn(n_genes, n_features).astype(np.float64)
    
    # Generate sparse expression data (typical for scRNA-seq)
    Y_raw_dense = np.random.randn(n_genes, n_cells).astype(np.float64)
    # Add sparsity (70% zeros - typical for scRNA-seq)
    sparsity_mask = np.random.rand(n_genes, n_cells) < 0.7
    Y_raw_dense[sparsity_mask] = 0
    
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    
    # Scale for standard method
    Y_scaled = scale_columns(Y_raw_dense)
    
    # =========================================================================
    # Method 1: Standard ridge (non-batch, dense)
    # =========================================================================
    print(f"\n1. Standard ridge (non-batch, dense Y, {backend})...")
    t_start = time.time()
    result_std = ridge(
        X, Y_scaled, 
        lambda_=LAMBDA, 
        n_rand=N_RAND, 
        seed=SEED, 
        backend=backend,
        verbose=False
    )
    t_std = time.time() - t_start
    print(f"   Time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense Y, {backend})...")
    batch_size = max(50, n_cells // 5)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=False
    )
    t_batch = time.time() - t_start
    print(f"   Time: {t_batch:.3f}s (batch_size={batch_size})")
    print(f"   Shape: {result_batch['beta'].shape}")
    
    # =========================================================================
    # Method 3: Sparse-preserving batch (sparse Y) - CPU only
    # =========================================================================
    print("\n3. Sparse-preserving batch (sparse Y, no densification, CPU)...")
    
    # Precompute from sparse
    stats = precompute_population_stats(Y_sparse)
    proj = precompute_projection_components(X, lambda_=LAMBDA)
    
    t_start = time.time()
    result_sparse = ridge_batch_sparse_preserving(
        proj, Y_sparse, stats,
        n_rand=N_RAND,
        seed=SEED,
        use_cache=True,
        verbose=False
    )
    t_sparse = time.time() - t_start
    print(f"   Time: {t_sparse:.3f}s")
    print(f"   Shape: {result_sparse['beta'].shape}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n" + "-" * 50)
    print("COMPARISONS:")
    
    cmp1 = compare_results(result_std, result_batch, "Standard", "Batch")
    cmp2 = compare_results(result_std, result_sparse, "Standard", "Sparse-preserving")
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY:")
    print(f"  Standard ({backend}):   {t_std:.3f}s")
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_st_comparison(n_spots: int = 500, use_gpu: bool = False):
    """
    Test batch vs non-batch for spatial transcriptomics.
    """
    print("\n" + "=" * 70)
    print(f"SPATIAL TRANSCRIPTOMICS: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create test data
    n_genes = 1000
    n_features = 50
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_spots: {n_spots}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
    # Generate signature matrix
    X = np.random.randn(n_genes, n_features).astype(np.float64)
    
    # Generate sparse expression data (typical for ST - Visium ~50% sparse, CosMx ~90% sparse)
    Y_raw_dense = np.random.randn(n_genes, n_spots).astype(np.float64)
    
    # CosMx-like sparsity (90% zeros)
    sparsity_mask = np.random.rand(n_genes, n_spots) < 0.9
    Y_raw_dense[sparsity_mask] = 0
    
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    
    # Scale for standard method
    Y_scaled = scale_columns(Y_raw_dense)
    
    # =========================================================================
    # Method 1: Standard ridge (non-batch, dense)
    # =========================================================================
    print(f"\n1. Standard ridge (non-batch, dense Y, {backend})...")
    t_start = time.time()
    result_std = ridge(
        X, Y_scaled, 
        lambda_=LAMBDA, 
        n_rand=N_RAND, 
        seed=SEED, 
        backend=backend,
        verbose=False
    )
    t_std = time.time() - t_start
    print(f"   Time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense Y, {backend})...")
    batch_size = max(50, n_spots // 5)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=False
    )
    t_batch = time.time() - t_start
    print(f"   Time: {t_batch:.3f}s (batch_size={batch_size})")
    print(f"   Shape: {result_batch['beta'].shape}")
    
    # =========================================================================
    # Method 3: Sparse-preserving batch (sparse Y) - CPU only
    # =========================================================================
    print("\n3. Sparse-preserving batch (sparse Y, no densification, CPU)...")
    
    # Precompute from sparse
    stats = precompute_population_stats(Y_sparse)
    proj = precompute_projection_components(X, lambda_=LAMBDA)
    
    t_start = time.time()
    result_sparse = ridge_batch_sparse_preserving(
        proj, Y_sparse, stats,
        n_rand=N_RAND,
        seed=SEED,
        use_cache=True,
        verbose=False
    )
    t_sparse = time.time() - t_start
    print(f"   Time: {t_sparse:.3f}s")
    print(f"   Shape: {result_sparse['beta'].shape}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print("\n" + "-" * 50)
    print("COMPARISONS:")
    
    cmp1 = compare_results(result_std, result_batch, "Standard", "Batch")
    cmp2 = compare_results(result_std, result_sparse, "Standard", "Sparse-preserving")
    
    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY:")
    print(f"  Standard ({backend}):   {t_std:.3f}s")
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_large_scale_comparison(n_samples: int = 5000):
    """
    Test with larger sample sizes to show batch processing benefits.
    """
    print("\n" + "=" * 70)
    print("LARGE-SCALE: Batch Processing Performance Test")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_genes = 1000
    n_features = 50
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_rand: {N_RAND}")
    
    # Generate data
    X = np.random.randn(n_genes, n_features).astype(np.float64)
    Y_raw_dense = np.random.randn(n_genes, n_samples).astype(np.float64)
    
    # 80% sparsity
    sparsity_mask = np.random.rand(n_genes, n_samples) < 0.8
    Y_raw_dense[sparsity_mask] = 0
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros")
    
    # Precompute for sparse-preserving
    stats = precompute_population_stats(Y_sparse)
    proj = precompute_projection_components(X, lambda_=LAMBDA)
    
    # =========================================================================
    # Test sparse-preserving with different batch sizes
    # =========================================================================
    print("\n" + "-" * 50)
    print("Sparse-Preserving Batch Processing:")
    
    batch_sizes = [100, 500, 1000, n_samples]  # Last one = no batching
    
    results = []
    for bs in batch_sizes:
        if bs > n_samples:
            continue
        
        label = "full" if bs == n_samples else f"batch={bs}"
        
        t_start = time.time()
        
        if bs == n_samples:
            # Single batch
            result = ridge_batch_sparse_preserving(
                proj, Y_sparse, stats,
                n_rand=N_RAND, seed=SEED,
                use_cache=True, verbose=False
            )
        else:
            # Multiple batches
            all_beta = []
            all_se = []
            all_zscore = []
            all_pvalue = []
            
            for start in range(0, n_samples, bs):
                end = min(start + bs, n_samples)
                Y_batch = Y_sparse[:, start:end]
                
                # For batch processing, we need stats for this batch
                batch_stats = precompute_population_stats(Y_batch)
                
                batch_result = ridge_batch_sparse_preserving(
                    proj, Y_batch, batch_stats,
                    n_rand=N_RAND, seed=SEED,
                    use_cache=True, verbose=False
                )
                
                all_beta.append(batch_result['beta'])
                all_se.append(batch_result['se'])
                all_zscore.append(batch_result['zscore'])
                all_pvalue.append(batch_result['pvalue'])
            
            result = {
                'beta': np.hstack(all_beta),
                'se': np.hstack(all_se),
                'zscore': np.hstack(all_zscore),
                'pvalue': np.hstack(all_pvalue),
            }
        
        elapsed = time.time() - t_start
        results.append((label, elapsed, result))
        print(f"  {label:12s}: {elapsed:.3f}s")
    
    # Compare all against full
    print("\n" + "-" * 50)
    print("Verification (all batched results vs full):")
    
    full_result = results[-1][2]  # Last one is full
    all_pass = True
    
    for label, elapsed, result in results[:-1]:
        max_diff = np.abs(result['zscore'] - full_result['zscore']).max()
        status = "✓" if max_diff < TOLERANCE else "✗"
        print(f"  {status} {label}: zscore max diff = {max_diff:.2e}")
        if max_diff >= TOLERANCE:
            all_pass = False
    
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Batch vs Non-Batch Comparison Test")
    parser.add_argument('--bulk', action='store_true', help='Test bulk RNA-seq')
    parser.add_argument('--scrnaseq', action='store_true', help='Test scRNA-seq')
    parser.add_argument('--st', action='store_true', help='Test spatial transcriptomics')
    parser.add_argument('--large', action='store_true', help='Test large-scale processing')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--gpu', action='store_true', help='Also run GPU tests (requires CuPy)')
    parser.add_argument('--gpu-only', action='store_true', help='Run only GPU tests')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--clear-cache', action='store_true', help='Clear permutation cache first')
    
    args = parser.parse_args()
    
    # Check GPU availability
    try:
        from secactpy import CUPY_AVAILABLE
    except ImportError:
        CUPY_AVAILABLE = False
    
    if (args.gpu or args.gpu_only) and not CUPY_AVAILABLE:
        print("WARNING: GPU requested but CuPy not available. Running CPU only.")
        args.gpu = False
        args.gpu_only = False
    
    # Default to all if nothing specified
    if not any([args.bulk, args.scrnaseq, args.st, args.large, args.all]):
        args.all = True
    
    if args.clear_cache:
        print("Clearing permutation cache...")
        clear_perm_cache()
    
    print("=" * 70)
    print("BATCH vs NON-BATCH PROCESSING COMPARISON")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  n_rand: {N_RAND}")
    print(f"  lambda: {LAMBDA}")
    print(f"  seed: {SEED}")
    print(f"  tolerance: {TOLERANCE}")
    print(f"  GPU available: {CUPY_AVAILABLE}")
    print(f"  GPU tests: {'yes' if args.gpu or args.gpu_only else 'no'}")
    
    results = []
    
    # CPU tests
    if not args.gpu_only:
        if args.bulk or args.all:
            results.append(("Bulk (CPU)", test_bulk_comparison(args.n_samples, use_gpu=False)))
        
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (CPU)", test_scrnaseq_comparison(args.n_samples, use_gpu=False)))
        
        if args.st or args.all:
            results.append(("ST (CPU)", test_st_comparison(args.n_samples, use_gpu=False)))
        
        if args.large or args.all:
            results.append(("Large-scale (CPU)", test_large_scale_comparison(min(args.n_samples * 10, 5000))))
    
    # GPU tests
    if args.gpu or args.gpu_only:
        print("\n" + "=" * 70)
        print("GPU TESTS")
        print("=" * 70)
        
        if args.bulk or args.all:
            results.append(("Bulk (GPU)", test_bulk_comparison(args.n_samples, use_gpu=True)))
        
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (GPU)", test_scrnaseq_comparison(args.n_samples, use_gpu=True)))
        
        if args.st or args.all:
            results.append(("ST (GPU)", test_st_comparison(args.n_samples, use_gpu=True)))
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED - Batch processing produces identical results!")
    else:
        print("SOME TESTS FAILED - Check output above for details.")
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
