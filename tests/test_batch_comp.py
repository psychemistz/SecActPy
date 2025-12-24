#!/usr/bin/env python3
"""
Batch vs Non-Batch Processing Comparison Test with Real Datasets.

Compares standard ridge inference with batch processing (including sparse-preserving)
for all data types: bulk RNA-seq, scRNA-seq, and spatial transcriptomics.

Uses real signature matrices (SecAct, CytoSig) and real/simulated expression data.
Resamples data if sample size is too small for meaningful batch comparison.

Usage:
    python tests/test_batch_comparison.py
    python tests/test_batch_comparison.py --bulk
    python tests/test_batch_comparison.py --scrnaseq
    python tests/test_batch_comparison.py --st
    python tests/test_batch_comparison.py --all --gpu
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
    CUPY_AVAILABLE,
)

# Test parameters
TOLERANCE = 1e-10
N_RAND = 1000  # Use realistic n_rand
SEED = 0
LAMBDA = 5e5

# Dataset paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset")
BULK_INPUT = os.path.join(DATASET_DIR, "input", "Ly86-Fc_vs_Vehicle_logFC.txt")
COSMX_INPUT = os.path.join(DATASET_DIR, "input", "LIHC_CosMx_data.h5ad")


def scale_columns(Y: np.ndarray) -> np.ndarray:
    """
    Scale columns of Y like R's scale() (mean=0, std=1, ddof=1).
    """
    mu = Y.mean(axis=0)
    sigma = Y.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y - mu) / sigma


def resample_columns(Y: np.ndarray, n_target: int, add_noise: bool = True) -> np.ndarray:
    """
    Resample columns to reach target count with optional noise.
    
    Parameters
    ----------
    Y : ndarray, shape (n_genes, n_samples)
        Original data
    n_target : int
        Target number of samples
    add_noise : bool
        If True, add small Gaussian noise to resampled columns
        
    Returns
    -------
    Y_resampled : ndarray, shape (n_genes, n_target)
    """
    n_genes, n_current = Y.shape
    
    if n_current >= n_target:
        return Y[:, :n_target].copy()
    
    # Randomly sample with replacement
    np.random.seed(42)
    indices = np.random.randint(0, n_current, size=n_target)
    Y_resampled = Y[:, indices].copy()
    
    if add_noise:
        # Add small noise to make samples slightly different
        noise_scale = np.std(Y) * 0.01
        noise = np.random.randn(n_genes, n_target) * noise_scale
        Y_resampled = Y_resampled + noise
    
    return Y_resampled


def load_real_bulk_data():
    """Load real bulk RNA-seq data."""
    if os.path.exists(BULK_INPUT):
        df = pd.read_csv(BULK_INPUT, sep='\t', index_col=0)
        return df
    return None


def load_cosmx_data(input_file, min_genes=50, verbose=True):
    """Load CosMx data from h5ad file."""
    import anndata as ad
    from scipy import sparse
    
    if not os.path.exists(input_file):
        if verbose:
            print(f"   CosMx file not found: {input_file}")
        return None
    
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
    
    # Return as sparse (genes × cells)
    if sparse.issparse(counts_matrix):
        counts_transposed = counts_matrix.T.tocsr()
    else:
        counts_transposed = sparse.csr_matrix(counts_matrix.T)
    
    return {
        'counts': counts_transposed,
        'gene_names': gene_names,
        'cell_names': cell_names,
    }


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


def test_bulk_comparison(n_samples: int = 1000, use_gpu: bool = False):
    """
    Test batch vs non-batch for bulk RNA-seq using real data.
    """
    print("=" * 70)
    print(f"BULK RNA-SEQ: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load real signature matrix
    print("\nLoading real SecAct signature matrix...")
    sig_df = load_signature('secact')
    X = sig_df.values.astype(np.float64)
    gene_names = sig_df.index.tolist()
    protein_names = sig_df.columns.tolist()
    n_genes_sig, n_features = X.shape
    print(f"  Signature: {n_genes_sig} genes × {n_features} proteins")
    
    # Load real bulk data
    print("\nLoading real bulk expression data...")
    bulk_df = load_real_bulk_data()
    
    if bulk_df is not None:
        # Find common genes
        common_genes = list(set(gene_names) & set(bulk_df.index))
        print(f"  Original: {bulk_df.shape[0]} genes × {bulk_df.shape[1]} samples")
        print(f"  Common genes with signature: {len(common_genes)}")
        
        # Subset to common genes
        X_common = sig_df.loc[common_genes].values.astype(np.float64)
        Y_base = bulk_df.loc[common_genes].values.astype(np.float64)
    else:
        print("  Real data not found, using simulated data...")
        np.random.seed(42)
        X_common = X
        Y_base = np.random.randn(n_genes_sig, 1).astype(np.float64)
        common_genes = gene_names
    
    n_genes = len(common_genes)
    
    # Resample to target number of samples
    print(f"\nResampling to {n_samples} samples...")
    Y_raw = resample_columns(Y_base, n_samples, add_noise=True)
    print(f"  Resampled data: {Y_raw.shape[0]} genes × {Y_raw.shape[1]} samples")
    
    # Scale Y
    Y_scaled = scale_columns(Y_raw)
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
    # =========================================================================
    # Method 1: Standard ridge (non-batch)
    # =========================================================================
    print(f"\n1. Standard ridge (non-batch, {backend})...")
    t_start = time.time()
    result_std = ridge(
        X_common, Y_scaled, 
        lambda_=LAMBDA, 
        n_rand=N_RAND, 
        seed=SEED, 
        backend=backend,
        verbose=True
    )
    t_std = time.time() - t_start
    print(f"   Total time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense, {backend})...")
    batch_size = max(100, n_samples // 10)
    t_start = time.time()
    result_batch = ridge_batch(
        X_common, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=True
    )
    t_batch = time.time() - t_start
    print(f"   Total time: {t_batch:.3f}s (batch_size={batch_size})")
    print(f"   Shape: {result_batch['beta'].shape}")
    
    # =========================================================================
    # Method 3: Sparse-preserving batch processing (CPU only)
    # =========================================================================
    print("\n3. Sparse-preserving batch processing (CPU)...")
    
    # Precompute components
    stats = precompute_population_stats(Y_raw)
    proj = precompute_projection_components(X_common, lambda_=LAMBDA)
    
    t_start = time.time()
    result_sparse = ridge_batch_sparse_preserving(
        proj, Y_raw, stats,
        n_rand=N_RAND,
        seed=SEED,
        use_cache=True,
        verbose=True
    )
    t_sparse = time.time() - t_start
    print(f"   Total time: {t_sparse:.3f}s")
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
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)" if t_batch > 0 else f"  Batch ({backend}):      {t_batch:.3f}s")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)" if t_sparse > 0 else f"  Sparse-preserving: {t_sparse:.3f}s")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_scrnaseq_comparison(n_cells: int = 5000, use_gpu: bool = False):
    """
    Test batch vs non-batch for scRNA-seq using real signature with simulated sparse data.
    """
    print("\n" + "=" * 70)
    print(f"scRNA-SEQ: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load real signature matrix
    print("\nLoading real SecAct signature matrix...")
    sig_df = load_signature('secact')
    X = sig_df.values.astype(np.float64)
    n_genes, n_features = X.shape
    print(f"  Signature: {n_genes} genes × {n_features} proteins")
    
    # Simulate scRNA-seq data (sparse, log-normalized counts)
    print(f"\nSimulating scRNA-seq data ({n_cells} cells)...")
    np.random.seed(42)
    
    # Simulate count-like data with realistic sparsity
    # Log-normal distribution for non-zero values
    Y_raw_dense = np.zeros((n_genes, n_cells), dtype=np.float64)
    
    # 70% dropout rate (typical for scRNA-seq)
    dropout_rate = 0.70
    non_zero_mask = np.random.rand(n_genes, n_cells) > dropout_rate
    
    # Fill non-zero entries with log-normal values
    n_nonzero = non_zero_mask.sum()
    Y_raw_dense[non_zero_mask] = np.exp(np.random.randn(n_nonzero) * 1.5 + 1)
    
    # Create sparse matrix
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"  Data shape: {Y_raw_dense.shape}")
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    print(f"  Memory (dense): {Y_raw_dense.nbytes / 1e6:.1f} MB")
    print(f"  Memory (sparse): {(Y_sparse.data.nbytes + Y_sparse.indices.nbytes + Y_sparse.indptr.nbytes) / 1e6:.1f} MB")
    
    # Scale for standard method
    Y_scaled = scale_columns(Y_raw_dense)
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_cells: {n_cells}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
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
        verbose=True
    )
    t_std = time.time() - t_start
    print(f"   Total time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense Y, {backend})...")
    batch_size = max(500, n_cells // 10)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=True
    )
    t_batch = time.time() - t_start
    print(f"   Total time: {t_batch:.3f}s (batch_size={batch_size})")
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
        verbose=True
    )
    t_sparse = time.time() - t_start
    print(f"   Total time: {t_sparse:.3f}s")
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
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)" if t_batch > 0 else f"  Batch ({backend}):      {t_batch:.3f}s")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)" if t_sparse > 0 else f"  Sparse-preserving: {t_sparse:.3f}s")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_st_comparison(n_spots: int = 10000, use_gpu: bool = False):
    """
    Test batch vs non-batch for spatial transcriptomics (CosMx-like sparsity).
    """
    print("\n" + "=" * 70)
    print(f"SPATIAL TRANSCRIPTOMICS: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load real signature matrix
    print("\nLoading real SecAct signature matrix...")
    sig_df = load_signature('secact')
    X = sig_df.values.astype(np.float64)
    n_genes, n_features = X.shape
    print(f"  Signature: {n_genes} genes × {n_features} proteins")
    
    # Simulate CosMx-like ST data (very sparse)
    print(f"\nSimulating CosMx-like ST data ({n_spots} spots)...")
    np.random.seed(42)
    
    # CosMx has ~90-95% sparsity
    Y_raw_dense = np.zeros((n_genes, n_spots), dtype=np.float64)
    
    # 90% dropout rate (typical for CosMx)
    dropout_rate = 0.90
    non_zero_mask = np.random.rand(n_genes, n_spots) > dropout_rate
    
    # Fill non-zero entries
    n_nonzero = non_zero_mask.sum()
    Y_raw_dense[non_zero_mask] = np.exp(np.random.randn(n_nonzero) * 1.0 + 0.5)
    
    # Create sparse matrix
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"  Data shape: {Y_raw_dense.shape}")
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    print(f"  Memory (dense): {Y_raw_dense.nbytes / 1e6:.1f} MB")
    print(f"  Memory (sparse): {(Y_sparse.data.nbytes + Y_sparse.indices.nbytes + Y_sparse.indptr.nbytes) / 1e6:.1f} MB")
    
    # Scale for standard method
    Y_scaled = scale_columns(Y_raw_dense)
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_spots: {n_spots}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
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
        verbose=True
    )
    t_std = time.time() - t_start
    print(f"   Total time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense Y, {backend})...")
    batch_size = max(1000, n_spots // 10)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=True
    )
    t_batch = time.time() - t_start
    print(f"   Total time: {t_batch:.3f}s (batch_size={batch_size})")
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
        verbose=True
    )
    t_sparse = time.time() - t_start
    print(f"   Total time: {t_sparse:.3f}s")
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
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)" if t_batch > 0 else f"  Batch ({backend}):      {t_batch:.3f}s")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)" if t_sparse > 0 else f"  Sparse-preserving: {t_sparse:.3f}s")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_cosmx_comparison(n_cells: int = None, use_gpu: bool = False):
    """
    Test batch vs non-batch for real CosMx spatial transcriptomics data.
    
    Parameters
    ----------
    n_cells : int, optional
        Number of cells to use. If None, use all cells.
    use_gpu : bool
        Whether to use GPU backend.
    """
    print("\n" + "=" * 70)
    print(f"COSMX REAL DATA: Batch vs Non-Batch Comparison {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load real signature matrix
    print("\nLoading real SecAct signature matrix...")
    sig_df = load_signature('secact')
    sig_genes = set(sig_df.index)
    print(f"  Signature: {sig_df.shape[0]} genes × {sig_df.shape[1]} proteins")
    
    # Load real CosMx data
    print("\nLoading real CosMx data...")
    cosmx_data = load_cosmx_data(COSMX_INPUT, min_genes=50, verbose=True)
    
    if cosmx_data is None:
        print("  ERROR: CosMx data not found!")
        print(f"  Expected at: {COSMX_INPUT}")
        return False
    
    counts_sparse = cosmx_data['counts']  # (genes × cells)
    gene_names = cosmx_data['gene_names']
    cell_names = cosmx_data['cell_names']
    
    # Find overlapping genes with signature
    print("\n  Checking gene overlap with signature...")
    data_genes = set(gene_names)
    overlap = sig_genes & data_genes
    print(f"  Data genes: {len(data_genes)}")
    print(f"  Signature genes: {len(sig_genes)}")
    print(f"  Overlap: {len(overlap)}")
    
    if len(overlap) < 10:
        print("  ERROR: Too few overlapping genes!")
        print(f"  Sample data genes: {list(data_genes)[:10]}")
        print(f"  Sample sig genes: {list(sig_genes)[:10]}")
        return False
    
    # Subset to overlapping genes
    overlap_list = sorted(list(overlap))
    gene_idx = [gene_names.index(g) for g in overlap_list]
    counts_subset = counts_sparse[gene_idx, :]
    
    # Subset signature to same genes
    X = sig_df.loc[overlap_list].values.astype(np.float64)
    n_genes, n_features = X.shape
    
    # Optionally subsample cells
    n_total_cells = counts_subset.shape[1]
    if n_cells is not None and n_cells < n_total_cells:
        np.random.seed(42)
        cell_idx = np.random.choice(n_total_cells, n_cells, replace=False)
        counts_subset = counts_subset[:, cell_idx]
        n_spots = n_cells
    else:
        n_spots = n_total_cells
    
    # Convert to dense for standard method
    Y_raw_dense = counts_subset.toarray() if sps.issparse(counts_subset) else counts_subset
    Y_sparse = sps.csr_matrix(Y_raw_dense) if not sps.issparse(counts_subset) else counts_subset
    
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"\n  Final data shape: {Y_raw_dense.shape} (genes × cells)")
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    print(f"  Memory (dense): {Y_raw_dense.nbytes / 1e6:.1f} MB")
    print(f"  Memory (sparse): {(Y_sparse.data.nbytes + Y_sparse.indices.nbytes + Y_sparse.indptr.nbytes) / 1e6:.1f} MB")
    
    # Scale for standard method
    Y_scaled = scale_columns(Y_raw_dense.astype(np.float64))
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_cells: {n_spots}")
    print(f"  n_rand: {N_RAND}")
    print(f"  backend: {backend}")
    
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
        verbose=True
    )
    t_std = time.time() - t_start
    print(f"   Total time: {t_std:.3f}s")
    print(f"   Shape: {result_std['beta'].shape}")
    
    # =========================================================================
    # Method 2: Batch processing (dense)
    # =========================================================================
    print(f"\n2. Batch processing (dense Y, {backend})...")
    batch_size = max(1000, n_spots // 10)
    t_start = time.time()
    result_batch = ridge_batch(
        X, Y_scaled,
        lambda_=LAMBDA,
        n_rand=N_RAND,
        seed=SEED,
        batch_size=batch_size,
        backend=backend,
        verbose=True
    )
    t_batch = time.time() - t_start
    print(f"   Total time: {t_batch:.3f}s (batch_size={batch_size})")
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
        verbose=True
    )
    t_sparse = time.time() - t_start
    print(f"   Total time: {t_sparse:.3f}s")
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
    print(f"  Batch ({backend}):      {t_batch:.3f}s ({t_std/t_batch:.1f}x)" if t_batch > 0 else f"  Batch ({backend}):      {t_batch:.3f}s")
    print(f"  Sparse-preserving: {t_sparse:.3f}s ({t_std/t_sparse:.1f}x)" if t_sparse > 0 else f"  Sparse-preserving: {t_sparse:.3f}s")
    
    all_pass = cmp1['pass'] and cmp2['pass']
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def test_large_scale_comparison(n_samples: int = 50000):
    """
    Test with large sample sizes to show batch processing benefits.
    Simulates million-cell scale by using representative subset.
    """
    print("\n" + "=" * 70)
    print("LARGE-SCALE: Batch Processing Performance Test")
    print("=" * 70)
    
    # Load real signature
    print("\nLoading real SecAct signature matrix...")
    sig_df = load_signature('secact')
    X = sig_df.values.astype(np.float64)
    n_genes, n_features = X.shape
    print(f"  Signature: {n_genes} genes × {n_features} proteins")
    
    print(f"\nSimulating large-scale data ({n_samples:,} samples)...")
    np.random.seed(42)
    
    # 85% sparsity (between scRNA-seq and CosMx)
    Y_raw_dense = np.zeros((n_genes, n_samples), dtype=np.float64)
    dropout_rate = 0.85
    non_zero_mask = np.random.rand(n_genes, n_samples) > dropout_rate
    n_nonzero = non_zero_mask.sum()
    Y_raw_dense[non_zero_mask] = np.exp(np.random.randn(n_nonzero) * 1.2 + 0.8)
    
    Y_sparse = sps.csr_matrix(Y_raw_dense)
    nnz_pct = 100 * Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1])
    
    print(f"  Data shape: {Y_raw_dense.shape}")
    print(f"  Sparsity: {100 - nnz_pct:.1f}% zeros ({Y_sparse.nnz:,} non-zeros)")
    print(f"  Memory (dense): {Y_raw_dense.nbytes / 1e9:.2f} GB")
    print(f"  Memory (sparse): {(Y_sparse.data.nbytes + Y_sparse.indices.nbytes + Y_sparse.indptr.nbytes) / 1e6:.1f} MB")
    
    print(f"\nTest setup:")
    print(f"  n_genes: {n_genes}")
    print(f"  n_features: {n_features}")
    print(f"  n_samples: {n_samples:,}")
    print(f"  n_rand: {N_RAND}")
    
    # Precompute for sparse-preserving
    print("\nPrecomputing projection components...")
    stats = precompute_population_stats(Y_sparse)
    proj = precompute_projection_components(X, lambda_=LAMBDA)
    
    # =========================================================================
    # Test sparse-preserving with different batch sizes
    # =========================================================================
    print("\n" + "-" * 50)
    print("Sparse-Preserving Batch Processing (varying batch sizes):")
    
    batch_sizes = [1000, 5000, 10000, n_samples]
    
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
            # Multiple batches - process each batch independently
            all_results = []
            
            for start in range(0, n_samples, bs):
                end = min(start + bs, n_samples)
                Y_batch = Y_sparse[:, start:end]
                
                # Compute stats for this batch
                batch_stats = precompute_population_stats(Y_batch)
                
                batch_result = ridge_batch_sparse_preserving(
                    proj, Y_batch, batch_stats,
                    n_rand=N_RAND, seed=SEED,
                    use_cache=True, verbose=False
                )
                all_results.append(batch_result)
            
            # Concatenate results
            result = {
                'beta': np.hstack([r['beta'] for r in all_results]),
                'se': np.hstack([r['se'] for r in all_results]),
                'zscore': np.hstack([r['zscore'] for r in all_results]),
                'pvalue': np.hstack([r['pvalue'] for r in all_results]),
            }
        
        elapsed = time.time() - t_start
        throughput = n_samples / elapsed
        results.append((label, elapsed, throughput, result))
        print(f"  {label:12s}: {elapsed:.2f}s ({throughput:,.0f} samples/sec)")
    
    # Compare all against full
    print("\n" + "-" * 50)
    print("Verification (batched results should be close to full):")
    print("  Note: Different batch stats may cause small differences")
    
    full_result = results[-1][3]  # Last one is full
    all_pass = True
    
    for label, elapsed, throughput, result in results[:-1]:
        # Use looser tolerance for batch processing (different stats per batch)
        max_diff = np.abs(result['zscore'] - full_result['zscore']).max()
        # Batched results may differ slightly due to per-batch statistics
        batch_tolerance = 0.1  # Allow some difference due to per-batch stats
        status = "✓" if max_diff < batch_tolerance else "✗"
        print(f"  {status} {label}: zscore max diff = {max_diff:.4f}")
        if max_diff >= batch_tolerance:
            all_pass = False
    
    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Batch vs Non-Batch Comparison Test (Real Datasets)")
    parser.add_argument('--bulk', action='store_true', help='Test bulk RNA-seq')
    parser.add_argument('--scrnaseq', action='store_true', help='Test scRNA-seq')
    parser.add_argument('--st', action='store_true', help='Test spatial transcriptomics (simulated)')
    parser.add_argument('--cosmx', action='store_true', help='Test real CosMx spatial transcriptomics')
    parser.add_argument('--large', action='store_true', help='Test large-scale processing')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--gpu', action='store_true', help='Also run GPU tests (requires CuPy)')
    parser.add_argument('--gpu-only', action='store_true', help='Run only GPU tests')
    parser.add_argument('--n-bulk', type=int, default=1000, help='Number of bulk samples')
    parser.add_argument('--n-cells', type=int, default=5000, help='Number of scRNA-seq cells')
    parser.add_argument('--n-spots', type=int, default=10000, help='Number of ST spots')
    parser.add_argument('--n-cosmx', type=int, default=None, help='Number of CosMx cells (None=all)')
    parser.add_argument('--n-large', type=int, default=50000, help='Number of large-scale samples')
    parser.add_argument('--clear-cache', action='store_true', help='Clear permutation cache first')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if (args.gpu or args.gpu_only) and not CUPY_AVAILABLE:
        print("WARNING: GPU requested but CuPy not available. Running CPU only.")
        args.gpu = False
        args.gpu_only = False
    
    # Default to all if nothing specified
    if not any([args.bulk, args.scrnaseq, args.st, args.cosmx, args.large, args.all]):
        args.all = True
    
    if args.clear_cache:
        print("Clearing permutation cache...")
        clear_perm_cache()
    
    print("=" * 70)
    print("BATCH vs NON-BATCH PROCESSING COMPARISON (REAL DATASETS)")
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
            results.append(("Bulk (CPU)", test_bulk_comparison(args.n_bulk, use_gpu=False)))
        
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (CPU)", test_scrnaseq_comparison(args.n_cells, use_gpu=False)))
        
        if args.st or args.all:
            results.append(("ST Simulated (CPU)", test_st_comparison(args.n_spots, use_gpu=False)))
        
        if args.cosmx:
            results.append(("CosMx Real (CPU)", test_cosmx_comparison(args.n_cosmx, use_gpu=False)))
        
        if args.large or args.all:
            results.append(("Large-scale (CPU)", test_large_scale_comparison(args.n_large)))
    
    # GPU tests
    if args.gpu or args.gpu_only:
        print("\n" + "=" * 70)
        print("GPU TESTS")
        print("=" * 70)
        
        if args.bulk or args.all:
            results.append(("Bulk (GPU)", test_bulk_comparison(args.n_bulk, use_gpu=True)))
        
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (GPU)", test_scrnaseq_comparison(args.n_cells, use_gpu=True)))
        
        if args.st or args.all:
            results.append(("ST Simulated (GPU)", test_st_comparison(args.n_spots, use_gpu=True)))
        
        if args.cosmx:
            results.append(("CosMx Real (GPU)", test_cosmx_comparison(args.n_cosmx, use_gpu=True)))
    
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
