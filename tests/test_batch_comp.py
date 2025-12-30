#!/usr/bin/env python3
"""
Batch vs Non-Batch Processing Comparison Test.

Compares standard ridge inference with batch processing for different data types.

Usage:
    python tests/test_batch_comp.py
    python tests/test_batch_comp.py --bulk --n-bulk 1000
    python tests/test_batch_comp.py --scrnaseq --n-cells 5000
    python tests/test_batch_comp.py --gpu-only --cosmx
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
    load_signature,
    clear_perm_cache,
    CUPY_AVAILABLE,
)

# Test parameters
TOLERANCE = 1e-10
N_RAND = 1000
SEED = 0
LAMBDA = 5e5

# Dataset paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset")
BULK_INPUT = os.path.join(DATASET_DIR, "input", "Ly86-Fc_vs_Vehicle_logFC.txt")
COSMX_INPUT = os.path.join(DATASET_DIR, "input", "LIHC_CosMx_data.h5ad")


def scale_columns(Y: np.ndarray) -> np.ndarray:
    """Scale columns of Y (mean=0, std=1, ddof=1)."""
    mu = Y.mean(axis=0)
    sigma = Y.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y - mu) / sigma


def resample_columns(Y: np.ndarray, n_target: int, seed: int = 42) -> np.ndarray:
    """Resample columns to target count with noise."""
    np.random.seed(seed)
    n_genes, n_current = Y.shape
    
    if n_current >= n_target:
        return Y[:, :n_target].copy()
    
    indices = np.random.randint(0, n_current, size=n_target)
    Y_resampled = Y[:, indices].copy()
    
    noise_scale = np.std(Y) * 0.01
    noise = np.random.randn(n_genes, n_target) * noise_scale
    return Y_resampled + noise


def compare_results(result1: dict, result2: dict, name1: str, name2: str) -> bool:
    """Compare two result dictionaries."""
    print(f"\n  Comparison: {name1} vs {name2}")
    all_pass = True
    
    for key in ['beta', 'se', 'zscore', 'pvalue']:
        if key in result1 and key in result2:
            diff = np.abs(result1[key] - result2[key]).max()
            status = "✓" if diff < TOLERANCE else "✗"
            print(f"    {status} {key}: max diff = {diff:.2e}")
            if diff >= TOLERANCE:
                all_pass = False
    
    return all_pass


def load_bulk_data():
    """Load real bulk RNA-seq data."""
    if not os.path.exists(BULK_INPUT):
        return None
    try:
        df = pd.read_csv(BULK_INPUT, sep=r'\s+')
        if df.iloc[:, 0].dtype == object:
            df = df.set_index(df.columns[0])
        df.index = df.index.astype(str)
        return df
    except Exception as e:
        print(f"  Warning: Failed to load bulk data: {e}")
        return None


def load_cosmx_data(min_genes=50):
    """Load CosMx data from h5ad file."""
    import anndata as ad
    
    if not os.path.exists(COSMX_INPUT):
        print(f"   CosMx file not found: {COSMX_INPUT}")
        return None
    
    adata = ad.read_h5ad(COSMX_INPUT)
    counts = adata.X
    gene_names = list(adata.var_names)
    
    print(f"   Loaded: {counts.shape} (cells × genes)")
    
    # QC filter
    if sps.issparse(counts):
        genes_per_cell = np.asarray((counts > 0).sum(axis=1)).ravel()
    else:
        genes_per_cell = (counts > 0).sum(axis=1)
    
    keep = genes_per_cell >= min_genes
    counts = counts[keep, :]
    print(f"   After QC: {counts.shape[0]} cells")
    
    # Transpose to (genes × cells)
    if sps.issparse(counts):
        counts = counts.T.tocsr()
    else:
        counts = sps.csr_matrix(counts.T)
    
    return {'counts': counts, 'gene_names': gene_names}


def test_bulk(n_samples: int = 1000, use_gpu: bool = False):
    """Test batch vs non-batch for bulk RNA-seq."""
    print("=" * 70)
    print(f"BULK RNA-SEQ: Batch vs Non-Batch {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load signature
    print("\nLoading signature matrix...")
    sig_df = load_signature('secact')
    print(f"  Signature: {sig_df.shape}")
    
    # Load or simulate data
    print("\nLoading bulk data...")
    bulk_df = load_bulk_data()
    
    if bulk_df is not None:
        # Find common genes
        common = sig_df.index.intersection(bulk_df.index)
        if len(common) >= 100:
            X = sig_df.loc[common].values.astype(np.float64)
            Y_base = bulk_df.loc[common].values.astype(np.float64)
            print(f"  Using real data: {len(common)} genes")
        else:
            X = sig_df.values.astype(np.float64)
            Y_base = np.random.randn(X.shape[0], 1)
            print(f"  Using simulated data (too few common genes)")
    else:
        X = sig_df.values.astype(np.float64)
        Y_base = np.random.randn(X.shape[0], 1)
        print(f"  Using simulated data")
    
    # Resample and scale
    Y_raw = resample_columns(Y_base, n_samples)
    Y = scale_columns(Y_raw)
    print(f"  Final shape: {Y.shape}")
    
    batch_size = max(100, n_samples // 10)
    
    # Method 1: Standard ridge
    print(f"\n1. Standard ridge ({backend})...")
    t1 = time.time()
    r1 = ridge(X, Y, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED, 
               backend=backend, use_cache=True, verbose=True)
    t1 = time.time() - t1
    print(f"   Time: {t1:.2f}s, Shape: {r1['beta'].shape}")
    
    # Method 2: Batch processing
    print(f"\n2. Batch processing ({backend}, batch_size={batch_size})...")
    t2 = time.time()
    r2 = ridge_batch(X, Y, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                     batch_size=batch_size, backend=backend, 
                     use_cache=True, verbose=True)
    t2 = time.time() - t2
    print(f"   Time: {t2:.2f}s, Shape: {r2['beta'].shape}")
    
    # Compare
    print("\n" + "-" * 50)
    passed = compare_results(r1, r2, "Standard", "Batch")
    
    print(f"\n  Summary: Standard={t1:.2f}s, Batch={t2:.2f}s")
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_scrnaseq(n_cells: int = 5000, use_gpu: bool = False):
    """Test batch vs non-batch for scRNA-seq (sparse data)."""
    print("\n" + "=" * 70)
    print(f"scRNA-SEQ: Batch vs Non-Batch {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load signature
    print("\nLoading signature matrix...")
    sig_df = load_signature('secact')
    X = sig_df.values.astype(np.float64)
    n_genes, n_features = X.shape
    print(f"  Signature: {n_genes} × {n_features}")
    
    # Simulate sparse scRNA-seq data
    print(f"\nSimulating scRNA-seq ({n_cells} cells, 70% sparsity)...")
    np.random.seed(42)
    Y_dense = np.zeros((n_genes, n_cells), dtype=np.float64)
    mask = np.random.rand(n_genes, n_cells) > 0.70
    Y_dense[mask] = np.exp(np.random.randn(mask.sum()) * 1.5 + 1)
    
    Y_sparse = sps.csr_matrix(Y_dense)
    Y_scaled = scale_columns(Y_dense)
    
    nnz = 100 * Y_sparse.nnz / Y_sparse.size
    print(f"  Shape: {Y_sparse.shape}, Non-zeros: {nnz:.1f}%")
    
    batch_size = max(500, n_cells // 10)
    
    # Method 1: Standard (dense)
    print(f"\n1. Standard ridge (dense, {backend})...")
    t1 = time.time()
    r1 = ridge(X, Y_scaled, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
               backend=backend, use_cache=True, verbose=True)
    t1 = time.time() - t1
    print(f"   Time: {t1:.2f}s")
    
    # Method 2: Batch (dense)
    print(f"\n2. Batch processing (dense, {backend}, batch_size={batch_size})...")
    t2 = time.time()
    r2 = ridge_batch(X, Y_scaled, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                     batch_size=batch_size, backend=backend,
                     use_cache=True, verbose=True)
    t2 = time.time() - t2
    print(f"   Time: {t2:.2f}s")
    
    # Method 3: Batch (sparse) - ridge_batch handles sparse input
    print(f"\n3. Batch processing (sparse, {backend}, batch_size={batch_size})...")
    t3 = time.time()
    r3 = ridge_batch(X, Y_sparse, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                     batch_size=batch_size, backend=backend,
                     use_cache=True, verbose=True)
    t3 = time.time() - t3
    print(f"   Time: {t3:.2f}s")
    
    # Compare
    print("\n" + "-" * 50)
    p1 = compare_results(r1, r2, "Standard", "Batch(dense)")
    p2 = compare_results(r1, r3, "Standard", "Batch(sparse)")
    
    print(f"\n  Summary: Standard={t1:.2f}s, Batch(dense)={t2:.2f}s, Batch(sparse)={t3:.2f}s")
    passed = p1 and p2
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_cosmx(n_cells: int = None, use_gpu: bool = False):
    """Test batch vs non-batch for real CosMx data."""
    print("\n" + "=" * 70)
    print(f"COSMX REAL DATA: Batch vs Non-Batch {'(GPU)' if use_gpu else '(CPU)'}")
    print("=" * 70)
    
    backend = 'cupy' if use_gpu else 'numpy'
    
    # Load signature
    print("\nLoading signature matrix...")
    sig_df = load_signature('secact')
    print(f"  Signature: {sig_df.shape}")
    
    # Load CosMx data
    print("\nLoading CosMx data...")
    data = load_cosmx_data()
    if data is None:
        print("  ERROR: CosMx data not found!")
        return False
    
    counts = data['counts']  # (genes × cells)
    gene_names = data['gene_names']
    
    # Find overlapping genes
    overlap = list(set(sig_df.index) & set(gene_names))
    print(f"  Overlap with signature: {len(overlap)} genes")
    
    if len(overlap) < 10:
        print("  ERROR: Too few overlapping genes!")
        return False
    
    # Subset data
    gene_idx = [gene_names.index(g) for g in overlap]
    Y_sparse = counts[gene_idx, :]
    X = sig_df.loc[overlap].values.astype(np.float64)
    
    # Optionally subsample cells
    n_total = Y_sparse.shape[1]
    if n_cells is not None and n_cells < n_total:
        np.random.seed(42)
        idx = np.random.choice(n_total, n_cells, replace=False)
        Y_sparse = Y_sparse[:, idx]
    
    n_spots = Y_sparse.shape[1]
    Y_dense = Y_sparse.toarray()
    Y_scaled = scale_columns(Y_dense.astype(np.float64))
    
    nnz = 100 * Y_sparse.nnz / Y_sparse.size
    print(f"  Final: {Y_sparse.shape}, Non-zeros: {nnz:.1f}%")
    
    batch_size = max(10000, n_spots // 10)
    
    # Method 1: Standard (dense)
    print(f"\n1. Standard ridge (dense, {backend})...")
    t1 = time.time()
    r1 = ridge(X, Y_scaled, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
               backend=backend, use_cache=True, verbose=True)
    t1 = time.time() - t1
    print(f"   Time: {t1:.2f}s")
    
    # Method 2: Batch (dense)
    print(f"\n2. Batch processing (dense, {backend}, batch_size={batch_size})...")
    t2 = time.time()
    r2 = ridge_batch(X, Y_scaled, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                     batch_size=batch_size, backend=backend,
                     use_cache=True, verbose=True)
    t2 = time.time() - t2
    print(f"   Time: {t2:.2f}s")
    
    # Method 3: Batch (sparse)
    print(f"\n3. Batch processing (sparse, {backend}, batch_size={batch_size})...")
    t3 = time.time()
    r3 = ridge_batch(X, Y_sparse, lambda_=LAMBDA, n_rand=N_RAND, seed=SEED,
                     batch_size=batch_size, backend=backend,
                     use_cache=True, verbose=True)
    t3 = time.time() - t3
    print(f"   Time: {t3:.2f}s")
    
    # Compare
    print("\n" + "-" * 50)
    p1 = compare_results(r1, r2, "Standard", "Batch(dense)")
    p2 = compare_results(r1, r3, "Standard", "Batch(sparse)")
    
    print(f"\n  Summary:")
    print(f"    Standard:      {t1:.2f}s")
    print(f"    Batch(dense):  {t2:.2f}s")
    print(f"    Batch(sparse): {t3:.2f}s")
    
    passed = p1 and p2
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="Batch vs Non-Batch Comparison Test")
    parser.add_argument('--bulk', action='store_true', help='Test bulk RNA-seq')
    parser.add_argument('--scrnaseq', action='store_true', help='Test scRNA-seq')
    parser.add_argument('--cosmx', action='store_true', help='Test real CosMx data')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--gpu', action='store_true', help='Also run GPU tests')
    parser.add_argument('--gpu-only', action='store_true', help='Run only GPU tests')
    parser.add_argument('--n-bulk', type=int, default=1000, help='Number of bulk samples')
    parser.add_argument('--n-cells', type=int, default=5000, help='Number of scRNA-seq cells')
    parser.add_argument('--n-cosmx', type=int, default=None, help='Number of CosMx cells')
    parser.add_argument('--clear-cache', action='store_true', help='Clear permutation cache')
    
    args = parser.parse_args()
    
    if (args.gpu or args.gpu_only) and not CUPY_AVAILABLE:
        print("WARNING: GPU requested but CuPy not available.")
        args.gpu = False
        args.gpu_only = False
    
    if not any([args.bulk, args.scrnaseq, args.cosmx, args.all]):
        args.all = True
    
    if args.clear_cache:
        print("Clearing permutation cache...")
        clear_perm_cache()
    
    print("=" * 70)
    print("BATCH vs NON-BATCH PROCESSING COMPARISON")
    print("=" * 70)
    print(f"\nParameters: n_rand={N_RAND}, lambda={LAMBDA}, seed={SEED}")
    print(f"GPU available: {CUPY_AVAILABLE}")
    
    results = []
    
    # CPU tests
    if not args.gpu_only:
        if args.bulk or args.all:
            results.append(("Bulk (CPU)", test_bulk(args.n_bulk, use_gpu=False)))
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (CPU)", test_scrnaseq(args.n_cells, use_gpu=False)))
        if args.cosmx:
            results.append(("CosMx (CPU)", test_cosmx(args.n_cosmx, use_gpu=False)))
    
    # GPU tests
    if args.gpu or args.gpu_only:
        print("\n" + "=" * 70)
        print("GPU TESTS")
        print("=" * 70)
        
        if args.bulk or args.all:
            results.append(("Bulk (GPU)", test_bulk(args.n_bulk, use_gpu=True)))
        if args.scrnaseq or args.all:
            results.append(("scRNA-seq (GPU)", test_scrnaseq(args.n_cells, use_gpu=True)))
        if args.cosmx:
            results.append(("CosMx (GPU)", test_cosmx(args.n_cosmx, use_gpu=True)))
    
    # Summary
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
    print("ALL TESTS PASSED!" if all_pass else "SOME TESTS FAILED!")
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
