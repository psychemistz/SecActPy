"""
Batch processing for large-scale ridge regression.

This module enables processing of million-sample datasets by:
1. Precomputing the projection matrix T once
2. Processing Y in memory-efficient batches
3. Optionally streaming results directly to disk (h5ad format)

Memory Management:
------------------
For a dataset with n_genes, n_features, and n_samples:
- T matrix: n_features × n_genes × 8 bytes
- Per batch: ~4 × n_features × batch_size × 8 bytes (results)
- Working memory: accumulation arrays during permutation

The `estimate_memory()` function helps determine optimal batch size.

Usage:
------
    >>> from secactpy.batch import ridge_batch, estimate_batch_size
    >>> 
    >>> # Estimate optimal batch size for available memory
    >>> batch_size = estimate_batch_size(n_genes=20000, n_features=50, 
    ...                                   available_gb=8.0)
    >>> 
    >>> # Run batch processing
    >>> result = ridge_batch(X, Y, batch_size=batch_size)
    >>> 
    >>> # Or stream directly to disk
    >>> ridge_batch(X, Y, batch_size=5000, output_path="results.h5ad")
"""

import numpy as np
from scipy import linalg
from typing import Optional, Literal, Dict, Any, Callable, Union
import time
import warnings
import gc
import math

from .rng import generate_permutation_table
from .ridge import CUPY_AVAILABLE, EPS, DEFAULT_LAMBDA, DEFAULT_NRAND, DEFAULT_SEED

# Try to import h5py for streaming output
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# CuPy setup
cp = None
if CUPY_AVAILABLE:
    try:
        import cupy as cp
    except ImportError:
        pass

__all__ = [
    'ridge_batch',
    'estimate_batch_size',
    'estimate_memory',
    'StreamingResultWriter',
]


# =============================================================================
# Memory Estimation
# =============================================================================

def estimate_memory(
    n_genes: int,
    n_features: int,
    n_samples: int,
    n_rand: int = 1000,
    batch_size: Optional[int] = None,
    include_gpu: bool = False
) -> Dict[str, float]:
    """
    Estimate memory requirements for ridge regression.
    
    Parameters
    ----------
    n_genes : int
        Number of genes/observations.
    n_features : int
        Number of features/proteins.
    n_samples : int
        Number of samples.
    n_rand : int
        Number of permutations.
    batch_size : int, optional
        Batch size. If None, assumes full dataset.
    include_gpu : bool
        Include GPU memory estimates.
    
    Returns
    -------
    dict
        Memory estimates in GB:
        - 'T_matrix': Projection matrix
        - 'Y_data': Input Y matrix
        - 'results': Output arrays (beta, se, zscore, pvalue)
        - 'working': Working memory during computation
        - 'total': Total estimated memory
        - 'per_batch': Memory per batch (if batch_size provided)
    """
    bytes_per_float = 8  # float64
    
    if batch_size is None:
        batch_size = n_samples
    
    # T matrix: (n_features, n_genes)
    T_bytes = n_features * n_genes * bytes_per_float
    
    # Y data: (n_genes, n_samples) - full dataset
    Y_bytes = n_genes * n_samples * bytes_per_float
    
    # Results: 4 arrays of (n_features, n_samples)
    results_bytes = 4 * n_features * n_samples * bytes_per_float
    
    # Working memory per batch: accumulation arrays
    # aver, aver_sq, pvalue_counts: 3 arrays of (n_features, batch_size)
    working_bytes = 3 * n_features * batch_size * bytes_per_float
    
    # Permutation table: (n_rand, n_genes) int32
    perm_bytes = n_rand * n_genes * 4
    
    # Y batch: (n_genes, batch_size)
    Y_batch_bytes = n_genes * batch_size * bytes_per_float
    
    # Beta batch: (n_features, batch_size)
    beta_batch_bytes = n_features * batch_size * bytes_per_float
    
    to_gb = lambda x: x / (1024 ** 3)
    
    estimates = {
        'T_matrix': to_gb(T_bytes),
        'Y_data': to_gb(Y_bytes),
        'results': to_gb(results_bytes),
        'working': to_gb(working_bytes + perm_bytes),
        'per_batch': to_gb(Y_batch_bytes + beta_batch_bytes + working_bytes),
        'total': to_gb(T_bytes + Y_bytes + results_bytes + working_bytes + perm_bytes)
    }
    
    if include_gpu:
        # GPU needs T + Y_batch + working arrays
        gpu_bytes = T_bytes + Y_batch_bytes + working_bytes + beta_batch_bytes
        estimates['gpu_per_batch'] = to_gb(gpu_bytes)
    
    return estimates


def estimate_batch_size(
    n_genes: int,
    n_features: int,
    available_gb: float = 4.0,
    n_rand: int = 1000,
    safety_factor: float = 0.7,
    min_batch: int = 100,
    max_batch: int = 50000
) -> int:
    """
    Estimate optimal batch size given available memory.
    
    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_features : int
        Number of features.
    available_gb : float
        Available memory in GB.
    n_rand : int
        Number of permutations.
    safety_factor : float
        Fraction of available memory to use (0-1).
    min_batch : int
        Minimum batch size.
    max_batch : int
        Maximum batch size.
    
    Returns
    -------
    int
        Recommended batch size.
    """
    bytes_per_float = 8
    available_bytes = available_gb * (1024 ** 3) * safety_factor
    
    # Fixed costs: T matrix + permutation table
    T_bytes = n_features * n_genes * bytes_per_float
    perm_bytes = n_rand * n_genes * 4
    fixed_bytes = T_bytes + perm_bytes
    
    # Available for batch processing
    batch_bytes = available_bytes - fixed_bytes
    
    if batch_bytes <= 0:
        warnings.warn(
            f"Available memory ({available_gb}GB) may be insufficient. "
            f"T matrix alone requires {T_bytes / 1e9:.2f}GB."
        )
        return min_batch
    
    # Per-sample cost: Y column + working arrays
    per_sample_bytes = (
        n_genes * bytes_per_float +           # Y column
        4 * n_features * bytes_per_float      # result arrays
    )
    
    # Estimate batch size
    batch_size = int(batch_bytes / per_sample_bytes)
    batch_size = max(min_batch, min(max_batch, batch_size))
    
    return batch_size


# =============================================================================
# Streaming Result Writer
# =============================================================================

class StreamingResultWriter:
    """
    Stream results directly to HDF5/h5ad file.
    
    Writes results incrementally to avoid keeping full arrays in memory.
    
    Parameters
    ----------
    path : str
        Output file path (.h5 or .h5ad).
    n_features : int
        Number of features (rows in output).
    n_samples : int
        Total number of samples (columns in output).
    feature_names : list, optional
        Feature names for labeling.
    sample_names : list, optional
        Sample names for labeling.
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', or None).
    
    Examples
    --------
    >>> writer = StreamingResultWriter("results.h5ad", n_features=50, n_samples=100000)
    >>> for i, batch_result in enumerate(batch_results):
    ...     writer.write_batch(batch_result, start_col=i * batch_size)
    >>> writer.close()
    """
    
    def __init__(
        self,
        path: str,
        n_features: int,
        n_samples: int,
        feature_names: Optional[list] = None,
        sample_names: Optional[list] = None,
        compression: Optional[str] = "gzip"
    ):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py required for streaming output. Install with: pip install h5py")
        
        self.path = path
        self.n_features = n_features
        self.n_samples = n_samples
        self.compression = compression
        self._closed = False
        
        # Create HDF5 file
        self._file = h5py.File(path, 'w')
        
        # Create datasets
        shape = (n_features, n_samples)
        chunks = (n_features, min(1000, n_samples))  # Chunk by columns
        
        self._datasets = {}
        for name in ['beta', 'se', 'zscore', 'pvalue']:
            self._datasets[name] = self._file.create_dataset(
                name,
                shape=shape,
                dtype='float64',
                chunks=chunks,
                compression=compression
            )
        
        # Store names as attributes
        if feature_names is not None:
            self._file.attrs['feature_names'] = np.array(feature_names, dtype='S')
        if sample_names is not None:
            self._file.attrs['sample_names'] = np.array(sample_names, dtype='S')
        
        self._samples_written = 0
    
    def write_batch(
        self,
        result: Dict[str, np.ndarray],
        start_col: Optional[int] = None
    ) -> None:
        """
        Write a batch of results.
        
        Parameters
        ----------
        result : dict
            Batch result with 'beta', 'se', 'zscore', 'pvalue' arrays.
        start_col : int, optional
            Starting column index. If None, appends after last written.
        """
        if self._closed:
            raise RuntimeError("Writer is closed")
        
        if start_col is None:
            start_col = self._samples_written
        
        batch_size = result['beta'].shape[1]
        end_col = start_col + batch_size
        
        if end_col > self.n_samples:
            raise ValueError(
                f"Batch would exceed dataset size: {end_col} > {self.n_samples}"
            )
        
        for name in ['beta', 'se', 'zscore', 'pvalue']:
            self._datasets[name][:, start_col:end_col] = result[name]
        
        self._samples_written = max(self._samples_written, end_col)
    
    def close(self) -> None:
        """Close the file."""
        if not self._closed:
            self._file.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Core Batch Functions
# =============================================================================

def _compute_T_numpy(X: np.ndarray, lambda_: float) -> np.ndarray:
    """Compute projection matrix T = (X'X + λI)^{-1} X' using NumPy."""
    n_features = X.shape[1]
    
    XtX = X.T @ X
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)
    
    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        warnings.warn("Cholesky failed, using pseudo-inverse")
        XtX_inv = linalg.pinv(XtX_reg)
    
    return XtX_inv @ X.T


def _compute_T_cupy(X_gpu, lambda_: float):
    """Compute projection matrix T on GPU."""
    n_features = X_gpu.shape[1]
    
    XtX = X_gpu.T @ X_gpu
    XtX_reg = XtX + lambda_ * cp.eye(n_features, dtype=cp.float64)
    
    try:
        L = cp.linalg.cholesky(XtX_reg)
        I_gpu = cp.eye(n_features, dtype=cp.float64)
        Z = cp.linalg.solve(L, I_gpu)
        XtX_inv = cp.linalg.solve(L.T, Z)
    except cp.linalg.LinAlgError:
        warnings.warn("GPU Cholesky failed, using pseudo-inverse")
        XtX_inv = cp.linalg.pinv(XtX_reg)
    
    return XtX_inv @ X_gpu.T


def _process_batch_numpy(
    T: np.ndarray,
    Y_batch: np.ndarray,
    perm_table: np.ndarray,
    n_rand: int
) -> Dict[str, np.ndarray]:
    """Process a single batch using NumPy."""
    n_features = T.shape[0]
    batch_size = Y_batch.shape[1]
    
    # Compute beta
    beta = T @ Y_batch
    
    # Permutation testing
    aver = np.zeros((n_features, batch_size), dtype=np.float64)
    aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
    abs_beta = np.abs(beta)
    
    for i in range(n_rand):
        perm_idx = perm_table[i]
        Y_perm = Y_batch[perm_idx, :]
        beta_perm = T @ Y_perm
        
        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2
    
    # Finalize statistics
    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)
    
    return {'beta': beta, 'se': se, 'zscore': zscore, 'pvalue': pvalue}


def _process_batch_cupy(
    T_gpu,
    Y_batch: np.ndarray,
    perm_table: np.ndarray,
    n_rand: int
) -> Dict[str, np.ndarray]:
    """Process a single batch using CuPy."""
    n_features = T_gpu.shape[0]
    batch_size = Y_batch.shape[1]
    
    # Transfer batch to GPU
    Y_gpu = cp.asarray(Y_batch, dtype=cp.float64)
    
    # Compute beta
    beta_gpu = T_gpu @ Y_gpu
    
    # Permutation testing on GPU
    aver = cp.zeros((n_features, batch_size), dtype=cp.float64)
    aver_sq = cp.zeros((n_features, batch_size), dtype=cp.float64)
    pvalue_counts = cp.zeros((n_features, batch_size), dtype=cp.float64)
    abs_beta = cp.abs(beta_gpu)
    
    for i in range(n_rand):
        perm_idx = perm_table[i]
        perm_gpu = cp.asarray(perm_idx, dtype=cp.int32)
        Y_perm = Y_gpu[perm_gpu, :]
        beta_perm = T_gpu @ Y_perm
        
        pvalue_counts += (cp.abs(beta_perm) >= abs_beta).astype(cp.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2
        
        del perm_gpu, Y_perm, beta_perm
    
    # Finalize statistics
    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se_gpu = cp.sqrt(cp.maximum(var, 0.0))
    zscore_gpu = cp.where(se_gpu > EPS, (beta_gpu - mean) / se_gpu, 0.0)
    pvalue_gpu = (pvalue_counts + 1.0) / (n_rand + 1.0)
    
    # Transfer back to CPU
    result = {
        'beta': cp.asnumpy(beta_gpu),
        'se': cp.asnumpy(se_gpu),
        'zscore': cp.asnumpy(zscore_gpu),
        'pvalue': cp.asnumpy(pvalue_gpu)
    }
    
    # Cleanup GPU memory
    del Y_gpu, beta_gpu, aver, aver_sq, pvalue_counts, abs_beta, mean, var
    del se_gpu, zscore_gpu, pvalue_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return result


# =============================================================================
# Main Batch Function
# =============================================================================

def ridge_batch(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float = DEFAULT_LAMBDA,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    batch_size: int = 5000,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    output_path: Optional[str] = None,
    feature_names: Optional[list] = None,
    sample_names: Optional[list] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Ridge regression with batch processing for large datasets.
    
    Computes T = (X'X + λI)^{-1} X' once, then processes Y in batches.
    Optionally streams results directly to disk to handle datasets
    that don't fit in memory.
    
    Parameters
    ----------
    X : ndarray, shape (n_genes, n_features)
        Design matrix (signature). Must be dense.
    Y : ndarray, shape (n_genes, n_samples)
        Response matrix (expression). Can be very large.
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations. Must be > 0 for batch processing.
    seed : int, default=0
        Random seed for permutations.
    batch_size : int, default=5000
        Number of samples per batch.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend.
    output_path : str, optional
        If provided, stream results to this HDF5 file instead of
        returning them in memory.
    feature_names : list, optional
        Feature names for output file.
    sample_names : list, optional
        Sample names for output file.
    progress_callback : callable, optional
        Function called with (batch_idx, n_batches) for progress tracking.
    verbose : bool, default=False
        Print progress information.
    
    Returns
    -------
    dict or None
        If output_path is None, returns results dictionary with:
        - beta, se, zscore, pvalue: ndarrays (n_features, n_samples)
        - method: backend used
        - time: execution time
        - n_batches: number of batches processed
        
        If output_path is provided, returns None (results written to file).
    
    Examples
    --------
    >>> # In-memory processing
    >>> result = ridge_batch(X, Y, batch_size=5000)
    >>> 
    >>> # Stream to disk for very large datasets
    >>> ridge_batch(X, Y, batch_size=10000, output_path="results.h5ad")
    >>> 
    >>> # With progress tracking
    >>> def show_progress(i, n):
    ...     print(f"Batch {i+1}/{n}")
    >>> result = ridge_batch(X, Y, progress_callback=show_progress)
    
    Notes
    -----
    For optimal performance:
    - Use `estimate_batch_size()` to determine appropriate batch_size
    - GPU backend provides significant speedup for large datasets
    - Streaming to disk allows processing datasets larger than RAM
    """
    start_time = time.time()
    
    # --- Input Validation ---
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of rows: {X.shape[0]} vs {Y.shape[0]}")
    if n_rand <= 0:
        raise ValueError("Batch processing requires n_rand > 0. Use ridge() for t-test.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    n_genes, n_features = X.shape
    n_samples = Y.shape[1]
    n_batches = math.ceil(n_samples / batch_size)
    
    if verbose:
        print(f"Ridge batch processing:")
        print(f"  Data: {n_genes} genes, {n_features} features, {n_samples} samples")
        print(f"  Batches: {n_batches} (size={batch_size})")
        mem = estimate_memory(n_genes, n_features, n_samples, n_rand, batch_size)
        print(f"  Estimated memory: {mem['total']:.2f} GB total, {mem['per_batch']:.3f} GB per batch")
    
    # --- Backend Selection ---
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        raise ImportError("CuPy backend requested but not available")
    
    use_gpu = (backend == "cupy")
    
    if verbose:
        print(f"  Backend: {backend}")
    
    # --- Setup Streaming Output ---
    writer = None
    if output_path is not None:
        if verbose:
            print(f"  Output: streaming to {output_path}")
        writer = StreamingResultWriter(
            output_path,
            n_features=n_features,
            n_samples=n_samples,
            feature_names=feature_names,
            sample_names=sample_names
        )
    
    # --- Compute T Matrix (once) ---
    if verbose:
        print("  Computing projection matrix T...")
    
    t_start = time.time()
    if use_gpu:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        T = _compute_T_cupy(X_gpu, lambda_)
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        T = _compute_T_numpy(X, lambda_)
    
    if verbose:
        print(f"  T matrix computed in {time.time() - t_start:.2f}s")
    
    # --- Generate Permutation Table (once) ---
    if verbose:
        print("  Generating permutation table...")
    
    perm_table = generate_permutation_table(n_genes, n_rand, seed)
    
    # --- Process Batches ---
    if verbose:
        print(f"  Processing {n_batches} batches...")
    
    results_list = [] if writer is None else None
    
    for batch_idx in range(n_batches):
        batch_start = time.time()
        
        # Get batch slice
        start_col = batch_idx * batch_size
        end_col = min(start_col + batch_size, n_samples)
        Y_batch = Y[:, start_col:end_col]
        
        # Process batch
        if use_gpu:
            batch_result = _process_batch_cupy(T, Y_batch, perm_table, n_rand)
        else:
            batch_result = _process_batch_numpy(T, Y_batch, perm_table, n_rand)
        
        # Store or write results
        if writer is not None:
            writer.write_batch(batch_result, start_col=start_col)
        else:
            results_list.append(batch_result)
        
        # Progress callback
        if progress_callback is not None:
            progress_callback(batch_idx, n_batches)
        
        if verbose:
            batch_time = time.time() - batch_start
            print(f"    Batch {batch_idx + 1}/{n_batches}: {end_col - start_col} samples in {batch_time:.2f}s")
        
        # Cleanup
        del Y_batch, batch_result
        gc.collect()
    
    # --- Finalize ---
    total_time = time.time() - start_time
    
    # Cleanup
    del T, perm_table
    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    
    if writer is not None:
        writer.close()
        if verbose:
            print(f"  Results written to {output_path}")
            print(f"  Completed in {total_time:.2f}s")
        return None
    
    # Concatenate in-memory results
    if verbose:
        print("  Concatenating results...")
    
    final_result = {
        'beta': np.hstack([r['beta'] for r in results_list]),
        'se': np.hstack([r['se'] for r in results_list]),
        'zscore': np.hstack([r['zscore'] for r in results_list]),
        'pvalue': np.hstack([r['pvalue'] for r in results_list]),
        'method': f"{backend}_batch",
        'time': total_time,
        'n_batches': n_batches
    }
    
    if verbose:
        print(f"  Completed in {total_time:.2f}s")
    
    return final_result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SecActPy Batch Module - Testing")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test parameters
    n_genes = 500
    n_features = 20
    n_samples = 1000
    batch_size = 200
    n_rand = 50
    
    X = np.random.randn(n_genes, n_features)
    Y = np.random.randn(n_genes, n_samples)
    
    print(f"\nTest data: X({n_genes}, {n_features}), Y({n_genes}, {n_samples})")
    print(f"batch_size={batch_size}, n_rand={n_rand}")
    
    # Test 1: Memory estimation
    print("\n1. Testing memory estimation...")
    mem = estimate_memory(n_genes, n_features, n_samples, n_rand, batch_size)
    print(f"   T matrix: {mem['T_matrix']:.4f} GB")
    print(f"   Per batch: {mem['per_batch']:.4f} GB")
    print(f"   Total: {mem['total']:.4f} GB")
    
    # Test 2: Batch size estimation
    print("\n2. Testing batch size estimation...")
    est_batch = estimate_batch_size(n_genes, n_features, available_gb=1.0, n_rand=n_rand)
    print(f"   Estimated batch size for 1GB: {est_batch}")
    
    # Test 3: Basic batch processing
    print("\n3. Testing batch processing (NumPy)...")
    result = ridge_batch(
        X, Y,
        lambda_=5e5,
        n_rand=n_rand,
        seed=0,
        batch_size=batch_size,
        backend='numpy',
        verbose=True
    )
    
    print(f"\n   Results:")
    print(f"   - beta shape: {result['beta'].shape}")
    print(f"   - pvalue range: [{result['pvalue'].min():.4f}, {result['pvalue'].max():.4f}]")
    print(f"   - n_batches: {result['n_batches']}")
    
    # Test 4: Compare with non-batch ridge
    print("\n4. Verifying consistency with standard ridge...")
    from secactpy.ridge import ridge
    result_standard = ridge(X, Y, lambda_=5e5, n_rand=n_rand, seed=0, backend='numpy')
    
    beta_match = np.allclose(result['beta'], result_standard['beta'], rtol=1e-10)
    pval_match = np.allclose(result['pvalue'], result_standard['pvalue'], rtol=1e-10)
    
    if beta_match and pval_match:
        print("   ✓ Batch results match standard ridge exactly")
    else:
        print("   ✗ Results differ!")
        print(f"     Max beta diff: {np.abs(result['beta'] - result_standard['beta']).max()}")
        print(f"     Max pval diff: {np.abs(result['pvalue'] - result_standard['pvalue']).max()}")
    
    # Test 5: Progress callback
    print("\n5. Testing progress callback...")
    progress_calls = []
    def track_progress(i, n):
        progress_calls.append((i, n))
    
    _ = ridge_batch(X, Y, n_rand=n_rand, batch_size=batch_size,
                    progress_callback=track_progress, verbose=False)
    print(f"   Progress callback called {len(progress_calls)} times")
    
    # Test 6: Streaming output (if h5py available)
    print(f"\n6. Testing streaming output (h5py available: {H5PY_AVAILABLE})...")
    if H5PY_AVAILABLE:
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_results.h5")
            
            ridge_batch(
                X, Y,
                n_rand=n_rand,
                seed=0,
                batch_size=batch_size,
                output_path=output_path,
                feature_names=[f"F{i}" for i in range(n_features)],
                sample_names=[f"S{i}" for i in range(n_samples)],
                verbose=True
            )
            
            # Verify file
            with h5py.File(output_path, 'r') as f:
                beta_streamed = f['beta'][:]
                print(f"   Streamed beta shape: {beta_streamed.shape}")
                
                if np.allclose(beta_streamed, result['beta'], rtol=1e-10):
                    print("   ✓ Streamed results match in-memory results")
                else:
                    print("   ✗ Streamed results differ!")
    else:
        print("   Skipped (h5py not installed)")
    
    # Test 7: GPU backend (if available)
    print(f"\n7. Testing GPU backend (CuPy available: {CUPY_AVAILABLE})...")
    if CUPY_AVAILABLE:
        result_gpu = ridge_batch(
            X, Y,
            n_rand=n_rand,
            seed=0,
            batch_size=batch_size,
            backend='cupy',
            verbose=True
        )
        
        if np.allclose(result['beta'], result_gpu['beta'], rtol=1e-10):
            print("   ✓ GPU results match CPU results")
        else:
            print("   ✗ GPU results differ!")
    else:
        print("   Skipped (CuPy not installed)")
    
    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
