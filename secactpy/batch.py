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
from scipy import sparse as sps
from typing import Optional, Literal, Any, Callable, Union
from dataclasses import dataclass
import time
import warnings
import gc
import math

from .rng import (
    GSLRNG,
    get_cached_inverse_perm_table,
)
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
    # Sparse-preserving batch processing
    'PopulationStats',
    'ProjectionComponents',
    'precompute_population_stats',
    'precompute_projection_components',
    'ridge_batch_sparse_preserving',
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
) -> dict[str, float]:
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
        result: dict[str, np.ndarray],
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
    inv_perm_table: np.ndarray,
    n_rand: int
) -> dict[str, np.ndarray]:
    """
    Process a single batch using NumPy with T-column permutation.

    Uses T-column permutation which is mathematically equivalent to Y-row
    permutation but more efficient (Y stays in place):
        T[:, inv_perm] @ Y == T @ Y[perm, :]
    """
    n_features = T.shape[0]
    batch_size = Y_batch.shape[1]

    # Ensure T is contiguous for efficient column indexing
    T = np.ascontiguousarray(T)

    # Compute beta
    beta = T @ Y_batch

    # Permutation testing with T-column permutation
    aver = np.zeros((n_features, batch_size), dtype=np.float64)
    aver_sq = np.zeros((n_features, batch_size), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, batch_size), dtype=np.float64)
    abs_beta = np.abs(beta)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        # Permute columns of T (Y stays in place)
        T_perm = T[:, inv_perm_idx]
        beta_perm = T_perm @ Y_batch

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
    inv_perm_table: np.ndarray,
    n_rand: int
) -> dict[str, np.ndarray]:
    """
    Process a single batch using CuPy with T-column permutation.

    Uses T-column permutation which is mathematically equivalent to Y-row
    permutation but more efficient for GPU (Y stays in place on device):
        T[:, inv_perm] @ Y == T @ Y[perm, :]
    """
    n_features = T_gpu.shape[0]
    batch_size = Y_batch.shape[1]

    # Transfer batch to GPU (stays in place during permutations)
    Y_gpu = cp.asarray(Y_batch, dtype=cp.float64)

    # Compute beta
    beta_gpu = T_gpu @ Y_gpu

    # Permutation testing on GPU with T-column permutation
    aver = cp.zeros((n_features, batch_size), dtype=cp.float64)
    aver_sq = cp.zeros((n_features, batch_size), dtype=cp.float64)
    pvalue_counts = cp.zeros((n_features, batch_size), dtype=cp.float64)
    abs_beta = cp.abs(beta_gpu)

    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]
        inv_perm_gpu = cp.asarray(inv_perm_idx, dtype=cp.intp)
        # Permute columns of T (Y stays in place on GPU)
        T_perm = T_gpu[:, inv_perm_gpu]
        beta_perm = T_perm @ Y_gpu

        pvalue_counts += (cp.abs(beta_perm) >= abs_beta).astype(cp.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

        del inv_perm_gpu, T_perm, beta_perm

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
    use_cache: bool = False,
    output_path: Optional[str] = None,
    feature_names: Optional[list] = None,
    sample_names: Optional[list] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False
) -> Optional[dict[str, Any]]:
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

    # --- Get Inverse Permutation Table for T-column permutation ---
    if verbose:
        print("  Loading inverse permutation table (T-column method)...")

    if use_cache:
        inv_perm_table = get_cached_inverse_perm_table(n_genes, n_rand, seed, verbose=verbose)
    else:
        rng = GSLRNG(seed)
        inv_perm_table = rng.inverse_permutation_table(n_genes, n_rand)

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

        # Process batch with T-column permutation
        if use_gpu:
            batch_result = _process_batch_cupy(T, Y_batch, inv_perm_table, n_rand)
        else:
            batch_result = _process_batch_numpy(T, Y_batch, inv_perm_table, n_rand)

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
    del T, inv_perm_table
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
# Sparse-Preserving Batch Processing
# =============================================================================

@dataclass
class PopulationStats:
    """
    Precomputed population statistics for sparse-preserving inference.

    For large datasets, compute these once from the full Y (or representative
    sample), then reuse for all batch processing.

    Attributes
    ----------
    mu : ndarray, shape (n_samples,)
        Column means of Y.
    sigma : ndarray, shape (n_samples,)
        Column standard deviations of Y (ddof=1).
    mu_over_sigma : ndarray, shape (n_samples,)
        Precomputed μ/σ ratio.
    n_genes : int
        Number of genes (rows of Y).
    """
    mu: np.ndarray
    sigma: np.ndarray
    mu_over_sigma: np.ndarray
    n_genes: int

    @classmethod
    def from_dense(cls, Y: np.ndarray, ddof: int = 1) -> 'PopulationStats':
        """Compute statistics from dense matrix."""
        mu = Y.mean(axis=0)
        sigma = Y.std(axis=0, ddof=ddof)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        mu_over_sigma = mu / sigma
        return cls(mu=mu, sigma=sigma, mu_over_sigma=mu_over_sigma, n_genes=Y.shape[0])

    @classmethod
    def from_sparse(cls, Y: sps.spmatrix, ddof: int = 1) -> 'PopulationStats':
        """Compute statistics from sparse matrix efficiently."""
        n_genes = Y.shape[0]

        # Column sums (sparse-efficient)
        col_sums = np.asarray(Y.sum(axis=0)).ravel()
        mu = col_sums / n_genes

        # Column sum of squares (sparse-efficient)
        Y_sq = Y.multiply(Y)  # Element-wise square, stays sparse
        col_sum_sq = np.asarray(Y_sq.sum(axis=0)).ravel()

        # Variance with ddof
        variance = (col_sum_sq - n_genes * mu**2) / (n_genes - ddof)
        variance = np.maximum(variance, 0)

        sigma = np.sqrt(variance)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        mu_over_sigma = mu / sigma

        return cls(mu=mu, sigma=sigma, mu_over_sigma=mu_over_sigma, n_genes=n_genes)


@dataclass
class ProjectionComponents:
    """
    Precomputed projection matrix components for batch processing.

    Attributes
    ----------
    T : ndarray, shape (n_features, n_genes)
        Projection matrix T = (X'X + λI)^{-1} X'.
    c : ndarray, shape (n_features,)
        Row sums of T, used for correction term: c = T @ ones.
    lambda_ : float
        Ridge regularization parameter.
    n_features : int
        Number of features (proteins/signatures).
    n_genes : int
        Number of genes.
    """
    T: np.ndarray
    c: np.ndarray
    lambda_: float
    n_features: int
    n_genes: int

    def correction_term(self, mu_over_sigma: np.ndarray) -> np.ndarray:
        """
        Compute correction term: outer(c, μ/σ).

        This term is constant across all permutations because neither c
        nor μ/σ depends on the permutation.

        Parameters
        ----------
        mu_over_sigma : ndarray, shape (n_samples,)
            Precomputed μ/σ from population statistics.

        Returns
        -------
        ndarray, shape (n_features, n_samples)
            Correction term to subtract from (T @ Y) / σ.
        """
        return np.outer(self.c, mu_over_sigma)


def precompute_population_stats(
    Y: Union[np.ndarray, sps.spmatrix],
    ddof: int = 1
) -> PopulationStats:
    """
    Precompute population statistics from Y.

    For large datasets, compute these from the full Y once, then reuse
    for all batch processing without modifying Y.

    Parameters
    ----------
    Y : ndarray or sparse, shape (n_genes, n_samples)
        Expression matrix (can be sparse).
    ddof : int, default=1
        Degrees of freedom for std calculation (1 matches R's scale()).

    Returns
    -------
    PopulationStats
        Precomputed statistics (μ, σ, μ/σ).

    Examples
    --------
    >>> stats = precompute_population_stats(Y_sparse)
    >>> print(f"μ range: [{stats.mu.min():.4f}, {stats.mu.max():.4f}]")
    >>> print(f"σ range: [{stats.sigma.min():.4f}, {stats.sigma.max():.4f}]")
    """
    if sps.issparse(Y):
        return PopulationStats.from_sparse(Y, ddof=ddof)
    else:
        return PopulationStats.from_dense(np.asarray(Y), ddof=ddof)


def precompute_projection_components(
    X: np.ndarray,
    lambda_: float = DEFAULT_LAMBDA
) -> ProjectionComponents:
    """
    Precompute projection matrix components from signature matrix.

    Computes T = (X'X + λI)^{-1} X' and c = T @ ones for sparse-preserving
    batch processing.

    Parameters
    ----------
    X : ndarray, shape (n_genes, n_features)
        Signature matrix (genes × features).
    lambda_ : float, default=5e5
        Ridge regularization parameter.

    Returns
    -------
    ProjectionComponents
        Precomputed T matrix and row sums c.

    Examples
    --------
    >>> proj = precompute_projection_components(sig_matrix, lambda_=5e5)
    >>> print(f"T shape: {proj.T.shape}")
    >>> print(f"c shape: {proj.c.shape}")
    """
    X = np.asarray(X, dtype=np.float64)
    n_genes, n_features = X.shape

    # T = (X'X + λI)^{-1} X'
    XtX = X.T @ X
    XtX_reg = XtX + lambda_ * np.eye(n_features, dtype=np.float64)

    try:
        L = linalg.cholesky(XtX_reg, lower=True)
        XtX_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
    except linalg.LinAlgError:
        warnings.warn("Cholesky decomposition failed, using pseudo-inverse")
        XtX_inv = linalg.pinv(XtX_reg)

    T = XtX_inv @ X.T  # (n_features, n_genes)
    T = np.ascontiguousarray(T)  # Ensure contiguous for efficient column indexing

    # c = T @ ones (row sums)
    c = T.sum(axis=1)

    return ProjectionComponents(
        T=T,
        c=c,
        lambda_=lambda_,
        n_features=n_features,
        n_genes=n_genes
    )


def ridge_batch_sparse_preserving(
    proj: ProjectionComponents,
    Y: Union[np.ndarray, sps.spmatrix],
    stats: PopulationStats,
    n_rand: int = DEFAULT_NRAND,
    seed: int = DEFAULT_SEED,
    use_cache: bool = False,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    verbose: bool = False
) -> dict[str, np.ndarray]:
    """
    Sparse-preserving ridge inference for a batch of samples.

    Uses the algebraic reformulation to keep Y sparse throughout:
        beta = (T @ Y) / σ - outer(c, μ/σ)

    The correction term outer(c, μ/σ) is constant across all permutations.

    Parameters
    ----------
    proj : ProjectionComponents
        Precomputed projection matrix from precompute_projection_components().
    Y : ndarray or sparse, shape (n_genes, n_samples)
        Expression batch (can be sparse, NOT scaled).
    stats : PopulationStats
        Population statistics from precompute_population_stats().
    n_rand : int, default=1000
        Number of permutations.
    seed : int, default=0
        Random seed for permutations.
    use_cache : bool, default=True
        Use cached permutation tables from /data/parks34/.cache/ridgesig_perm_tables.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend. "auto" uses GPU if available.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict
        Results with keys: 'beta', 'se', 'zscore', 'pvalue'.
        Each has shape (n_features, n_samples).

    Notes
    -----
    Key insight: The correction term outer(c, μ/σ) is constant across all
    permutations because:
    - c = T @ ones depends only on T
    - μ/σ depends only on Y (population statistics)
    - Neither changes during permutation

    For permuted beta:
        beta_perm = (T[:, inv_perm] @ Y) / σ - correction

    The correction term is the same for all permutations!

    Examples
    --------
    >>> # 1. Precompute statistics (once for full dataset)
    >>> stats = precompute_population_stats(Y_full)
    >>> proj = precompute_projection_components(X, lambda_=5e5)
    >>>
    >>> # 2. Process batches (Y stays sparse)
    >>> for batch_start in range(0, n_samples, batch_size):
    ...     Y_batch = Y_full[:, batch_start:batch_start+batch_size]
    ...     result = ridge_batch_sparse_preserving(proj, Y_batch, stats, backend='cupy')
    """
    # Determine backend
    if backend == "auto":
        backend = "cupy" if CUPY_AVAILABLE else "numpy"
    elif backend == "cupy" and not CUPY_AVAILABLE:
        warnings.warn("CuPy not available, falling back to NumPy")
        backend = "numpy"

    use_gpu = (backend == "cupy")

    T = proj.T
    c = proj.c
    n_features, n_genes = T.shape

    # Validate dimensions
    if Y.shape[0] != n_genes:
        raise ValueError(f"Y rows ({Y.shape[0]}) must match T columns ({n_genes})")

    n_samples = Y.shape[1]

    # Get statistics for this batch
    if len(stats.sigma) >= n_samples:
        sigma = stats.sigma[:n_samples]
        mu_over_sigma = stats.mu_over_sigma[:n_samples]
    else:
        # Stats were computed for smaller batch, need to slice from Y
        sigma = stats.sigma
        mu_over_sigma = stats.mu_over_sigma

    # Convert Y if sparse
    is_sparse = sps.issparse(Y)

    # --- GPU path ---
    if use_gpu:
        return _ridge_batch_sparse_preserving_gpu(
            T, c, Y, sigma, mu_over_sigma, n_rand, seed, use_cache, is_sparse, verbose
        )

    # --- CPU path ---
    # Compute correction term (constant across permutations!)
    correction = np.outer(c, mu_over_sigma)

    if is_sparse:
        Y_csr = Y.tocsr() if not isinstance(Y, sps.csr_matrix) else Y
    else:
        Y_arr = np.asarray(Y, dtype=np.float64)

    # --- Compute observed beta ---
    if verbose:
        print(f"  Computing beta (sparse={is_sparse}, n_samples={n_samples})...")

    if is_sparse:
        # Efficient sparse @ dense: (Y.T @ T.T).T
        beta_raw = (Y_csr.T @ T.T).T
        if sps.issparse(beta_raw):
            beta_raw = beta_raw.toarray()
    else:
        beta_raw = T @ Y_arr

    beta = beta_raw / sigma - correction

    # --- Permutation testing ---
    if n_rand == 0:
        return {
            'beta': beta,
            'se': np.zeros_like(beta),
            'zscore': np.zeros_like(beta),
            'pvalue': np.ones_like(beta)
        }

    if verbose:
        print(f"  Running {n_rand} permutations (T-column method)...")

    # Get inverse permutation table
    if use_cache:
        inv_perm_table = get_cached_inverse_perm_table(
            n_genes, n_rand, seed, verbose=verbose
        )
    else:
        rng = GSLRNG(seed)
        inv_perm_table = rng.inverse_permutation_table(n_genes, n_rand)

    # Accumulators
    aver = np.zeros((n_features, n_samples), dtype=np.float64)
    aver_sq = np.zeros((n_features, n_samples), dtype=np.float64)
    pvalue_counts = np.zeros((n_features, n_samples), dtype=np.float64)
    abs_beta = np.abs(beta)

    # Permutation loop (Y stays unchanged!)
    for i in range(n_rand):
        inv_perm_idx = inv_perm_table[i]

        # T-column permutation
        T_perm = T[:, inv_perm_idx]

        # Compute permuted beta
        if is_sparse:
            beta_raw_perm = (Y_csr.T @ T_perm.T).T
            if sps.issparse(beta_raw_perm):
                beta_raw_perm = beta_raw_perm.toarray()
        else:
            beta_raw_perm = T_perm @ Y_arr

        # Apply scaling (correction is constant!)
        beta_perm = beta_raw_perm / sigma - correction

        # Accumulate statistics
        pvalue_counts += (np.abs(beta_perm) >= abs_beta).astype(np.float64)
        aver += beta_perm
        aver_sq += beta_perm ** 2

    # --- Finalize statistics ---
    if verbose:
        print("  Finalizing statistics...")

    mean = aver / n_rand
    var = (aver_sq / n_rand) - (mean ** 2)
    se = np.sqrt(np.maximum(var, 0.0))
    zscore = np.where(se > EPS, (beta - mean) / se, 0.0)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)

    return {
        'beta': beta,
        'se': se,
        'zscore': zscore,
        'pvalue': pvalue
    }


def _ridge_batch_sparse_preserving_gpu(
    T: np.ndarray,
    c: np.ndarray,
    Y: Union[np.ndarray, sps.spmatrix],
    sigma: np.ndarray,
    mu_over_sigma: np.ndarray,
    n_rand: int,
    seed: int,
    use_cache: bool,
    is_sparse: bool,
    verbose: bool
) -> dict[str, np.ndarray]:
    """GPU implementation of sparse-preserving ridge batch processing."""
    if cp is None:
        raise RuntimeError("CuPy not available")

    n_features, n_genes = T.shape
    n_samples = Y.shape[1]

    if verbose:
        print(f"  Transferring data to GPU...")

    # Transfer to GPU
    T_gpu = cp.asarray(T, dtype=cp.float64)
    c_gpu = cp.asarray(c, dtype=cp.float64)
    sigma_gpu = cp.asarray(sigma, dtype=cp.float64)
    mu_over_sigma_gpu = cp.asarray(mu_over_sigma, dtype=cp.float64)

    # Handle Y (dense on GPU even if sparse on CPU for better performance)
    if is_sparse:
        Y_gpu = cp.asarray(Y.toarray(), dtype=cp.float64)
    else:
        Y_gpu = cp.asarray(Y, dtype=cp.float64)

    # Compute correction term (constant across permutations!)
    correction_gpu = cp.outer(c_gpu, mu_over_sigma_gpu)

    # --- Compute observed beta ---
    if verbose:
        print(f"  Computing beta on GPU (n_samples={n_samples})...")

    beta_raw_gpu = T_gpu @ Y_gpu
    beta_gpu = beta_raw_gpu / sigma_gpu - correction_gpu

    # --- Permutation testing ---
    if n_rand == 0:
        beta = cp.asnumpy(beta_gpu)
        return {
            'beta': beta,
            'se': np.zeros_like(beta),
            'zscore': np.zeros_like(beta),
            'pvalue': np.ones_like(beta)
        }

    if verbose:
        print(f"  Running {n_rand} permutations on GPU (T-column method)...")

    # Get inverse permutation table (on CPU)
    if use_cache:
        inv_perm_table = get_cached_inverse_perm_table(
            n_genes, n_rand, seed, verbose=verbose
        )
    else:
        rng = GSLRNG(seed)
        inv_perm_table = rng.inverse_permutation_table(n_genes, n_rand)

    # Transfer entire permutation table to GPU
    inv_perm_table_gpu = cp.asarray(inv_perm_table, dtype=cp.int32)

    # Accumulators on GPU
    aver_gpu = cp.zeros((n_features, n_samples), dtype=cp.float64)
    aver_sq_gpu = cp.zeros((n_features, n_samples), dtype=cp.float64)
    pvalue_counts_gpu = cp.zeros((n_features, n_samples), dtype=cp.float64)
    abs_beta_gpu = cp.abs(beta_gpu)

    # Permutation loop on GPU
    for i in range(n_rand):
        inv_perm_idx = inv_perm_table_gpu[i]

        # T-column permutation
        T_perm_gpu = T_gpu[:, inv_perm_idx]

        # Compute permuted beta
        beta_raw_perm_gpu = T_perm_gpu @ Y_gpu

        # Apply scaling (correction is constant!)
        beta_perm_gpu = beta_raw_perm_gpu / sigma_gpu - correction_gpu

        # Accumulate statistics
        pvalue_counts_gpu += (cp.abs(beta_perm_gpu) >= abs_beta_gpu).astype(cp.float64)
        aver_gpu += beta_perm_gpu
        aver_sq_gpu += beta_perm_gpu ** 2

    # --- Finalize statistics on GPU ---
    if verbose:
        print("  Finalizing statistics on GPU...")

    mean_gpu = aver_gpu / n_rand
    var_gpu = (aver_sq_gpu / n_rand) - (mean_gpu ** 2)
    se_gpu = cp.sqrt(cp.maximum(var_gpu, 0.0))
    zscore_gpu = cp.where(se_gpu > EPS, (beta_gpu - mean) / se_gpu, 0.0)
    pvalue_gpu = (pvalue_counts_gpu + 1.0) / (n_rand + 1.0)

    # Transfer results to CPU
    if verbose:
        print("  Transferring results to CPU...")

    result = {
        'beta': cp.asnumpy(beta_gpu),
        'se': cp.asnumpy(se_gpu),
        'zscore': cp.asnumpy(zscore_gpu),
        'pvalue': cp.asnumpy(pvalue_gpu)
    }

    # Cleanup GPU memory
    del T_gpu, c_gpu, Y_gpu, sigma_gpu, mu_over_sigma_gpu, correction_gpu
    del beta_raw_gpu, beta_gpu, inv_perm_table_gpu
    del aver_gpu, aver_sq_gpu, pvalue_counts_gpu, abs_beta_gpu
    del mean_gpu, var_gpu, se_gpu, zscore_gpu, pvalue_gpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return result


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
