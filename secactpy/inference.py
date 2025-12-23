"""
High-level SecAct activity inference API.

This module provides the main user-facing functions for inferring
secreted protein activity from gene expression data.

Usage:
------
    >>> from secactpy import secact_activity
    >>> 
    >>> # Basic usage with expression DataFrame
    >>> result = secact_activity(expression_df, signature_df)
    >>> 
    >>> # Access results as DataFrames
    >>> activity = result['zscore']  # Activity z-scores
    >>> pvalues = result['pvalue']   # Significance

The main function `secact_activity()` handles:
- Gene overlap detection between expression and signature
- Z-score normalization of expression data
- Ridge regression with permutation testing
- Result formatting with proper row/column labels
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, Dict, Any, Literal
import time

from .ridge import ridge, compute_projection_matrix, CUPY_AVAILABLE

__all__ = [
    'secact_activity',
    'prepare_data',
    'scale_columns',
]


# =============================================================================
# Data Preparation Functions
# =============================================================================

def scale_columns(
    df: pd.DataFrame,
    method: Literal["zscore", "center", "none"] = "zscore",
    epsilon: float = 1e-12
) -> pd.DataFrame:
    """
    Scale DataFrame columns (samples).
    
    Parameters
    ----------
    df : DataFrame
        Data with genes as rows, samples as columns.
    method : {"zscore", "center", "none"}
        - "zscore": Standardize to mean=0, std=1 per column
        - "center": Center to mean=0 per column (no scaling)
        - "none": No transformation
    epsilon : float
        Small value added to std to prevent division by zero.
    
    Returns
    -------
    DataFrame
        Scaled data with same shape and labels.
    """
    if method == "none":
        return df.copy()
    
    values = df.values.astype(np.float64)
    
    # Column-wise centering
    means = np.nanmean(values, axis=0, keepdims=True)
    centered = values - means
    
    if method == "zscore":
        stds = np.nanstd(values, axis=0, keepdims=True)
        # Warn about near-zero std columns
        zero_std_mask = stds.ravel() < epsilon
        if zero_std_mask.any():
            n_zero = zero_std_mask.sum()
            warnings.warn(
                f"{n_zero} column(s) have near-zero variance. "
                "Z-scores for these will be 0.",
                RuntimeWarning
            )
        scaled = centered / (stds + epsilon)
    else:  # center
        scaled = centered
    
    # Handle NaN/Inf
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def prepare_data(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    scale: Literal["zscore", "center", "none"] = "zscore",
    min_genes: int = 10
) -> tuple:
    """
    Prepare expression and signature matrices for ridge regression.
    
    Handles:
    - Finding common genes between expression and signature
    - Aligning matrices to common gene order
    - Optional z-score scaling of expression data
    
    Parameters
    ----------
    expression : DataFrame, shape (n_genes, n_samples)
        Gene expression data. Rows are genes, columns are samples.
    signature : DataFrame, shape (n_genes, n_features)
        Signature matrix. Rows are genes, columns are proteins/features.
    scale : {"zscore", "center", "none"}
        How to scale expression columns.
    min_genes : int
        Minimum number of common genes required.
    
    Returns
    -------
    tuple
        (X, Y, feature_names, sample_names, gene_names)
        - X : ndarray (n_common_genes, n_features) - Signature matrix
        - Y : ndarray (n_common_genes, n_samples) - Expression matrix
        - feature_names : list - Column names from signature
        - sample_names : list - Column names from expression
        - gene_names : list - Common gene names
    
    Raises
    ------
    ValueError
        If fewer than min_genes common genes are found.
    """
    # Ensure string indices for matching
    expr_idx = expression.index.astype(str)
    sig_idx = signature.index.astype(str)
    
    # Find common genes
    common_genes = expr_idx.intersection(sig_idx)
    n_common = len(common_genes)
    
    if n_common < min_genes:
        raise ValueError(
            f"Only {n_common} common genes found between expression and signature. "
            f"Minimum required: {min_genes}. "
            "Check that gene identifiers match (e.g., both use gene symbols)."
        )
    
    if n_common < len(sig_idx):
        pct = 100 * n_common / len(sig_idx)
        warnings.warn(
            f"Using {n_common}/{len(sig_idx)} ({pct:.1f}%) signature genes. "
            f"Missing genes will reduce inference accuracy.",
            RuntimeWarning
        )
    
    # Align to common genes
    # Create temporary DataFrames with string indices
    expr_aligned = expression.copy()
    expr_aligned.index = expr_idx
    sig_aligned = signature.copy()
    sig_aligned.index = sig_idx
    
    # Subset to common genes (in signature order for reproducibility)
    common_genes_ordered = [g for g in sig_idx if g in common_genes]
    Y_df = expr_aligned.loc[common_genes_ordered]
    X_df = sig_aligned.loc[common_genes_ordered]
    
    # Scale expression data
    if scale != "none":
        Y_df = scale_columns(Y_df, method=scale)
    
    # Extract arrays
    X = X_df.values.astype(np.float64)
    Y = Y_df.values.astype(np.float64)
    
    # Store names for result labeling
    feature_names = list(X_df.columns)
    sample_names = list(Y_df.columns)
    gene_names = list(common_genes_ordered)
    
    return X, Y, feature_names, sample_names, gene_names


# =============================================================================
# Main Inference Function
# =============================================================================

def secact_activity(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    scale: Literal["zscore", "center", "none"] = "zscore",
    backend: Literal["auto", "numpy", "cupy"] = "auto",
    min_genes: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Infer secreted protein activity from gene expression data.
    
    This is the main user-facing function that combines data preparation
    and ridge regression into a single convenient call.
    
    Parameters
    ----------
    expression : DataFrame, shape (n_genes, n_samples)
        Gene expression data. Rows are genes (e.g., gene symbols),
        columns are samples.
    signature : DataFrame, shape (n_genes, n_features)
        Signature matrix mapping genes to proteins/cytokines.
        Rows are genes, columns are protein names.
        Can be loaded from built-in signatures (see `load_cytosig()`).
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations for significance testing.
        Set to 0 for faster t-test based inference.
    seed : int, default=0
        Random seed for reproducibility.
        Use 0 for exact compatibility with RidgeR.
    scale : {"zscore", "center", "none"}, default="zscore"
        How to scale expression data before inference.
        - "zscore": Standardize each sample to mean=0, std=1
        - "center": Center each sample to mean=0
        - "none": No scaling
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Computation backend.
    min_genes : int, default=10
        Minimum number of overlapping genes required.
    verbose : bool, default=False
        Print progress information.
    
    Returns
    -------
    dict
        Results dictionary containing:
        
        - **beta** : DataFrame (n_features, n_samples)
            Regression coefficients (activity estimates).
        - **se** : DataFrame (n_features, n_samples)
            Standard errors.
        - **zscore** : DataFrame (n_features, n_samples)
            Z-scores (activity significance).
        - **pvalue** : DataFrame (n_features, n_samples)
            P-values from permutation test (or t-test if n_rand=0).
        - **n_genes** : int
            Number of genes used in inference.
        - **genes** : list
            Gene names used.
        - **method** : str
            Backend used ("numpy" or "cupy").
        - **time** : float
            Total execution time in seconds.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from secactpy import secact_activity
    >>> 
    >>> # Load your data
    >>> expression = pd.read_csv("expression.csv", index_col=0)
    >>> signature = pd.read_csv("signature.csv", index_col=0)
    >>> 
    >>> # Run inference
    >>> result = secact_activity(expression, signature)
    >>> 
    >>> # Get significant activities
    >>> significant = result['pvalue'] < 0.05
    >>> top_activities = result['zscore'][significant].stack().sort_values()
    
    >>> # Quick analysis with t-test (faster, less accurate)
    >>> result_fast = secact_activity(expression, signature, n_rand=0)
    
    Notes
    -----
    The function performs the following steps:
    
    1. Find common genes between expression and signature
    2. Align matrices to common gene order
    3. Optionally z-score normalize expression columns
    4. Run ridge regression: β = (X'X + λI)^{-1} X' Y
    5. Compute significance via permutation testing
    6. Return results as labeled DataFrames
    
    For compatibility with R's SecAct/RidgeR package, use the default
    parameters (lambda_=5e5, n_rand=1000, seed=0, scale="zscore").
    """
    start_time = time.time()
    
    # --- Input Validation ---
    if not isinstance(expression, pd.DataFrame):
        raise TypeError(
            f"expression must be a pandas DataFrame, got {type(expression).__name__}"
        )
    if not isinstance(signature, pd.DataFrame):
        raise TypeError(
            f"signature must be a pandas DataFrame, got {type(signature).__name__}"
        )
    
    if verbose:
        print(f"SecAct Activity Inference")
        print(f"  Expression: {expression.shape[0]} genes × {expression.shape[1]} samples")
        print(f"  Signature: {signature.shape[0]} genes × {signature.shape[1]} features")
    
    # --- Prepare Data ---
    if verbose:
        print("  Preparing data...")
    
    X, Y, feature_names, sample_names, gene_names = prepare_data(
        expression=expression,
        signature=signature,
        scale=scale,
        min_genes=min_genes
    )
    
    n_genes_used = len(gene_names)
    if verbose:
        print(f"  Using {n_genes_used} common genes")
    
    # --- Run Ridge Regression ---
    if verbose:
        print(f"  Running ridge regression (n_rand={n_rand})...")
    
    ridge_result = ridge(
        X=X,
        Y=Y,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        backend=backend,
        verbose=verbose
    )
    
    # --- Format Results as DataFrames ---
    if verbose:
        print("  Formatting results...")
    
    beta_df = pd.DataFrame(
        ridge_result['beta'],
        index=feature_names,
        columns=sample_names
    )
    se_df = pd.DataFrame(
        ridge_result['se'],
        index=feature_names,
        columns=sample_names
    )
    zscore_df = pd.DataFrame(
        ridge_result['zscore'],
        index=feature_names,
        columns=sample_names
    )
    pvalue_df = pd.DataFrame(
        ridge_result['pvalue'],
        index=feature_names,
        columns=sample_names
    )
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"  Completed in {total_time:.2f}s")
    
    # --- Build Result Dictionary ---
    result = {
        # Main results as DataFrames
        'beta': beta_df,
        'se': se_df,
        'zscore': zscore_df,
        'pvalue': pvalue_df,
        
        # Metadata
        'n_genes': n_genes_used,
        'genes': gene_names,
        'features': feature_names,
        'samples': sample_names,
        
        # Execution info
        'method': ridge_result['method'],
        'time': total_time,
        'ridge_time': ridge_result['time'],
        
        # Parameters used
        'params': {
            'lambda_': lambda_,
            'n_rand': n_rand,
            'seed': seed,
            'scale': scale,
            'backend': backend
        }
    }
    
    # Add t-test df if applicable
    if 'df' in ridge_result:
        result['df'] = ridge_result['df']
    
    return result


# =============================================================================
# Differential Expression Helper
# =============================================================================

def compute_differential(
    treatment: pd.DataFrame,
    control: Optional[pd.DataFrame] = None,
    paired: bool = False,
    aggregate: bool = True
) -> pd.DataFrame:
    """
    Compute differential expression profile.
    
    Parameters
    ----------
    treatment : DataFrame
        Treatment expression (genes × samples).
    control : DataFrame, optional
        Control expression (genes × samples).
        If None, centers treatment by row means.
    paired : bool, default=False
        If True and control provided, compute paired differences
        (requires matching column names).
    aggregate : bool, default=True
        If True, average across samples to get single profile.
    
    Returns
    -------
    DataFrame
        Differential expression profile.
    """
    treatment = treatment.copy()
    treatment.index = treatment.index.astype(str)
    
    if control is None:
        # Center by row means
        row_means = treatment.mean(axis=1)
        diff = treatment.subtract(row_means, axis=0)
    else:
        control = control.copy()
        control.index = control.index.astype(str)
        
        # Find common genes
        common_genes = treatment.index.intersection(control.index)
        if len(common_genes) == 0:
            raise ValueError("No common genes between treatment and control")
        
        treatment = treatment.loc[common_genes]
        control = control.loc[common_genes]
        
        if paired:
            # Paired differences (matching samples)
            common_samples = treatment.columns.intersection(control.columns)
            if len(common_samples) == 0:
                raise ValueError("No matching sample names for paired analysis")
            diff = treatment[common_samples] - control[common_samples]
        else:
            # Difference from control mean
            control_mean = control.mean(axis=1)
            diff = treatment.subtract(control_mean, axis=0)
    
    if aggregate:
        diff = pd.DataFrame({'differential': diff.mean(axis=1)})
    
    # Handle NaN
    diff = diff.fillna(0)
    
    return diff


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SecActPy Inference Module - Testing")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    n_genes = 200
    n_features = 15
    n_samples = 10
    
    # Gene names (some shared, some not)
    all_genes = [f"GENE_{i}" for i in range(n_genes + 50)]
    expr_genes = all_genes[:n_genes]
    sig_genes = all_genes[25:n_genes + 25]  # Offset to test overlap
    
    # Create DataFrames
    expression = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=expr_genes,
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )
    signature = pd.DataFrame(
        np.random.randn(len(sig_genes), n_features),
        index=sig_genes,
        columns=[f"Protein_{i}" for i in range(n_features)]
    )
    
    print(f"\nTest data:")
    print(f"  Expression: {expression.shape}")
    print(f"  Signature: {signature.shape}")
    
    # Test 1: Basic inference
    print("\n1. Testing basic inference...")
    result = secact_activity(
        expression, signature,
        n_rand=100, seed=0,
        verbose=True
    )
    
    print(f"\n   Results:")
    print(f"   - beta shape: {result['beta'].shape}")
    print(f"   - pvalue range: [{result['pvalue'].values.min():.4f}, {result['pvalue'].values.max():.4f}]")
    print(f"   - n_genes used: {result['n_genes']}")
    print(f"   - method: {result['method']}")
    
    # Verify DataFrame structure
    assert result['beta'].index.tolist() == result['features']
    assert result['beta'].columns.tolist() == result['samples']
    print("   ✓ DataFrame structure correct")
    
    # Test 2: T-test mode
    print("\n2. Testing t-test mode (n_rand=0)...")
    result_ttest = secact_activity(
        expression, signature,
        n_rand=0, seed=0
    )
    print(f"   - df: {result_ttest.get('df', 'N/A')}")
    print(f"   - time: {result_ttest['time']:.3f}s (vs {result['time']:.3f}s for permutation)")
    print("   ✓ T-test mode works")
    
    # Test 3: Scaling options
    print("\n3. Testing scaling options...")
    for scale in ["zscore", "center", "none"]:
        r = secact_activity(expression, signature, n_rand=50, scale=scale)
        print(f"   - scale='{scale}': beta mean={r['beta'].values.mean():.6f}")
    print("   ✓ All scaling options work")
    
    # Test 4: Reproducibility
    print("\n4. Testing reproducibility...")
    r1 = secact_activity(expression, signature, n_rand=100, seed=0)
    r2 = secact_activity(expression, signature, n_rand=100, seed=0)
    
    if np.allclose(r1['beta'].values, r2['beta'].values) and \
       np.allclose(r1['pvalue'].values, r2['pvalue'].values):
        print("   ✓ Results reproducible with same seed")
    else:
        print("   ✗ Results not reproducible!")
    
    # Test 5: Differential expression helper
    print("\n5. Testing differential expression helper...")
    control = pd.DataFrame(
        np.random.randn(n_genes, 5),
        index=expr_genes,
        columns=[f"Ctrl_{i}" for i in range(5)]
    )
    
    diff = compute_differential(expression, control, aggregate=True)
    print(f"   - Differential shape: {diff.shape}")
    
    diff_paired = compute_differential(
        expression.iloc[:, :5].rename(columns=lambda x: x.replace("Sample", "Pair")),
        control.rename(columns=lambda x: x.replace("Ctrl", "Pair")),
        paired=True, aggregate=False
    )
    print(f"   - Paired differential shape: {diff_paired.shape}")
    print("   ✓ Differential expression helper works")
    
    # Test 6: Edge case - few genes
    print("\n6. Testing edge case (few common genes)...")
    small_sig = signature.iloc[:15]  # Only 15 genes
    try:
        r_small = secact_activity(expression, small_sig, n_rand=50, min_genes=10)
        print(f"   - Used {r_small['n_genes']} genes")
        print("   ✓ Works with few genes")
    except ValueError as e:
        print(f"   - Correctly raised error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete.")
    print("=" * 60)
