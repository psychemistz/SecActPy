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
    'secact_activity_inference',
    'prepare_data',
    'scale_columns',
    'compute_differential',
    'group_signatures',
    'expand_rows',
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
# Signature Grouping (matching R's SecAct.activity.inference)
# =============================================================================

def group_signatures(
    X: pd.DataFrame,
    cor_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Group similar signatures by Pearson correlation.
    
    Matches R's .group_signatures function from RidgeR:
    - Calculate correlation-based distance
    - Hierarchical clustering (complete linkage)
    - Cut tree at 1 - cor_threshold
    - Average signatures within groups
    - Name groups as "A|B|C"
    
    Parameters
    ----------
    X : DataFrame
        Signature matrix (genes × proteins)
    cor_threshold : float, default=0.9
        Correlation threshold for grouping.
        Signatures with correlation >= threshold are grouped together.
    
    Returns
    -------
    DataFrame
        Grouped signature matrix with averaged values and pipe-delimited names.
    
    Examples
    --------
    >>> sig = load_signature('secact')
    >>> grouped = group_signatures(sig, cor_threshold=0.9)
    >>> print(f"Reduced from {sig.shape[1]} to {grouped.shape[1]} groups")
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    n_features = X.shape[1]
    
    # Handle edge case: single column
    if n_features <= 1:
        return X.copy()
    
    # Calculate correlation distance (1 - correlation)
    # pdist expects samples as rows, so we transpose X
    # metric='correlation' computes 1 - pearson_correlation
    try:
        dist_condensed = pdist(X.T.values, metric='correlation')
        # Handle NaN in distances (from constant columns)
        dist_condensed = np.nan_to_num(dist_condensed, nan=1.0)
    except Exception:
        # Fallback: compute manually
        corr_matrix = X.corr(method='pearson')
        corr_matrix = corr_matrix.fillna(0).clip(-1, 1)
        n = len(corr_matrix)
        dist_condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_condensed.append(1 - corr_matrix.iloc[i, j])
        dist_condensed = np.array(dist_condensed)
    
    # Hierarchical clustering with complete linkage (matching R)
    Z = linkage(dist_condensed, method='complete')
    
    # Cut tree at distance = 1 - cor_threshold
    cut_height = 1 - cor_threshold
    group_labels = fcluster(Z, t=cut_height, criterion='distance')
    
    # Create mapping from protein to group
    protein_names = list(X.columns)
    protein_groups = dict(zip(protein_names, group_labels))
    
    # Build new signature matrix with grouped signatures
    # Use list to collect columns, then concat at once (avoiding fragmentation warning)
    group_data = {}
    
    for group_id in sorted(set(group_labels)):
        # Get proteins in this group (sorted for consistent naming)
        proteins_in_group = sorted([p for p, g in protein_groups.items() if g == group_id])
        
        # Create group name (e.g., "A|B|C")
        group_name = "|".join(proteins_in_group)
        
        # Average signatures within group
        group_data[group_name] = X[proteins_in_group].mean(axis=1)
    
    new_sig = pd.DataFrame(group_data, index=X.index)
    
    return new_sig


def expand_rows(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows with pipe-delimited names.
    
    Matches R's .expand_rows function from RidgeR:
    Expands rows where index contains "|" (grouped signatures) into
    separate rows with duplicated values.
    
    Parameters
    ----------
    mat : DataFrame
        Matrix with potentially grouped row names (e.g., "A|B|C")
    
    Returns
    -------
    DataFrame
        Matrix with expanded rows
    
    Examples
    --------
    >>> # If mat has row "IL6|IL6R" with values [1, 2]
    >>> # Result will have two rows: "IL6" with [1, 2] and "IL6R" with [1, 2]
    >>> expanded = expand_rows(mat)
    """
    new_rows = []
    new_names = []
    
    for idx in mat.index:
        idx_str = str(idx)
        if "|" in idx_str:
            # Split the grouped name
            split_names = idx_str.split("|")
            for name in split_names:
                new_rows.append(mat.loc[idx].values)
                new_names.append(name)
        else:
            new_rows.append(mat.loc[idx].values)
            new_names.append(idx_str)
    
    result = pd.DataFrame(new_rows, index=new_names, columns=mat.columns)
    return result


# =============================================================================
# Full Inference Function (matching R's SecAct.activity.inference)
# =============================================================================

def secact_activity_inference(
    input_profile: pd.DataFrame,
    input_profile_control: pd.DataFrame = None,
    is_differential: bool = False,
    is_paired: bool = False,
    is_single_sample_level: bool = False,
    sig_matrix: str = "secact",
    is_group_sig: bool = True,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    sig_filter: bool = False,
    backend: str = "numpy",
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Secreted Protein Activity Inference (matching R's SecAct.activity.inference).
    
    This function provides full compatibility with the R RidgeR package,
    including signature grouping and row expansion.
    
    Parameters
    ----------
    input_profile : DataFrame
        Gene expression matrix (genes × samples).
    input_profile_control : DataFrame, optional
        Control expression matrix (genes × samples).
        If None and is_differential=False, uses mean of input_profile as control.
    is_differential : bool, default=False
        If True, input_profile is already differential expression.
    is_paired : bool, default=False
        If True, perform paired differential calculation.
    is_single_sample_level : bool, default=False
        If True, calculate per-sample activity.
    sig_matrix : str or DataFrame, default="secact"
        Signature matrix. Either "secact", "cytosig", path to file, or DataFrame.
    is_group_sig : bool, default=True
        Whether to group similar signatures by correlation.
    is_group_cor : float, default=0.9
        Correlation threshold for grouping.
    lambda_ : float, default=5e5
        Ridge regularization parameter.
    n_rand : int, default=1000
        Number of permutations.
    seed : int, default=0
        Random seed for reproducibility.
    sig_filter : bool, default=False
        If True, filter signatures by available genes.
    backend : str, default="numpy"
        Computation backend ("numpy" or "cupy").
    verbose : bool, default=True
        Print progress information.
    
    Returns
    -------
    dict
        Results dictionary containing:
        - beta : DataFrame (proteins × samples) - Regression coefficients
        - se : DataFrame (proteins × samples) - Standard errors
        - zscore : DataFrame (proteins × samples) - Z-scores
        - pvalue : DataFrame (proteins × samples) - P-values
    
    Examples
    --------
    >>> # Load differential expression data
    >>> expr_diff = pd.read_csv("Ly86-Fc_vs_Vehicle_logFC.txt", sep="\\t", index_col=0)
    >>> 
    >>> # Run inference
    >>> result = secact_activity_inference(expr_diff, is_differential=True)
    >>> 
    >>> # Access results
    >>> print(result['zscore'].head())
    """
    from .signature import load_signature
    
    # --- Step 1: Load signature matrix ---
    if isinstance(sig_matrix, pd.DataFrame):
        X = sig_matrix.copy()
    elif isinstance(sig_matrix, str):
        if sig_matrix.lower() in ["secact", "cytosig"]:
            X = load_signature(sig_matrix)
        else:
            # Assume it's a file path
            X = pd.read_csv(sig_matrix, sep='\t', index_col=0)
    else:
        raise ValueError("sig_matrix must be 'secact', 'cytosig', a file path, or a DataFrame")
    
    if verbose:
        print(f"  Loaded signature: {X.shape[0]} genes × {X.shape[1]} proteins")
    
    # --- Step 2: Compute differential expression if needed ---
    if is_differential:
        Y = input_profile.copy()
        if Y.shape[1] == 1 and Y.columns[0] != "Change":
            Y.columns = ["Change"]
    else:
        if input_profile_control is None:
            # Center by row means
            row_means = input_profile.mean(axis=1)
            Y = input_profile.subtract(row_means, axis=0)
        else:
            if is_paired:
                # Paired differences
                common_samples = input_profile.columns.intersection(input_profile_control.columns)
                Y = input_profile[common_samples] - input_profile_control[common_samples]
            else:
                # Difference from control mean
                control_mean = input_profile_control.mean(axis=1)
                Y = input_profile.subtract(control_mean, axis=0)
            
            if not is_single_sample_level:
                # Aggregate to single column
                Y = pd.DataFrame({"Change": Y.mean(axis=1)})
    
    # --- Step 3: Filter signatures if requested ---
    if sig_filter:
        available_genes = set(Y.index)
        X = X.loc[:, X.columns.isin(available_genes)]
    
    # --- Step 4: Group similar signatures if requested ---
    if is_group_sig:
        if verbose:
            print(f"  Grouping signatures (cor_threshold={is_group_cor})...")
        X = group_signatures(X, cor_threshold=is_group_cor)
        if verbose:
            print(f"  Grouped into {X.shape[1]} signature groups")
    
    # --- Step 5: Find overlapping genes ---
    common_genes = Y.index.intersection(X.index)
    
    if verbose:
        print(f"  Common genes: {len(common_genes)}")
    
    if len(common_genes) < 2:
        raise ValueError(
            f"Too few overlapping genes ({len(common_genes)}) between expression and signature matrices! "
            "Check that gene identifiers match (e.g., both use gene symbols)."
        )
    
    # --- Step 6: Subset to common genes ---
    X_aligned = X.loc[common_genes].astype(np.float64)
    Y_aligned = Y.loc[common_genes].astype(np.float64)
    
    # --- Step 7: Scale (z-score normalize columns) ---
    # R's scale() function: (x - mean) / sd
    X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=0)
    Y_scaled = (Y_aligned - Y_aligned.mean()) / Y_aligned.std(ddof=0)
    
    # --- Step 8: Replace NaN with 0 (from constant columns) ---
    X_scaled = X_scaled.fillna(0)
    Y_scaled = Y_scaled.fillna(0)
    
    # --- Step 9: Run ridge regression ---
    if verbose:
        print(f"  Running ridge regression (n_rand={n_rand})...")
    
    result = ridge(
        X=X_scaled.values,
        Y=Y_scaled.values,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        backend=backend,
        verbose=False
    )
    
    # --- Step 10: Create DataFrames with proper labels ---
    feature_names = X_scaled.columns.tolist()
    sample_names = Y_scaled.columns.tolist()
    
    beta_df = pd.DataFrame(result['beta'], index=feature_names, columns=sample_names)
    se_df = pd.DataFrame(result['se'], index=feature_names, columns=sample_names)
    zscore_df = pd.DataFrame(result['zscore'], index=feature_names, columns=sample_names)
    pvalue_df = pd.DataFrame(result['pvalue'], index=feature_names, columns=sample_names)
    
    # --- Step 11: Expand grouped signatures back to individual rows ---
    if is_group_sig:
        if verbose:
            print("  Expanding grouped signatures...")
        beta_df = expand_rows(beta_df)
        se_df = expand_rows(se_df)
        zscore_df = expand_rows(zscore_df)
        pvalue_df = expand_rows(pvalue_df)
        
        # Sort by row name (matching R's behavior)
        row_order = sorted(beta_df.index)
        beta_df = beta_df.loc[row_order]
        se_df = se_df.loc[row_order]
        zscore_df = zscore_df.loc[row_order]
        pvalue_df = pvalue_df.loc[row_order]
    
    if verbose:
        print(f"  Result shape: {beta_df.shape}")
    
    return {
        'beta': beta_df,
        'se': se_df,
        'zscore': zscore_df,
        'pvalue': pvalue_df
    }


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
