"""
SecActPy: Secreted Protein Activity Inference

A Python package for inferring secreted protein activity from 
gene expression data using ridge regression with permutation testing.

Compatible with R's SecAct/RidgeR package - produces identical results.

Quick Start:
------------
    >>> import pandas as pd
    >>> from secactpy import secact_activity, load_signature
    >>> 
    >>> # Load expression data and signature
    >>> expression = pd.read_csv("expression.csv", index_col=0)
    >>> signature = load_signature()  # Default: SecAct
    >>> 
    >>> # Run inference
    >>> result = secact_activity(expression, signature)
    >>> 
    >>> # Access results as DataFrames
    >>> activity = result['zscore']    # Activity z-scores
    >>> pvalues = result['pvalue']     # Significance
    >>> 
    >>> # Find significant activities
    >>> significant = result['pvalue'] < 0.05

For large datasets (>100k samples):
-----------------------------------
    >>> from secactpy import ridge_batch, estimate_batch_size
    >>> 
    >>> # Estimate optimal batch size
    >>> batch_size = estimate_batch_size(n_genes=20000, n_features=50)
    >>> 
    >>> # Run batch processing with streaming output
    >>> ridge_batch(X, Y, batch_size=batch_size, output_path="results.h5ad")
    >>> 
    >>> # Load results
    >>> from secactpy import load_results, results_to_anndata
    >>> results = load_results("results.h5ad")
    >>> adata = results_to_anndata(results)  # For scanpy integration
"""

__version__ = "0.1.0"

# High-level API (most users need only these)
from .inference import (
    secact_activity,
    prepare_data,
    scale_columns,
    compute_differential,
)

# Signature loading
from .signatures import (
    load_signature,
    load_secact,
    load_cytosig,
    list_signatures,
    get_signature_info,
    AVAILABLE_SIGNATURES,
)

# Lower-level ridge functions (for advanced users)
from .ridge import (
    ridge,
    compute_projection_matrix,
    ridge_with_precomputed_T,
    CUPY_AVAILABLE,
    DEFAULT_LAMBDA,
    DEFAULT_NRAND,
    DEFAULT_SEED,
)

# Batch processing for large datasets
from .batch import (
    ridge_batch,
    estimate_batch_size,
    estimate_memory,
    StreamingResultWriter,
)

# I/O utilities
from .io import (
    load_results,
    save_results,
    results_to_anndata,
    load_as_anndata,
    results_to_dataframes,
    get_file_info,
    concatenate_results,
    H5PY_AVAILABLE,
    ANNDATA_AVAILABLE,
)

# RNG (for reproducibility testing)
from .rng import GSLRNG, generate_permutation_table

__all__ = [
    # Main API
    "secact_activity",
    "prepare_data",
    "scale_columns",
    "compute_differential",
    
    # Signature loading
    "load_signature",
    "load_secact",
    "load_cytosig",
    "list_signatures",
    "get_signature_info",
    "AVAILABLE_SIGNATURES",
    
    # Ridge functions
    "ridge",
    "compute_projection_matrix",
    "ridge_with_precomputed_T",
    
    # Batch processing
    "ridge_batch",
    "estimate_batch_size",
    "estimate_memory",
    "StreamingResultWriter",
    
    # I/O
    "load_results",
    "save_results",
    "results_to_anndata",
    "load_as_anndata",
    "results_to_dataframes",
    "get_file_info",
    "concatenate_results",
    
    # RNG
    "GSLRNG",
    "generate_permutation_table",
    
    # Availability flags
    "CUPY_AVAILABLE",
    "H5PY_AVAILABLE",
    "ANNDATA_AVAILABLE",
    
    # Constants
    "DEFAULT_LAMBDA",
    "DEFAULT_NRAND",
    "DEFAULT_SEED",
    
    # Version
    "__version__",
]
