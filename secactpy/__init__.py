"""
SecActPy: Secreted Protein Activity Inference

A Python package for inferring secreted protein activity from 
gene expression data using ridge regression with permutation testing.

Compatible with R's SecAct/RidgeR package - produces identical results.

Quick Start (Bulk RNA-seq):
---------------------------
    >>> from secactpy import secact_activity_inference
    >>> 
    >>> # From file path (auto-detect format: CSV, TSV, TXT)
    >>> result = secact_activity_inference(
    ...     "diff_expression.csv",  # or .tsv, .txt
    ...     is_differential=True,
    ...     verbose=True
    ... )
    >>> 
    >>> # If genes are in first column (not row names)
    >>> result = secact_activity_inference(
    ...     "data.csv",
    ...     gene_col=0,  # genes in first column
    ...     is_differential=True
    ... )
    >>> 
    >>> # Or from DataFrame
    >>> import pandas as pd
    >>> diff_expr = pd.read_csv("diff_expression.csv", index_col=0)
    >>> result = secact_activity_inference(diff_expr, is_differential=True)
    >>> 
    >>> # Access results
    >>> activity = result['zscore']    # Activity z-scores
    >>> pvalues = result['pvalue']     # Significance

Flexible Data Loading:
----------------------
    >>> from secactpy import load_expression_data
    >>> 
    >>> # Auto-detect format
    >>> expr = load_expression_data("data.csv")
    >>> expr = load_expression_data("data.tsv")
    >>> expr = load_expression_data("data.txt")
    >>> 
    >>> # Genes in first column (not row names)
    >>> expr = load_expression_data("data.csv", gene_col=0)

scRNA-seq Analysis:
-------------------
    >>> import anndata as ad
    >>> from secactpy import secact_activity_inference_scrnaseq
    >>> 
    >>> # Load AnnData (h5ad file)
    >>> adata = ad.read_h5ad("scrnaseq_data.h5ad")
    >>> 
    >>> # Run pseudo-bulk analysis by cell type
    >>> result = secact_activity_inference_scrnaseq(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=False,  # Aggregate by cell type
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × cell_types)

Spatial Transcriptomics:
------------------------
    >>> from secactpy import secact_activity_inference_st, load_visium_10x
    >>> 
    >>> # Load 10X Visium data
    >>> result = secact_activity_inference_st(
    ...     "path/to/visium_folder/",
    ...     min_genes=1000,
    ...     scale_factor=1e5,
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × spots)

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
    secact_activity_inference,
    secact_activity_inference_scrnaseq,
    secact_activity_inference_st,
    load_visium_10x,
    load_expression_data,
    prepare_data,
    scale_columns,
    compute_differential,
    group_signatures,
    expand_rows,
)

# Signature loading
from .signature import (
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
    save_st_results_to_h5ad,
    H5PY_AVAILABLE,
    ANNDATA_AVAILABLE,
)

# RNG (for reproducibility testing)
from .rng import GSLRNG, generate_permutation_table

__all__ = [
    # Main API
    "secact_activity",
    "secact_activity_inference",
    "secact_activity_inference_scrnaseq",
    "secact_activity_inference_st",
    "load_visium_10x",
    "load_expression_data",
    "prepare_data",
    "scale_columns",
    "compute_differential",
    "group_signatures",
    "expand_rows",
    
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
    "save_st_results_to_h5ad",
    
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
