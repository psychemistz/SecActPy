"""
SecActPy utility functions.

This module provides core utilities for:
- Sparse matrix operations
- Gene symbol handling
- Survival analysis
"""

from .sparse import (
    sweep_sparse,
    normalize_sparse,
    sparse_column_scale,
    sparse_log1p,
    sparse_row_vars,
    sparse_col_vars,
)

from .genes import (
    transfer_symbol,
    rm_duplicates,
    expand_rows,
    scalar1,
    filter_genes,
    match_genes,
)

from .survival import (
    coxph_best_separation,
    survival_analysis,
    kaplan_meier_plot,
    log_rank_test,
)

__all__ = [
    # Sparse utilities
    "sweep_sparse",
    "normalize_sparse",
    "sparse_column_scale",
    "sparse_log1p",
    "sparse_row_vars",
    "sparse_col_vars",
    # Gene utilities
    "transfer_symbol",
    "rm_duplicates",
    "expand_rows",
    "scalar1",
    "filter_genes",
    "match_genes",
    # Survival utilities
    "coxph_best_separation",
    "survival_analysis",
    "kaplan_meier_plot",
    "log_rank_test",
]
